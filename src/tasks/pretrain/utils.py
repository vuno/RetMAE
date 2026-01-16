import math
from typing import Dict, List, Optional

import einops
import matplotlib.cm as cm
import numpy as np
import torch
import torchvision.transforms.v2 as T
from aim.sdk.objects.image import Image as AimImage
from lightning import pytorch as pl
from sklearn.decomposition import PCA
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn import functional as F


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split images into non-overlapped patches.

    Args:
        imgs (torch.Tensor): A batch of images. The shape should
            be :math:`(B, C, H, W)`.

    Returns:
        torch.Tensor: Patchified images. The shape is
        :math:`(B, L, \text{patch_size}^2 \times C)`.
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = einops.rearrange(x, "n c h p w q -> n h w p q c")
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
    return x


def unpatchify(
    x: torch.Tensor,
    patch_size: int,
    in_channels: int = 3,
) -> torch.Tensor:
    """Combine non-overlapped patches into images.

    Args:
        x (torch.Tensor): The shape is
            :math:`(B, L, \text{patch_size}^2 \times C)`.
        patch_size (int): The size of the patches.
        in_channels (int, optional): The number of channels in the input tensor. Defaults to 3.

    Returns:
        torch.Tensor: The shape is :math:`(B, C, H, W)`.
    """
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, in_channels))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], in_channels, h * p, h * p))
    return imgs


def min_max_scale_1d(x: torch.Tensor) -> torch.Tensor:
    """Apply min-max scaling to tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Scaled tensor
    """
    assert x.ndim < 4 and x.ndim > 1, f"x dim should be (B, *, C), but got {x.shape}"
    x_min = x.min(dim=-1, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min + 1e-6)


def min_max_scale_2d(x: torch.Tensor) -> torch.Tensor:
    """Apply min-max scaling to tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Scaled tensor
    """
    assert x.ndim == 4, f"x dim should be (B, C, H, W), but got {x.shape}"
    x_flat = x.reshape(x.shape[0], x.shape[1] * x.shape[-2] * x.shape[-1])
    x_flat = min_max_scale_1d(x_flat)
    return x_flat.reshape(x.shape)


def destandardize_images(
    images: torch.Tensor,
    input_standardization_mean: torch.Tensor,
    input_standardization_std: torch.Tensor,
) -> torch.Tensor:
    """Destandardize images using mean and std from outputs.

    Args:
        images (torch.Tensor): Input images
        input_standardization_mean (torch.Tensor): Mean of the input images
        input_standardization_std (torch.Tensor): Standard deviation of the input images

    Returns:
        torch.Tensor: Destandardized images
    """
    assert images.ndim == 4, f"image dim should be (B, C, H, W), but got {images.shape}"
    img_flat = images.reshape(images.shape[0], images.shape[1], images.shape[2] * images.shape[3])
    denorm = (
        img_flat.cpu() * input_standardization_std[None, :, None]
        + input_standardization_mean[None, :, None]
    )
    return denorm.reshape(images.shape)


def resize_attention_map(attn: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """Resize attention map to match target size.

    Args:
        attn (torch.Tensor): Attention map
        target_size (tuple): Target size (H, W)

    Returns:
        torch.Tensor: Resized attention map
    """
    # attn: (B, HD, N, HW)
    B, HD, HW = attn.shape

    # Reshape to (B*HD*N, HW)
    attn_flat = attn.reshape(-1, HW)

    # Unpatchify to (B*HD*N, 1, H, W)
    h = w = int(math.sqrt(HW))
    attn_unpatchified = unpatchify(attn_flat, 1, 1)

    # Reshape back to (B, HD, H, W)
    attn_unpatchified = attn_unpatchified.reshape(B, HD, h, w)

    # Expand height and width dimensions
    H, W = target_size
    return attn_unpatchified.repeat_interleave(H // h, dim=2).repeat_interleave(W // w, dim=3)


def combine_images_and_heatmaps(
    images: torch.Tensor,
    heatmaps: torch.Tensor,
    margin: torch.Tensor,
) -> torch.Tensor:
    """Combine images and attention heatmaps.

    Args:
        images (torch.Tensor): Input images
        heatmaps (torch.Tensor): Attention maps
        margin (torch.Tensor): Margin
    Returns:
        torch.Tensor: Combined visualization
    """
    if images.max() > 1:
        images = images / 255

    concat_list = []
    for img, heatmap in zip(images, heatmaps):
        HD, H, W = heatmap.shape
        nh_heatmap = []
        for hd in range(HD):
            heatmap_per_hd = min_max_scale_1d(heatmap[hd].reshape(1, H * W)).reshape(H, W)
            heatmap_per_hd = convert_attn_to_heatmap(heatmap_per_hd)
            nh_heatmap.append(heatmap_per_hd)  # (3, H, W)

        max_heatmap = min_max_scale_1d(heatmap.max(dim=0)[0].reshape(1, H * W)).reshape(H, W)
        max_heatmap = convert_attn_to_heatmap(max_heatmap)
        mean_heatmap = min_max_scale_1d(heatmap.mean(dim=0).reshape(1, H * W)).reshape(H, W)
        mean_heatmap = convert_attn_to_heatmap(mean_heatmap)

        # Concatenate heatmaps horizontally
        heatmap = torch.cat(
            nh_heatmap + [margin, max_heatmap, margin, mean_heatmap], dim=2
        )  # (3, H, W*HD)

        # Concatenate image and heatmap horizontally
        img_heatmap = torch.cat([img.to("cpu"), margin, heatmap], dim=2)  # (3, H, W+W*HD)
        concat_list.append(img_heatmap)

    return torch.stack(concat_list, dim=0)  # (B, 3, H, W+W*HD)


def convert_attn_to_heatmap(attn: torch.Tensor, colormap: str = "viridis") -> torch.Tensor:
    """Convert attention values to RGB heatmap.

    Args:
        attn (torch.Tensor): Attention values
        colormap (str, optional): Matplotlib colormap name. Defaults to "viridis".

    Returns:
        torch.Tensor: RGB heatmap
    """
    numpy_array = attn.cpu().numpy()
    colored_array = cm.get_cmap(colormap)(numpy_array)
    return torch.from_numpy(colored_array[:, :, :3]).permute(2, 0, 1)


def apply_mask(
    out: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]
) -> np.ndarray:
    """Applies mask to the image.

    Args:
        out (np.ndarray): Input image array
        mask (np.ndarray): Mask array to apply
        color (tuple[float, float, float]): RGB color tuple

    Returns:
        np.ndarray: Image array with mask applied
    """
    t = 0.2
    mi = np.min(mask)
    ma = np.max(mask)
    mask = (mask - mi) / (ma - mi + 1e-6)

    for c in range(3):
        out[:, :, c] += np.sqrt(mask) * (mask > t) * color[c]

    return out


def head_colormap_attn(
    attn: torch.Tensor,
    patch_size: int = 14,
    attn_th: float = 0.6,
) -> np.ndarray:
    """Head-wise colormap attention map.

    Args:
        attn (torch.Tensor): Attention tensor (B, H, L)
        patch_size (int, optional): Patch size. Defaults to 14.
        attn_th (float, optional): Attention threshold. Defaults to 0.6.

    Returns:
        np.ndarray: Head-wise colormap attention map
    """
    assert attn.ndim == 3, "attn must be 3D tensor (B, H, L)"

    B, D, L = attn.shape
    H = int(np.sqrt(L))
    attn_reg_reshape = attn.reshape(B, D, H, H).detach()
    base_colors = [
        (255, 0, 0),  # Red
        (0, 0, 255),  # Blue
        (0, 255, 0),  # Green
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (255, 0, 128),  # Hot pink
        (0, 255, 128),  # Mint
        (128, 255, 0),  # Lime
        (0, 128, 255),  # Sky blue
    ]

    colors = []
    for i in range(D):
        color_idx = i % len(base_colors)
        colors.append(base_colors[color_idx])
    colors = [(float(c[0]) / 255, float(c[1]) / 255, float(c[2]) / 255) for c in colors]

    outputs = []
    for reg_i in range(B):
        _attn = attn[reg_i]
        val, idx = torch.sort(_attn)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - attn_th)
        idx2 = torch.argsort(idx)
        for head in range(D):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(D, H, H).float()
        attn_proc = attn_reg_reshape[reg_i]
        for head in range(D):
            m = attn_proc[head]
            m *= th_attn[head]
            attn_proc[head] = m

        mask = np.stack([attn_proc[head] for head in range(D)])

        for head in range(D):
            mask[head] *= mask[head] == np.amax(mask, axis=0)
        a = np.cumsum(mask, axis=0)
        for head in range(D):
            mask[head] *= mask[head] == a[head]

        out = np.zeros((H, H, 3))
        for head in range(D):
            color = colors[head]
            _mask = mask[head]
            out = apply_mask(out, _mask, color)

        outputs.append(out.transpose(2, 0, 1)[np.newaxis, ...])

    return np.concatenate(outputs, axis=0)


def compute_pca_features(
    patch_tokens: torch.Tensor,
    fg_mask: Optional[torch.Tensor] = None,
    n_components: int = 3,
) -> torch.Tensor:
    """Compute PCA features and convert them to RGB image for visualization

    Args:
        patch_tokens (torch.Tensor): Patch tokens from a single image
        th (float): Threshold for foreground/background separation
        fg_mask (Optional[torch.Tensor]): Field of view mask

    Returns:
        torch.Tensor: RGB image containing PCA features
    """
    assert patch_tokens.ndim == 2

    pca = PCA(n_components=n_components)

    # Convert patch tokens to numpy array
    tokens = patch_tokens.cpu().detach().numpy()

    # First PCA - Separate foreground/background
    pca.fit(tokens)
    feats = pca.transform(tokens)

    if fg_mask is not None:
        # Second PCA - Extract foreground features
        mask_1d = einops.rearrange(fg_mask, "b h w -> (b h w)")
        mask_1d = mask_1d.bool().cpu().numpy()  # shape: (N_patches,)
        masked_tokens = tokens[mask_1d]  # shape: (N_fg, D)
        pca2 = PCA(n_components=n_components)
        pca2.fit(masked_tokens)
        fg_feats = pca2.transform(masked_tokens)  # shape: (N_fg, 3)
        fg_feats = (fg_feats - fg_feats.min(axis=0)) / (
            fg_feats.max(axis=0) - fg_feats.min(axis=0) + 1e-6
        )
        rgb_feats = np.zeros_like(feats)  # shape: (N_patches, 3)
        rgb_feats[mask_1d] = fg_feats

    else:
        feats = (feats - feats.min(axis=0)) / (feats.max(axis=0) - feats.min(axis=0) + 1e-6)
        rgb_feats = feats

    return torch.from_numpy(rgb_feats).to(patch_tokens.device)


def step_scheduler(
    epochs: int,
    inactive_epochs: int,
    warmup_epochs: int = 0,
    base_value: float = 0.0,
) -> np.ndarray:
    """Generate a step learning rate scheduler with exactly 'inactive_epochs' 0 values.

    Args:
        base_value (float): Value for inactive epochs
        epochs (int): Total number of epochs
        inactive_epochs (int): Number of inactive epochs
        warmup_epochs (int, optional): Number of warmup epochs. Defaults to 0
    """
    inactive_schedule = np.full(inactive_epochs, base_value)
    warmup_schedule = np.linspace(base_value, 1, warmup_epochs + 1, endpoint=False)[1:]
    active_schedule = np.ones(epochs - inactive_epochs - warmup_epochs + 1)
    schedule = np.concatenate([inactive_schedule, warmup_schedule, active_schedule])
    return schedule


def tanh_increase_scheduler(epochs: int, scale: int = 100) -> np.ndarray:
    """Generate a hyperbolic tangent increasing scheduler.

    Args:
        epochs (int): Total number of epochs
        scale (int, optional): Scaling factor for the rate of increase. Defaults to 100.

    Returns:
        np.ndarray: Schedule array containing values that increase according to tanh curve
    """
    return np.tanh(np.linspace(0, epochs, epochs) / scale)


def warmup_scheduler(
    base_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0,
) -> np.ndarray:
    """Generate a warmup learning rate scheduler.

    Args:
        base_value (float): Initial learning rate value
        epochs (int): Total number of epochs
        niter_per_ep (int): Number of iterations per epoch
        warmup_epochs (int, optional): Number of warmup epochs. Defaults to 0
        start_warmup_value (float, optional): Starting warmup value. Defaults to 0

    Returns:
        np.ndarray: Schedule array containing learning rates for each step
    """
    # Initialize warmup schedule
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters + 1)
    schedule = np.array([base_value for _ in iters])

    # Combine warmup and cosine schedules
    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep + 1
    return schedule


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
    warmup_steps: int = -1,
    start_epoch: int = 0,
    start_step: int = 0,
) -> np.ndarray:
    assert epochs > 0 and niter_per_ep > 0
    assert 0 <= start_step < niter_per_ep, "`start_step` must be in [0, niter_per_ep)."

    total_steps = epochs * niter_per_ep
    start_offset = start_epoch * niter_per_ep + start_step
    effective_total = total_steps - start_offset

    def _pad(seq: np.ndarray) -> np.ndarray:
        prefix = np.full(start_offset, start_warmup_value or base_value, dtype=float)
        out = np.concatenate([prefix, seq])
        expected = total_steps + 1
        if len(out) != expected:
            if len(out) < expected:
                out = np.concatenate([out, np.full(expected - len(out), seq[-1], dtype=float)])
        return out

    if effective_total <= 0:
        seq = np.array([final_value], dtype=float)
        return _pad(seq)

    warmup_iters = (
        min(warmup_steps, effective_total)
        if warmup_steps > 0
        else min(warmup_epochs * niter_per_ep, effective_total)
    )

    warmup_schedule = (
        np.linspace(start_warmup_value, base_value, warmup_iters, endpoint=False, dtype=float)
        if warmup_iters > 0
        else np.array([], dtype=float)
    )

    cosine_iters = effective_total - warmup_iters
    if cosine_iters <= 0:
        seq = np.concatenate([warmup_schedule, np.array([final_value], dtype=float)])
        return _pad(seq)

    t = np.arange(cosine_iters + 1, dtype=float)
    cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
        1.0 + np.cos(math.pi * t / max(1.0, float(cosine_iters)))
    )

    seq = np.concatenate([warmup_schedule, cosine_schedule])
    return _pad(seq)


class LoggingUtil:
    """Utility class for logging image tensors to various loggers"""

    def log_image_tensor(
        self, pl_module: pl.LightningModule, images: torch.Tensor, name: str
    ) -> None:
        """Log images to various loggers.

        Args:
            pl_module: Lightning module to log to
            images: Image tensor to log (shape: C x H x W)
            name: Name/tag for the logged image
        """
        assert isinstance(images, torch.Tensor), type(images)
        assert images.ndim == 3, images.shape
        if images.max() > 1:
            images = images / 255

        for _logger in pl_module.loggers:
            logger_name = _logger.__class__.__name__
            if logger_name == "AimLogger":
                _logger.experiment.track(AimImage(images), name=name, step=pl_module.current_epoch)
            elif logger_name == "MLFlowLogger":
                np_images = (images.numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
                _logger.experiment.log_image(
                    _logger.run_id,
                    key=name,
                    image=np_images,
                    step=pl_module.current_epoch,
                )


def ToTensor(image: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy array to torch tensor.

    Args:
        image: Input numpy array
        dtype: Target tensor dtype

    Returns:
        Converted torch tensor
    """
    func = T.Compose([T.ToImage(), T.ToDtype(dtype)])
    return func(image)


def gaussian_blur(image: torch.Tensor, sigma: float = 1.6, radius: int = 3) -> torch.Tensor:
    """
    Efficiently applies Gaussian Blur to the input image.

    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        sigma (float): Standard deviation of the Gaussian kernel (default: 1.6).
        radius (int): Radius of the Gaussian kernel (default: 3, i.e., 7x7 kernel).

    Returns:
        torch.Tensor: Blurred image.
    """
    kernel_size = 2 * radius + 1
    coords = torch.arange(kernel_size, device=image.device) - radius
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    # Repeat shape to (C, 1, k, k) for applying the same kernel per channel
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(image.shape[1], 1, 1, 1)
    padding = radius
    # Perform convolution simultaneously for each channel using the groups parameter
    return F.conv2d(image, kernel, padding=padding, groups=image.shape[1])


# Global cache for Butterworth filters to avoid redundant computations.
_butterworth_filter_cache = {}


def get_gaussian_kernel1d(sigma, device, dtype):
    radius = int(6 * sigma + 1)
    if radius % 2 == 0:
        radius += 1
    coords = torch.arange(radius, device=device, dtype=dtype) - radius // 2
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d /= g1d.sum()
    return g1d.view(1, 1, -1), radius // 2  # (1,1,K), padding


def soft_edge_mask(
    images: torch.Tensor, threshold_frac: float = 0.01, sigma: float = 50
) -> torch.Tensor:
    """
    GPU-efficient soft edge mask.
    """
    B, C, H, W = images.shape
    device, dtype = images.device, images.dtype

    # Grayscale
    gray = images.mean(dim=1, keepdim=True)  # (B,1,H,W)

    # Threshold
    thresholds = threshold_frac * gray.amax(dim=(2, 3), keepdim=True)
    binary_mask = (gray > thresholds).float()

    # Gaussian blur separable convolution
    kernel1d, pad = get_gaussian_kernel1d(sigma, device, dtype)

    # Horizontal blur
    binary_mask_padded = F.pad(binary_mask, (pad, pad, 0, 0), mode="constant", value=0)
    blurred = F.conv2d(binary_mask_padded, kernel1d.unsqueeze(2))  # (B,1,H,W)

    # Vertical blur
    blurred_padded = F.pad(blurred, (0, 0, pad, pad), mode="constant", value=0)
    blurred = F.conv2d(blurred_padded, kernel1d.unsqueeze(3))  # (B,1,H,W)

    soft_mask = blurred.clamp(0, 1)

    return soft_mask


def butterworth_high_pass_filter(
    image: torch.Tensor,
    fov_mask: Optional[torch.Tensor] = None,
    blur_sigma: float = 1.0,
    blur_radius: int = 3,
    cutoff: float = 30,
    order: int = 2,
    boundary_cut: float = 0.5,
    normalize: bool = True,
    center_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Butterworth high-pass filter to image in frequency domain,
    with optional elliptical center-mask that keeps only the interior."""
    B, C, H, W = image.shape

    # === (1) Min-max normalization with stability ===
    eps = 1e-6
    img_min = image.amin(dim=(-2, -1), keepdim=True)
    img_max = image.amax(dim=(-2, -1), keepdim=True)
    denom = (img_max - img_min).clamp(min=eps)
    image = (image - img_min) / denom

    # === (2) Gaussian blur + FFT ===
    if fov_mask is not None:
        blurred = gaussian_blur(image * fov_mask, sigma=blur_sigma, radius=blur_radius)
    else:
        blurred = gaussian_blur(image, sigma=blur_sigma, radius=blur_radius)
    freq = torch.fft.fft2(blurred.float())

    # === (3) Apply Butterworth high-pass filter ===
    H_hp = get_butterworth_filter(H, W, cutoff, order, image.device)
    freq_shift = torch.fft.fftshift(freq, dim=(-2, -1))
    freq_filtered = freq_shift * H_hp.view(1, 1, H, W)

    # === (4) Apply elliptical center-mask (outer masking) ===
    if center_mask is not None:
        freq_filtered = freq_filtered * center_mask.view(1, 1, H, W)

    # === (5) Inverse FFT ===
    filtered = torch.fft.ifftshift(freq_filtered, dim=(-2, -1))
    filtered = torch.fft.ifft2(filtered, dim=(-2, -1))
    filtered = torch.abs(filtered)

    # === (6) FOV mask & normalize ===
    if fov_mask is not None:
        binary_mask = (fov_mask > boundary_cut).float()
        filtered = filtered * binary_mask

    if normalize:
        filt_min = filtered.amin(dim=(-2, -1), keepdim=True)
        filt_max = filtered.amax(dim=(-2, -1), keepdim=True)
        filt_denom = (filt_max - filt_min).clamp(min=eps)
        filtered = (filtered - filt_min) / filt_denom

    return {
        "input_image": image,
        "high_freq_image": filtered,
        "high_freq": freq_filtered.squeeze(1),
    }


def get_center_ellipse_mask(
    H: int,
    W: int,
    center_mask_radius_ratio_x: float,
    center_mask_radius_ratio_y: float,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Get center ellipse mask."""
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    center_y, center_x = H / 2, W / 2
    ry = center_mask_radius_ratio_y * (H / 2)
    rx = center_mask_radius_ratio_x * (W / 2)
    ellipse_eq = ((yy - center_y) ** 2) / (ry**2) + ((xx - center_x) ** 2) / (rx**2)
    return (ellipse_eq <= 1).float()


def get_butterworth_filter(
    H: int, W: int, cutoff: int, order: int, device: torch.device
) -> torch.Tensor:
    """
    Caches or recalculates Butterworth high-pass filter for the given size.
    """
    key = (H, W, cutoff, order, device)
    if key in _butterworth_filter_cache:
        return _butterworth_filter_cache[key]

    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    center_y, center_x = H / 2, W / 2
    distance = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    H_filter = 1 - 1 / (1 + (distance / cutoff) ** (2 * order))
    H_filter = H_filter.unsqueeze(0).unsqueeze(0)
    _butterworth_filter_cache[key] = H_filter
    return H_filter


def thresholded_mask(
    input_tensor: torch.Tensor,
    mask_ratio: float = 0.0,
) -> torch.Tensor:
    """Create binary mask based on threshold with consistent masking across batch.

    Args:
        input_tensor (torch.Tensor): Tensor shape (B, H, W) or (B, 1, H, W)
        mask_ratio (float): Mask ratio (0.0~1.0)

    Returns:
        torch.Tensor: Mask tensor (B, H, W)
    """
    if input_tensor.ndim == 4:
        if input_tensor.shape[1] != 1:
            raise ValueError("Input must be grayscale with channel=1.")
        input_tensor = input_tensor.squeeze(1)

    B, H, W = input_tensor.shape
    flat_tensor = input_tensor.view(B, -1)

    # Calculate exact number of pixels to mask per batch
    num_pixels = H * W
    num_mask = int(mask_ratio * num_pixels)

    if num_mask == 0:
        raise ValueError("Mask ratio too small, results in zero masked pixels.")

    # argsort to get indices sorted by value (ascending)
    sorted_indices = torch.argsort(flat_tensor, dim=1, descending=True)

    # select threshold per batch to ensure fixed number of masked pixels
    threshold_indices = sorted_indices[:, :num_mask]

    # create mask initialized to zeros
    mask = torch.zeros_like(flat_tensor, device=input_tensor.device)

    # scatter ones to the masked positions
    mask = mask.scatter(1, threshold_indices, 1.0).view(B, H, W)
    return mask


def patchified_binary_mask(
    input_tensor: torch.Tensor,
    patch_size: int = 16,
    mask_ratio: float = 0.5,
) -> torch.Tensor:
    """Generate binary mask by applying average pooling followed by percentile thresholding.

    Args:
        input_tensor (torch.Tensor): Grayscale image tensor, shape (B, H, W) or (B, 1, H, W)
        patch_size (int): Patch size for average pooling
        mask_ratio (float): Threshold mask ratio value (0.0 ~ 1.0)

    Returns:
        torch.Tensor: Binary mask tensor after pooling, shape (B, H_p, W_p)
    """
    # Add channel dimension for 3D tensor to make (B, 1, H, W)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(1)
    elif input_tensor.ndim == 4 and input_tensor.shape[1] != 1:
        raise ValueError("Input image must be grayscale with channel dimension of 1")
    elif input_tensor.ndim != 4:
        raise ValueError("Input image must be of shape (B, H, W) or (B, 1, H, W)")
    # Apply average pooling (downsampling by patch size)
    pooled = F.avg_pool2d(input_tensor, kernel_size=patch_size, stride=patch_size)
    # Generate binary mask from pooled result using percentile threshold
    return thresholded_mask(pooled, mask_ratio)


def patchified_values(
    input_tensor: torch.Tensor,
    patch_size: int = 16,
    mask_ratio: float = 0.5,
) -> torch.Tensor:
    """Generate binary mask by applying average pooling followed by percentile thresholding.

    Args:
        input_tensor (torch.Tensor): Grayscale image tensor, shape (B, H, W) or (B, 1, H, W)
        patch_size (int): Patch size for average pooling
        mask_ratio (float): Threshold mask ratio value (0.0 ~ 1.0)

    Returns:
        torch.Tensor: Binary mask tensor after pooling, shape (B, H_p, W_p)
    """
    # Add channel dimension for 3D tensor to make (B, 1, H, W)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(1)
    elif input_tensor.ndim == 4 and input_tensor.shape[1] != 1:
        raise ValueError("Input image must be grayscale with channel dimension of 1")
    elif input_tensor.ndim != 4:
        raise ValueError("Input image must be of shape (B, H, W) or (B, 1, H, W)")
    # Apply average pooling (downsampling by patch size)
    pooled = F.avg_pool2d(input_tensor, kernel_size=patch_size, stride=patch_size)
    return pooled


def calculate_grad_cosine_similarity(
    grad_dict1: Dict[str, torch.Tensor], grad_dict2: Dict[str, torch.Tensor]
) -> float:
    """Calculate cosine similarity between two gradient dictionaries.

    Computes the cosine similarity between two dictionaries of gradients,
    treating each dictionary as a flattened vector.

    Args:
        grad_dict1: First dictionary of gradients with parameter names as keys
        grad_dict2: Second dictionary of gradients with parameter names as keys

    Returns:
        float: Cosine similarity value between -1 and 1, where 1 means perfectly aligned,
              0 means orthogonal, and -1 means opposite directions
    """

    dot_product = 0
    norm1 = 0
    norm2 = 0

    for name in grad_dict1:
        if name in grad_dict2:
            grad1 = grad_dict1[name].view(-1)
            grad2 = grad_dict2[name].view(-1)

            dot_product += torch.sum(grad1 * grad2).item()
            norm1 += torch.sum(grad1 * grad1).item()
            norm2 += torch.sum(grad2 * grad2).item()

    if dot_product == 0 and norm1 == 0 and norm2 == 0:
        print("[WARN] No overlapping parameters in gradient dicts")

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))


def folded_freq_quadrants(high_freq: torch.Tensor):
    """
    image: (B, 1, H, W) 실수 이미지
    return: (B, 4, H/2, W/2) - real/imag 쌍의 Q1+Q4 접은 결과
    """
    B, H, W = high_freq.shape
    assert H % 2 == 0 and W % 2 == 0

    real = high_freq.real  # (B, H, W)
    imag = high_freq.imag

    real_q1, real_q4 = get_quadrants_freq(real, H, W)
    imag_q1, imag_q4 = get_quadrants_freq(imag, H, W)

    # 3. Q4를 x축 기준으로 flip → Q1과 접기
    real_q4 = torch.flip(real_q4, dims=[1])
    imag_q4 = torch.flip(imag_q4, dims=[1])

    # 4. 채널 구성: [real_Q1, real_Q4_flipped, imag_Q1, imag_Q4_flipped]
    folded = torch.stack([real_q1, real_q4, imag_q1, imag_q4], dim=1)  # (B, 4, H/2, W/2)

    return folded  # shape: (B, 4, H/2, W/2)


# 2. Q1, Q4 추출 (real/imag 각각)
def get_quadrants_freq(t, H, W):  # t: (B, H, W)
    Q1 = t[:, : H // 2, W // 2 :]  # top-right
    Q4 = t[:, H // 2 :, W // 2 :]  # bottom-right
    return Q1, Q4


def standardize_input(
    x: torch.Tensor,
) -> torch.Tensor:
    """Standardize input image tensor using mean and std values.

    Args:
        x (torch.Tensor): Input image tensor of shape [B, C, H, W]
        mean (List[float], optional): Mean values for each channel.
            Defaults to IMAGENET_DEFAULT_MEAN.
        std (List[float], optional): Standard deviation values for each channel.
            Defaults to IMAGENET_DEFAULT_STD.

    Returns:
        torch.Tensor: Standardized image tensor of shape [B, C, H, W]
    """
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # Convert to tensor and reshape for broadcasting
    mean_tensor = torch.tensor(mean).to(x.device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std).to(x.device).view(1, -1, 1, 1)

    # Standardize: (x - mean) / std
    x = (x - mean_tensor) / std_tensor
    return x
