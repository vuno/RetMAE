"""RET-CLIP model implementation."""

from typing import Dict, List, Optional, OrderedDict

import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.transform import resize
from torch.utils.checkpoint import checkpoint

from src.tasks.pretrain.utils import standardize_input
from src.utils import RankedLogger, ckpt_path as _ckpt

log = RankedLogger(__name__, rank_zero_only=True)


# ============================ RET-CLIP ============================
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """Quick GELU activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Residual attention block with self-attention and MLP.

    Args:
        d_model: Model dimension
        n_head: Number of attention heads
        attn_mask: Optional attention mask
        use_flash_attention: Whether to use flash attention
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: Optional[torch.Tensor] = None,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.use_flash_attention = use_flash_attention

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to input.

        Args:
            x: Input tensor

        Returns:
            Attention output
        """
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        if self.use_flash_attention:
            return self.attn(x.transpose(1, 0))[0].transpose(1, 0)
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual attention block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Transformer encoder with multiple residual attention blocks.

    Args:
        width: Model width/hidden dimension
        layers: Number of transformer layers
        heads: Number of attention heads
        attn_mask: Optional attention mask
        use_flash_attention: Whether to use flash attention
    """

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: Optional[torch.Tensor] = None,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(width, heads, attn_mask, use_flash_attention)
                for _ in range(layers)
            ]
        )
        print("transformer int finished")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for r in self.resblocks:
                x = checkpoint(r, x)
            return x
        return self.resblocks(x)

    def forward_intermediates(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feature extraction with intermediate outputs.

        Args:
            x: Input tensor

        Returns:
            Final output and list of intermediate features
        """
        intermediates = []
        for i, r in enumerate(self.resblocks):
            x = r(x)
            intermediates.append(x.permute(1, 0, 2)[:, 1:])  # LND -> NLD
        return x, intermediates


class RETClipBackbone(nn.Module):
    """RET-CLIP model backbone.

    Args:
        input_resolution: Input image resolution
        patch_size: Patch size for patch embedding
        width: Model width/hidden dimension
        layers: Number of transformer layers
        heads: Number of attention heads
        output_dim: Output dimension
        use_flash_attention: Whether to use flash attention
        use_rn_for_embed: Whether to use ResNet for embedding
        ckpt_path: Path to pretrained checkpoint
    """

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        use_flash_attention: bool = False,
        use_rn_for_embed: bool = False,
        ckpt_path: str | None = None,
    ):
        super().__init__()
        if ckpt_path is None:
            ckpt_path = str(_ckpt("retclip", "ret-clip.pt"))
        self.num_prefix_tokens = 1
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.grid_size = (
            self.input_resolution // patch_size,
            self.input_resolution // patch_size,
        )
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.out_dim = self.embed_dim = output_dim

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads, use_flash_attention=use_flash_attention
        )
        self.head = heads

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        print("vit int finished")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        new_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith("module.visual"):
                new_k = k.replace("module.visual.", "")
                new_ckpt[new_k] = v

        self.load_state_dict(new_ckpt)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing."""
        self.transformer.grad_checkpointing = enable

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Apply random masking to input sequence."""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int((L - 1) * (1 - mask_ratio))

        noise = torch.rand(N, L - 1, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) + torch.ones(
            N, L - 1, device=x.device, dtype=int
        )
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        x0 = x[:, 0, :]
        x0 = x0.reshape(N, 1, D)
        x_masked_add = torch.cat([x0, x_masked], axis=1)
        return x_masked_add

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x: Input tensor
            mask_ratio: Masking ratio for MAE

        Returns:
            Dictionary containing encoded features
        """
        return {
            "features": self._forward(x, mask_ratio),
        }

    def _forward(self, x: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        """Internal forward pass implementation.

        Args:
            x: Input tensor
            mask_ratio: Masking ratio for MAE

        Returns:
            Encoded features
        """
        x = resize(x, (224, 224))
        x = standardize_input(x)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if mask_ratio != 0:
            x = self.random_masking(x, mask_ratio)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0:1, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for feature extraction.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing encoded features
        """
        return self.forward(x)

    @torch.no_grad()
    def get_intermediate_features_all(
        self, x: torch.Tensor, output_type: str = "all", flatten: bool = False
    ) -> List[torch.Tensor]:
        """Extract intermediate features from all layers.

        Args:
            x: Input tensor
            output_type: Type of features to extract ('all', 'cls', or 'patch')
            flatten: Whether to flatten the output features

        Returns:
            List of intermediate feature tensors
        """
        assert output_type in ["all", "cls", "patch"]
        features = []

        def forward_hook(module, input, output):
            """Hook function to capture intermediate features."""
            nonlocal features

            if not features:
                out = (
                    output.permute(0, 2, 3, 1)
                    .reshape(output.shape[0], -1, output.shape[1])
                    .detach()
                )
            else:
                out = output.permute(1, 0, 2).detach()[:, 1:]
            features.append(out)

        # Register forward hooks
        hooks = []
        target_layers = "transformer.resblocks."
        for name, module in self.named_modules():
            append = False
            if not isinstance(module, torch.nn.Module):
                continue
            # Patch Embedding Layer
            if name.startswith("conv1"):
                append = True

            # Transformer Layers
            elif any(name.startswith(layer) for layer in target_layers):
                if name.split(".")[-1].isdigit():
                    append = True

            if append:
                hooks.append(module.register_forward_hook(forward_hook))

        # forward pass
        is_train = self.training

        if is_train:
            self.eval()

        x = resize(x, (224, 224))
        x = standardize_input(x)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)

        if is_train and not self.training:
            self.train()

        # remove hook
        for hook in hooks:
            hook.remove()

        return features


# ============================ RETFound ============================
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """Generate 2D sine-cosine positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate sine-cosine positional embeddings from a 2D grid."""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate sine-cosine positional embeddings for 1D positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
