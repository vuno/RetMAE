import collections.abc
import importlib.util
import logging
import math
from collections import OrderedDict
from itertools import repeat
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from kornia.geometry.transform import resize
from torch import nn
from torch.utils.checkpoint import checkpoint

if importlib.util.find_spec("flash_attn"):
    FlashMHA = importlib.import_module("flash_attn.flash_attention").FlashMHA

from src.tasks.pretrain.utils import standardize_input


class Bottleneck(nn.Module):
    """
    Bottleneck block used in ResNet architectures.

    This module implements the bottleneck layer, as used in ResNet architectures, with optional downsampling.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        """
        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of intermediate channels for the bottleneck.
            stride (int, optional): Stride for the convolution. Default: 1.
        """
        super().__init__()
        # All conv layers have stride 1. An avgpool is performed after the second convolution when stride > 1.
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # Downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1.
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the bottleneck block.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after bottleneck processing.
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    """
    2D Attention Pooling layer.

    This module applies multi-head attention-based pooling to 2D feature maps.
    """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        """
        Args:
            spacial_dim (int): Spatial dimension (height/width of input feature map).
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            output_dim (int, optional): Output dimension. Defaults to None, which means output_dim=embed_dim.
        """
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        """
        Forward pass for attention pooling.

        Args:
            x (torch.Tensor): Input feature map of shape (N, C, H, W).
        Returns:
            torch.Tensor: Pooled feature tensor.
        """
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    Modified ResNet implementation.

    This class is similar to torchvision's ResNet but with the following changes:
        - Uses 3 "stem" convolutions instead of 1, with an average pool instead of a max pool.
        - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1.
        - The final pooling layer is a QKV attention instead of an average pool.
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        """
        Args:
            layers (list[int] or tuple[int]): Number of Bottleneck blocks for each of the 4 residual layers.
            output_dim (int): Output embedding dimension.
            heads (int): Number of attention heads for the attention pooling.
            input_resolution (int): Input image resolution.
            width (int): Base channel width.
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # The 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self._inplanes = width  # Mutable variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # The ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        """
        Construct a layer of bottleneck blocks.

        Args:
            planes (int): Number of output channels for the bottleneck.
            blocks (int): Number of Bottleneck blocks in this layer.
            stride (int): Stride for the first block.
        Returns:
            nn.Sequential: Sequential container of Bottleneck blocks.
        """
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """
        Dummy method for interface consistency; non-transformer ResNet does not support grad checkpointing.
        """
        # FIXME support for non-transformer
        pass

    def forward(self, x):
        """
        Forward pass through the ModifiedResNet.

        Args:
            x (torch.Tensor): Input images, shape (N, C, H, W).
        Returns:
            torch.Tensor: Output features after attention pooling.
        """

        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3),
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """
    LayerNorm subclass to handle fp16 inputs.

    Ensures normalization is performed in float32 for numerical stability.
    """

    def forward(self, x: torch.Tensor):
        """
        LayerNorm forward with support for fp16 input.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Normalized output tensor, of original dtype.
        """
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    QuickGELU activation function.
    See: https://arxiv.org/abs/2002.05202
    """

    def forward(self, x: torch.Tensor):
        """
        Applies the QuickGELU activation: x * sigmoid(1.702 * x)

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Activated tensor.
        """
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Residual Attention Block for the Vision Transformer.

    Consists of a multi-head self-attention, followed by a feed-forward MLP block, each preceded by layer norm and wrapped in residual connections.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        use_flash_attention: bool = False,
    ):
        """
        Args:
            d_model (int): The input/output dimension.
            n_head (int): Number of attention heads.
            attn_mask (torch.Tensor, optional): Attention mask tensor.
            use_flash_attention (bool): If True, use Flash Attention implementation (if available).
        """
        super().__init__()

        self.attn = (
            nn.MultiheadAttention(d_model, n_head)
            if not use_flash_attention
            else FlashMHA(d_model, n_head)
        )
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

    def attention(self, x: torch.Tensor):
        """
        Applies self-attention with optional mask and Flash Attention.

        Args:
            x (torch.Tensor): Input tensor (L, N, D).
        Returns:
            torch.Tensor: Output of the self-attention layer.
        """
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        if self.use_flash_attention:
            # Batch first is needed for FlashAttention. See https://github.com/HazyResearch/flash-attention/issues/84 for more information.
            return self.attn(x.transpose(1, 0))[0].transpose(1, 0)
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the residual attention block.

        Args:
            x (torch.Tensor): Input tensor (L, N, D).
        Returns:
            torch.Tensor: Output tensor after self-attention and MLP.
        """
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Stacked Transformer encoder for vision transformer architectures.
    """

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        use_flash_attention: bool = False,
    ):
        """
        Args:
            width (int): Input/output feature dimension.
            layers (int): Number of transformer blocks.
            heads (int): Number of self-attention heads.
            attn_mask (torch.Tensor, optional): Attention mask tensor.
            use_flash_attention (bool): Whether to use flash attention.
        """
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

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the transformer stacks.

        Args:
            x (torch.Tensor): Input tensor (L, N, D).
        Returns:
            torch.Tensor: Output embedding.
        """
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for r in self.resblocks:
                x = checkpoint(r, x)
            return x
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    """
    Vision Transformer model.

    Applies patch embedding with a Conv2d stem, and a series of Transformer blocks, for vision tasks.
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
    ):
        """
        Args:
            input_resolution (int): Spatial size of input images.
            patch_size (int): Patch size for patch embedding.
            width (int): Embedding dimension.
            layers (int): Number of Transformer layers.
            heads (int): Number of attention heads.
            output_dim (int): Output embedding dimension.
            use_flash_attention (bool): Use Flash Attention in transformer.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.grid_size = (self.input_resolution // patch_size, self.input_resolution // patch_size)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads, use_flash_attention=use_flash_attention
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """
        Enable or disable gradient checkpointing (activation checkpointing).

        Args:
            enable (bool): If True, enable activation checkpointing.
        """
        self.transformer.grad_checkpointing = enable

    def random_masking(self, x, mask_ratio):
        """
        Apply random token masking for masked image modeling during training.

        Args:
            x (torch.Tensor): Input tokens, shape [N, L, D] (batch, length, dim).
            mask_ratio (float): Ratio of patches to mask.
        Returns:
            torch.Tensor: Masked sequence (with class token always retained).
        """
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

    def for_linear_probing(self, x: torch.Tensor, mask_ratio: float = 0.0):
        """
        Forward for extracting the output after vision transformer for linear probing.

        Args:
            x (torch.Tensor): Input images [B, C, H, W].
            mask_ratio (float): Mask ratio if using random patch masking.
        Returns:
            torch.Tensor: Feature tensor.
        """
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

        x = self.ln_post(x[:, 0, :])

        return x

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
        """
        Forward pass for vision transformer. Optionally supports masked patch modeling.

        Args:
            x (torch.Tensor): Input images [B, C, H, W].
            mask_ratio (float, optional): Mask ratio for random masking (default: 0.0).
        Returns:
            torch.Tensor: Output image features (global pooled embedding).
        """
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

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionEncoder(nn.Module):
    """
    Vision-only encoder, simplified from CLIP. Selects a ResNet or Vision Transformer backbone based on parameters.
    """

    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        vision_head_width: int = 64,
        use_flash_attention: bool = False,
    ):
        """
        Args:
            embed_dim (int): Output embedding dimension.
            image_resolution (int): Input image resolution.
            vision_layers (tuple or int): If tuple/list, use ResNet backbone (stage depths); if int, number of ViT layers.
            vision_width (int): Width/hidden dimension.
            vision_patch_size (int): Patch size for ViT, ignored for ResNet.
            vision_head_width (int, optional): Width per attention head. Default: 64.
            use_flash_attention (bool, optional): Use flash attention if possible.
        """
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // vision_head_width
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // vision_head_width
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                use_flash_attention=use_flash_attention,
            )

        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize model parameters specific to the backbone.
        """
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """
        Enable or disable activation checkpointing for the vision backbone.

        Args:
            enable (bool): If True, enable checkpointing.
        """
        self.visual.set_grad_checkpointing(enable)

    @property
    def dtype(self):
        """
        Returns the dtype of the input convolution (for proper input casting).
        """
        if isinstance(self.visual, ModifiedResNet):
            return self.visual.conv1.weight.dtype
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, mask_ratio=0):
        """
        Encode input images into global image embeddings.

        Args:
            image (torch.Tensor): Input image tensor.
            mask_ratio (float, optional): Mask ratio for patch masking.
        Returns:
            torch.Tensor: Image features vector.
        """
        if isinstance(self.visual, ModifiedResNet):
            return self.visual(image.type(self.dtype))
        return self.visual(image.type(self.dtype), mask_ratio)

    def encode_image_featExt(self, image, mask_ratio=0):
        """
        Extract image features for external use (e.g. feature visualization or linear probing).

        Args:
            image (torch.Tensor): Input image tensor.
            mask_ratio (float, optional): Mask ratio for patch masking.
        Returns:
            torch.Tensor: Image features.
        """
        if isinstance(self.visual, ModifiedResNet):
            img_feat = self.visual(image.type(self.dtype))
            return img_feat
        return self.visual.for_linear_probing(image.type(self.dtype), mask_ratio)

    def forward(self, image, mask_ratio=0):
        """
        Forward pass for VisionEncoder.

        Args:
            image (torch.Tensor): Batch of input images.
            mask_ratio (float, optional): Mask ratio for patch masking.
        Returns:
            torch.Tensor: Encoded image embeddings.
        """
        return self.encode_image(image, mask_ratio)


def convert_models_to_fp32(model):
    """
    Convert all model parameters to float32.

    Args:
        model (nn.Module): Model to convert.
    """
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def convert_weights(model: nn.Module):
    """
    Convert applicable model parameters to float16.

    Useful when preparing for mixed-precision inference/training.

    Args:
        model (nn.Module): Target model for weight conversion.
    """

    def _convert_weights_to_fp16(input_layer: nn.Module):
        if isinstance(input_layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            input_layer.weight.data = input_layer.weight.data.half()
            if input_layer.bias is not None:
                input_layer.bias.data = input_layer.bias.data.half()

        if isinstance(input_layer, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(input_layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(input_layer, name):
                attr = getattr(input_layer, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def convert_state_dict(state_dict):
    """
    Adapt Vision Transformer state dicts for Flash Attention compatibility.

    If the weight naming uses one style, convert it to the other.

    Args:
        state_dict (dict): State dictionary.
    Returns:
        dict: Modified state dictionary.
    """
    if not state_dict:
        return state_dict

    prefix = "module." if list(state_dict.keys())[0].startswith("module") else ""

    if f"{prefix}visual.transformer.resblocks.0.attn.in_proj_weight" in state_dict:
        for k in list(state_dict.keys()):
            if "attn.in_proj_weight" in k:
                state_dict[k.replace("attn.in_proj_weight", "attn.Wqkv.weight")] = state_dict.pop(
                    k
                )
            elif "attn.in_proj_bias" in k:
                state_dict[k.replace("attn.in_proj_bias", "attn.Wqkv.bias")] = state_dict.pop(k)
    elif f"{prefix}visual.transformer.resblocks.0.attn.Wqkv.weight" in state_dict:
        for k in list(state_dict.keys()):
            if "attn.Wqkv.weight" in k:
                state_dict[k.replace("attn.Wqkv.weight", "attn.in_proj_weight")] = state_dict.pop(
                    k
                )
            elif "attn.Wqkv.bias" in k:
                state_dict[k.replace("attn.Wqkv.bias", "attn.in_proj_bias")] = state_dict.pop(k)

    return state_dict


def resize_pos_embed(state_dict, model, interpolation: str = "bicubic", seq_dim=1, prefix=""):
    """
    Rescale the grid of position embeddings when loading a checkpoint.

    Args:
        state_dict (dict): State dict loaded from checkpoint.
        model (nn.Module): Model whose position embeddings to match.
        interpolation (str): Interpolation mode (e.g., 'bicubic').
        seq_dim (int): Sequence dimension.
        prefix (str): Prefix for the position embedding key.
    """
    old_pos_embed = state_dict.get(prefix + "visual.positional_embedding", None)
    model = model.module if hasattr(model, "module") else model
    if old_pos_embed is None or not hasattr(model.visual, "grid_size"):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (i.e., no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info("Resizing position embedding grid-size from %s to %s", old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(
        0, 3, 1, 2
    )
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict[prefix + "visual.positional_embedding"] = new_pos_embed


# From PyTorch internals
def _ntuple(n):
    """
    Factory for parsing tuple-like arguments.

    Args:
        n (int): Tuple length.
    Returns:
        callable: Function that transforms input into tuple of length n.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class VilRef(nn.Module):
    """
    Vision Language Reference Model (VilRef).

    Wraps a vision encoder, loads pretrained ViLReF weights, and performs standardized inference.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        image_resolution: int = 224,
        vision_layers: int = 12,
        vision_width: int = 768,
        vision_patch_size: int = 16,
        use_flash_attention: bool = False,
    ):
        """
        Args:
            embed_dim (int): Embedding dimension for output.
            image_resolution (int): Input image resolution.
            vision_layers (int): Number of vision transformer/ResNet layers.
            vision_width (int): Width/hidden dimension.
            vision_patch_size (int): Patch size for ViT.
            use_flash_attention (bool): Whether to use flash attention.
        """
        super().__init__()
        self.model = VisionEncoder(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            use_flash_attention=use_flash_attention,
        )

        # Load checkpoint and extract only visual weights
        full_ckpt = torch.load(
            "your_path/dev/weights/ssl/ViLReF_ViT.pt", map_location="cpu"
        )["state_dict"]
        visual_ckpt = {}
        for k, v in full_ckpt.items():
            if k.startswith("module.visual."):
                new_k = k.replace("module.visual.", "")
                visual_ckpt[new_k] = v

        # Load visual weights into the model
        self.model.visual.load_state_dict(visual_ckpt, strict=True)
        self.model.eval()

        self.patch_size = vision_patch_size
        self.input_resolution = image_resolution
        self.grid_size = (
            self.input_resolution // self.patch_size,
            self.input_resolution // self.patch_size,
        )
        self.out_dim = self.embed_dim = vision_width

    def forward(self, x: torch.Tensor):
        """
        Forward pass for VilRef model.

        Args:
            x (torch.Tensor): Input images [B, C, H, W].
        Returns:
            dict: {'features': (image features as tensor)}
        """
        x = resize(x, (224, 224))
        x = standardize_input(x)
        return {
            "features": self.model.encode_image_featExt(x),
        }

    def forward_features(self, x: torch.Tensor):
        """
        Alternative forward for feature extraction.

        Args:
            x (torch.Tensor): Input images [B, C, H, W].
        Returns:
            dict: Output of self.forward(x).
        """
        return self.forward(x)

    @torch.no_grad()
    def get_intermediate_features_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract intermediate features from all layers.

        Args:
            x: Input tensor
            output_type: Type of features to extract ('all', 'cls', or 'patch')
            flatten: Whether to flatten the output features

        Returns:
            List of intermediate feature tensors
        """
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
        for name, module in self.model.visual.named_modules():
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
            self.model.visual.eval()

        x = resize(x, (224, 224))
        x = standardize_input(x)

        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.model.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)

        if is_train and not self.training:
            self.model.visual.train()

        # remove hook
        for hook in hooks:
            hook.remove()

        return features
