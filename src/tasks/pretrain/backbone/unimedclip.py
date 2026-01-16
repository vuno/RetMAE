"""UniMedCLIP Vision Encoder

Vision-only encoder extracted from UniMedCLIP.
Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import collections.abc
import logging
import math
from collections import OrderedDict
from itertools import repeat
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from kornia.geometry.transform import resize
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.tasks.pretrain.utils import standardize_input


# Utility functions
def _ntuple(n):
    """Factory for parsing tuple-like arguments.

    Args:
        n: Number of elements in the tuple

    Returns:
        Parse function that converts input to n-tuple
    """

    def parse(x):
        """Parse input to n-tuple.

        Args:
            x: Input value or iterable

        Returns:
            Tuple of n elements
        """
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        """Forward pass with fp16 support.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    """QuickGELU activation function."""

    def forward(self, x: torch.Tensor):
        """Forward pass through QuickGELU activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Residual attention block for Vision Transformer.

    Args:
        d_model: Model dimension
        n_head: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        act_layer: Activation layer class
    """

    def __init__(
        self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """Apply self-attention to input.

        Args:
            x: Input tensor
            attn_mask: Optional attention mask

        Returns:
            Attention output
        """
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """Forward pass through the residual attention block.

        Args:
            x: Input tensor
            attn_mask: Optional attention mask

        Returns:
            Output tensor
        """
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Transformer encoder stack.

    Args:
        width: Model width/hidden dimension
        layers: Number of transformer layers
        heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        act_layer: Activation layer class
    """

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """Forward pass through the transformer.

        Args:
            x: Input tensor
            attn_mask: Optional attention mask

        Returns:
            Output tensor
        """
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisualTransformer(nn.Module):
    """Vision Transformer for image encoding.

    Args:
        image_size: Input image size
        patch_size: Patch size for patch embedding
        width: Model width/hidden dimension
        layers: Number of transformer layers
        heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        output_dim: Output dimension
        act_layer: Activation layer class
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        output_dim: int,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
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
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing.

        Args:
            enable: Whether to enable gradient checkpointing
        """
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor):
        """Forward pass through the vision transformer.

        Args:
            x: Input tensor

        Returns:
            Encoded features
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
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


def resize_pos_embed(state_dict, model, interpolation: str = "bicubic", seq_dim=1):
    """Rescale the grid of position embeddings when loading from state_dict.

    Args:
        state_dict: State dictionary containing position embeddings
        model: Model to load embeddings into
        interpolation: Interpolation mode for resizing
        seq_dim: Sequence dimension
    """
    old_pos_embed = state_dict.get("visual.positional_embedding", None)
    if old_pos_embed is None or not hasattr(model.visual, "grid_size"):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
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
    state_dict["visual.positional_embedding"] = new_pos_embed


class UniMedCLIP(nn.Module):
    """
    UniMedCLIP Vision Encoder Wrapper.

    Wraps a vision encoder, loads pretrained UniMedCLIP weights, and performs standardized inference.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        image_size: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        """
        Args:
            embed_dim (int): Output embedding dimension.
            image_size (int): Input image resolution.
            patch_size (int): Patch size for ViT.
            width (int): Transformer width/hidden dimension.
            layers (int): Number of transformer layers.
            heads (int): Number of attention heads.
            mlp_ratio (float): MLP ratio for transformer blocks.
        """
        super().__init__()

        # Create vision transformer
        self.visual = VisualTransformer(
            image_size=image_size,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            output_dim=embed_dim,
            act_layer=nn.GELU,
        )

        # Load checkpoint and extract only visual weights
        full_ckpt = torch.load(
            "your_path/dev/weights/ssl/unimed_clip_vit_b16.pt", map_location="cpu"
        )["state_dict"]

        # Extract visual weights
        visual_ckpt = {}
        for k, v in full_ckpt.items():
            if k.startswith("module.visual."):
                new_k = k.replace("module.visual.", "")
                visual_ckpt[new_k] = v

        # Load visual weights into the model
        self.visual.load_state_dict(visual_ckpt, strict=True)
        self.visual.eval()

        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_size = (
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
        )
        self.out_dim = self.embed_dim = width

    def forward(self, x: torch.Tensor):
        """
        Forward pass for UniMedCLIP model.

        Args:
            x (torch.Tensor): Input images [B, C, H, W].
        Returns:
            dict: {'features': (image features as tensor)}
        """
        x = resize(x, (224, 224))
        x = standardize_input(x)

        # Extract features before projection
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.visual.ln_post(x[:, 0, :])

        return {
            "features": x,
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
        target_layers = "visual.transformer.resblocks."
        for name, module in self.named_modules():
            append = False
            if not isinstance(module, torch.nn.Module):
                continue
            # Patch Embedding Layer
            if name.startswith("visual.conv1"):
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

        # Extract features before projection
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)

        if is_train and not self.training:
            self.train()

        # remove hook
        for hook in hooks:
            hook.remove()

        return features
