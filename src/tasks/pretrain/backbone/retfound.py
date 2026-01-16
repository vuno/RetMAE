"""RETFound model implementation."""

from functools import partial
from typing import Dict

import numpy as np
import timm
import torch
import torch.nn as nn
from kornia.geometry.transform import resize
from timm.models.vision_transformer import Block, PatchEmbed, VisionTransformer

from src.tasks.pretrain.utils import standardize_input
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """Generate 2D sine-cosine positional embeddings.

    Args:
        embed_dim: Embedding dimension
        grid_size: Grid height and width (assumes square grid)
        cls_token: Whether to include CLS token

    Returns:
        np.ndarray: Positional embedding of shape [grid_size*grid_size, embed_dim]
                    or [1+grid_size*grid_size, embed_dim]
    """
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
    """Generate sine-cosine positional embeddings from a 2D grid.

    Args:
        embed_dim: Embedding dimension
        grid: 2D position grid

    Returns:
        np.ndarray: Positional embedding of shape [H*W, D]
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate sine-cosine positional embeddings for 1D positions.

    Args:
        embed_dim: Output dimension for each position
        pos: List of positions to encode, shape (M,)

    Returns:
        np.ndarray: Positional embedding of shape [M, D]
    """
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


class RETFoundBackbone(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone.

    Args:
        img_size: Input image size
        patch_size: Patch size for patch embedding
        in_chans: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Number of decoder blocks
        decoder_num_heads: Number of decoder attention heads
        mlp_ratio: MLP hidden dimension ratio
        norm_layer: Normalization layer
        norm_pix_loss: Whether to normalize pixel loss
        ckpt_path: Path to pretrained checkpoint
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,
        ckpt_path: str = "your_path/save/weights/ssl/RETFound_cfp_weights.pth",
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_size = patch_size
        self.img_size = img_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.out_dim = self.embed_dim = embed_dim

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

        ckpt = torch.load(ckpt_path, map_location="cpu")["model"]
        new_ckpt = {}
        for k, v in ckpt.items():
            if ("decoder" not in k) and ("mask_token" not in k):
                new_ckpt[k] = v

        self.load_state_dict(new_ckpt, strict=True)

    def initialize_weights(self) -> None:
        """Initialize model weights."""
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize weights for linear and layer norm modules.

        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Performs encoder forward pass.

        Args:
            x: Input tensor
            mask_ratio: Masking ratio for MAE

        Returns:
            Encoded features
        """
        x = resize(x, (self.img_size, self.img_size))
        x = standardize_input(x)

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        """Performs model forward pass.

        Args:
            imgs: Input images
            mask_ratio: Masking ratio for MAE

        Returns:
            Dictionary containing encoded features
        """
        return {"features": self.forward_encoder(imgs, mask_ratio)}

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Performs forward pass for feature extraction.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing encoded features
        """
        return self(x)


def interpolate_pos_embed(model, checkpoint_model):
    """Interpolate position embeddings when model and checkpoint sizes differ.

    Args:
        model: The model to load weights into
        checkpoint_model: Dictionary containing checkpoint weights
    """
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(
                0, 3, 1, 2
            )
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


class RETFound(nn.Module):
    """RETFound model with backbone for fundus image analysis.

    Args:
        patch_size: Patch size for patch embedding
        img_size: Input image size
        embed_dim: Embedding dimension
        ckpt_path: Path to pretrained checkpoint
    """

    def __init__(
        self,
        patch_size: int = 14,
        img_size: int = 224,
        embed_dim: int = 1024,
        model_type: str = "mae",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim

        if model_type == "dinov2":
            ckpt_path = "your_path/dev/weights/ssl/RETFound_dinov2_shanghai.pth"
            ckpt = torch.load(ckpt_path, map_location="cpu")
            new_ckpt = {}
            for k, v in ckpt["teacher"].items():
                if k.startswith("backbone") and "mask_token" not in k:
                    k = k.replace("backbone.", "")
                    new_ckpt[k] = v
            self.model = timm.create_model(
                "vit_large_patch14_dinov2.lvd142m",
                pretrained=True,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
            )
        elif model_type == "mae":
            ckpt_path = "your_path/dev/weights/ssl/RETFound_mae_shanghai.pth"
            ckpt = torch.load(ckpt_path, map_location="cpu")
            new_ckpt = {}
            for k, v in ckpt["model"].items():
                if "mask_token" not in k and "decoder" not in k:
                    k = k.replace("backbone.", "")
                    new_ckpt[k] = v

            self.model = VisionTransformer(
                patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model.head = nn.Identity()

        interpolate_pos_embed(self.model, new_ckpt)
        self.model.load_state_dict(new_ckpt, strict=True)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Performs encoder forward pass.

        Args:
            x: Input tensor

        Returns:
            Encoded features
        """
        x = resize(x, (self.img_size, self.img_size))
        x = standardize_input(x)
        return self.model.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model forward pass.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing encoded features
        """
        return {"features": self.forward_encoder(x)}

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Performs forward pass for feature extraction.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing encoded features
        """
        return self(x)
