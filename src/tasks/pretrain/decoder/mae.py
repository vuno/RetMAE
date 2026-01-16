# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import math
from typing import Dict, Optional, Type

import einops
import numpy as np
import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import Block


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, num_prefix_tokens: int = 1
) -> np.ndarray:
    """Generate 2D positional embeddings

    Args:
        embed_dim: Embedding dimension
        grid_size: Height and width of the grid
        num_prefix_tokens: Number of prefix tokens
    Returns:
        np.ndarray: Positional embeddings of shape [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if num_prefix_tokens:
        pos_embed = np.concatenate([np.zeros([num_prefix_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate 2D sine-cosine positional embeddings from grid

    Args:
        embed_dim: Embedding dimension
        grid: Position grid

    Returns:
        np.ndarray: Positional embeddings of shape (H*W, D)
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sine-cosine positional embeddings

    Args:
        embed_dim: Output dimension for each position
        pos: List of positions to be encoded (size: M)

    Returns:
        np.ndarray: Positional embeddings of shape (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = einops.einsum(pos, omega, "m, d -> m d")  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MAEDecoder(nn.Module):
    """Masked Autoencoder with Vision Transformer backbone

    This model performs self-supervised learning by masking portions of the input image
    and learning to reconstruct the original image.
    """

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        reg_tokens: int = 0,
        no_reg_tokens: bool = False,
    ) -> None:
        """
        Args:
            image_size: Size of input image
            patch_size: Size of each patch
            in_chans: Number of input channels
            embed_dim: Encoder embedding dimension
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of transformer blocks in decoder
            decoder_num_heads: Number of attention heads in decoder
            mlp_ratio: MLP expansion ratio
            norm_layer: Type of normalization layer
            reg_tokens: Number of regularization tokens
            no_reg_tokens: Whether to use no regularization tokens
        """
        super().__init__()
        # MAE decoder specifics
        self.num_patches = (image_size // patch_size) ** 2
        self.num_prefix_tokens = 1 + reg_tokens if not no_reg_tokens else 0
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_out_dim = patch_size**2 * in_chans
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + self.num_prefix_tokens, decoder_embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.decoder_out_dim, bias=True)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize model weights"""
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            num_prefix_tokens=self.num_prefix_tokens,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights of a specific module

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

    def forward_decoder(
        self, x: torch.Tensor, ids_restore_mim: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of decoder

        Args:
            x: Encoded input tensor
            ids_restore_mim: Original positions of masked tokens

        Returns:
            torch.Tensor: Reconstructed image patches
        """
        B, N, _ = x.shape

        # 임베딩 및 마스크 토큰 처리
        x = self.decoder_embed(x)
        if ids_restore_mim is not None:
            num_masked = ids_restore_mim.shape[1] + self.num_prefix_tokens - N
            # 마스크 토큰과 시각 토큰 결합 최적화
            patch_tokens = torch.cat(
                [
                    x[:, self.num_prefix_tokens :],
                    self.mask_token.expand(B, num_masked, -1),
                ],
                dim=1,
            )
            # 인덱스 복원 최적화
            restore_index = ids_restore_mim.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            patch_tokens = torch.gather(patch_tokens, dim=1, index=restore_index)
            # prefix 토큰과 결합
            x = torch.cat([x[:, : self.num_prefix_tokens], patch_tokens], dim=1)

        # 위치 임베딩 추가 및 트랜스포머 처리
        try:
            x = x + self.decoder_pos_embed
        except ValueError:
            _, n, _ = x.shape
            h = w = int(math.sqrt(n - self.num_prefix_tokens))
            H = W = int(math.sqrt(self.num_patches))
            x = x + resample_abs_pos_embed(
                self.decoder_pos_embed,
                new_size=(h, w),
                old_size=(H, W),
                num_prefix_tokens=self.num_prefix_tokens,
            )

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_pred(self.decoder_norm(x))

        return {
            "prefix_tokens": x[:, : self.num_prefix_tokens],
            "patch": x[:, self.num_prefix_tokens :],
        }

    def forward(
        self, latent: torch.Tensor, ids_restore_mim: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the model

        Args:
            latent: Encoded latent representation
            ids_restore_mim: Original positions of masked tokens

        Returns:
            torch.Tensor: Reconstructed image patches
        """
        return self.forward_decoder(latent, ids_restore_mim)
