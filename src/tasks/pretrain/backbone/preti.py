# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.transform import resize
from timm.models.vision_transformer import Block, PatchEmbed

from src.tasks.pretrain.utils import standardize_input

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sine-cosine positional embeddings.

    Args:
        embed_dim (int): The dimensionality of the embedding.
        grid_size (int): The grid height and width.
        cls_token (bool): If True, add a class token embedding at the beginning.

    Returns:
        np.ndarray: Positional embedding of shape [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim].
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Build 2D sine-cosine positional embedding from a grid of positions.

    Args:
        embed_dim (int): Embedding dimension.
        grid (np.ndarray): Position grid of shape [2, 1, grid_size, grid_size].

    Returns:
        np.ndarray: Positional embedding of shape [grid_size*grid_size, embed_dim].
    """
    assert embed_dim % 2 == 0

    # Use half of the dimensions for height, half for width
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sine-cosine positional embeddings from grid positions.

    Args:
        embed_dim (int): Embedding dimension for each position.
        pos (np.ndarray): Positions to be encoded, shape (M, ).

    Returns:
        np.ndarray: Positional embedding of shape (M, embed_dim).
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


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    """
    Interpolate positional embeddings from a pre-trained checkpoint
    if the current model has a different number of patches.

    Args:
        model: Target model whose positional embedding will be interpolated.
        checkpoint_model (dict): State dict from checkpoint containing "pos_embed".
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


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with Vision Transformer (ViT) backbone."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        """
        Initialize the MaskedAutoencoderViT.

        Args:
            img_size (int): Image size (height/width).
            patch_size (int): Size of one patch.
            in_chans (int): Number of input image channels.
            embed_dim (int): Embedding dimension for encoder.
            depth (int): Number of encoder transformer blocks.
            num_heads (int): Number of attention heads for encoder.
            decoder_embed_dim (int): Embedding dim for decoder.
            decoder_depth (int): Number of decoder transformer blocks.
            decoder_num_heads (int): Number of decoder attention heads.
            mlp_ratio (float): MLP hidden dim ratio in transformer.
            norm_layer (nn.Module): Normalization layer.
            norm_pix_loss (bool): Whether to normalize the pixel reconstruction loss.
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # # --------------------------------------------------------------------------
        # # MAE decoder specifics
        # self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.decoder_pos_embed = nn.Parameter(
        #     torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        # )  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList(
        #     [
        #         Block(
        #             decoder_embed_dim,
        #             decoder_num_heads,
        #             mlp_ratio,
        #             qkv_bias=True,
        #             norm_layer=norm_layer,
        #         )
        #         for i in range(decoder_depth)
        #     ]
        # )

        # self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(
        #     decoder_embed_dim, patch_size**2 * in_chans, bias=True
        # )  # decoder to patch
        # # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        # self.initialize_weights()

    def count_params(self, params: Iterable[nn.Parameter]):
        """
        Count number of model parameters.
        """
        return sum(p.numel() for p in params)

    @property
    def n_params(self, unit=1e6, ndigits=4):
        """
        Return number of model parameters (total and component breakdown).

        Args:
            unit (float): Unit to divide parameter counts (default: 1e6 for 'M').
            ndigits (int): Number of decimal digits to round.

        Returns:
            dict: Dictionary with total and per-component parameter counts.
        """
        return {
            "total_n_params": self.count_params(self.parameters()),
            "total_n_trainable_params": self.count_params(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "total_n_trainable_params_encoder": self.count_params(
                p.numel() for p in self.blocks.parameters()
            ),
            "total_n_trainable_params_decoder": self.count_params(
                p.numel() for p in self.decoder_blocks.parameters()
            ),
        }

    #####################################################################################################
    def initialize_decoder_weights(self):
        """
        Initialize the decoder only, leaving the encoder weights unchanged.
        This function sets positional embeddings, mask token, and initializes decoder blocks and projection.
        """
        # Decoder positional embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Initialize decoder blocks (nn.Linear / nn.LayerNorm)
        for module in self.decoder_blocks:
            module.apply(self._init_weights)

        self.decoder_pred.apply(self._init_weights)
        self.decoder_norm.apply(self._init_weights)

    #####################################################################################################

    def initialize_weights(self):
        """
        Initialize all model weights:
        - Set fixed sine-cosine position embeddings for encoder and decoder.
        - Initialize patch embedding weights.
        - Initialize special tokens and all Linear/LayerNorm layers.
        """
        # initialization (freeze pos_embed with sin-cos)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (not nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Truncated normal is approximately normal for ViT
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Initialize nn.Linear and nn.LayerNorm in all modules
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights for Linear and LayerNorm layers.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            # Use Xavier uniform initialization for Linear layers (official JAX ViT).
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Convert a batch of images into patches.

        Args:
            imgs (torch.Tensor): Image tensor of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Patch tensor of shape (N, L, patch_size**2 * 3).
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        Convert patch vectors back to images.

        Args:
            x (torch.Tensor): Patch tensor of shape (N, L, patch_size**2 * 3).

        Returns:
            torch.Tensor: Reconstructed images of shape (N, 3, H, W).
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    # ============================================================================>
    def random_masking(self, x, mask_ratio):
        """
        Perform random per-sample masking via shuffling.

        Args:
            x (torch.Tensor): Input sequence tensor of shape [N, L, D].
            mask_ratio (float): Ratio of the patches to mask (between 0 and 1).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked sequence [N, len_keep, D],
            binary mask [N, L], and restore indices [N, L].
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        ids_shuffle = torch.argsort(noise, dim=1)  # ascending: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # to restore original order

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # ============================================================================>
    def roi_random_masking(self, x, mask, mask_ratio):
        """
        Perform random masking only within Regions Of Interest (ROI).

        Args:
            x (torch.Tensor): Input sequence [N, L, D].
            mask (torch.Tensor): ROI mask [N, L]: 1 for valid ROI, 0 for background.
            mask_ratio (float): Masking ratio (e.g., 0.75).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked sequence, binary mask, and restoration indices.
        """
        N, L, D = x.shape  # batch, length, dim

        num_roi = mask.sum(dim=1)
        num_keep = int(L * (1 - mask_ratio))
        num_mask = L - num_keep

        noise = (
            torch.rand(N, L, device=x.device) * mask
        )  # noise in [0, 1], only ROI portions have noise

        ids_shuffle = torch.argsort(
            noise, dim=1, descending=True
        )  # descending: high is keep, low is background/remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate mask (0: keep, 1: remove)
        final_mask = torch.ones([N, L], device=x.device)
        final_mask[:, :num_keep] = 0
        final_mask = torch.gather(final_mask, dim=1, index=ids_restore)

        return x_masked, final_mask, ids_restore

    # ============================================================================>

    def forward_encoder(self, x, mask_ratio):
        """
        Forward pass through ViT encoder with random masking and positional encoding.

        Args:
            x (torch.Tensor): Input image tensor.
            mask_ratio (float): Ratio of patches to mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Latent features (with CLS), binary mask, restoration indices.
        """
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Forward pass through the decoder, reconstructing masked patches.

        Args:
            x (torch.Tensor): Latent tokens from encoder [N, L+1, D].
            ids_restore (torch.Tensor): Indices to restore order of masked sequence.

        Returns:
            torch.Tensor: Patch predictions [N, L, patch_size*patch_size*channels].
        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Compute the reconstruction loss for masked patches.

        Args:
            imgs (torch.Tensor): Input images [N, 3, H, W].
            pred (torch.Tensor): Predicted patches [N, L, p*p*3].
            mask (torch.Tensor): Mask, 0 is keep, 1 is remove [N, L].

        Returns:
            torch.Tensor: Mean squared error loss over masked patches.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """
        Full MAE forward: encoding, masking, decoding, and compute loss.

        Args:
            imgs (torch.Tensor): Batch of input images.
            mask_ratio (float): Ratio of masked patches (default 0.75).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Loss, predictions, binary mask.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_small_dec512d8b(patch_size=16, **kwargs):
    """
    Build a small ViT MAE with decoder: 256 dim, 4 blocks.

    Args:
        patch_size (int): Patch size.

    Returns:
        MaskedAutoencoderViT: Instantiated model.
    """
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    """
    Build a base ViT MAE with decoder: 512 dim, 8 blocks.

    Returns:
        MaskedAutoencoderViT: Instantiated model.
    """
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    """
    Build a large ViT MAE with decoder: 512 dim, 8 blocks.

    Returns:
        MaskedAutoencoderViT: Instantiated model.
    """
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    """
    Build a huge ViT MAE with decoder: 512 dim, 8 blocks.

    Returns:
        MaskedAutoencoderViT: Instantiated model.
    """
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


class PreTI(nn.Module):
    """
    PreTI wrapper for feature extraction using a pretrained MAE ViT backbone.
    Follows VilRef pattern for simple feature output.
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
        image_resolution: int = 224,
    ):
        """
        Initialize the PreTI feature extractor.

        Args:
            patch_size (int): Patch size to use.
            embed_dim (int): Embedding dimensionality.
            image_resolution (int): Size (height/width) used for input images.
        """
        super().__init__()

        # Create ViT-Base model
        self.model = mae_vit_base_patch16()

        # Load checkpoint
        ckpt = torch.load("your_path/dev/weights/ssl/PRETI.pth", map_location="cpu")[
            "model"
        ]
        new_ckpt = {}
        for k, v in ckpt.items():
            if (
                k.startswith("decoder")
                or k.startswith("mask_token")
                or k.startswith("age")
                or k.startswith("gender")
                or k.startswith("perceptual")
            ):
                continue
            new_ckpt[k] = v

        self.model.load_state_dict(new_ckpt, strict=True)
        self.model.eval()

        self.patch_size = patch_size
        self.input_resolution = image_resolution
        self.grid_size = (
            self.input_resolution // self.patch_size,
            self.input_resolution // self.patch_size,
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        """
        Forward pass for feature extraction.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            dict: Dictionary containing extracted features under 'features' key.
        """
        x = resize(x, (224, 224))
        x = standardize_input(x)
        # Extract features without masking (mask_ratio=0)
        latent, _, _ = self.model.forward_encoder(x, mask_ratio=0.0)

        # Return CLS token features
        return {
            "features": latent,
        }

    def forward_features(self, x: torch.Tensor):
        """
        Alias for forward. Extract features from input images.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            dict: Extracted features.
        """
        return self.forward(x)
