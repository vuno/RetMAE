"""MAE (Masked Autoencoder) related models."""

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.transform import resize
from timm.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.tasks.pretrain.utils import (
    butterworth_high_pass_filter,
    folded_freq_quadrants,
    patchified_binary_mask,
    soft_edge_mask,
    standardize_input,
)
from src.utils import RankedLogger

from .base import BackboneBase

log = RankedLogger(__name__, rank_zero_only=True)


class MAEVisionTransformer(BackboneBase):
    """Backbone model for MAE pre-training.

    Args:
        model_name (str): Name of the model to load.
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        ckpt_path (Optional[str]): Checkpoint path. Defaults to None.
        high_freq_masking (bool): Whether to use high frequency masking. Defaults to True.
        load_ckpt_strict (bool): Whether to load checkpoint strictly. Defaults to True.
        **model_kwargs: Additional keyword arguments for timm.create_model.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        ckpt_path: Optional[str] = None,
        high_freq_masking: bool = True,
        load_ckpt_strict: bool = True,
        **model_kwargs,
    ):
        super().__init__(ckpt_path=ckpt_path)

        for k, v in model_kwargs.items():
            setattr(self, k, v)

        if model_name == "vit":
            self.model = VisionTransformer(**model_kwargs)
        else:
            import timm

            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                pretrained_strict=load_ckpt_strict,
                num_classes=0,
                **model_kwargs,
            )

        if not hasattr(self, "num_prefix_tokens"):
            self.num_prefix_tokens = self.model.num_prefix_tokens

        self.out_dim = self.model.embed_dim
        self.patch_size = self.model.patch_embed.patch_size
        self.high_freq_masking = high_freq_masking
        self._setup(load_ckpt_strict=load_ckpt_strict)

    def forward_features(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
        fixed_mask: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass function for feature extraction.

        Args:
            x (torch.Tensor): Input image tensor
            mask_ratio (float, optional): Ratio of patches to mask. Defaults to 0.75.
            fixed_mask (bool, optional): Whether to use fixed mask. Defaults to False.
        Returns:
            Dict[str, torch.Tensor]:
                - Encoded features
                - Mask
                - Indices for restoration
        """
        if self.img_size != x.shape[-2]:
            x = resize(x, (self.img_size, self.img_size))

        x = standardize_input(x)

        # model forward
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)

        B = x.shape[0]
        L = x.shape[1] - self.num_prefix_tokens

        mask_mim = torch.zeros((B, L), device=x.device)
        if mask_ratio is not None and mask_ratio > 0:
            # Masking
            register_tokens = x[:, : self.num_prefix_tokens, :]
            patch_tokens = x[:, self.num_prefix_tokens :, :]
            patch_tokens, mask_mim, ids_restore_mim = self.random_masking(
                x=patch_tokens,
                mask_ratio=mask_ratio,
                fixed_mask=fixed_mask,
            )
            x = torch.cat([register_tokens, patch_tokens], dim=1)
        else:
            ids_restore_mim = None

        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return {
            "features": x,
            "mask_mim": mask_mim,
            "restore_ids_mim": ids_restore_mim,
        }

    def forward_intermediates(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor

        Returns:
            Intermediate features or tuple of output and intermediate features
        """
        x = standardize_input(x)
        return self.model.forward_intermediates(x)

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
        fixed_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate mask for MAE pre-training.

        Args:
            x (torch.Tensor): Image tensor with data augmentation applied (B x L x C)
            mask_ratio (float, optional): Ratio of total patches to mask. Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Masked image, mask, and indices to restore original image
                - x_masked (torch.Tensor): Masked image
                - mask_mim (torch.Tensor): Mask used for masking
                - ids_restore (torch.Tensor): Indices to restore original image
                - fixed_mask (bool): Whether to use fixed mask
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # Generate indices and mask at once
        if fixed_mask:
            # Use manual seed for reproducible noise
            generator = torch.Generator(device=x.device).manual_seed(42)
            noise = torch.rand(N, L, device=x.device, generator=generator)
        else:
            noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        # Optimize mask generation
        mask_mim = torch.zeros(N, L, device=x.device)
        mask_mim.scatter_(1, ids_shuffle[:, len_keep:], 1.0)

        # Optimize gather operation
        expand_shape = (-1, -1, D)
        gather_idx = ids_keep.unsqueeze(-1).expand(expand_shape)
        x_masked = x.gather(1, gather_idx)

        return x_masked, mask_mim, ids_restore

    @torch.inference_mode()
    def get_self_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get self-attention map from the last transformer block using hook."""

        attention_maps = {}
        stage = ["early", "middle", "late"]
        stage_idx = 0

        def attention_hook(module, input, output):
            nonlocal attention_maps
            nonlocal stage_idx
            nonlocal stage

            # qkv 계산
            B, N, _ = input[0].shape
            qkv = (
                module.qkv(input[0])
                .reshape(B, N, 3, module.num_heads, module.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, _ = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)

            q = q * module.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attention_maps[stage[stage_idx]] = attn.detach().cpu()
            stage_idx += 1

        # attention 모듈 자체에 hook 등록
        # early stage
        hook_early = self.model.blocks[0].attn.register_forward_hook(attention_hook)
        # middle stage
        hook_middle = self.model.blocks[len(self.model.blocks) // 2].attn.register_forward_hook(
            attention_hook
        )
        # late stage
        hook_late = self.model.blocks[-1].attn.register_forward_hook(attention_hook)

        # Forward pass
        is_train = self.model.training
        if is_train:
            self.model.eval()
        _ = self.forward_features(x)
        if is_train and not self.model.training:
            self.model.train()

        # Remove hook
        hook_early.remove()
        hook_middle.remove()
        hook_late.remove()

        return attention_maps

    @torch.inference_mode()
    def get_cosine_similarity_map(self, x: torch.Tensor) -> torch.Tensor:
        """Get cosine similarity map from the last transformer block using hook."""

        similarity_maps = None

        def attention_hook(module, input, output):
            nonlocal similarity_maps
            # L2 정규화를 통한 cosine similarity 계산
            output_norm = torch.nn.functional.normalize(output, p=2, dim=-1)
            similarity_maps = output_norm @ output_norm.transpose(-2, -1)
            similarity_maps = similarity_maps.detach()

        # attention 모듈 자체에 hook 등록
        hook = self.model.blocks[-1].register_forward_hook(attention_hook)

        # Forward pass
        is_train = self.model.training
        if is_train:
            self.model.eval()
        _ = self.forward_features(x)
        if is_train and not self.model.training:
            self.model.train()

        # Remove hook
        hook.remove()

        return similarity_maps

    @torch.inference_mode()
    def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features from the last transformer block using hook."""
        features = {}
        stage = ["patch", "early", "middle", "late"]
        stage_idx = 0

        def forward_hook(module, input, output):
            nonlocal features
            nonlocal stage_idx
            nonlocal stage

            features[stage[stage_idx]] = output.detach().cpu()
            stage_idx += 1
            if stage_idx == len(stage):
                stage_idx = 0

        # forward hook 등록
        hook_patch = self.model.patch_embed.register_forward_hook(forward_hook)
        hook_early = self.model.blocks[0].register_forward_hook(forward_hook)
        hook_middle = self.model.blocks[len(self.model.blocks) // 2].register_forward_hook(
            forward_hook
        )
        hook_late = self.model.blocks[-1].register_forward_hook(forward_hook)

        # forward pass
        is_train = self.model.training
        if is_train:
            self.model.eval()
        _ = self.forward_features(x)
        if is_train and not self.model.training:
            self.model.train()

        # remove hook
        hook_patch.remove()
        hook_early.remove()
        hook_middle.remove()
        hook_late.remove()

        return features

    @torch.no_grad()
    def get_intermediate_features_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        def forward_hook(module, input, output):
            nonlocal features

            out = output.detach()
            features.append(out)

        # forward hook 등록
        hooks = []
        target_layers = "blocks."
        for name, module in self.model.named_modules():
            append = False
            if not isinstance(module, torch.nn.Module):
                continue
            # Patch Embedding Layer
            if name == "patch_embed":
                append = True

            # Transformer Layers
            elif any(name.startswith(layer) for layer in target_layers):
                if name.split(".")[-1].isdigit():
                    append = True

            if append:
                hooks.append(module.register_forward_hook(forward_hook))

        # forward pass
        is_train = self.model.training
        if is_train:
            self.model.eval()
        _ = self.forward_features(x)
        if is_train and not self.model.training:
            self.model.train()

        # remove hook
        for hook in hooks:
            hook.remove()

        new_features = []
        for i, feature in enumerate(features):
            if i == 0:
                new_features.append(feature)
            else:
                new_features.append(feature[:, self.num_prefix_tokens :])
        return new_features


class RetMAE(MAEVisionTransformer):
    HIGH_FREQ_MASKING_PARAMS = {
        "butterworth_cutoff": 14.0470391901718,
        "butterworth_order": 2,
        "soft_mask_th": 0.0868832016680601,
        "soft_mask_sigma": 10.133165328972893,
        "boundary_cutoff": 0.6701278573157876,
        "blur_sigma": 0.5185059445302515,
        "blur_radius": 1,
        "mask_ratio": 0.2506265193179896,
        "center_mask_radius_ratio_x": 0.38237365605666845,
        "center_mask_radius_ratio_y": 0.41296138198996574,
    }

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        ckpt_path: Optional[str] = None,
        load_ckpt_strict: bool = True,
        img_size: int = 224,
        high_freq_modeling_1: bool = False,
        high_freq_modeling_2: bool = False,
        **model_kwargs,
    ):
        self.high_freq_modeling_1 = high_freq_modeling_1
        self.high_freq_modeling_2 = high_freq_modeling_2
        if self.high_freq_modeling_2:
            model_kwargs["in_chans"] = 4

        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            ckpt_path=ckpt_path,
            load_ckpt_strict=load_ckpt_strict,
            **model_kwargs,
        )
        self.img_size = img_size

    def forward_features(
        self,
        x: torch.Tensor,
        masking: bool = False,
        random_masking: bool = False,
        fixed_mask: bool = False,
        visible_ratio: float = 0.5,
        mask_ratio: float = 0.3,
        infer_high_freq: bool = False,
        infer_low_freq: bool = False,
        infer_random: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the frequency-aware MAE Vision Transformer.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            masking (bool, optional): Whether to apply masking. Defaults to False.
            random_masking (bool, optional): Whether to use random masking instead of frequency-aware masking. Defaults to False.
            fixed_mask (bool, optional): Whether to use a fixed random seed for reproducible masks. Defaults to False.
            visible_ratio (float, optional): Fraction of tokens to keep during masking. Defaults to 0.5.
            mask_ratio (float, optional): Ratio of tokens to mask. Defaults to 0.3.
            infer_high_freq (bool, optional): Whether to only process high-frequency tokens. Defaults to False.
            infer_low_freq (bool, optional): Whether to only process low-frequency tokens. Defaults to False.
            infer_random (bool, optional): Whether to only process random tokens. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - features: Final feature representations
                - mask_mim: Mask tensor for reconstruction
                - restore_ids_mim: Indices to restore original token order

        Raises:
            AssertionError: If both infer_high_freq and infer_low_freq are True.
        """
        assert not (
            infer_high_freq and infer_low_freq
        ), "infer_high_freq and infer_low_freq cannot be True at the same time"

        if self.img_size != x.shape[-2]:
            x = resize(x, (self.img_size, self.img_size))

        with torch.no_grad():
            high_freq_res = self.get_high_freq(x)

        high_freq_mask = high_freq_res["high_freq_mask"]

        x = standardize_input(x)
        if self.high_freq_modeling_2:
            x = torch.cat([x, high_freq_res["high_freq_image"]], dim=1)

        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)

        B, L, D = x.shape
        L -= self.model.num_prefix_tokens

        mask_mim = None
        ids_restore_mim = None
        if masking and mask_ratio > 0:
            # 5) Separate prefix and patch tokens
            prefix, patches = (
                x[:, : self.model.num_prefix_tokens, :],
                x[:, self.model.num_prefix_tokens :, :],
            )

            # 6) Apply masking
            if self.high_freq_modeling_1:
                patches, mask_mim, ids_restore_mim = self.high_freq_prioritized_masking(
                    x=patches,
                    high_freq_mask=high_freq_mask,
                    fixed_mask=fixed_mask,
                )

            else:
                patches, mask_mim, ids_restore_mim = self.random_masking(
                    x=patches,
                    mask_ratio=mask_ratio,
                    fixed_mask=fixed_mask,
                )
            # 7) Reassemble tokens
            x = torch.cat([prefix, patches], dim=1)

        elif infer_high_freq:
            prefix, patches = (
                x[:, : self.model.num_prefix_tokens, :],
                x[:, self.model.num_prefix_tokens :, :],
            )
            x = torch.cat(
                [
                    prefix,
                    patches.masked_select(
                        high_freq_mask.bool().view(B, patches.shape[1], 1).expand(-1, -1, D)
                    ).reshape(B, -1, D),
                ],
                dim=1,
            )

        elif infer_low_freq:
            prefix, patches = (
                x[:, : self.model.num_prefix_tokens, :],
                x[:, self.model.num_prefix_tokens :, :],
            )
            x = torch.cat(
                [
                    prefix,
                    patches.masked_select(
                        ~high_freq_mask.bool().view(B, patches.shape[1], 1).expand(-1, -1, D)
                    ).reshape(B, -1, D),
                ],
                dim=1,
            )

        elif infer_random:
            prefix, patches = (
                x[:, : self.model.num_prefix_tokens, :],
                x[:, self.model.num_prefix_tokens :, :],
            )
            patches, _, _ = self.random_masking(
                x=patches,
                mask_ratio=mask_ratio,
                fixed_mask=fixed_mask,
            )
            x = torch.cat([prefix, patches], dim=1)

        # 8) Pre-normalization
        x = self.model.norm_pre(x)

        # 9) Transformer blocks with optional checkpointing for memory
        if getattr(self, "use_gradient_checkpointing", False):
            x = checkpoint(lambda inp: self.model.blocks(inp), x)
        else:
            x = self.model.blocks(x)

        # 10) Post-normalization
        x = self.model.norm(x)

        return {
            "features": x,
            "mask_mim": mask_mim,
            "restore_ids_mim": ids_restore_mim,
            "high_freq": high_freq_res["high_freq"],
        }

    def get_high_freq(
        self, x: torch.Tensor, return_mask: bool = False, out_res: int = 224
    ) -> torch.Tensor:
        """
        Get high-frequency mask for the input tensor.
        """
        fov_mask = soft_edge_mask(
            images=x,
            threshold_frac=self.HIGH_FREQ_MASKING_PARAMS["soft_mask_th"],
            sigma=self.HIGH_FREQ_MASKING_PARAMS["soft_mask_sigma"],
        )
        high_freq_res = butterworth_high_pass_filter(
            image=x[:, 1:2],
            fov_mask=fov_mask,
            cutoff=self.HIGH_FREQ_MASKING_PARAMS["butterworth_cutoff"],
            order=self.HIGH_FREQ_MASKING_PARAMS["butterworth_order"],
            boundary_cut=self.HIGH_FREQ_MASKING_PARAMS["boundary_cutoff"],
            blur_sigma=self.HIGH_FREQ_MASKING_PARAMS["blur_sigma"],
            blur_radius=self.HIGH_FREQ_MASKING_PARAMS["blur_radius"],
        )

        high_freq_mask = patchified_binary_mask(
            high_freq_res["high_freq_image"],
            patch_size=self.patch_size,
            mask_ratio=self.HIGH_FREQ_MASKING_PARAMS["mask_ratio"],
        )
        if return_mask:
            return high_freq_mask
        else:
            if high_freq_res["high_freq_image"].ndim == 3:  # [B, H, W]
                high_freq_image = F.interpolate(
                    high_freq_res["high_freq_image"].unsqueeze(1),
                    size=(out_res, out_res),
                    mode="bilinear",
                ).squeeze(1)
            else:
                high_freq_image = high_freq_res["high_freq_image"]

            return {
                "high_freq": folded_freq_quadrants(high_freq_res["high_freq"]),  # (B, 4, H/2, W/2)
                "high_freq_mask": high_freq_mask,
                "high_freq_image": high_freq_image,
                "fov_mask": fov_mask,
            }

    def high_freq_prioritized_masking(
        self,
        x: torch.Tensor,
        high_freq_mask: torch.Tensor,
        fixed_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """High-frequency aware masking for MAE pre-training.

        Args:
            x (torch.Tensor): Image tensor with data augmentation applied (B x L x C)
            high_freq_mask (torch.Tensor): Binary mask indicating high-frequency regions (B x L)
                                           1 = high frequency, 0 = low frequency
            fixed_mask (bool, optional): Whether to use fixed mask. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Masked image, mask, and indices to restore original image
                - x_masked (torch.Tensor): Masked image
                - mask_mim (torch.Tensor): Mask used for masking
                - ids_restore (torch.Tensor): Indices to restore original image
        """
        N, L, D = x.shape  # batch, length, dim

        # high_freq_mask 형태 정규화
        # - (N, L) 또는 (N, H, W) (단, H * W == L 이어야 함)
        if high_freq_mask.dim() == 3:
            n, h, w = high_freq_mask.shape
            assert (
                n == N and h * w == L
            ), f"Expected high_freq_mask shape ({N}, {L}) or ({N}, H, W) with H*W={L}, got {high_freq_mask.shape}"
            high_freq_mask = high_freq_mask.view(N, L)
        else:
            assert high_freq_mask.shape == (
                N,
                L,
            ), f"Expected high_freq_mask shape ({N}, {L}), got {high_freq_mask.shape}"

        # 각 배치에 대해 마스킹 수행
        mask_mim = torch.zeros(N, L, device=x.device)
        ids_restore_list = []
        ids_keep_list = []

        for i in range(N):
            # High frequency 영역 (1인 부분)과 Low frequency 영역 (0인 부분) 인덱스 찾기
            high_freq_indices = torch.where(high_freq_mask[i] == 1)[0]
            low_freq_indices = torch.where(high_freq_mask[i] == 0)[0]

            # High frequency 영역에서 절반 선택 (우선적으로 마스킹)
            num_high_to_mask = len(high_freq_indices) // 2

            if fixed_mask:
                # Use manual seed for reproducible selection
                generator = torch.Generator(device=x.device).manual_seed(42)
                perm_high = torch.randperm(
                    len(high_freq_indices), device=x.device, generator=generator
                )
                perm_low = torch.randperm(
                    len(low_freq_indices), device=x.device, generator=generator
                )
            else:
                perm_high = torch.randperm(len(high_freq_indices), device=x.device)
                perm_low = torch.randperm(len(low_freq_indices), device=x.device)

            # High frequency에서 절반 선택
            selected_high_indices = high_freq_indices[perm_high[:num_high_to_mask]]

            # ------------------------------------------------------------------
            # 나머지(High frequency가 아닌 영역)에서 전체 토큰 기준 70%가 되도록 추가 마스킹
            # 예) high_freq 영역이 전체의 30%라면,
            #     - high_freq 절반  = 15% (위에서 선택)
            #     - low_freq 쪽에서 55%를 추가로 선택 → 총 70% 마스킹
            # ------------------------------------------------------------------
            target_total_mask_ratio = 0.70
            target_total_mask = int(L * target_total_mask_ratio)
            remaining_to_mask = max(target_total_mask - num_high_to_mask, 0)
            num_low_to_mask = min(remaining_to_mask, len(low_freq_indices))
            selected_low_indices = low_freq_indices[perm_low[:num_low_to_mask]]

            # 선택된 인덱스들을 마스킹
            masked_indices = torch.cat([selected_high_indices, selected_low_indices])
            mask_mim[i, masked_indices] = 1.0

            # Keep되는 인덱스 (마스킹되지 않은 인덱스)
            all_indices = torch.arange(L, device=x.device)
            mask_bool = mask_mim[i].bool()
            ids_keep = all_indices[~mask_bool]

            # Restore를 위한 인덱스 생성
            # ids_shuffle 생성: [kept indices, masked indices]
            ids_masked = all_indices[mask_bool]
            ids_shuffle = torch.cat([ids_keep, ids_masked])

            # ids_restore: ids_shuffle의 역순서
            ids_restore = torch.argsort(ids_shuffle)

            ids_keep_list.append(ids_keep)
            ids_restore_list.append(ids_restore)

        # Stack all ids_keep and ids_restore
        max_keep_len = max(len(ids) for ids in ids_keep_list)

        # Pad ids_keep to same length and gather x
        x_masked_list = []
        for i in range(N):
            ids_keep = ids_keep_list[i]
            gather_idx = ids_keep.unsqueeze(-1).expand(-1, D)
            x_masked_i = x[i].gather(0, gather_idx)
            x_masked_list.append(x_masked_i)

        # Stack to create batch
        x_masked = torch.stack(x_masked_list, dim=0)
        ids_restore = torch.stack(ids_restore_list, dim=0)

        return x_masked, mask_mim, ids_restore
