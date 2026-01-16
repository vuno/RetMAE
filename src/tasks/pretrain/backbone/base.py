"""Base classes and common utilities for backbone models."""

from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn
from kornia.geometry.transform import resize

from src.tasks.pretrain.utils import standardize_input
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class BackboneBase(nn.Module):
    """Base class for backbone models.

    This class provides common functionality for loading checkpoints and removing classifier layers.

    Attributes:
        REMOVE_LAYER_NAMES (List[str]): List of possible classifier layer names to remove
        standardize (bool): Whether to standardize input images
        # Standardization uses ImageNet constants from timm.data.constants
        model (nn.Module): The underlying model instance
        remove_layer_name (str): Name of the classifier layer that was removed
    """

    REMOVE_LAYER_NAMES = ["classifier", "fc", "head", "fc_norm"]

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        standardize: bool = True,
    ) -> None:
        """Initialize the backbone model.

        Args:
            ckpt_path: Path to checkpoint file to load weights from. Defaults to None.
            standardize: Whether to standardize input images using ImageNet stats. Defaults to True.
        """
        super().__init__()
        self.ckpt_path = ckpt_path
        self.standardize = standardize
        self.remove_layer_name = None
        self.model = None

    def _setup(self, load_ckpt_strict: bool = True) -> None:
        """Model setup function.

        Finds and removes the classifier layer from the model during initial setup.
        The classifier layer name is stored in self.remove_layer_name.

        Raises:
            ValueError: If no classifier layer could be found in the model.
        """
        # Find and remove all classifier layers
        removed_layers = []
        for name in self.REMOVE_LAYER_NAMES:
            if hasattr(self.model, name):
                delattr(self.model, name)
                removed_layers.append(name)

        if not removed_layers:
            raise ValueError(
                f"Could not find any of the remove layer names: {self.REMOVE_LAYER_NAMES}"
            )

        self.remove_layer_name = removed_layers  # Store all removed layer names

        # Standardization is handled by the standardize_input function from utils

        if self.ckpt_path is not None:
            ckpt = torch.load(self.ckpt_path, map_location="cpu")
            if ckpt.get("state_dict") is not None:
                ckpt = ckpt["state_dict"]

            new_ckpt = {}
            for k, v in ckpt.items():
                if "decoder" in k:
                    continue
                if "_orig_mod." in k:
                    k = k.replace("_orig_mod.", "")
                if "model." in k:
                    k = k.replace("model.", "")
                if "encoder.backbone" in k:
                    k = k.replace("encoder.backbone.", "")
                if "mi_max" in k:
                    continue

                new_ckpt[k] = v

            self.model.load_state_dict(new_ckpt, strict=load_ckpt_strict)
            log.info(f"Loaded checkpoint from {self.ckpt_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TimmViTBackbone(torch.nn.Module):
    """timm 라이브러리의 Vision Transformer 모델을 위한 백본 클래스입니다.

    Args:
        model_name (str): timm 모델 이름
        standardize (bool, optional): 입력 이미지 표준화 여부. Defaults to True.
        **model_kwargs: timm.create_model에 전달할 추가 인자들

    Attributes:
        model (nn.Module): 기본 Vision Transformer 모델
        img_size (int): 입력 이미지 크기
        num_prefix_tokens (int): 프리픽스 토큰 개수 (DINOv2 모델용)
        standardize (bool): 입력 표준화 여부
        # Standardization uses ImageNet constants from timm.data.constants
    """

    def __init__(
        self,
        model_name: str,
        standardize: bool = True,
        **model_kwargs,
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, **model_kwargs)
        del self.model.head

        if "eva02" in model_name:
            self.img_size = 224
        elif "dinov2" in model_name:
            self.img_size = 518
            self.num_prefix_tokens = 5
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.standardize = standardize
        # Standardization is handled by the standardize_input function from utils

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_features(x)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """특징 추출을 위한 순전파를 수행합니다.

        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            Dict[str, torch.Tensor]: 특징 맵을 포함하는 딕셔너리
        """
        if self.img_size != x.shape[-2]:
            x = resize(x, (self.img_size, self.img_size))

        if self.standardize:
            x = standardize_input(x)

        return {
            "features": self.model.forward_features(x),
        }

    @torch.inference_mode()
    def get_self_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """마지막 트랜스포머 블록의 self-attention map을 추출합니다.

        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            Dict[str, torch.Tensor]: 각 스테이지별 attention map을 포함하는 딕셔너리
        """
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
        is_train = self.training
        if is_train:
            self.eval()
        _ = self.forward_features(x)
        if is_train and not self.training:
            self.train()

        # Remove hook
        hook_early.remove()
        hook_middle.remove()
        hook_late.remove()

        return attention_maps

    @torch.inference_mode()
    def get_cosine_similarity_map(self, x: torch.Tensor) -> torch.Tensor:
        """마지막 트랜스포머 블록의 cosine similarity map을 추출합니다.

        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            torch.Tensor: Cosine similarity map
        """
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
    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """각 스테이지의 중간 특징을 추출합니다.

        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            Dict[str, torch.Tensor]: 각 스테이지별 특징을 포함하는 딕셔너리
        """
        features = {}
        stage = ["patch", "early", "middle", "late"]
        stage_idx = 0

        def forward_hook(module, input, output):
            nonlocal features
            nonlocal stage_idx
            nonlocal stage

            features[stage[stage_idx]] = output.detach().cpu()
            stage_idx += 1

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
