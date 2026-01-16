"""FLAIR and related Vision Transformer models."""

from typing import Dict, List, Optional, Union

import torch
import torchvision
from kornia.geometry.transform import resize

from src.utils import RankedLogger

from .base import BackboneBase

log = RankedLogger(__name__, rank_zero_only=True)


class FLAIRProjectionLayer(torch.nn.Module):
    """FLAIR 모델의 프로젝션 레이어입니다.

    Args:
        layer (torch.nn.Module): 프로젝션에 사용할 레이어
        projection (bool): 프로젝션 적용 여부. Defaults to True.
        norm (bool): 정규화 적용 여부. Defaults to True.
    """

    def __init__(self, layer, projection=True, norm=True):
        super().__init__()

        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 텐서에 프로젝션과 정규화를 적용합니다.

        Args:
            x (torch.Tensor): 입력 텐서

        Returns:
            torch.Tensor: 프로젝션과 정규화가 적용된 출력 텐서
        """
        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)

        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x


class FLAIR(torch.nn.Module):
    """FLAIR 모델 구현체입니다.

    Args:
        vision_type (str): 비전 인코더 타입. ["resnet_v1", "resnet_v2", "efficientnet"] 중 하나. Defaults to "resnet".
        pretrained (bool): 사전학습된 가중치 사용 여부. Defaults to True.
        proj_dim (int): 프로젝션 차원. Defaults to 512.
        proj_bias (bool): 프로젝션 레이어의 편향 사용 여부. Defaults to False.
        projection (bool): 프로젝션 적용 여부. Defaults to True.
        norm (bool): 정규화 적용 여부. Defaults to True.
        ckpt_path (str): 체크포인트 경로. Defaults to "your_path/weights/ssl/flair_resnet.pth".
    """

    def __init__(
        self,
        vision_type="resnet",
        pretrained=True,
        proj_dim=512,
        proj_bias=False,
        projection=True,
        norm=True,
        ckpt_path="your_path/dev/weights/ssl/flair_resnet.pth",
    ):
        super().__init__()
        self.proj_dim = self.embed_dim = proj_dim

        # Assert vision encoders
        if vision_type not in ["resnet_v1", "resnet_v2", "efficientnet"]:
            print("Vision model should be one of resnet/efficientnet... using resnet.")
            vision_type = "resnet_v1"

        # Set vision encoder architecture and pretrained weights
        if vision_type == "resnet_v1" or vision_type == "resnet_v2":
            # Set pretrained weights from Imagenet and get model
            if vision_type == "resnet_v1":
                weights = "IMAGENET1K_V1" if pretrained else None
            elif vision_type == "resnet_v2":
                weights = "IMAGENET1K_V2" if pretrained else None
            else:
                weights = "IMAGENET1K_V1" if pretrained else None
            print("Pretrained weights: " + str(weights))
            self.model = torchvision.models.resnet50(weights=weights)
            # Set number of extracted features
            self.vision_dim = 2048
            # Replace classifier by Identity layer
            self.model.fc = torch.nn.Identity()
        elif vision_type == "efficientnet":
            weights = "IMAGENET1K_V1" if pretrained else None
            self.model = torchvision.models.efficientnet_b7(weights=weights)
            self.vision_dim = 2096

        # Set output dimension
        if projection:
            self.out_dim = self.proj_dim
        else:
            self.out_dim = self.vision_dim

        # Set projection head
        self.projection_head_vision = FLAIRProjectionLayer(
            layer=torch.nn.Linear(self.vision_dim, self.proj_dim, bias=proj_bias),
            projection=projection,
            norm=norm,
        )

        ckpt = torch.load(ckpt_path)

        new_ckpt = {}
        for k, v in ckpt.items():
            if not k.startswith("vision_model."):
                continue
            else:
                new_ckpt[k.replace("vision_model.", "")] = v

        self.load_state_dict(new_ckpt)

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """입력 이미지에 대한 특징을 추출합니다.

        Args:
            pixel_values (torch.Tensor): 입력 이미지 텐서

        Returns:
            Dict[str, torch.Tensor]: 추출된 특징을 포함하는 딕셔너리
                - features: 추출된 이미지 특징
        """
        if pixel_values.shape[2] != 512:
            pixel_values = resize(pixel_values, (512, 512))

        # Forwards trough vision encoder
        embed = self.model(pixel_values)

        # Compute projection from vision embedding to multi-modal projection
        embed = self.projection_head_vision(embed)
        return {
            "features": embed,
        }

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

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

        # forward hook 등록
        hook_patch = self.model.maxpool.register_forward_hook(forward_hook)
        hook_early = self.model.layer1.register_forward_hook(forward_hook)
        hook_middle = self.model.layer3.register_forward_hook(forward_hook)
        hook_late = self.model.layer4.register_forward_hook(forward_hook)

        # forward pass
        is_train = self.model.training
        if is_train:
            self.model.eval()
        _ = self(x)
        if is_train and not self.model.training:
            self.model.train()

        # remove hook
        hook_patch.remove()
        hook_early.remove()
        hook_middle.remove()
        hook_late.remove()

        return features


class KeepFIT(FLAIR):
    """KeepFIT 모델 클래스입니다.

    FLAIR 모델을 확장하여 KeepFIT 사전학습 가중치를 로드하고 사용합니다.

    Args:
        **kwargs: FLAIR 모델에 전달할 추가 인자들
    """

    def __init__(self, **kwargs):
        super().__init__(
            ckpt_path="your_path/dev/weights/ssl/KeepFIT (flair+MM).pth",
            **kwargs,
        )
        self.num_prefix_tokens = 1
