"""FLAIR and related Vision Transformer models."""

from typing import Dict, List, Optional, Union

import torch
import torchvision
from kornia.geometry.transform import resize

from src.utils import RankedLogger, ckpt_path as _ckpt

from .base import BackboneBase

log = RankedLogger(__name__, rank_zero_only=True)


class FLAIRProjectionLayer(torch.nn.Module):
    """Projection layer used inside the FLAIR head.

    Args:
        layer: projection module.
        projection: whether to apply projection.
        norm: whether to L2-normalise inputs/outputs.
    """

    def __init__(self, layer, projection=True, norm=True):
        super().__init__()

        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply projection and (optionally) L2 normalisation to ``x``."""
        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)

        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x


class FLAIR(torch.nn.Module):
    """FLAIR vision-language pre-trained encoder.

    Args:
        vision_type: one of ``{"resnet_v1", "resnet_v2", "efficientnet"}``.
        pretrained: load ImageNet-1K weights for the trunk.
        proj_dim: projection dimension.
        proj_bias: whether the projection layer uses a bias.
        projection: enable the projection head.
        norm: apply L2 normalisation in the projection layer.
        ckpt_path: optional override; defaults to ``$CKPT_ROOT/flair/flair_resnet.pth``.
    """

    def __init__(
        self,
        vision_type="resnet",
        pretrained=True,
        proj_dim=512,
        proj_bias=False,
        projection=True,
        norm=True,
        ckpt_path: str | None = None,
    ):
        super().__init__()
        if ckpt_path is None:
            ckpt_path = str(_ckpt("flair", "flair_resnet.pth"))
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
    """KeepFIT (FLAIR + MM) pre-trained encoder.

    Loads the same trunk as :class:`FLAIR` but with KeepFIT weights from
    ``$CKPT_ROOT/keepfit/KeepFIT (flair+MM).pth``.
    """

    def __init__(self, **kwargs):
        super().__init__(
            ckpt_path=str(_ckpt("keepfit", "KeepFIT (flair+MM).pth")),
            **kwargs,
        )
        self.num_prefix_tokens = 1
