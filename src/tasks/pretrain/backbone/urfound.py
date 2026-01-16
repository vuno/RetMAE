"""UrFound model implementation."""

from functools import partial
from typing import List, Union

import torch
import torch.nn as nn
from kornia.geometry.transform import resize
from timm.models.vision_transformer import VisionTransformer

from src.tasks.pretrain.utils import standardize_input
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class UrFound(nn.Module):
    def __init__(
        self,
        ckpt_path: str = "your_path/save/weights/ssl/urfound_mm.pth",
    ) -> None:
        """UrFound 모델의 초기화 함수입니다."""
        super().__init__()

        self.model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.patch_size = 16
        del self.model.head

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]

        new_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith("bert") or k.startswith("decoder") or "mask_token" in k:
                continue
            else:
                new_ckpt[k] = v

        # Standardization is handled by the standardize_input function from utils

        self.model.load_state_dict(new_ckpt, strict=True)
        self.out_dim = self.embed_dim = self.model.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """특징을 추출하는 순전파 함수입니다."""
        if x.shape[2] != 224:
            x = resize(x, (224, 224))

        x = standardize_input(x)
        return {
            "features": self.model.forward_features(x),
        }

    def forward_intermediates(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward features that returns intermediates."""
        if x.shape[2] != 224:
            x = resize(x, (224, 224))

        x = standardize_input(x)
        return self.model.forward_intermediates(x)
