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
from typing import Optional, Type

import rootutils
import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.tasks.pretrain.decoder import MAEDecoder


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(
    nlayers: int,
    in_dim: int,
    bottleneck_dim: int,
    hidden_dim: Optional[int] = None,
    use_bn: bool = False,
    bias: bool = True,
) -> nn.Sequential:
    """MLP 모듈을 구성합니다.

    Args:
        nlayers: MLP의 레이어 수
        in_dim: 입력 차원
        bottleneck_dim: 병목 차원 (출력 차원)
        hidden_dim: 은닉층의 차원. Defaults to None.
        use_bn: 배치 정규화 사용 여부. Defaults to False.
        bias: 선형 레이어의 편향 사용 여부. Defaults to True.

    Returns:
        nn.Sequential: 구성된 MLP 모듈
    """
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


class DINOv2Head(nn.Module):
    """DINOv2 헤드 모듈.

    레지스터 토큰과 패치 토큰에 대해 각각 별도의 DINO 헤드를 적용합니다.

    Args:
        in_dim: 입력 차원
        out_dim: 출력 차원
        use_bn: 배치 정규화 사용 여부. Defaults to False.
        nlayers: MLP의 레이어 수. Defaults to 3.
        hidden_dim: 은닉층의 차원. Defaults to 2048.
        bottleneck_dim: 병목 차원. Defaults to 256.
        mlp_bias: MLP의 편향 사용 여부. Defaults to True.
        reg_tokens: 레지스터 토큰의 수. Defaults to 4.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        mlp_bias: bool = True,
        reg_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.decoder_class = DINOHead(
            in_dim, out_dim, use_bn, nlayers, hidden_dim, bottleneck_dim, mlp_bias
        )
        self.decoder_patch = DINOHead(
            in_dim, out_dim, use_bn, nlayers, hidden_dim, bottleneck_dim, mlp_bias
        )
        self.num_prefix_tokens = reg_tokens + 1

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """순전파를 수행합니다.

        Args:
            x: 입력 텐서. Shape: (B, N, D)
                B는 배치 크기, N은 토큰 수, D는 임베딩 차원

        Returns:
            dict[str, torch.Tensor]: 클래스 토큰과 패치 토큰에 대한 출력을 포함하는 딕셔너리
                - class: 클래스 토큰의 출력
                - patch: 패치 토큰의 출력
        """
        cls_token = x[:, 0]
        patch_token = x[:, self.num_prefix_tokens :]

        return {
            "class": self.decoder_class(cls_token),
            "patch": self.decoder_patch(patch_token),
        }


class DINOv2MAEHead(MAEDecoder):
    """DINOv2 MAE 헤드 모듈.

    MAE 디코더를 확장하여 DINOv2 스타일의 디코딩을 수행합니다.
    레지스터 토큰과 패치 토큰에 대해 각각 별도의 디코더를 적용할 수 있습니다.

    Args:
        image_size: 입력 이미지의 크기. Defaults to 512.
        patch_size: 패치의 크기. Defaults to 16.
        in_chans: 입력 채널 수. Defaults to 1.
        embed_dim: 임베딩 차원. Defaults to 1024.
        decoder_embed_dim: 디코더의 임베딩 차원. Defaults to 512.
        decoder_depth: 디코더의 깊이. Defaults to 8.
        decoder_num_heads: 디코더의 어텐션 헤드 수. Defaults to 16.
        mlp_ratio: MLP의 확장 비율. Defaults to 4.0.
        norm_layer: 정규화 레이어. Defaults to nn.LayerNorm.
        reg_tokens: 레지스터 토큰의 수. Defaults to 4.
        untying_register_decoder: 레지스터 토큰별로 별도의 디코더 사용 여부. Defaults to False.
    """

    def __init__(
        self,
        image_size: int = 512,
        embed_dim: int = 1024,
        pred_dim: int = 384,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        reg_tokens: int = 4,
    ) -> None:
        super().__init__(
            image_size=image_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            reg_tokens=reg_tokens,
        )
        self.decoder_norm = None
        self.decoder_pred = None
        # decoder for class token
        self.decoder_class = weight_norm(nn.Linear(decoder_embed_dim, pred_dim, bias=True))
        self.decoder_class.parametrizations.weight.original0.data.fill_(1)

        # decoder for patch token
        self.decoder_patch = weight_norm(nn.Linear(decoder_embed_dim, pred_dim, bias=True))
        self.decoder_patch.parametrizations.weight.original0.data.fill_(1)
        self.initialize_weights()

    def forward_decoder(
        self, x: torch.Tensor, ids_restore_mim: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """디코더의 순전파를 수행합니다.

        Args:
            x: 인코더의 출력 텐서. Shape: (B, N, D)
                B는 배치 크기, N은 토큰 수, D는 임베딩 차원
            ids_restore_mim: 마스킹된 토큰의 원래 위치. Shape: (B, L)
                B는 배치 크기, L은 전체 패치 수. Defaults to None.

        Returns:
            dict[str, torch.Tensor]: 레지스터와 패치 토큰의 디코딩 결과를 포함하는 딕셔너리
                - register: 레지스터 토큰의 디코딩 결과
                - patch: 패치 토큰의 디코딩 결과
        """
        B, N, _ = x.shape

        # 임베딩 및 마스크 토큰 처리
        x = self.decoder_embed(x)

        if ids_restore_mim is not None:
            # 마스크 토큰과 시각 토큰 결합 최적화
            num_masked = ids_restore_mim.shape[1] + self.num_prefix_tokens - N
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
        except RuntimeError:
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

        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)

        # 최종 예측
        o_class = x[:, 0]
        o_patch = x[:, self.num_prefix_tokens :]

        return {
            "class": self.decoder_class(o_class),
            "patch": self.decoder_patch(o_patch),
        }

    def forward(
        self, x: torch.Tensor, ids_restore_mim: Optional[torch.Tensor] = None, **kwargs
    ) -> dict[str, torch.Tensor]:
        """디코더의 순전파를 수행합니다.

        Args:
            x: 인코더의 출력 텐서. Shape: (B, N, D)
                B는 배치 크기, N은 토큰 수, D는 임베딩 차원
            ids_restore_mim: 마스킹된 토큰의 원래 위치. Shape: (B, L)
                B는 배치 크기, L은 전체 패치 수. Defaults to None.

        Returns:
            dict[str, torch.Tensor]: 레지스터와 패치 토큰의 디코딩 결과를 포함하는 딕셔너리
                - register: 레지스터 토큰의 디코딩 결과
                - patch: 패치 토큰의 디코딩 결과
        """
        return self.forward_decoder(x, ids_restore_mim)
