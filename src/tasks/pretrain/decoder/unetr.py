# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import rootutils
import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import (
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)
from monai.utils import ensure_tuple_rep
from torch.nn import functional as F

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class BackboneWrapper(nn.Module):
    """Wrapper class for backbone models to use in UNETR architecture.

    This class wraps a backbone model and optionally freezes its parameters.
    It forwards input through the backbone's forward_intermediates method.

    Attributes:
        backbone (nn.Module): The backbone model to wrap
    """

    def __init__(self, backbone: nn.Module, freeze: bool = False):
        """Initialize the backbone wrapper.

        Args:
            backbone (nn.Module): The backbone model to wrap
            freeze (bool): Whether to freeze the backbone parameters. Defaults to False.
        """
        super().__init__()
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()

        self.backbone = backbone
        self.backbone.num_prefix_tokens = getattr(backbone, "num_prefix_tokens", 1)

    def check_grad(self) -> bool:
        """Check if backbone parameters require gradients.

        Returns:
            bool: True if any backbone parameter requires gradients, False otherwise.
        """
        return any(p.requires_grad for p in self.backbone.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output from backbone's forward_intermediates method
        """
        x = self.backbone.get_intermediate_features_all(x)
        return x


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: Sequence[int] | int = (224, 224),
        feature_size: int = 14,
        hidden_size: int = 768,
        num_heads: int = 12,
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 2,
        encoder_freeze: bool = True,
        num_layer: int = 12,
    ) -> None:
        """
        Args:
            backbone: backbone model.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.
            encoder_freeze: whether to freeze the encoder parameters. Defaults to True.
        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), proj_type='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(img_size[0] // feature_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = BackboneWrapper(backbone, encoder_freeze)
        self.num_layers = [i for i in range(num_layer // 4, num_layer + 1, num_layer // 4)]

        # Check if any parameters in self.vit are trainable
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the feature tensor to the desired shape.

        Args:
            x: Input tensor from the vision transformer

        Returns:
            Projected tensor with proper dimensions for decoder
        """
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNETR model.

        Args:
            x_in: Input image tensor

        Returns:
            Output tensor after passing through the encoder-decoder architecture
        """
        hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[self.num_layers[0]]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[self.num_layers[1]]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[self.num_layers[2]]

        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(hidden_states_out[self.num_layers[3]])
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        if (dec1.shape[-1] * 2) != enc1.shape[-1]:
            dec1 = F.interpolate(
                dec1, size=enc1.shape[-1] // 2, mode="bilinear", align_corners=False
            )
        out = self.decoder2(dec1, enc1)
        return self.out(out)


if __name__ == "__main__":
    from src.tasks.pretrain.backbone import RetMAE

    backbone = RetMAE(
        model_name="timm/vit_base_patch16_224.augreg_in21k",
        pretrained=True,
        ckpt_path=None,
        load_ckpt_strict=True,
    )
    unetr = UNETR(backbone, out_channels=3)
    x = torch.randn(1, 3, 224, 224)
    y = unetr(x)
    print(y.shape)
