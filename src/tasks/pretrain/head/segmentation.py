"""
This file is a derivative work based on the file "decoder.py" from the
segmentation_models.pytorch project:
https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/base/heads.py#L5

Original work citation:
    @misc{Iakubovskii:2019,
      Author = {Pavel Iakubovskii},
      Title = {Segmentation Models Pytorch},
      Year = {2019},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {https://github.com/qubvel/segmentation_models.pytorch}
    }

Copyright (c) [Your Name], [Year]

Licensed under the MIT License.
You may obtain a copy of the License at: https://opensource.org/licenses/MIT
"""

import torch.nn as nn


class SegmentationHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        activation: nn.Module = nn.Sigmoid,
        upsampling: int = 1,
    ) -> None:
        """Initialize segmentation head.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            activation: Activation function to use
            upsampling: Upsampling factor for bilinear interpolation
        """
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        )
        activation = activation()
        super().__init__(conv2d, upsampling, activation)
        self.initialize_decoder(self)

    def initialize_decoder(self, module: nn.Module) -> None:
        """Initialize decoder weights.

        Args:
            module: Module to initialize
        """
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
