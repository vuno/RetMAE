from copy import deepcopy
from math import cos, pi
from typing import Dict, Optional

import rootutils
import torch
import torch.nn as nn
from torch import Tensor

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.tasks.pretrain.utils import patchify


class TargetGenerator(nn.Module):
    """Base class for target generation.

    This is a base class that takes images as input and generates target tokens.
    """

    def __init__(self) -> None:
        """Initialize the target generator."""
        super().__init__()

    def forward(self, imgs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the target generator.

        Args:
            imgs: Input image tensor

        Returns:
            torch.Tensor: Generated target tokens
        """
        return NotImplementedError


class NormalizedPixelsGenerator(TargetGenerator):
    """Normalized pixels target generator.

    This generator uses the input images directly as targets.

    Args:
        num_prefix_tokens: Number of prefix tokens
        patch_size: Size of patches to extract
    """

    def __init__(self, patch_size: int) -> None:
        """Initialize the normalized pixels generator.

        Args:
            patch_size: Size of patches to extract
        """
        super().__init__()
        self.patch_size = patch_size

    def forward(self, imgs: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass of the normalized pixels generator.

        Args:
            imgs: Input image tensor

        Returns:
            Dict[str, torch.Tensor]: Target, mean, variance
        """
        patchified_imgs = patchify(imgs, self.patch_size)
        means = patchified_imgs.mean(dim=-1, keepdim=True)
        vars = patchified_imgs.var(dim=-1, keepdim=True)
        return (patchified_imgs - means) / torch.sqrt(vars + 1e-6)


# class MAETargetGenerator(MAEIdentityTargetGenerator):
#     """Target generator for MAE.

#     This generator uses a pre-trained model to generate targets.

#     Args:
#         target_generator: Model for target generation
#         ckpt_path: Path to checkpoint file
#         num_prefix_tokens: Number of prefix tokens
#     """

#     def __init__(
#         self, target_generator: nn.Module, ckpt_path: str, num_prefix_tokens: int
#     ) -> None:
#         """Initialize the MAE target generator.

#         Args:
#             target_generator: Model for target generation
#             ckpt_path: Path to checkpoint file
#             num_prefix_tokens: Number of prefix tokens
#         """
#         super().__init__(num_prefix_tokens)
#         self.target_generator = target_generator
#         ckpt = torch.load(
#             ckpt_path, map_location="cpu"
#         )  # Use map_location for safe loading
#         if not isinstance(ckpt, dict):
#             raise ValueError("Checkpoint file is not in correct format")
#         self.target_generator.load_state_dict(ckpt)

#     def forward(self, imgs: torch.Tensor, **kwargs) -> torch.Tensor:
#         """Forward pass of the MAE target generator.

#         Args:
#             imgs: Input image tensor

#         Returns:
#             torch.Tensor: Visual tokens
#         """
#         tokens = self.target_generator(imgs)
#         visual_tokens = tokens[:, self.num_prefix_tokens :, :]
#         return visual_tokens


class BaseAveragedModel(nn.Module):
    """A base class for averaging model weights.

    Weight averaging, such as SWA and EMA, is a widely used technique for
    training neural networks. This class implements the averaging process
    for a model. All subclasses must implement the `avg_func` method.
    This class creates a copy of the provided module model
    on the device and allows computing running averages of the
    parameters of the model.

    The code is referenced from: https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py.

    Different from the `AveragedModel` in PyTorch, we use in-place operation
    to improve the parameter updating speed, which is about 5 times faster
    than the non-in-place version.

    In mmengine, we provide two ways to use the model averaging:

    1. Use the model averaging module in hook:
       We provide an mmengine.hooks.EMAHook to apply the model
       averaging during training. Add ``custom_hooks=[dict(type='EMAHook')]``
       to the config or the runner.

    2. Use the model averaging module directly in the algorithm. Take the ema
       teacher in semi-supervise as an example:

       >>> from mmengine.model import ExponentialMovingAverage
       >>> student = ResNet(depth=50)
       >>> # use ema model as teacher
       >>> ema_teacher = ExponentialMovingAverage(student)

    Args:
        model: The model to be averaged.
        interval: Interval between two updates. Defaults to 1.
        update_buffers: if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: E501

    def __init__(
        self,
        model: nn.Module,
        interval: int = 1,
        update_buffers: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.module = deepcopy(model).requires_grad_(False)
        self.interval = interval
        device = self.module.device if device is None else device
        self.module = self.module.to(device)
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long, device=device))
        self.update_buffers = update_buffers
        if update_buffers:
            self.avg_parameters = self.module.state_dict()
        else:
            self.avg_parameters = dict(self.module.named_parameters())

    def avg_func(self, averaged_param: Tensor, source_param: Tensor, steps: int) -> None:
        """Use in-place operation to compute the average of the parameters. All
        subclasses must implement this method.

        Args:
            averaged_param: The averaged parameters.
            source_param: The source parameters.
            steps: The number of times the parameters have been
                updated.
        """

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward method of the averaged model."""
        try:
            out = self.module(input_tensor, masking=False, **kwargs)
        except TypeError:
            out = self.module(input_tensor, **kwargs)

        if isinstance(out, tuple):
            features, mask_tokens, restore_ids = out
            return features, mask_tokens, restore_ids
        return out

    def update_parameters(self, model: nn.Module) -> None:
        """Update the parameters of the model. This method will execute the
        avg_func to compute the new parameters and update the model's
        parameters.

        Args:
            model: The model whose parameters will be averaged.
        """
        src_parameters = (
            model.state_dict() if self.update_buffers else dict(model.named_parameters())
        )
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                p_avg.data.copy_(src_parameters[k].data)
        elif self.steps % self.interval == 0:
            for k, p_avg in self.avg_parameters.items():
                if p_avg.dtype.is_floating_point:
                    device = p_avg.device
                    self.avg_func(p_avg.data, src_parameters[k].data.to(device), self.steps)
        if not self.update_buffers:
            # If not update the buffers,
            # keep the buffers in sync with the source model.
            for b_avg, b_src in zip(self.module.buffers(), model.buffers()):
                b_avg.data.copy_(b_src.data.to(b_avg.device))
        self.steps += 1


class StochasticWeightAverage(BaseAveragedModel):
    """Implements the stochastic weight averaging (SWA) of the model.

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization, UAI 2018.
    <https://arxiv.org/abs/1803.05407>`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson.
    """

    def avg_func(self, averaged_param: Tensor, source_param: Tensor, steps: int) -> None:
        """Compute the average of the parameters using stochastic weight
        average.

        Args:
            averaged_param: The averaged parameters.
            source_param: The source parameters.
            steps: The number of times the parameters have been
                updated.
        """
        averaged_param.add_(
            source_param - averaged_param, alpha=1 / float(steps // self.interval + 1)
        )


class ExponentialMovingAverage(BaseAveragedModel):
    """Implements the exponential moving average (EMA) of the model.

    All parameters are updated by the formula as below:

        .. math::

            Xema_{t+1} = (1 - momentum) * Xema_{t} +  momentum * X_t

    Note:
        This momentum argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically,
        Xema_{t+1} is the moving average and X_t is the
        new observed value. The value of momentum is usually a small number,
        allowing observed values to slowly update the ema parameters.

    Args:
        model: The model to be averaged.
        momentum: The momentum used for updating ema parameter.
            Defaults to 0.0002.
            Ema's parameter are updated with the formula
            averaged_param = (1-momentum) * averaged_param +
            momentum * source_param.
        interval: Interval between two updates. Defaults to 1.
        update_buffers: if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: W605

    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.0002,
        interval: int = 1,
        update_buffers: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model, interval, update_buffers, device)
        assert 0.0 < momentum < 1.0, "momentum must be in range (0.0, 1.0)" f"but got {momentum}"
        if momentum > 0.5:
            print(
                "The value of momentum in EMA is usually a small number,"
                "which is different from the conventional notion of "
                f"momentum but got {momentum}. Please make sure the "
                f"value is correct."
            )

        self.momentum = momentum

    def avg_func(self, averaged_param: Tensor, source_param: Tensor, steps: int) -> None:
        """Compute the moving average of the parameters using exponential
        moving average.

        Args:
            averaged_param: The averaged parameters.
            source_param: The source parameters.
            steps: The number of times the parameters have been
                updated.
        """
        averaged_param.lerp_(source_param, self.momentum)


class MomentumAnnealingEMA(ExponentialMovingAverage):
    """Exponential moving average (EMA) with momentum annealing strategy.

    Args:
        model: The model to be averaged.
        momentum: The momentum used for updating ema parameter.
            Defaults to 0.0002.
            Ema's parameter are updated with the formula
            averaged_param = (1-momentum) * averaged_param +
            momentum * source_param.
        gamma: Use a larger momentum early in training and gradually
            annealing to a smaller value to update the ema model smoothly. The
            momentum is calculated as max(momentum, gamma / (gamma + steps))
            Defaults to 100.
        interval: Interval between two updates. Defaults to 1.
        update_buffers: if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.0002,
        gamma: int = 100,
        interval: int = 1,
        update_buffers: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            model=model,
            momentum=momentum,
            interval=interval,
            update_buffers=update_buffers,
            device=device,
        )
        assert gamma > 0, f"gamma must be greater than 0, but got {gamma}"
        self.gamma = gamma

    def avg_func(self, averaged_param: Tensor, source_param: Tensor, steps: int) -> None:
        """Compute the moving average of the parameters using the linear
        momentum strategy.

        Args:
            averaged_param: The averaged parameters.
            source_param: The source parameters.
            steps: The number of times the parameters have been
                updated.
        """
        momentum = max(self.momentum, self.gamma / (self.gamma + self.steps.item()))
        averaged_param.lerp_(source_param, momentum)


class CosineEMA(ExponentialMovingAverage):
    """CosineEMA is implemented for updating momentum parameter, used in BYOL,
    MoCoV3, etc.

    All parameters are updated by the formula as below:

    .. math::

        X'_{t+1} = (1 - m) * X'_t + m * X_t

    Where m is the momentum parameter. And it's updated with cosine
    annealing, including momentum adjustment following:

    .. math::
        m = m_{end} + (m_{end} - m_{start}) * (\\cos\\frac{k\\pi}{K} + 1) / 2

    where k is the current step, K is the total steps.

    Note:
        This momentum argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically,
        X'_{t} is the moving average and X_t is the new
        observed value. The value of momentum is usually a small number,
        allowing observed values to slowly update the ema parameters.

    Args:
        model: The model to be averaged.
        momentum: The start momentum value. Defaults to 0.004.
        end_momentum: The end momentum value for cosine annealing.
            Defaults to 0.
        interval: Interval between two updates. Defaults to 1.
        update_buffers: if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.004,
        end_momentum: float = 0.0,
        interval: int = 1,
        update_buffers: bool = False,
        device: Optional[torch.device] = None,
        max_iters: int = 100,
    ) -> None:
        super().__init__(
            model=model,
            momentum=momentum,
            interval=interval,
            update_buffers=update_buffers,
            device=device,
        )
        self.end_momentum = end_momentum
        self.max_iters = max_iters

    def avg_func(
        self,
        averaged_param: torch.Tensor,
        source_param: torch.Tensor,
        steps: int,
    ) -> None:
        """Compute the moving average of the parameters using the cosine
        momentum strategy.

        Args:
            averaged_param: The averaged parameters.
            source_param: The source parameters.
            steps: The number of times the parameters have been
                updated.

        Returns:
            Tensor: The averaged parameters.
        """
        cosine_annealing = (cos(pi * steps / float(self.max_iters)) + 1) / 2
        momentum = self.end_momentum - (self.end_momentum - self.momentum) * cosine_annealing
        averaged_param.mul_(1 - momentum).add_(source_param, alpha=momentum)
