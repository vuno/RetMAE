from functools import partial
from typing import Callable

from torch.optim import Optimizer


class OptimizerWrapper:
    """Class wrapping the optimizer.

    Automatically scales the learning rate based on batch size and gradient accumulation steps.

    Args:
        optimizer (Callable): Optimizer class or function
        batch_size (int): Batch size
        gradient_accumulation_steps (int): Number of gradient accumulation steps
    """

    def __init__(
        self,
        optimizer: Callable,
        batch_size: int,
        num_gpus: int,
        num_nodes: int,
        gradient_accumulation_steps: int,
        base_lr: float = 5e-4,
    ) -> None:
        lr = base_lr * (batch_size * num_gpus * num_nodes * gradient_accumulation_steps) / 256
        self.optimizer = partial(optimizer, lr=lr)

    def __call__(
        self,
        **kwargs,
    ) -> Optimizer:
        """Returns the optimizer.

        Returns:
            Optimizer: Optimizer with the configured learning rate
        """
        return self.optimizer(**kwargs)
