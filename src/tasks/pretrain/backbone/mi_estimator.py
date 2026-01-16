"""Mutual Information Estimators including MINE and InfoNCE."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class MINE(nn.Module):
    """MINE (Mutual Information Neural Estimation) module for mutual information estimation.

    This module implements the MINE loss which provides a lower bound on mutual information
    between two random variables. It uses a neural network to learn a critic function that
    distinguishes between positive pairs and negative samples.

    Args:
        x_dim (int): Dimension of the first input variable
        y_dim (int): Dimension of the second input variable
        hidden_size (int, optional): Hidden dimension of the critic network. Defaults to 1024.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.F_func = nn.Sequential(
            nn.Linear(2 * y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.input_proj = nn.Linear(x_dim, y_dim)
        self.input_norm = nn.LayerNorm(y_dim, eps=1e-5)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize network weights using truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        """Compute the MINE lower bound on mutual information.

        Args:
            input_feats (torch.Tensor): First set of samples with shape [sample_size, x_dim]
            target_feats (torch.Tensor): Second set of samples with shape [sample_size, y_dim]

        Returns:
            torch.Tensor: Lower bound estimate of mutual information
        """
        sample_size = target_feats.shape[0]
        if target_feats.ndim == 3:
            target_feats = target_feats[:, 0]

        input_feats = self.input_norm(self.input_proj(input_feats))
        x_tile = input_feats.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = target_feats.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([input_feats, target_feats], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return lower_bound

    def learning_loss(self, input_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        return -self.forward(input_feats, target_feats)


class InfoNCE(nn.Module):
    """
    Memory-efficient InfoNCE:
      - No 2N×2N matrix; use two N×N cross-view logits (x@y^T and y@x^T).
      - Optional block processing along batch dimension to avoid materializing full N×N logits.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        x_dim: int = 768,
        y_dim: int = 512,
        block_size: int | None = None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.input_proj = nn.Sequential(
            nn.Linear(x_dim, 2 * y_dim),
            nn.GELU(),
            nn.Linear(2 * y_dim, y_dim),
        )
        self.temperature = temperature
        self.block_size = block_size
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _make_targets(self, n: int, device: torch.device) -> torch.Tensor:
        # positives are on the diagonal for cross-view logits
        return torch.arange(n, device=device)

    def _ce_blocks(self, input_norm: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy over logits = (input_norm @ y_norm.T) / T
        in blocks along the batch dimension to save memory.
        """
        n = input_norm.size(0)
        targets = self._make_targets(n, input_norm.device)
        bs = self.block_size or n  # fall back to full-batch if block_size is None
        losses = []
        for start in range(0, n, bs):
            end = min(start + bs, n)
            # [bs, d] @ [d, n] -> [bs, n]
            logits = (input_norm[start:end] @ y_norm.T) / self.temperature
            loss = F.cross_entropy(logits, targets[start:end], reduction="sum")
            losses.append(loss)
        # average over all samples
        return torch.stack(losses).sum() / n

    def forward(self, input_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_feats: [N, x_dim]
            target_feats: [N, y_dim]
        Returns:
            scalar InfoNCE loss (mean of x→y and y→x)
        """
        # project x to y-dim (y is assumed already in y_dim)
        x = self.input_proj(input_feats)

        # L2-normalize features so dot product = cosine similarity
        input_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(target_feats, dim=1)

        # cross-view losses (no self-masking needed)
        loss_xy = self._ce_blocks(input_norm, y_norm)  # positives on diag
        loss_yx = self._ce_blocks(y_norm, input_norm)  # symmetric direction

        return 0.5 * (loss_xy + loss_yx)

    def learning_loss(self, input_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        return self.forward(input_feats, target_feats)
