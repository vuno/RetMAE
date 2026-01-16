import io
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union
from weakref import proxy

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import torchvision.transforms as T
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.tasks.pretrain.utils import LoggingUtil, ToTensor


class EncoderCheckpoint(ModelCheckpoint):
    """
    Custom ModelCheckpoint for saving only the encoder during pretraining.
    """

    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[Union[bool, Literal["link"]]] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
    ):
        """
        Args:
            dirpath: Directory path for saving checkpoints.
            filename: Name pattern for checkpoint files.
            monitor: Metric to monitor for saving.
            verbose: If True, print logs.
            save_last: Whether to always save the last checkpoint.
            save_top_k: Number of best checkpoints to keep.
            save_weights_only: If True, only save model weights.
            mode: 'min' or 'max' for the monitored metric.
            auto_insert_metric_name: Automatically include metric name in file.
            every_n_train_steps: Save every n train steps.
            train_time_interval: Time interval to save checkpoints.
            every_n_epochs: Save every n epochs.
            save_on_train_epoch_end: Save at end of train epoch.
            enable_version_counter: Unique version suffix for files.
        """
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called at the end of every training epoch.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: LightningModule being trained.
        """
        if trainer.is_global_zero and getattr(self, "_save_on_train_epoch_end", False):
            current_epoch = trainer.current_epoch
            every_n_epochs = getattr(self, "_every_n_epochs", 1)

            if (current_epoch + 1) % every_n_epochs == 0:
                # Manually create and save checkpoint for the encoder
                filename = f"encoder_epoch_{current_epoch:07d}.pth"
                filepath = Path(self.dirpath) / filename
                self._save_checkpoint(trainer, str(filepath))

        super().on_train_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        Called at the end of every validation epoch.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: LightningModule being validated.
        """
        super().on_validation_epoch_end(trainer, pl_module)

    @rank_zero_only
    def save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """
        Save the checkpoint using the base implementation.

        Args:
            trainer: Lightning Trainer instance.
            filepath: Where to save the checkpoint.
        """
        super().save_checkpoint(trainer, filepath)

    @rank_zero_only
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """
        Actually saves the encoder model weights to disk.

        Args:
            trainer: Lightning Trainer instance.
            filepath: Where to save the checkpoint.
        """
        if not Path(filepath).parent.exists():
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model = trainer.model.module.encoder.backbone.model
        torch.save(model.state_dict(), str(filepath))

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # Notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


class LatestModelCheckpoint(Callback):
    """
    Custom callback to keep only the latest model checkpoint for each epoch.
    """

    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        every_n_epochs: int = 1,
        save_on_train_epoch_end: bool = True,
        save_weights_only: bool = False,
    ):
        """
        Args:
            dirpath: Directory path for saving checkpoints.
            filename: Name pattern for checkpoint files.
            every_n_epochs: Save every n epochs.
            save_on_train_epoch_end: Save at end of train epoch (else validation end).
            save_weights_only: If True, only save model weights.
        """
        super().__init__()
        self.dirpath = dirpath if dirpath else "./checkpoints"
        self.filename = filename if filename else "latest_model_epoch_{epoch:07d}"
        self.every_n_epochs = every_n_epochs
        self.save_on_train_epoch_end = save_on_train_epoch_end
        self.save_weights_only = save_weights_only

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called at the end of every training epoch. Saves a checkpoint if configured.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: LightningModule being trained.
        """
        if not self.save_on_train_epoch_end:
            return

        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.every_n_epochs == 0:
            self._save_checkpoint(trainer, pl_module, current_epoch)

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        Called at the end of every validation epoch. Saves a checkpoint if configured.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: LightningModule being validated.
        """
        if self.save_on_train_epoch_end:
            return

        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.every_n_epochs == 0:
            self._save_checkpoint(trainer, pl_module, current_epoch)

    def _save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", epoch: int
    ) -> None:
        """
        Save a checkpoint for the latest epoch and remove older ones.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: LightningModule being saved.
            epoch: Current epoch number.
        """
        if not Path(self.dirpath).exists():
            Path(self.dirpath).mkdir(parents=True, exist_ok=True)

        # Remove existing checkpoints (keep only the newest one)
        dest_dir = Path(self.dirpath)
        for old_file in dest_dir.glob("lm_epoch_*.ckpt"):
            old_file.unlink()

        # Create checkpoint file name and save
        filename = self.filename.format(epoch=epoch)
        filepath = Path(self.dirpath) / f"{filename}.ckpt"

        try:
            if self.save_weights_only:
                torch.save(pl_module.state_dict(), str(filepath))
            else:
                checkpoint_dict = {
                    "state_dict": pl_module.state_dict(),
                    "epoch": epoch,
                    "global_step": trainer.global_step,
                }
                torch.save(checkpoint_dict, str(filepath))
        except Exception:
            # Continue training even if saving fails
            raise Exception(f"Failed to save checkpoint: {filepath}")


class AttentionScoreHistogramCallback(Callback):
    """Callback to visualize the distribution of attention scores as a histogram."""

    def __init__(
        self,
        log_train: bool = True,
        log_val: bool = True,
        log_test: bool = True,
    ) -> None:
        """
        Args:
            log_train: Whether to log for training data
            log_val: Whether to log for validation data
            log_test: Whether to log for test data
        """
        self.log_train = log_train
        self.log_val = log_val
        self.log_test = log_test
        self.bins = np.linspace(0, 0.002, 101)  # 100 bins from 0 to 0.002
        self.hist_dict = defaultdict(lambda: np.zeros(len(self.bins) - 1))  # Store count for each bin
        self.logging_util = LoggingUtil()

    @torch.no_grad()
    def _batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        split: str,
    ) -> None:
        """Updates the histogram of attention scores at the end of a batch."""
        img = batch["img"]

        attn = pl_module.encoder.get_self_attention(img)["late"]  # (B, HD, N, N)

        # Update histogram for each attention type
        attn_types = {
            f"{split}/attn_score/patch-to-patch": attn[
                :,
                :,
                pl_module.encoder.backbone.num_prefix_tokens :,
                pl_module.encoder.backbone.num_prefix_tokens :,
            ],
            f"{split}/attn_score/patch-to-class": attn[
                :,
                :,
                :1,
                :1,
            ],
            f"{split}/attn_score/class-to-patch": attn[
                :,
                :,
                :1,
                pl_module.encoder.backbone.num_prefix_tokens :,
            ],
        }

        hist_min = float(torch.stack([attn_map.min() for attn_map in attn_types.values()]).min())
        hist_max = float(torch.stack([attn_map.max() for attn_map in attn_types.values()]).max())
        for name, attn_map in attn_types.items():
            hist = torch.histc(
                attn_map.flatten(),
                bins=len(self.bins) - 1,
                min=hist_min,
                max=hist_max,
            )
            hist = hist.cpu().numpy()
            self.hist_dict[name] += hist

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Method called at the end of a training batch."""
        if self.log_train:
            self._batch_end(trainer, pl_module, outputs, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Method called at the end of a validation batch."""
        if self.log_val:
            self._batch_end(trainer, pl_module, outputs, batch, batch_idx, split="val")

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Method called at the end of a test batch."""
        if self.log_test:
            self._batch_end(trainer, pl_module, outputs, batch, batch_idx, split="test")

    def _epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", split: str
    ) -> None:
        """Visualizes and logs the histogram at the end of an epoch."""
        plt.rcParams["font.family"] = "Times New Roman"
        # Define a color palette with good visibility
        fig, ax = plt.subplots(figsize=(8, 6))
        color_idx = 0
        for key, hist_values in self.hist_dict.items():
            if not key.startswith(split):
                continue

            # Draw normalized histogram using Set1 colormap
            colors = plt.cm.Set1(np.linspace(0, 1, 9))  # Extract 9 colors from Set1 colormap
            normalized_hist = hist_values / (hist_values.sum() + 1e-8)
            ax.bar(
                self.bins[:-1],  # Use bin start points
                normalized_hist,  # Normalized histogram
                width=np.diff(self.bins),
                align="edge",  # Use edge instead of center
                alpha=0.5,  # Set transparency
                color=colors[color_idx % len(colors)],  # Cycle through Set1 colormap colors
                label=key.split("/")[-1],
            )
            color_idx += 1

        ax.set_xlabel("Attention Score", fontsize=16, labelpad=10)
        ax.set_ylabel("Density", fontsize=16, labelpad=10)
        ax.tick_params(labelsize=14)

        # Show only 5 x-axis ticks
        xticks = np.linspace(self.bins[0], self.bins[-1], 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:0.2e}" for x in xticks])

        # Add legend
        ax.legend(fontsize=14, loc="upper right")

        plt.tight_layout()

        # Convert figure to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)

        image = Image.open(buf)
        image_tensor = ToTensor(image)
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min() + 1e-6
        )

        # Log as a single image
        self.logging_util.log_image_tensor(pl_module, image_tensor, f"{split}/attention_score")

        # Calculate and log mean and variance of each histogram
        for key, hist_values in self.hist_dict.items():
            if not key.startswith(split):
                continue

            # Calculate mean and variance from normalized histogram
            normalized_hist = hist_values / (hist_values.sum() + 1e-8)
            bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
            mean = np.sum(bin_centers * normalized_hist)
            var = np.sum((bin_centers - mean) ** 2 * normalized_hist)

            # Logging
            pl_module.log(f"attention_score/mean/{key}", mean, sync_dist=True, rank_zero_only=True)
            pl_module.log(
                f"attention_score/variance/{key}",
                var,
                sync_dist=True,
                rank_zero_only=True,
            )

        # Reset histograms for this split
        keys_to_reset = [k for k in self.hist_dict.keys() if k.startswith(split)]
        for k in keys_to_reset:
            self.hist_dict[k] = np.zeros(len(self.bins) - 1)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Method called at the end of a training epoch."""
        if self.log_train:
            self._epoch_end(trainer, pl_module, split="train")

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Method called at the end of a validation epoch."""
        if self.log_val:
            self._epoch_end(trainer, pl_module, split="val")

    @rank_zero_only
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Method called at the end of a test epoch."""
        if self.log_test:
            self._epoch_end(trainer, pl_module, split="test")


class FeatureCosineSimilarityCallback(AttentionScoreHistogramCallback):
    """Callback to visualize the distribution of cosine similarity between feature vectors as a histogram."""

    def __init__(
        self,
        log_train: bool = True,
        log_val: bool = True,
        log_test: bool = True,
    ):
        """
        Args:
            log_train: Whether to log for training data
            log_val: Whether to log for validation data
            log_test: Whether to log for test data
        """
        super().__init__(
            log_train=log_train,
            log_val=log_val,
            log_test=log_test,
        )
        self.bins = np.linspace(0, 1, 101)  # 100 bins from 0 to 1
        self.hist_dict = defaultdict(lambda: np.zeros(len(self.bins) - 1))  # Store count for each bin

    @torch.no_grad()
    def _batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        split: str,
    ) -> None:
        """Updates the histogram of cosine similarity at the end of a batch."""
        img = batch["img"]

        sim_map = pl_module.encoder.get_cosine_similarity_map(img)  # (B, N, N)

        # Update histogram for each attention type
        attn_types = {
            f"{split}/feature_cosine_similarity/patch-to-patch": sim_map[
                :,
                pl_module.encoder.backbone.num_prefix_tokens :,
                pl_module.encoder.backbone.num_prefix_tokens :,
            ],
            f"{split}/feature_cosine_similarity/patch-to-class": sim_map[
                :,
                pl_module.encoder.backbone.num_prefix_tokens :,
                :1,
            ],
        }

        for name, sim_map in attn_types.items():
            sim_map = sim_map.flatten()

            hist = torch.histc(
                sim_map, bins=len(self.bins) - 1, min=self.bins[0], max=self.bins[-1]
            )
            hist = hist.cpu().numpy()
            self.hist_dict[name] += hist

    def _epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", split: str
    ) -> None:
        """Visualizes and logs the histogram at the end of an epoch."""
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(figsize=(8, 6))
        color_idx = 0
        for key, hist_values in self.hist_dict.items():
            if not key.startswith(split):
                continue
            # Draw normalized histogram using Set1 colormap
            colors = plt.cm.Set1(np.linspace(0, 1, 9))  # Extract 9 colors from Set1 colormap
            ax.bar(
                self.bins[:-1],  # Use bin start points
                hist_values / (hist_values.sum() + 1e-8),  # Normalize
                width=np.diff(self.bins),
                align="edge",  # Use edge instead of center
                alpha=0.5,  # Set transparency
                color=colors[color_idx % len(colors)],  # Cycle through Set1 colormap colors
                label=key.split("/")[-1],
            )

            color_idx += 1

        ax.set_xlabel("Cosine Similarity", fontsize=16, labelpad=10)
        ax.set_ylabel("Density", fontsize=16, labelpad=10)
        ax.tick_params(labelsize=14)

        # Set y-axis range
        # ax.set_ylim(0, 0.2)

        # Set x-axis range
        ax.set_xlim(0, 1)

        # Limit x-axis ticks to 5
        ax.set_xticks(np.linspace(0, 1, 5))

        # Add legend
        ax.legend(fontsize=14, loc="upper left")

        plt.tight_layout()

        # Convert figure to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)

        image = Image.open(buf)
        image_tensor = ToTensor(image)
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min() + 1e-6
        )

        # Log as a single image
        self.logging_util.log_image_tensor(
            pl_module, image_tensor, f"{split}/feature_cosine_similarity"
        )

        # Calculate and log mean and variance of each histogram
        for key, hist_values in self.hist_dict.items():
            if not key.startswith(split):
                continue

            # Calculate mean and variance from normalized histogram
            normalized_hist = hist_values / (hist_values.sum() + 1e-8)
            bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
            mean = np.sum(bin_centers * normalized_hist)
            var = np.sum((bin_centers - mean) ** 2 * normalized_hist)

            # Logging
            pl_module.log(
                f"feature_cosine_similarity/mean/{key}",
                mean,
                sync_dist=True,
                rank_zero_only=True,
            )
            pl_module.log(
                f"feature_cosine_similarity/variance/{key}",
                var,
                sync_dist=True,
                rank_zero_only=True,
            )

        # Reset histograms for this split
        keys_to_reset = [k for k in self.hist_dict.keys() if k.startswith(split)]
        for k in keys_to_reset:
            self.hist_dict[k] = np.zeros(len(self.bins) - 1)
