import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import rootutils
import torch
import torchmetrics as tm
import torchvision
from lightning import LightningModule
from lightning.pytorch.trainer import Trainer
from torch import nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.common.callbacks_abc_metric import BaseMetricCallback
from src.common.data.classes import TARGET_CLASS_LIST
from src.tasks.pretrain.transferability_estimation import LogME_Score, PED_Score


# ===============================
# Validation 메트릭
# ===============================
class LogMEScoreMetric(BaseMetricCallback):
    """A metric callback for computing LogME scores during model training and validation.

    This class implements the LogME (Log Maximum Evidence) score calculation, which is used
    to evaluate the transferability of learned features. It supports evaluation on training,
    validation, and test datasets, with configurable pooling strategies and NaN handling.

    Attributes:
        VALIDATION_DATASET (List[str]): List of validation dataset names
        best (float): Best score achieved
        best_post (float): Best post-processing score
        metric_settings (Dict[str, bool]): Settings for which splits to evaluate
        metrics (nn.ModuleDict): Dictionary of metrics for each split and dataset
        metric_name (str): Name of the metric
        pool_type (str): Type of pooling to use ('mean' or 'cls')
        fill_nan (int): Value to fill NaN values with
    """

    VALIDATION_DATASET: List[str] = ["BH"]

    def __init__(
        self,
        metric_train: bool = False,
        metric_val: bool = True,
        metric_test: bool = False,
        pool_type: str = "mean",
        fill_nan: int = -1,
    ) -> None:
        """Initialize the LogME score metric callback.

        Args:
            metric_train: Whether to evaluate on training data. Defaults to False.
            metric_val: Whether to evaluate on validation data. Defaults to True.
            metric_test: Whether to evaluate on test data. Defaults to False.
            pool_type: Type of pooling to use ('mean' or 'cls'). Defaults to "mean".
            fill_nan: Value to fill NaN values with. Defaults to -1.
        """
        super().__init__(metric_train, metric_val, metric_test)
        self.best = self.best_post = -1
        self.metric_settings: Dict[str, bool] = {
            "train": metric_train,
            "val": metric_val,
            "test": metric_test,
        }
        self.metrics: nn.ModuleDict = self.init_metrics()
        self.metric_name: str = "LogME"
        if pool_type not in ["mean", "cls"]:
            raise ValueError("pool_type must be 'mean' or 'cls'")
        self.pool_type: str = pool_type
        self.fill_nan: int = fill_nan

    def init_metrics(self, **kwargs) -> nn.ModuleDict:
        """Initialize metrics for each split and dataset.

        Returns:
            nn.ModuleDict: Dictionary containing initialized metrics for each split and dataset
        """
        init_metrics = nn.ModuleDict()
        target_classes = TARGET_CLASS_LIST["finding"] + TARGET_CLASS_LIST["detail_dx"]

        # Initialize metrics only for enabled splits
        for split, enabled in self.metric_settings.items():
            if not enabled:
                continue

            for dataset in self.VALIDATION_DATASET:
                base_key = f"{split}/{dataset}"
                init_metrics[f"{base_key}/feats"] = tm.CatMetric()
                init_metrics.update(
                    {f"{base_key}/labels/{c}": tm.CatMetric() for c in target_classes}
                )

        return init_metrics

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Called when training begins. Moves metrics to the correct device.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        self.metrics.to(pl_module.device)

    def on_test_start(self, trainer: Any, pl_module: Any) -> None:
        """Called when testing begins. Moves metrics to the correct device.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        self.metrics.to(pl_module.device)

    def update_metrics(
        self, feats: torch.Tensor, labels: torch.Tensor, split: str, dataset_name: str
    ) -> None:
        """Update metrics with new feature vectors and labels.

        Args:
            feats: Feature vectors from the model
            labels: Ground truth labels
            split: Data split (train/val/test)
            dataset_name: Name of the dataset
        """
        if not getattr(self, f"metric_{split}"):
            return

        try:
            target_classes = TARGET_CLASS_LIST["finding"] + TARGET_CLASS_LIST["detail_dx"]
            with torch.cuda.device(feats.device):
                labels = torch.where(torch.isnan(labels), self.fill_nan, labels)
                for name, feat, label in zip(dataset_name, feats, labels):
                    # Process features in batch
                    base_key = f"{split}/{name}"
                    self.metrics[f"{base_key}/feats"].update(feat.unsqueeze(0))

                    for i, class_name in enumerate(target_classes):
                        self.metrics[f"{base_key}/labels/{class_name}"].update(
                            label[i].unsqueeze(0)
                        )
        finally:
            # Clean up GPU memory
            torch.cuda.empty_cache()

    def compute_metrics(self, pl_module: Any, split: str) -> None:
        """Compute and log metrics for the specified split.

        Args:
            pl_module: PyTorch Lightning module
            split: Data split (train/val/test)
        """
        if not getattr(self, f"metric_{split}"):
            return

        scores: List[float] = []
        for dataset_name in self.VALIDATION_DATASET:
            metrics = self._compute_dataset_metrics(pl_module, split, dataset_name)
            scores.append(metrics)

        avg_score = sum(scores) / len(scores)
        pl_module.log(
            f"{split}/avg/{self.metric_name}_Score",
            avg_score,
            sync_dist=True,
            rank_zero_only=True,
        )

    def _compute_dataset_metrics(self, pl_module: Any, split: str, dataset_name: str) -> float:
        """Compute metrics for a specific dataset.

        Args:
            pl_module: PyTorch Lightning module
            split: Data split (train/val/test)
            dataset_name: Name of the dataset

        Returns:
            float: Average LogME score for the dataset
        """
        feats = self.metrics[f"{split}/{dataset_name}/feats"].compute()
        self.metrics[f"{split}/{dataset_name}/feats"].reset()

        scores: List[float] = []
        for c in TARGET_CLASS_LIST["finding"] + TARGET_CLASS_LIST["detail_dx"]:
            labels = self.metrics[f"{split}/{dataset_name}/labels/{c}"].compute()
            self.metrics[f"{split}/{dataset_name}/labels/{c}"].reset()

            score = self._compute_score(feats, labels)
            if math.isnan(score):
                continue

            scores.append(score)
            pl_module.log(
                f"{split}/{dataset_name}/{c}/{self.metric_name}_Score",
                score,
                sync_dist=True,
                rank_zero_only=True,
            )

        if len(scores) == 0:
            raise ValueError("No valid scores found")

        avg_score = sum(scores) / len(scores)
        pl_module.log(
            f"{split}/{dataset_name}/avg/{self.metric_name}_Score",
            avg_score,
            sync_dist=True,
            rank_zero_only=True,
        )
        return avg_score

    def _compute_score(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute LogME score for given features and labels.

        Args:
            feats: Feature vectors from the model
            labels: Ground truth labels

        Returns:
            torch.Tensor: Computed LogME score
        """
        # Convert to numpy arrays for LogME computation
        feats_np = feats
        labels_np = labels

        # Calculate LogME score
        score = LogME_Score(feats_np, labels_np)

        torch.cuda.empty_cache()
        return torch.tensor(score)

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Called when validation batch ends. Updates metrics with batch results.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            outputs: Model outputs
            batch: Input batch
            batch_idx: Batch index
        """
        if self.metric_val:
            feats = pl_module.encoder(batch["img"], masking=False)["features"]
            if self.pool_type == "cls":
                feats = feats[:, 0]
            else:
                feats = feats[:, pl_module.encoder.backbone.num_prefix_tokens :].mean(dim=1)

            self.update_metrics(
                feats,
                batch["label"],
                split="val",
                dataset_name=batch["dataset_name"],
            )


class PEDScoreMetric(LogMEScoreMetric):
    def __init__(
        self,
        metric_train: bool = False,
        metric_val: bool = True,
        metric_test: bool = False,
        score_type: str = "SFDA",
        pool_type: str = "cls",
        fill_nan: int = -1,
    ) -> None:
        """Initialize the PED score metric callback.

        Args:
            metric_train: Whether to evaluate on training data. Defaults to False.
            metric_val: Whether to evaluate on validation data. Defaults to True.
            metric_test: Whether to evaluate on test data. Defaults to False.
            score_type: Type of PED score to compute. Must be one of ['SFDA', 'NLEEP', 'LOGME', 'PARC']. Defaults to "SFDA".
            pool_type: Type of pooling to use ('mean' or 'cls'). Defaults to "cls".
            fill_nan: Value to fill NaN values with. Defaults to -1.
        """
        super().__init__(metric_train, metric_val, metric_test, pool_type, fill_nan)
        assert score_type in ["SFDA", "NLEEP", "LOGME", "PARC"]
        self.score_type: str = score_type
        self.metric_name: str = f"PED_{score_type}"

    def _compute_score(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute PED score using the specified score type.

        Args:
            feats: Feature vectors from the model
            labels: Ground truth labels

        Returns:
            Computed PED score, or 0.0 if computation fails
        """
        # Convert to numpy arrays for LogME computation
        feats_np = feats
        labels_np = labels
        score = PED_Score(feats_np, labels_np, type=self.score_type)

        torch.cuda.empty_cache()
        return score


class ValidationLossMetric(LogMEScoreMetric):
    """A metric callback for computing validation loss during model training.

    This class implements validation loss tracking using PyTorch Lightning's MeanMetric.
    It inherits from LogMEScoreMetric but focuses specifically on loss computation
    during the validation phase.

    Attributes:
        metric_name (str): Name of the metric ('loss')
        metrics (tm.MeanMetric): Mean metric for loss computation
    """

    def __init__(self) -> None:
        """Initialize validation loss metric.

        Sets up the metric to only track validation data and initializes
        the metric name as 'loss'.
        """
        super().__init__(False, True, False)
        self.metric_name = "loss"

    def init_metrics(self, **kwargs) -> tm.MeanMetric:
        """Initialize mean metric for loss computation.

        Returns:
            tm.MeanMetric: Initialized mean metric for tracking average loss
        """
        init_metrics = tm.MeanMetric()
        return init_metrics

    def update_metrics(self, loss: torch.Tensor, split: str) -> None:
        """Update metrics with the current loss value.

        Args:
            loss (torch.Tensor): Current loss value to be added to the running average
            split (str): Dataset split being processed (e.g., 'train', 'val', 'test')
        """
        if not getattr(self, f"metric_{split}"):
            return

        self.metrics.update(loss)

    def compute_metrics(self, pl_module: pl.LightningModule, split: str) -> None:
        """Compute final metrics and log results to the Lightning module.

        Args:
            pl_module (pl.LightningModule): Lightning module for logging results
            split (str): Dataset split being processed
        """
        if not getattr(self, f"metric_{split}"):
            return

        loss = self.metrics.compute()
        self.metrics.reset()
        pl_module.log(f"val/{self.metric_name}", loss, sync_dist=True, rank_zero_only=True)

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Process validation batch results and update loss metrics.

        This method is called at the end of each validation batch to compute
        and track the model's loss.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer instance
            pl_module (pl.LightningModule): PyTorch Lightning module instance
            outputs (Dict[str, Any]): Model outputs from the current batch
            batch (Dict[str, Any]): Input batch data
            batch_idx (int): Index of the current batch
        """
        loss = pl_module.compute_loss(batch)
        self.update_metrics(loss["total_loss"], split="val")


# ===============================
# Transfer Learning Metrics
# ===============================
class TLValidationAUPRCMetric(BaseMetricCallback):
    """Validation AUPRC metric callback."""

    def __init__(self, val_dataset_list: List[str]) -> None:
        """Initialize validation AUPRC metric callback."""
        super().__init__(False, True, False)
        self.val_dataset_list = val_dataset_list
        self.metric: tm.AveragePrecision = self.init_metrics()

    def init_metrics(self, **kwargs) -> tm.AveragePrecision:
        """Initialize metrics."""
        return tm.AveragePrecision(task="binary", ignore_index=-1)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when fitting starts."""
        pass

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation starts.

        Moves the metric to the same device as the model.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance
            pl_module (LightningModule): PyTorch Lightning module instance
        """
        self.metric.to(pl_module.device)

    def update_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """Update metrics with prediction and label."""
        for pred, label in zip(preds, labels):
            self.metric.update(pred, label.to(torch.long))

    def compute_metrics(self, pl_module: LightningModule, split: str) -> None:
        """Compute metrics."""
        metric_result = self.metric.compute()
        self.metric.reset()

        pl_module.log(
            "val/auprc",
            metric_result,
            sync_dist=True,
            rank_zero_only=True,
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Called at the end of each validation batch.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance
            pl_module (LightningModule): PyTorch Lightning module instance
            outputs (Any): Outputs from the validation step
            batch (Dict[str, Any]): Current batch of data containing:
                - img: Input images
                - label: Ground truth labels
                - dataset_name: Names of datasets for each sample
            batch_idx (int): Index of the current batch
        """
        img, label = batch["img"], batch["label"]
        preds = pl_module(img, mode="val")
        if isinstance(preds, tuple):
            preds = sum(preds) / len(preds)

        self.update_metrics(preds, label)


# ===============================
class TLValidationAUROCMetric(BaseMetricCallback):
    """Validation AUROC metric callback."""

    def __init__(self, val_dataset_list: List[str]) -> None:
        """Initialize validation AUROC metric callback."""
        super().__init__(False, True, False)
        self.val_dataset_list = val_dataset_list
        self.metric: tm.AUROC = self.init_metrics()

    def init_metrics(self, **kwargs) -> tm.AUROC:
        """Initialize metrics."""
        return tm.AUROC(task="binary", ignore_index=-1)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when fitting starts."""
        pass

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation starts.

        Moves the metric to the same device as the model.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance
            pl_module (LightningModule): PyTorch Lightning module instance
        """
        self.metric.to(pl_module.device)

    def update_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """Update metrics with prediction and label."""
        for pred, label in zip(preds, labels):
            self.metric.update(pred, label.to(torch.long))

    def compute_metrics(self, pl_module: LightningModule, split: str) -> None:
        """Compute metrics."""
        metric_result = self.metric.compute()
        self.metric.reset()

        pl_module.log(
            "val/auroc",
            metric_result,
            sync_dist=True,
            rank_zero_only=True,
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Called at the end of each validation batch.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance
            pl_module (LightningModule): PyTorch Lightning module instance
            outputs (Any): Outputs from the validation step
            batch (Dict[str, Any]): Current batch of data containing:
                - img: Input images
                - label: Ground truth labels
                - dataset_name: Names of datasets for each sample
            batch_idx (int): Index of the current batch
        """
        img, label = batch["img"], batch["label"]
        preds = pl_module(img, mode="val")
        if isinstance(preds, tuple):
            preds = sum(preds) / len(preds)

        self.update_metrics(preds, label)


class TLClassificationMetric(BaseMetricCallback):
    """Callback class for calculating classification metrics.

    Calculates Accuracy, AUROC, and AUPRC for the test dataset.

    Args:
        test_dataset_list (List[str]): List of dataset names to test
        task_name (str): Name of the task
    """

    def __init__(self, test_dataset_list: List[str], task_name: str) -> None:
        super().__init__(
            metric_train=False,
            metric_val=False,
            metric_test=True,
        )
        self.test_dataset_list = test_dataset_list
        self.task_name = task_name
        self.metrics: nn.ModuleDict = self.init_metrics()
        self.dataset_name: Optional[str] = None

    def init_metrics(self, **kwargs) -> nn.ModuleDict:
        """Initializes metric objects.

        Returns:
            nn.ModuleDict: ModuleDict containing initialized metric objects
        """
        init_metrics = nn.ModuleDict()
        for dataset_name in self.test_dataset_list:
            _metrics = nn.ModuleDict()
            _metrics["preds"] = tm.CatMetric()
            _metrics["labels"] = tm.CatMetric()
            init_metrics[f"{dataset_name}"] = _metrics
        return init_metrics

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Method called at the beginning of training."""
        pass

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Moves metrics to the appropriate device when testing begins."""
        self.metrics.to(pl_module.device)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initializes dataset_name at the end of testing and calls the parent class method."""
        self.dataset_name = None
        super().on_test_end(trainer, pl_module)

    def update_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """Updates metrics with predictions and actual values.

        Args:
            preds (torch.Tensor): Model's predictions
            labels (torch.Tensor): Actual labels
        """
        for k, v in self.metrics[self.dataset_name].items():
            if "preds" == k:
                v.update(preds)
            elif "labels" == k:
                v.update(labels)

    def compute_metrics(self, pl_module: LightningModule, split: str) -> None:
        """Computes and logs all metrics for the current dataset.

        Args:
            pl_module (LightningModule): Lightning module
            split (str): Dataset split (train/val/test)
        """
        m = self.metrics[self.dataset_name]
        preds = m["preds"].compute()
        labels = m["labels"].compute()
        m["preds"].reset()
        m["labels"].reset()
        preds = torch.sigmoid(preds)

        # Find optimal threshold using Youden's J statistic
        fpr, tpr, thresholds = tm.functional.roc(
            preds, labels.to(torch.long), task="binary", ignore_index=-1
        )
        youden_j = tpr - fpr
        best_threshold = thresholds[torch.argmax(youden_j)]

        # Calculate AUROC, AUPRC
        auroc = tm.functional.auroc(preds, labels.to(torch.long), task="binary", ignore_index=-1)
        auprc = tm.functional.average_precision(
            preds, labels.to(torch.long), task="binary", ignore_index=-1
        )

        pl_module.log(
            f"test/{self.dataset_name}/auroc",
            auroc,
            sync_dist=True,
            rank_zero_only=True,
        )
        pl_module.log(
            f"test/{self.dataset_name}/auprc",
            auprc,
            sync_dist=True,
            rank_zero_only=True,
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Calculates and logs metrics at the end of each test batch.

        Args:
            trainer (Trainer): Lightning trainer
            pl_module (LightningModule): Lightning module
            outputs (Any): Model outputs
            batch (Dict[str, Any]): Input batch
            batch_idx (int): Batch index
            dataloader_idx (int): Dataloader index
        """
        img, label = batch["img"], batch["label"]
        if self.dataset_name is None:
            self.dataset_name = batch["dataset_name"][0]

        preds = pl_module(img, mode="test")
        if isinstance(preds, tuple):
            preds = sum(preds) / len(preds)

        self.update_metrics(preds, label)


# --- Segmentation Metric ---
##################################################################
def get_stats(
    output: Union[torch.LongTensor, torch.FloatTensor],
    target: torch.LongTensor,
    mode: str,
    ignore_index: Optional[int] = None,
    threshold: Optional[Union[float, List[float]]] = None,
    num_classes: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Compute true positive, false positive, false negative, true negative 'pixels'
    for each image and each class.

    Args:
        output (Union[torch.LongTensor, torch.FloatTensor]): Model output with following
            shapes and types depending on the specified ``mode``:

            'binary'
                shape (N, 1, ...) and ``torch.LongTensor`` or ``torch.FloatTensor``

            'multilabel'
                shape (N, C, ...) and ``torch.LongTensor`` or ``torch.FloatTensor``

            'multiclass'
                shape (N, ...) and ``torch.LongTensor``

        target (torch.LongTensor): Targets with following shapes depending on the specified ``mode``:

            'binary'
                shape (N, 1, ...)

            'multilabel'
                shape (N, C, ...)

            'multiclass'
                shape (N, ...)

        mode (str): One of ``'binary'`` | ``'multilabel'`` | ``'multiclass'``
        ignore_index (Optional[int]): Label to ignore on for metric computation.
            **Not** supported for ``'binary'`` and ``'multilabel'`` modes.  Defaults to None.
        threshold (Optional[float, List[float]]): Binarization threshold for
            ``output`` in case of ``'binary'`` or ``'multilabel'`` modes. Defaults to None.
        num_classes (Optional[int]): Number of classes, necessary attribute
            only for ``'multiclass'`` mode. Class values should be in range 0..(num_classes - 1).
            If ``ignore_index`` is specified it should be outside the classes range, e.g. ``-1`` or
            ``255``.

    Raises:
        ValueError: in case of misconfiguration.

    Returns:
        Tuple[torch.LongTensor]: true_positive, false_positive, false_negative,
            true_negative tensors (N, C) shape each.

    """

    if torch.is_floating_point(target):
        raise ValueError(f"Target should be one of the integer types, got {target.dtype}.")

    if torch.is_floating_point(output) and threshold is None:
        raise ValueError(
            f"Output should be one of the integer types if ``threshold`` is not None, got {output.dtype}."
        )

    if torch.is_floating_point(output) and mode == "multiclass":
        raise ValueError(
            f"For ``multiclass`` mode ``output`` should be one of the integer types, got {output.dtype}."
        )

    if mode not in {"binary", "multiclass", "multilabel"}:
        raise ValueError(
            f"``mode`` should be in ['binary', 'multiclass', 'multilabel'], got mode={mode}."
        )

    if mode == "multiclass" and threshold is not None:
        raise ValueError("``threshold`` parameter does not supported for this 'multiclass' mode")

    if output.shape != target.shape:
        raise ValueError(
            "Dimensions should match, but ``output`` shape is not equal to ``target`` "
            + f"shape, {output.shape} != {target.shape}"
        )

    if mode != "multiclass" and ignore_index is not None:
        raise ValueError(f"``ignore_index`` parameter is not supported for '{mode}' mode")

    if mode == "multiclass" and num_classes is None:
        raise ValueError("``num_classes`` attribute should be not ``None`` for 'multiclass' mode.")

    if ignore_index is not None and 0 <= ignore_index <= num_classes - 1:
        raise ValueError(
            f"``ignore_index`` should be outside the class values range, but got class values in range "
            f"0..{num_classes - 1} and ``ignore_index={ignore_index}``. Hint: if you have ``ignore_index = 0``"
            f"consirder subtracting ``1`` from your target and model output to make ``ignore_index = -1``"
            f"and relevant class values started from ``0``."
        )

    if mode == "multiclass":
        tp, fp, fn, tn = _get_stats_multiclass(output, target, num_classes, ignore_index)
    else:
        if threshold is not None:
            output = torch.where(output >= threshold, 1, 0)
            target = torch.where(target >= threshold, 1, 0)
        tp, fp, fn, tn = _get_stats_multilabel(output, target)

    return tp, fp, fn, tn


@torch.no_grad()
def _get_stats_multiclass(
    output: torch.LongTensor,
    target: torch.LongTensor,
    num_classes: int,
    ignore_index: Optional[int],
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Calculate true positives, false positives, false negatives and true negatives for multiclass classification.

    Args:
        output (torch.LongTensor): Predicted labels of shape (batch_size, *dims)
        target (torch.LongTensor): Ground truth labels of shape (batch_size, *dims)
        num_classes (int): Number of classes
        ignore_index (Optional[int]): Index to ignore in target

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
            True positives, false positives, false negatives and true negatives counts
            of shape (batch_size, num_classes)
    """
    batch_size, *dims = output.shape
    num_elements = torch.prod(torch.tensor(dims)).long()

    if ignore_index is not None:
        ignore = target == ignore_index
        output = torch.where(ignore, -1, output)
        target = torch.where(ignore, -1, target)
        ignore_per_sample = ignore.view(batch_size, -1).sum(1)

    tp_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    fp_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    fn_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    tn_count = torch.zeros(batch_size, num_classes, dtype=torch.long)

    for i in range(batch_size):
        target_i = target[i]
        output_i = output[i]
        mask = output_i == target_i
        matched = torch.where(mask, target_i, -1)
        tp = torch.histc(matched.float(), bins=num_classes, min=0, max=num_classes - 1)
        fp = torch.histc(output_i.float(), bins=num_classes, min=0, max=num_classes - 1) - tp
        fn = torch.histc(target_i.float(), bins=num_classes, min=0, max=num_classes - 1) - tp
        tn = num_elements - tp - fp - fn
        if ignore_index is not None:
            tn = tn - ignore_per_sample[i]
        tp_count[i] = tp.long()
        fp_count[i] = fp.long()
        fn_count[i] = fn.long()
        tn_count[i] = tn.long()

    return tp_count, fp_count, fn_count, tn_count


@torch.no_grad()
def _get_stats_multilabel(
    output: torch.LongTensor, target: torch.LongTensor
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Calculate true positives, false positives, false negatives and true negatives for multilabel classification.

    Args:
        output (torch.LongTensor): Predicted labels of shape (batch_size, num_classes, *dims)
        target (torch.LongTensor): Ground truth labels of shape (batch_size, num_classes, *dims)

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
            True positives, false positives, false negatives and true negatives counts
            of shape (batch_size, num_classes)
    """
    batch_size, num_classes, *dims = target.shape
    output = output.view(batch_size, num_classes, -1)
    target = target.view(batch_size, num_classes, -1)

    tp = (output * target).sum(2)
    fp = output.sum(2) - tp
    fn = target.sum(2) - tp
    tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)

    return tp, fp, fn, tn


###################################################################################################
# Metrics computation
###################################################################################################


def _handle_zero_division(x, zero_division):
    """Handle zero division in metric calculation.

    Args:
        x (torch.Tensor): Input tensor
        zero_division (Union[str, float]): Value to use when division by zero occurs

    Returns:
        torch.Tensor: Tensor with zero divisions handled
    """
    nans = torch.isnan(x)
    if torch.any(nans) and zero_division == "warn":
        warnings.warn("Zero division in metric calculation!")
    value = zero_division if zero_division != "warn" else 0
    value = torch.tensor(value, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x


def _compute_metric(
    metric_fn,
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division="warn",
    **metric_kwargs,
) -> float:
    """Compute metric with specified reduction strategy.

    Args:
        metric_fn: Function to compute metric
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives
        reduction (Optional[str]): Reduction strategy
        class_weights (Optional[List[float]]): Class weights for weighted reduction
        zero_division (Union[str, float]): Value to use when division by zero occurs
        **metric_kwargs: Additional arguments for metric function

    Returns:
        float: Computed metric value

    Raises:
        ValueError: If class weights are not provided for weighted reduction
        ValueError: If reduction strategy is invalid
    """
    if class_weights is None and reduction is not None and "weighted" in reduction:
        raise ValueError(f"Class weights should be provided for `{reduction}` reduction")

    class_weights = class_weights if class_weights is not None else 1.0
    class_weights = torch.tensor(class_weights).to(tp.device)
    class_weights = class_weights / class_weights.sum()

    if reduction == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)

    elif reduction == "macro":
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score * class_weights).mean()

    elif reduction == "weighted":
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score * class_weights).sum()

    elif reduction == "micro-imagewise":
        tp = tp.sum(1)
        fp = fp.sum(1)
        fn = fn.sum(1)
        tn = tn.sum(1)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = score.mean()

    elif reduction == "macro-imagewise" or reduction == "weighted-imagewise":
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score.mean(0) * class_weights).mean()

    elif reduction == "none" or reduction is None:
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)

    else:
        raise ValueError(
            "`reduction` should be in [micro, macro, weighted, micro-imagewise,"
            + "macro-imagesize, weighted-imagewise, none, None]"
        )

    return score


# Logic for metric computation, all metrics are with the same interface


def _fbeta_score(tp, fp, fn, tn, beta=1):
    """Calculate F-beta score.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives
        beta (float): Beta parameter for F-beta score

    Returns:
        torch.Tensor: F-beta score
    """
    beta_tp = (1 + beta**2) * tp
    beta_fn = (beta**2) * fn
    score = beta_tp / (beta_tp + beta_fn + fp)
    return score


def _iou_score(tp, fp, fn, tn):
    """Calculate IoU score.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: IoU score
    """
    return tp / (tp + fp + fn)


def _accuracy(tp, fp, fn, tn):
    """Calculate accuracy.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: Accuracy
    """
    return (tp + tn) / (tp + fp + fn + tn)


def _sensitivity(tp, fp, fn, tn):
    """Calculate sensitivity.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: Sensitivity
    """
    return tp / (tp + fn)


def _specificity(tp, fp, fn, tn):
    """Calculate specificity.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: Specificity
    """
    return tn / (tn + fp)


def _balanced_accuracy(tp, fp, fn, tn):
    """Calculate balanced accuracy.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: Balanced accuracy
    """
    return (_sensitivity(tp, fp, fn, tn) + _specificity(tp, fp, fn, tn)) / 2


def _positive_predictive_value(tp, fp, fn, tn):
    """Calculate positive predictive value.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: Positive predictive value
    """
    return tp / (tp + fp)


def _negative_predictive_value(tp, fp, fn, tn):
    """Calculate negative predictive value.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: Negative predictive value
    """
    return tn / (tn + fn)


def _false_negative_rate(tp, fp, fn, tn):
    """Calculate false negative rate.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: False negative rate
    """
    return 1 - _sensitivity(tp, fp, fn, tn)


def _false_positive_rate(tp, fp, fn, tn):
    """Calculate false positive rate.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: False positive rate
    """
    return 1 - _specificity(tp, fp, fn, tn)


def _false_discovery_rate(tp, fp, fn, tn):
    """Calculate false discovery rate.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: False discovery rate
    """
    return 1 - _positive_predictive_value(tp, fp, fn, tn)


def _false_omission_rate(tp, fp, fn, tn):
    """Calculate false omission rate.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: False omission rate
    """
    return 1 - _negative_predictive_value(tp, fp, fn, tn)


def _positive_likelihood_ratio(tp, fp, fn, tn):
    """Calculate positive likelihood ratio.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: Positive likelihood ratio
    """
    return _sensitivity(tp, fp, fn, tn) / _false_positive_rate(tp, fp, fn, tn)


def _negative_likelihood_ratio(tp, fp, fn, tn):
    """Calculate negative likelihood ratio.

    Args:
        tp (torch.LongTensor): True positives
        fp (torch.LongTensor): False positives
        fn (torch.LongTensor): False negatives
        tn (torch.LongTensor): True negatives

    Returns:
        torch.Tensor: Negative likelihood ratio
    """
    return _false_negative_rate(tp, fp, fn, tn) / _specificity(tp, fp, fn, tn)


def fbeta_score(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    beta: float = 1.0,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """F beta score"""
    return _compute_metric(
        _fbeta_score,
        tp,
        fp,
        fn,
        tn,
        beta=beta,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def f1_score(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """F1 score"""
    return _compute_metric(
        _fbeta_score,
        tp,
        fp,
        fn,
        tn,
        beta=1.0,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def iou_score(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """IoU score or Jaccard index"""  # noqa
    return _compute_metric(
        _iou_score,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def accuracy(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Accuracy"""
    return _compute_metric(
        _accuracy,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def sensitivity(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Sensitivity, recall, hit rate, or true positive rate (TPR)"""
    return _compute_metric(
        _sensitivity,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def specificity(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Specificity, selectivity or true negative rate (TNR)"""
    return _compute_metric(
        _specificity,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def balanced_accuracy(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Balanced accuracy"""
    return _compute_metric(
        _balanced_accuracy,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def positive_predictive_value(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Precision or positive predictive value (PPV)"""
    return _compute_metric(
        _positive_predictive_value,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def negative_predictive_value(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Negative predictive value (NPV)"""
    return _compute_metric(
        _negative_predictive_value,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_negative_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Miss rate or false negative rate (FNR)"""
    return _compute_metric(
        _false_negative_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_positive_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Fall-out or false positive rate (FPR)"""
    return _compute_metric(
        _false_positive_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_discovery_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """False discovery rate (FDR)"""  # noqa
    return _compute_metric(
        _false_discovery_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_omission_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """False omission rate (FOR)"""  # noqa
    return _compute_metric(
        _false_omission_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def positive_likelihood_ratio(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Positive likelihood ratio (LR+)"""
    return _compute_metric(
        _positive_likelihood_ratio,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def negative_likelihood_ratio(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Negative likelihood ratio (LR-)"""
    return _compute_metric(
        _negative_likelihood_ratio,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


precision = positive_predictive_value
recall = sensitivity


class TLSegmentationMetric(TLClassificationMetric):
    """A class for computing segmentation metrics in transfer learning tasks.

    This class extends TLClassificationMetric to handle segmentation-specific metrics
    including IoU, Dice score, and AUPRC. It also provides functionality to visualize
    predictions by overlaying masks on images.

    Attributes:
        test_dataset_list (List[str]): List of test dataset names
        task_name (str): Name of the current task
        target_class (List[str]): List of target classes for segmentation
        metrics (nn.ModuleDict): Dictionary containing metric objects
        dataset_name (Optional[str]): Current dataset name being processed
    """

    def __init__(
        self, test_dataset_list: List[str], task_name: str, target_class: List[str], save_dir: str
    ) -> None:
        super().__init__(
            test_dataset_list=test_dataset_list,
            task_name=task_name,
        )
        self.test_dataset_list = test_dataset_list
        self.task_name = task_name
        self.metrics: nn.ModuleDict = self.init_metrics()
        self.dataset_name: Optional[str] = None
        self.target_class = target_class
        self.save_dir = save_dir

    def init_metrics(self, **kwargs) -> nn.ModuleDict:
        """Initialize metric objects for each dataset.

        Returns:
            nn.ModuleDict: Dictionary containing initialized metric objects
        """
        init_metrics = nn.ModuleDict()
        for dataset_name in self.test_dataset_list:
            _metrics = nn.ModuleDict()
            _metrics["preds"] = tm.CatMetric()
            _metrics["labels"] = tm.CatMetric()
            _metrics["images"] = tm.CatMetric()
            _metrics["valid_mask"] = tm.CatMetric()
            init_metrics[f"{dataset_name}"] = _metrics
        return init_metrics

    def update_metrics(
        self,
        images: torch.Tensor,
        preds: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """Update metrics with predictions and ground truth labels.

        Args:
            images (torch.Tensor): Input images
            preds (torch.Tensor): Model predictions
            labels (torch.Tensor): Ground truth labels
        """
        for k, v in self.metrics[self.dataset_name].items():
            if "preds" == k:
                v.update(preds)
            elif "images" == k:
                v.update(images)
            elif "labels" == k:
                v.update(labels)
            elif "valid_mask" == k:
                v.update(valid_mask)

    def compute_metrics(self, pl_module: LightningModule, split: str) -> None:
        """Compute and log all metrics for the current dataset.

        Args:
            pl_module (LightningModule): PyTorch Lightning module
            split (str): Dataset split (train/val/test)
        """
        m = self.metrics[self.dataset_name]
        images = m["images"].compute()  # (B, 3, H, W)
        preds = m["preds"].compute()
        preds = preds.sigmoid()
        labels = m["labels"].compute()
        valid_mask = m["valid_mask"].compute()
        m["preds"].reset()
        m["labels"].reset()
        m["valid_mask"].reset()

        for c_i, c in enumerate(sorted(self.target_class)):
            valid_mask_i = valid_mask[:, c_i]
            img_i = images[valid_mask_i == 1]
            preds_i = preds[:, c_i][valid_mask_i == 1]
            labels_i = labels[:, c_i][valid_mask_i == 1]
            if preds_i.numel() == 0 or labels_i.numel() == 0:
                continue

            # Calculate optimal threshold using F1 score
            precision, recall, thresholds = tm.functional.precision_recall_curve(
                preds_i, labels_i.long(), task="binary"
            )
            f1_scores = (
                2 * (precision * recall) / (precision + recall + 1e-7)
            )  # Add small epsilon to avoid division by zero
            optimal_idx = torch.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            binary_preds = (preds_i > optimal_threshold).float()

            # Save binary predictions as images
            save_dir = Path(f"{self.save_dir}/{self.task_name}/{self.dataset_name}/{c}_preds")
            save_dir.mkdir(parents=True, exist_ok=True)

            def overlay_mask_on_image(image, mask, color, alpha=0.5):
                """Overlay a binary mask on an image with specified color and transparency.

                Args:
                    image (torch.Tensor): Input image tensor of shape (3, H, W) in [0,1] range
                    mask (torch.Tensor): Binary mask tensor of shape (H, W)
                    color (tuple): RGB color tuple for the overlay
                    alpha (float): Transparency of the overlay

                Returns:
                    torch.Tensor: Image with overlaid mask
                """
                image_np = image.cpu().numpy()
                mask_np = mask.cpu().numpy()

                overlay = np.copy(image_np)
                for c in range(3):
                    overlay[c] = np.where(
                        mask_np > 0, (1 - alpha) * image_np[c] + alpha * color[c], image_np[c]
                    )
                return torch.tensor(overlay, device=image.device)

            # Batch loop
            concat_images = []
            for i in range(img_i.shape[0]):
                img = img_i[i]  # (3, H, W)
                # Squeeze if binary_preds and labels are (B, 1, H, W)
                pred_mask = binary_preds[i]
                if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                    pred_mask = pred_mask.squeeze(0)  # (H, W)

                gt_mask = labels_i[i]
                if gt_mask.ndim == 3 and gt_mask.shape[0] == 1:
                    gt_mask = gt_mask.squeeze(0)  # (H, W)

                img_pred_overlay = overlay_mask_on_image(img, pred_mask, (0, 0, 1), 0.5)
                img_gt_overlay = overlay_mask_on_image(img, gt_mask, (0, 1, 0), 0.5)

                concat_img = torch.cat(
                    [img, img_pred_overlay, img_gt_overlay], dim=-1
                )  # (3, H, 3*W)
                concat_images.append(concat_img)

            for i, img in enumerate(concat_images):
                save_path = save_dir / f"sample_{i}.png"
                torchvision.utils.save_image(img, save_path)

            tp, fp, fn, tn = get_stats(
                preds_i, labels_i.long(), mode="binary", threshold=optimal_threshold
            )
            iou = iou_score(tp, fp, fn, tn, reduction="macro")
            dice = f1_score(tp, fp, fn, tn, reduction="macro")
            auprc = tm.functional.average_precision(
                preds_i, labels_i.long(), task="binary", average="macro"
            )
            auroc = tm.functional.auroc(preds_i, labels_i.long(), task="binary", average="macro")
            sens = sensitivity(tp, fp, fn, tn, reduction="macro")
            spec = specificity(tp, fp, fn, tn, reduction="macro")
            acc = accuracy(tp, fp, fn, tn, reduction="macro")
            pl_module.log(
                f"test/{self.dataset_name}/iou/{c}",
                iou,
                sync_dist=True,
                rank_zero_only=True,
            )
            pl_module.log(
                f"test/{self.dataset_name}/dice/{c}",
                dice,
                sync_dist=True,
                rank_zero_only=True,
            )
            pl_module.log(
                f"test/{self.dataset_name}/auprc/{c}",
                auprc,
                sync_dist=True,
                rank_zero_only=True,
            )
            pl_module.log(
                f"test/{self.dataset_name}/auroc/{c}",
                auroc,
                sync_dist=True,
                rank_zero_only=True,
            )
            pl_module.log(
                f"test/{self.dataset_name}/sens/{c}",
                sens,
                sync_dist=True,
                rank_zero_only=True,
            )
            pl_module.log(
                f"test/{self.dataset_name}/spec/{c}",
                spec,
                sync_dist=True,
                rank_zero_only=True,
            )
            pl_module.log(
                f"test/{self.dataset_name}/acc/{c}",
                acc,
                sync_dist=True,
                rank_zero_only=True,
            )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Update metrics at the end of each test batch.

        Args:
            trainer (Trainer): PyTorch Lightning trainer
            pl_module (LightningModule): PyTorch Lightning module
            outputs (Any): Model outputs
            batch (Dict[str, Any]): Input batch
            batch_idx (int): Batch index
        """
        img, label, valid_mask = batch["img"], batch["mask"], batch["valid_mask"]
        if self.dataset_name is None:
            self.dataset_name = batch["dataset_name"][0]

        img, preds, label = pl_module(img, label, mode="test")
        if isinstance(preds, tuple):
            preds = sum(preds) / len(preds)
        preds = preds.sigmoid()

        self.update_metrics(img, preds, label, valid_mask)


if __name__ == "__main__":
    from pytorch_lightning import Trainer

    batch_size = 4
    feat_dim = 128
    num_classes = len(TARGET_CLASS_LIST["finding"] + TARGET_CLASS_LIST["detail_dx"])

    # 임의의 특징 벡터와 레이블 생성
    dummy_feats = torch.randn(batch_size, 5, feat_dim).to("cuda")
    dummy_labels = torch.tensor(
        [
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        ]
    ).to("cuda")
    dummy_dataset_names = ["BH"] * batch_size

    # 테스트용 배치 데이터 생성
    batch = {"label": dummy_labels, "dataset_name": dummy_dataset_names}

    outputs = {"latent_register_tokens": dummy_feats}

    # PEDScoreCallback 초기화 및 테스트
    pl_module = pl.LightningModule()
    callback = PEDScoreMetric(metric_val=True, score_type="LOGME", pool_type="first")

    # 콜백 테스트
    print("=== PEDScoreCallback 테스트 시작 ===")

    print("=== 첫 번째 업데이트 ===")
    callback.update_metrics(
        outputs["latent_register_tokens"][:2],
        batch["label"][:2],
        split="val",
        dataset_name=batch["dataset_name"][:2],
    )

    print("=== 두 번째 업데이트 ===")
    callback.update_metrics(
        outputs["latent_register_tokens"][2:],
        batch["label"][2:],
        split="val",
        dataset_name=batch["dataset_name"][2:],
    )

    print("=== 메트릭 계산 ===")
    callback.compute_metrics(pl_module, "val")

    print("\n테스트 성공적으로 완료!")
