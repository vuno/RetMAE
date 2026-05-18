import gc
import os
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import hydra
import kornia.augmentation as K
import lightning as L
import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.tasks.pretrain.decoder.unetr import UNETR
from src.tasks.pretrain.utils import cosine_scheduler
from src.tasks.transfer import safe_release
from src.utils import (
    RankedLogger,
    data_path,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)


def _seg_csv(name: str, split: str) -> str:
    return str(data_path("DATASET_EXTERNAL_EVAL_224", name, f"seg_labels_{split}.csv"))

log = RankedLogger(__name__, rank_zero_only=True)


def compute_threshold(preds: np.ndarray, labels: np.ndarray, default_th: float = 0.5) -> float:
    """
    preds: (N,1,H,W) or (N,H,W) sigmoid probabilities in [0,1]
    labels: (N,1,H,W) or (N,H,W) {0,1} or grayscale -> binarized with >0
    return: threshold that maximizes F1 (Dice)
    """
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if labels.ndim == 4:
        labels = labels.squeeze(1)

    p = preds.reshape(-1)
    y = labels.reshape(-1)

    # Safety guards on inputs.
    p = np.clip(p, 0.0, 1.0)
    y = (y > 0).astype(np.uint8)  # 명시적 이진화

    pos = y.sum()
    neg = y.size - pos
    if pos == 0 or neg == 0:
        # Skip optimization when only one class is present.
        return float(default_th)

    from sklearn.metrics import precision_recall_curve

    precision, recall, ths = precision_recall_curve(y, p)
    # thresholds has length len(precision) - 1 == len(recall) - 1.
    if ths.size == 0:
        # All prediction scores identical or no positives detected.
        return float(default_th)

    numer = 2 * precision[:-1] * recall[:-1]
    denom = precision[:-1] + recall[:-1]
    f1_scores = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)

    idx = int(np.nanargmax(f1_scores))
    best_th = float(ths[idx])
    if not np.isfinite(best_th):
        return float(default_th)
    return best_th


class TransferLearningModule(LightningModule):
    """
    PyTorch Lightning module for transfer learning.

    This module loads a pretrained encoder and performs either fine-tuning or linear probing
    on a new task. Both multi-class and multi-label classification are supported.

    Args:
        encoder (nn.Module): Pretrained encoder model.
        num_classes (int, optional): Number of target classes. Default is 1.
        transfer_type (str, optional): 'linear_probing' or 'finetuning'. Default is 'finetuning'.
        pool_type (str, optional): Pooling strategy: one of 'cls_token', 'avg', 'flatten', 'none', 'patch_token'. Default is 'none'.
        learning_rate (float, optional): Learning rate. Default is 1e-3.
        weight_decay (float, optional): Weight decay for optimizer. Default is 1e-4.
        warmup_epochs (int, optional): Number of warmup epochs. Default is 20.
        target_size (int, optional): Target image size for data augmentation. Default is 224.
    """

    def __init__(
        self,
        encoder: nn.Module,
        transfer_type: str = "finetuning",
        pool_type: str = "none",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 20,
        target_size: int = 224,
        data_augmentation: bool = True,
    ):
        """
        Initialize the TransferLearningModule with model settings and optimizer configuration.
        """
        super().__init__()
        assert transfer_type in [
            "linear_probing",
            "finetuning",
        ], f"transfer_type must be in ['linear_probing', 'finetuning'], but got {transfer_type}"
        assert pool_type in [
            "cls_token",
            "avg",
            "flatten",
            "none",
            "patch_token",
        ], f"pool_type must be in ['cls_token', 'avg', 'flatten', 'none', 'patch_token'], but got {pool_type}"

        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder
        self.target_size = target_size
        self.data_augmentation = data_augmentation
        if transfer_type == "linear_probing":
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        # Linear classifier for both linear_probing and finetuning
        self.decoder = UNETR(encoder, out_channels=1)

        self.criterion = F.binary_cross_entropy_with_logits

        self.transform_train = self._get_transform("train")
        self.lr_schedule = None
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.automatic_optimization = False

    def _dice_loss(
        self, logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5
    ) -> torch.Tensor:
        """
        Binary soft Dice loss.
        - logits: (N,1,H,W) raw logits
        - target: (N,1,H,W) in {0,1}
        """
        # 시그모이드로 확률화
        probs = torch.sigmoid(logits)

        # 타입/shape 정리
        if target.dtype != probs.dtype:
            target = target.to(probs.dtype)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # 배치별 dice 계산 (평균)
        # intersection = sum(p*t), denominator = sum(p)+sum(t)
        dims = (2, 3)  # H,W에 대해 합
        intersection = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)

        dice = (2.0 * intersection + smooth) / (denom + smooth)  # (N,1) or (N,)
        dice_loss = 1.0 - dice.mean()
        return dice_loss

    def on_train_epoch_start(self):
        """Reset the sampler epoch so that shuffling differs between epochs."""
        if hasattr(self.trainer.train_dataloader, "sampler") and hasattr(
            self.trainer.train_dataloader.sampler, "set_epoch"
        ):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def forward(self, img: torch.Tensor, split: str = "train") -> torch.Tensor:
        """Run the decoder on the provided batch and return raw logits."""
        assert split in ["train", "val", "test"], "split must be in ['train', 'val', 'test']"
        if self.encoder.training and self.hparams.transfer_type == "linear_probing":
            raise ValueError("encoder should be in eval mode when transfer_type is linear_probing")
        return self.decoder(img)

    def _get_transform(self, mode: str):
        """Return the augmentation pipeline corresponding to the requested mode."""
        assert mode in [
            "train",
            "val",
            "test",
        ], "mode must be in ['train', 'val', 'test']"
        if mode == "train":
            transform = nn.Sequential(
                K.RandomAffine(
                    degrees=(-10, 10),
                    scale=(1, 1),
                    translate=None,
                    shear=None,
                    padding_mode="zeros",
                    p=1.0,
                    same_on_batch=False,
                    align_corners=False,
                ),
                K.RandomResizedCrop(
                    size=(self.target_size, self.target_size),
                    scale=(0.7, 1.0),
                    ratio=(1.0, 1.0),
                    resample="bilinear",
                    p=1.0,
                    same_on_batch=False,
                ),
                K.RandomHorizontalFlip(p=0.5),
            )
        else:
            transform = None
        return transform

    def configure_optimizers(self) -> Dict[str, Any]:
        """Create the AdamW optimizer with parameter groups for weight decay."""
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if any(nd in name for nd in ["pos_embed", "cls_token", "reg_token", "norm", "bias"]):
                no_decay.append(param)
            else:
                decay.append(param)

        params = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.99, 0.999),
        )
        return {"optimizer": optimizer}

    def _calculate_loss(self, batch: dict, split: str = "train") -> torch.Tensor:
        """Compute BCE and Dice losses for a batch and log the intermediate values."""
        img, label = batch["img"], batch["label"]
        if self.data_augmentation and split == "train":
            img_concat = torch.cat([img, label.unsqueeze(1)], dim=1)
            img_concat = self.transform_train(img_concat)
            img, label = img_concat[:, :3], img_concat[:, 3:]

        if label.dim() == 3:
            label = label.unsqueeze(1)

        # BCE (logits 입력)
        preds = self(img, split)
        bce = self.criterion(preds, label.float())

        # Dice (sigmoid 내부에서 계산)
        dice = self._dice_loss(preds, label.float())

        total = bce + dice

        # 로깅(선택)
        self.log(
            f"{split}/bce",
            bce,
            on_step=(split == "train"),
            on_epoch=(split != "train"),
            sync_dist=True,
            rank_zero_only=False,
        )
        self.log(
            f"{split}/dice",
            dice,
            on_step=(split == "train"),
            on_epoch=(split != "train"),
            sync_dist=True,
            rank_zero_only=False,
        )
        self.log(
            f"{split}/total_loss",
            total,
            on_step=(split == "train"),
            on_epoch=(split != "train"),
            sync_dist=True,
            rank_zero_only=False,
        )
        return total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Perform one optimization step with manual optimizer control."""
        loss = self._calculate_loss(batch, split="train")
        opt = self.optimizers()
        opt.zero_grad()
        loss.backward()
        if self.hparams.transfer_type == "finetuning":
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0, norm_type=2)
        self._log_grad_norm(self.encoder.parameters(), split="train", prefix="encoder")
        self._log_grad_norm(self.decoder.parameters(), split="train", prefix="decoder")
        opt.step()
        if self.lr_schedule is not None and self.global_step < len(self.lr_schedule):
            opt.param_groups[0]["lr"] = self.lr_schedule[self.global_step]
            self.log(
                "lr",
                self.lr_schedule[self.global_step],
                sync_dist=True,
                rank_zero_only=True,
                on_epoch=False,
                on_step=True,
            )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Run the validation loss computation to monitor generalization."""
        _ = self._calculate_loss(batch, split="val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Evaluate the loss terms on the held-out test set."""
        _ = self._calculate_loss(batch, split="test")

    @torch.inference_mode()
    def _log_grad_norm(
        self,
        parameters: Union[Iterable[torch.nn.Parameter], torch.nn.Parameter],
        norm_type: int = 2,
        split: str = "train",
        prefix: str = "",
    ) -> float:
        """Compute and log the gradient norm for the provided parameters."""
        if isinstance(parameters, torch.nn.Parameter):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        total_norm = torch.zeros(1, device=self.device)
        try:
            for p in parameters:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.pow(norm_type)
            total_norm = total_norm.pow(1.0 / norm_type)
        except Exception as e:
            print(f"⚠️ Error occurred during gradient norm calculation: {e}")
            total_norm = torch.tensor(float("nan"), device=self.device)
        if prefix:
            title = f"{split}/{prefix}_grad_norm"
        else:
            title = f"{split}/grad_norm"
        self.log(
            title,
            total_norm.item(),
            sync_dist=True,
            rank_zero_only=True,
            on_step=True,
            on_epoch=False,
        )


TRAIN_DATASET = {"IDRiD": _seg_csv("IDRiD", "train")}
TEST_DATASET = {"IDRiD": _seg_csv("IDRiD", "test")}

TARGETS = {
    "IDRiD": ["HE", "EX", "CWP"],
}


class CustomDataset(Dataset):
    """Dataset wrapper used for both training and evaluation splits."""

    def __init__(
        self,
        csv_path: str,
        dataset_name: str,
        indices: Optional[List[int]] = None,
        target_size: int = 224,
        target_label: str = "HE",
    ):
        assert target_label in ["HE", "EX", "CWP"]
        self.dataset_name = dataset_name
        self.target_size = target_size
        self.df = pd.read_csv(csv_path)
        self.target_label = target_label
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[self.target_label]).reset_index(drop=True)
        dropped_count = initial_count - len(self.df)
        if dropped_count > 0:
            log.info(
                f"Dropped {dropped_count} rows with NaN mask paths for {target_label} (remaining: {len(self.df)})"
            )
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.filepaths = self.df["filepath"].tolist()
        self.mask_paths = self.df[self.target_label].values
        ssd_root = str(data_path("DATASET_EXTERNAL_EVAL_224", dataset_name))
        original_root = str(data_path("DATASET", dataset_name))
        if os.path.exists(ssd_root):
            self.root = ssd_root
            log.info(f"Using SSD path for {dataset_name}: {self.root}")
        else:
            self.root = original_root
            log.info(f"Using original path for {dataset_name}: {self.root}")

    def __len__(self):
        """Return the number of samples available after filtering."""
        return len(self.filepaths)

    def __getitem__(self, idx):
        """Load a single RGB fundus image and its segmentation mask."""
        img_path = os.path.join(self.root, self.filepaths[idx])
        mask_path = os.path.join(self.root, self.mask_paths[idx])
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise RuntimeError(f"Failed to load image or mask: {img_path} or {mask_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0 if img.max() > 1.0 else img.astype(np.float32)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.float32) / 255.0
        return {
            "img": img,
            "filepath": self.filepaths[idx],
            "label": torch.tensor(mask, dtype=torch.float32),
        }


def calculate_dice_iou(
    y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5
) -> Tuple[float, float]:
    """Calculate mean Dice and IoU scores after thresholding predictions."""
    # (N,1,H,W) -> (N,H,W)
    if len(y_pred.shape) == 4:
        y_pred = y_pred.squeeze(1)
    if len(y_true.shape) == 4:
        y_true = y_true.squeeze(1)

    # Apply binarization to both predictions and ground truth.
    y_pred = (y_pred > threshold).astype(bool)
    y_true = (y_true > 0).astype(bool)  # GT가 0/1이면 .astype(bool)로 대체 가능

    dice_scores, iou_scores = [], []
    for pred, true in zip(y_pred, y_true):
        intersection = np.logical_and(pred, true).sum()
        a, b = pred.sum(), true.sum()
        union_iou = np.logical_or(pred, true).sum()

        # Exact Dice without smoothing; handle the empty-mask case explicitly.
        if (a + b) == 0:
            dice = 1.0
        else:
            dice = 2.0 * intersection / (a + b)

        if union_iou == 0:
            iou = 1.0
        else:
            iou = intersection / union_iou

        dice_scores.append(dice)
        iou_scores.append(iou)

    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


def worker_init_fn(worker_id):
    """Seed numpy and torch PRNGs per worker for deterministic behavior."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def collate_fn(batch):
    """Convert a list of samples from the dataset into a stacked batch."""
    imgs = torch.stack(
        [
            (
                torch.from_numpy(item["img"]).permute(2, 0, 1)
                if isinstance(item["img"], np.ndarray)
                else item["img"]
            )
            for item in batch
        ]
    )
    filepaths = [item["filepath"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    return {
        "img": imgs,
        "filepath": filepaths,
        "label": labels,
    }


def get_predictions_and_labels(
    model: LightningModule, dataloader: DataLoader, dataset_name: str, split: str = "test"
) -> Tuple[np.ndarray, List, np.ndarray]:
    """Run inference on a dataloader and collect predictions, filenames, and labels."""
    model.eval()
    all_preds = []
    all_filepaths = []
    all_labels = []
    for batch in tqdm(dataloader, desc=f"Getting predictions and labels for {dataset_name}"):
        if not isinstance(batch, dict) or "img" not in batch:
            raise ValueError(
                f"Invalid batch format for dataset {dataset_name}. Expected dict with 'img' key, got {type(batch)}"
            )
        filepaths = batch["filepath"]
        images = batch["img"]
        labels = batch["label"]
        if not filepaths or images.size(0) == 0:
            raise ValueError(f"Empty batch for dataset {dataset_name}")
        device = next(model.parameters()).device
        images = images.to(device)
        preds = model.forward(images)
        preds = torch.sigmoid(preds)
        all_preds.append(preds.cpu().numpy())
        all_filepaths.extend(filepaths)
        all_labels.append(labels.cpu().numpy())
        del images, preds, labels
    if not all_preds:
        raise RuntimeError(
            f"No predictions extracted for dataset {dataset_name}. Check dataloader and model."
        )
    preds_array = np.concatenate(all_preds, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    return preds_array, all_filepaths, labels_array


@rank_zero_only
@torch.inference_mode()
def evaluate_model_on_test_set(
    model: LightningModule,
    val_loader: DataLoader,
    train_dataset_name: str,
    target_label: str,
    target_size: int,
    collate_fn,
    worker_init_fn,
    cfg: DictConfig,
    run: int,
    dice_scores: list,
    iou_scores: list,
    output_dir: str,
) -> None:
    """Evaluate the segmentation model on the held-out test set and persist metrics/predictions."""
    val_preds, _, val_labels = get_predictions_and_labels(
        model, val_loader, train_dataset_name, split="val"
    )
    best_th = compute_threshold(val_preds, val_labels)
    log.info(
        f"[F1] Best threshold on validation for {train_dataset_name}/{target_label}/run{run+1}: {best_th:.4f}"
    )

    test_csv_path = TEST_DATASET[train_dataset_name]
    test_dataset = CustomDataset(
        test_csv_path, train_dataset_name, target_size=target_size, target_label=target_label
    )
    log.info(
        f"Evaluating {train_dataset_name} target {target_label} for run {run + 1} - Test dataset: {len(test_dataset)} samples"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=False,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,
    )
    model.eval()
    preds_array, filepaths, labels_array = get_predictions_and_labels(
        model, test_loader, train_dataset_name, split="test"
    )
    if preds_array is None or len(preds_array) == 0:
        raise RuntimeError(
            f"preds_array is empty for dataset {train_dataset_name} target {target_label} run {run + 1}. "
            f"This might be due to: "
            f"1) Empty test dataset (len={len(test_dataset)}), "
            f"2) No valid batches in dataloader, "
            f"3) All image loading failed, or "
            f"4) Model inference failed for all samples."
        )
    if not filepaths or len(filepaths) == 0:
        raise RuntimeError(
            f"filepaths is empty for dataset {train_dataset_name} target {target_label} run {run + 1}"
        )
    if labels_array is None or len(labels_array) == 0:
        raise RuntimeError(
            f"labels_array is empty for dataset {train_dataset_name} target {target_label} run {run + 1}"
        )

    # Threshold predictions with the validation-optimal Dice/F1 threshold and compute summary metrics.
    dice, iou_score = calculate_dice_iou(preds_array, labels_array, threshold=best_th)
    dice_scores.append(dice)
    iou_scores.append(iou_score)
    log.info(
        f"Run {run + 1} target {target_label} - (th={best_th:.4f}) Dice: {dice:.4f}, IoU: {iou_score:.4f}"
    )

    # Persist per-image mean prediction scores (legacy behaviour).
    if len(preds_array.shape) == 4:
        preds_flat = preds_array.squeeze(1).reshape(len(preds_array), -1).mean(axis=1)
    else:
        preds_flat = preds_array.reshape(len(preds_array), -1).mean(axis=1)

    del test_loader, test_dataset
    gc.collect()
    try:
        model.to("cpu")
        log.info("Moved model to CPU after evaluation")
    except Exception as e:
        log.warning(f"Failed to move model to CPU: {e}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        dev = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(dev) / 1024**2
        reserved = torch.cuda.memory_reserved(dev) / 1024**2
        log.info(f"GPU {dev} memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


@rank_zero_only
def save_target_metrics(
    all_metrics_data: List[Dict[str, Any]],
    train_dataset_name: str,
    target_label: str,
    dice_scores: List[float],
    iou_scores: List[float],
) -> None:
    dice_mean = np.mean(dice_scores)
    dice_std = np.std(dice_scores)
    iou_mean = np.mean(iou_scores)
    iou_std = np.std(iou_scores)
    all_metrics_data.append(
        {
            "dataset": train_dataset_name,
            "target": target_label,
            "Dice": dice_mean,
            "Dice_std": dice_std,
            "IoU": iou_mean,
            "IoU_std": iou_std,
        }
    )
    log.info(
        f"Dataset {train_dataset_name} target {target_label} - "
        f"Dice: {dice_mean:.4f}±{dice_std:.4f}, "
        f"IoU: {iou_mean:.4f}±{iou_std:.4f}"
    )


@rank_zero_only
def save_all_metrics_and_config(
    all_metrics_data: List[Dict[str, Any]], output_dir: str, cfg: DictConfig
) -> None:
    if not all_metrics_data:
        raise RuntimeError("No metrics collected from any dataset")
    metrics_df = pd.DataFrame(all_metrics_data)
    metrics_path = os.path.join(output_dir, f"{cfg.model_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"Saved all metrics to {metrics_path}")
    cfg_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_path)
    log.info(f"Saved config to {cfg_path}")


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train the segmentation model across datasets, targets, folds, and runs."""
    base_seed = cfg.get("seed", 42)
    log.info(f"Using base seed: {base_seed}")
    L.seed_everything(base_seed, workers=True)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    rank = 0
    world_size = 1
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    output_dir = cfg.paths.output_dir
    if rank == 0:
        log.info(f"Using output directory with timestamp: {output_dir}")
    all_metrics_data = []

    for train_dataset_name, train_csv_path in TRAIN_DATASET.items():
        log.info(f"Processing dataset: {train_dataset_name}")
        targets = TARGETS.get(train_dataset_name)
        assert targets is not None
        for target_label in targets:
            log.info(f"Processing target: {target_label}")
            dice_scores = []
            iou_scores = []
            target_size = cfg.model.get("target_size", 224)
            full_dataset = CustomDataset(
                train_csv_path,
                train_dataset_name,
                target_size=target_size,
                target_label=target_label,
            )
            dataset_size = len(full_dataset)
            log.info(
                f"  Full dataset loaded: {len(full_dataset)} samples for target {target_label}"
            )
            kf = KFold(n_splits=5, shuffle=True, random_state=base_seed)
            all_indices = np.arange(len(full_dataset))
            fold_indices = list(kf.split(all_indices))
            log.info(f"Created 5 folds using KFold with base_seed={base_seed}")
            for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
                log.info(
                    f"  Fold {fold_idx + 1}: Train={len(train_idx)} samples ({len(train_idx)/len(full_dataset)*100:.1f}%), "
                    f"Val={len(val_idx)} samples ({len(val_idx)/len(full_dataset)*100:.1f}%)"
                )
            for run in range(5):
                run_seed = base_seed + run
                log.info(
                    f"Starting run {run + 1}/5 for {train_dataset_name} target {target_label} (fold {run + 1}, seed {run_seed})"
                )
                L.seed_everything(run_seed, workers=True)
                np.random.seed(run_seed)
                torch.manual_seed(run_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(run_seed)
                    torch.cuda.manual_seed_all(run_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                encoder = hydra.utils.instantiate(cfg.model.encoder)
                train_indices, val_indices = fold_indices[run]
                train_dataset = CustomDataset(
                    train_csv_path,
                    train_dataset_name,
                    train_indices.tolist(),
                    target_size=target_size,
                    target_label=target_label,
                )
                val_dataset = CustomDataset(
                    train_csv_path,
                    train_dataset_name,
                    val_indices.tolist(),
                    target_size=target_size,
                    target_label=target_label,
                )
                log.info(
                    f"Dataset sizes for {train_dataset_name} target {target_label} run {run + 1} (fold {run + 1}):"
                )
                log.info(f"  Total dataset: {dataset_size} samples")
                log.info(
                    f"  Train dataset: {len(train_dataset)} samples ({len(train_dataset)/dataset_size*100:.1f}%)"
                )
                log.info(
                    f"  Validation dataset: {len(val_dataset)} samples ({len(val_dataset)/dataset_size*100:.1f}%)"
                )
                if "transfer_type" not in cfg.model:
                    raise ValueError("transfer_type must be specified in cfg.model")
                if "pool_type" not in cfg.model:
                    raise ValueError("pool_type must be specified in cfg.model")
                model = TransferLearningModule(
                    encoder=encoder,
                    transfer_type=cfg.model.transfer_type,
                    pool_type=cfg.model.pool_type,
                    learning_rate=cfg.model.get("learning_rate", 1e-3),
                    weight_decay=cfg.model.get("weight_decay", 1e-4),
                    warmup_epochs=cfg.model.get("warmup_epochs", 20),
                    target_size=target_size,
                )
                logger = instantiate_loggers(cfg.get("logger"))
                callbacks = [
                    ModelCheckpoint(
                        monitor="val/total_loss",
                        mode="min",
                        save_top_k=1,
                        save_last=True,
                        filename="epoch{epoch:02d}-val_total_loss{val_total_loss:.4f}",
                        auto_insert_metric_name=False,
                    ),
                    EarlyStopping(
                        monitor="val/total_loss",
                        mode="min",
                        patience=10,
                        min_delta=0.0,
                        check_on_train_epoch_end=False,
                        verbose=True,
                    ),
                ]
                trainer: Trainer = hydra.utils.instantiate(
                    cfg.trainer, callbacks=callbacks, logger=logger
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=4,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    persistent_workers=False,
                    prefetch_factor=2,
                    worker_init_fn=worker_init_fn,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    persistent_workers=False,
                    prefetch_factor=2,
                    worker_init_fn=worker_init_fn,
                )
                model.max_epochs = cfg.trainer.max_epochs
                model.steps_per_epoch = len(train_loader)
                if (
                    hasattr(model, "lr_schedule")
                    and hasattr(model, "learning_rate")
                    and model.steps_per_epoch is not None
                ):
                    model.lr_schedule = cosine_scheduler(
                        base_value=model.learning_rate,
                        final_value=1e-12,
                        epochs=model.max_epochs,
                        niter_per_ep=model.steps_per_epoch,
                        warmup_epochs=model.hparams.warmup_epochs,
                        start_warmup_value=1e-9,
                    )
                trainer.fit(
                    model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=cfg.get("ckpt_path"),
                )
                log.info(f"All callbacks: {[type(cb).__name__ for cb in trainer.callbacks]}")
                checkpoint_callback = None
                for callback in trainer.callbacks:
                    if hasattr(callback, "best_model_path"):
                        checkpoint_callback = callback
                        break
                if checkpoint_callback is None:
                    log.error(
                        "No ModelCheckpoint callback found! Check if callback is properly configured."
                    )
                    raise RuntimeError("ModelCheckpoint callback not found in trainer.callbacks")
                ckpt_path = checkpoint_callback.best_model_path
                log.info(f"Checkpoint callback found: {type(checkpoint_callback).__name__}")
                log.info(f"Best checkpoint path: {ckpt_path}")
                log.info(f"Checkpoint dirpath: {getattr(checkpoint_callback, 'dirpath', 'N/A')}")
                if ckpt_path and os.path.exists(ckpt_path):
                    log.info(f"Loading best checkpoint: {ckpt_path}")
                    model = TransferLearningModule.load_from_checkpoint(
                        ckpt_path,
                        encoder=encoder,
                        transfer_type=cfg.model.transfer_type,
                        pool_type=cfg.model.pool_type,
                        learning_rate=cfg.model.get("learning_rate", 1e-3),
                        weight_decay=cfg.model.get("weight_decay", 1e-4),
                        warmup_epochs=cfg.model.get("warmup_epochs", 20),
                        target_size=target_size,
                    )
                    log.info("Successfully loaded checkpoint")
                else:
                    log.warning(f"Checkpoint file not found: {ckpt_path}")
                    checkpoint_dir = os.path.dirname(ckpt_path) if ckpt_path else None
                    if checkpoint_dir:
                        log.info(f"Checking checkpoint directory: {checkpoint_dir}")
                        if os.path.exists(checkpoint_dir):
                            log.info(
                                f"Directory exists. Files in directory: {os.listdir(checkpoint_dir)}"
                            )
                        else:
                            log.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
                    if trainer.checkpoint_callback:
                        log.info("Checkpoint callback attributes:")
                        log.info(
                            f"  - dirpath: {getattr(trainer.checkpoint_callback, 'dirpath', 'N/A')}"
                        )
                        log.info(
                            f"  - filename: {getattr(trainer.checkpoint_callback, 'filename', 'N/A')}"
                        )
                        log.info(
                            f"  - best_model_path: {getattr(trainer.checkpoint_callback, 'best_model_path', 'N/A')}"
                        )
                        log.info(
                            f"  - best_model_score: {getattr(trainer.checkpoint_callback, 'best_model_score', 'N/A')}"
                        )
                    log.info("Using current trained model for evaluation")
                if torch.cuda.is_available() and rank == 0:
                    log.info(
                        f"GPU memory before evaluation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                    )

                # Evaluate using the validation-derived Dice/F1 threshold.
                evaluate_model_on_test_set(
                    model=model,
                    val_loader=val_loader,
                    train_dataset_name=train_dataset_name,
                    target_label=target_label,
                    target_size=target_size,
                    collate_fn=collate_fn,
                    worker_init_fn=worker_init_fn,
                    cfg=cfg,
                    run=run,
                    dice_scores=dice_scores,
                    iou_scores=iou_scores,
                    output_dir=output_dir,
                )
                if torch.cuda.is_available() and rank == 0:
                    log.info(
                        f"GPU memory after evaluation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                    )
                log.info(f"Cleaning up resources after run {run + 1}")
                safe_release(
                    trainer=trainer,
                    model=model,
                    extra_to_del=(
                        encoder,
                        callbacks,
                        logger,
                        train_loader,
                        val_loader,
                        train_dataset,
                        val_dataset,
                    ),
                    empty_cuda_cache=True,
                )
                log.info(f"Resource cleanup completed for run {run + 1}")
                if torch.cuda.is_available() and rank == 0:
                    log.info(
                        f"GPU memory after cleanup: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                    )
            del full_dataset
            save_target_metrics(
                all_metrics_data, train_dataset_name, target_label, dice_scores, iou_scores
            )
    save_all_metrics_and_config(all_metrics_data, output_dir, cfg)
    if rank == 0 and torch.cuda.is_available():
        log.info("=" * 80)
        log.info("Final GPU memory cleanup")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            log.info(f"GPU {i} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        log.info("=" * 80)
    return {"output_dir": output_dir}, {}


@rank_zero_only
def cleanup_checkpoints_and_logs(output_dir: str) -> None:
    try:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            log.info(f"Deleting checkpoint directory: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
            log.info("Checkpoint directory deleted successfully")
        else:
            log.info(f"Checkpoint directory not found: {checkpoint_dir}")
        log_dir = os.path.join(output_dir, "logs")
        if os.path.exists(log_dir):
            log.info(f"Deleting log directory: {log_dir}")
            shutil.rmtree(log_dir)
            log.info("Log directory deleted successfully")
        else:
            log.info(f"Log directory not found: {log_dir}")
        lightning_log_dir = os.path.join(output_dir, "lightning_logs")
        if os.path.exists(lightning_log_dir):
            log.info(f"Deleting lightning logs directory: {lightning_log_dir}")
            shutil.rmtree(lightning_log_dir)
            log.info("Lightning logs directory deleted successfully")
        else:
            log.info(f"Lightning logs directory not found: {lightning_log_dir}")
    except Exception as e:
        log.error(f"Error during cleanup: {e}")
        log.error("Cleanup failed, but this does not affect the main results")


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Entry point that launches training and cleans up artifacts."""
    extras(cfg)
    try:
        train_results, _ = train(cfg)
        output_dir = train_results.get("output_dir")
        if output_dir:
            log.info("Training completed successfully. Starting cleanup...")
            cleanup_checkpoints_and_logs(output_dir)
            log.info("Cleanup completed successfully.")
        else:
            log.warning("output_dir not found in train results, skipping cleanup")
    except Exception as e:
        log.error(f"Training failed with error: {e}")
        log.info("Attempting cleanup despite training failure...")
        try:
            log.warning("Cannot perform cleanup without output_dir information")
        except Exception as cleanup_error:
            log.error(f"Cleanup also failed: {cleanup_error}")
        raise


if __name__ == "__main__":
    main()
