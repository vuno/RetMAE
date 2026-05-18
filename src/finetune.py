import gc
import os
import shutil
from datetime import datetime
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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from lightning import LightningModule, Trainer
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.tasks.pretrain.utils import cosine_scheduler
from src.tasks.transfer import safe_release
from src.utils import (
    RankedLogger,
    data_path,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)

_EVAL_DATASETS = ["MESSIDOR2", "FIVES", "IDRiD", "PAPILA", "REFUGE", "RFMiD2"]


def _eval_csv(name: str, split: str) -> str:
    return str(data_path("DATASET_EXTERNAL_EVAL_224", name, f"labels_{split}.csv"))

log = RankedLogger(__name__, rank_zero_only=True)


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
        is_multilabel (bool, optional): Whether task is multi-label classification. Default is False.
        target_size (int, optional): Target image size for data augmentation. Default is 224.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 1,
        transfer_type: str = "finetuning",
        pool_type: str = "none",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 20,
        is_multilabel: bool = False,
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
        self.is_multilabel = is_multilabel
        self.target_size = target_size
        self.data_augmentation = data_augmentation
        if transfer_type == "linear_probing":
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        in_features = encoder.embed_dim

        # Linear classifier for both linear_probing and finetuning
        self.decoder = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

        # Set loss function according to task type (multi-label or multi-class)
        if is_multilabel:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            # Multi-class: always >= 2 classes
            self.criterion = F.cross_entropy

        self.transform_train = self._get_transform("train")
        self.lr_schedule = None
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.automatic_optimization = False

        # TorchMetrics를 사용한 AUROC 계산 (DDP에서 자동으로 gather)
        if is_multilabel:
            # Multi-label: 각 클래스별로 binary AUROC 평균
            self.val_auroc = AUROC(task="multilabel", num_labels=num_classes, average="macro")
        else:
            # Multi-class: OVR 방식
            self.val_auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro")

    def on_train_epoch_start(self):
        """
        Set random seed for distributed sampler at start of epoch (if available).
        """
        if hasattr(self.trainer.train_dataloader, "sampler") and hasattr(
            self.trainer.train_dataloader.sampler, "set_epoch"
        ):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def forward(self, img: torch.Tensor, split: str = "train") -> torch.Tensor:
        """
        Forward pass for the model.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Logits (unnormalized predictions).
        """
        assert split in ["train", "val", "test"], "split must be in ['train', 'val', 'test']"

        if self.encoder.training and self.hparams.transfer_type == "linear_probing":
            raise ValueError("encoder should be in eval mode when transfer_type is linear_probing")

        # Normalize image to 0-1 if necessary
        img = img.float() / 255.0 if img.max() > 1.0 else img.float()
        if self.data_augmentation and split == "train":
            img = self.transform_train(img)

        latents = self.encoder(img)["features"]

        # Apply selected pooling strategy for extracted features
        if self.hparams.pool_type == "cls_token":
            latents = latents[:, 0, :]
        elif self.hparams.pool_type == "avg":
            assert (
                latents.ndim == 3
            ), f"latents must be in [B, N, C] shape, but got {latents.shape}"
            latents = latents.mean(dim=1)
        elif self.hparams.pool_type == "flatten":
            latents = latents.flatten(start_dim=1)
        elif self.hparams.pool_type == "patch_token":
            latents = latents[:, self.encoder.backbone.num_prefix_tokens :, :]
            latents = latents.mean(dim=1)
        elif self.hparams.pool_type == "none":
            pass

        preds = self.decoder(latents)
        return preds

    def _get_transform(self, mode: str):
        """
        Return the default augmentation transform for the requested mode.

        Args:
            mode (str): Either 'train', 'val', or 'test'.

        Returns:
            nn.Sequential or None: Transformation pipeline or None if not used.
        """
        assert mode in [
            "train",
            "val",
            "test",
        ], "mode must be in ['train', 'val', 'test']"
        if mode == "train":
            transform = nn.Sequential(
                K.RandomAffine(
                    degrees=(-5, 5),
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
                    scale=(0.9, 1.0),
                    ratio=(1.0, 1.0),
                    resample="bilinear",
                    p=1.0,
                    same_on_batch=False,
                ),
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            )
        else:
            transform = None

        return transform

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Set up the optimizer and return optimizer configuration.

        Returns:
            Dict[str, Any]: Optimizer configuration dictionary.
        """
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
        """
        Compute and return the loss for a given batch.

        Args:
            batch (dict): Batch dictionary containing 'img' and 'label'.

        Returns:
            torch.Tensor: Calculated loss value.
        """
        img, label = batch["img"], batch["label"]
        preds = self(img, split)

        # Compute loss according to task type
        if self.is_multilabel:
            # Multi-label: use binary cross-entropy for each class
            loss = self.criterion(preds, label)
        else:
            # Multi-class: convert one-hot encoding to class indices
            target = torch.argmax(label, dim=1).long()  # [N, C] -> [N]
            loss = self.criterion(preds, target)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Perform a single optimization (training) step.

        Args:
            batch (dict): Batch dictionary containing 'img' and 'label'.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Calculated training loss.
        """
        loss = self._calculate_loss(batch, split="train")

        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            rank_zero_only=True,
            on_epoch=False,
            on_step=True,
        )

        opt = self.optimizers()
        opt.zero_grad()
        loss.backward()

        if self.hparams.transfer_type == "finetuning":
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0, norm_type=2)

        self._log_grad_norm(self.encoder.parameters(), split="train", prefix="encoder")
        self._log_grad_norm(self.decoder.parameters(), split="train", prefix="decoder")

        opt.step()

        # Learning rate scheduling
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
        """
        Perform a single validation step.

        Args:
            batch (dict): Batch dictionary containing 'img' and 'label'.
            batch_idx (int): Batch index.
        """
        img, label = batch["img"], batch["label"]
        preds = self(img, split="val")

        loss = self._calculate_loss(batch, split="val")

        # DDP 환경에서 모든 rank의 loss를 평균내어 동기화된 메트릭 사용
        self.log(
            "val/loss",
            loss,
            sync_dist=True,  # 모든 rank의 loss를 평균
            rank_zero_only=False,  # 모든 rank가 동기화된 값을 받아야 콜백이 제대로 동작
            on_epoch=True,
            on_step=False,
            reduce_fx="mean",  # 명시적으로 평균 사용
        )

        # TorchMetrics AUROC 업데이트 (DDP에서 자동으로 gather됨)
        # 확률값으로 변환
        if self.is_multilabel:
            # Multi-label: sigmoid
            probs = torch.sigmoid(preds)
            # Multi-label은 int로 변환 (0 또는 1)
            target = label.int()
        else:
            # Multi-class: softmax
            probs = F.softmax(preds, dim=1)
            # Multi-class는 one-hot을 class index로 변환
            target = torch.argmax(label, dim=1)

        # TorchMetrics 업데이트
        self.val_auroc.update(probs, target)

    def on_validation_epoch_end(self) -> None:
        """
        Validation epoch이 끝날 때 AUROC를 계산합니다.
        TorchMetrics가 DDP에서 자동으로 모든 랭크의 데이터를 gather하여 계산합니다.
        """
        # TorchMetrics compute() 호출 - DDP에서 자동으로 모든 랭크의 예측/레이블을 gather
        auroc = self.val_auroc.compute()

        # AUROC 로깅
        self.log(
            "val/auroc",
            auroc,
            sync_dist=False,  # TorchMetrics가 이미 동기화했으므로 False
            rank_zero_only=False,
            on_epoch=True,
        )
        log.info(f"Rank {self.global_rank} AUROC: {auroc}")

        # 메트릭 리셋 (다음 epoch를 위해)
        self.val_auroc.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """
        Perform a single test (evaluation) step.

        Args:
            batch (dict): Batch dictionary containing 'img' and 'label'.
            batch_idx (int): Batch index.
        """
        loss = self._calculate_loss(batch, split="test")

        self.log(
            "test/loss",
            loss,
            sync_dist=True,
            rank_zero_only=True,
            on_epoch=False,
            on_step=True,
        )

    @torch.inference_mode()
    def _log_grad_norm(
        self,
        parameters: Union[Iterable[torch.nn.Parameter], torch.nn.Parameter],
        norm_type: int = 2,
        split: str = "train",
        prefix: str = "",
    ) -> float:
        """
        Compute and log the gradient norm of parameters.

        Args:
            parameters (Iterable[torch.nn.Parameter] or torch.nn.Parameter): Model parameters.
            norm_type (int): Type of norm to use (default: 2).
            split (str): Train/val/test split for logging context.
            prefix (str): Optional prefix for logging key.

        Returns:
            float: Calculated norm value.
        """
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


# Per-dataset CSVs of labels for the downstream evaluation sets. Resolved from
# $DATA_ROOT/DATASET_EXTERNAL_EVAL_224/<NAME>/labels_{train,test}.csv at runtime.
TRAIN_DATASET = {name: _eval_csv(name, "train") for name in _EVAL_DATASETS}
TEST_DATASET = {name: _eval_csv(name, "test") for name in _EVAL_DATASETS}

# Multi-label classification datasets.
MULTILABEL_DATASETS = {"FIVES", "RFMiD2"}


def pad_and_resize_image(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    이미지를 zero padding하여 정사각형으로 만든 후 리사이즈합니다.

    Args:
        img: 입력 이미지 (H, W, C)
        target_size: 리사이즈할 크기 (기본값: 224)

    Returns:
        전처리된 이미지 (target_size, target_size, C)
    """
    h, w, c = img.shape

    # 가로/세로 중 작은 쪽을 찾아서 정사각형으로 만들기
    max_dim = max(h, w)

    # zero padding으로 정사각형 생성
    if h < w:
        # 세로가 더 작은 경우, 위아래에 padding 추가
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        padded_img = np.pad(
            img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="constant", constant_values=0
        )
    elif w < h:
        # 가로가 더 작은 경우, 좌우에 padding 추가
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        padded_img = np.pad(
            img, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=0
        )
    else:
        # 이미 정사각형인 경우
        padded_img = img.copy()

    # 224x224로 리사이즈
    resized_img = cv2.resize(
        padded_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )

    return resized_img


class CustomDataset(Dataset):
    """데이터셋 클래스 (학습/테스트 공용)

    이미지를 zero padding하여 정사각형으로 만들고 지정된 크기로 리사이즈합니다.

    Args:
        csv_path: CSV 파일 경로
        dataset_name: 데이터셋 이름
        indices: 사용할 인덱스 리스트 (None이면 전체 사용)
        target_size: 리사이즈할 이미지 크기 (기본값: 224)
    """

    def __init__(
        self,
        csv_path: str,
        dataset_name: str,
        indices: Optional[List[int]] = None,
        target_size: int = 224,
    ):
        self.dataset_name = dataset_name
        self.target_size = target_size
        self.df = pd.read_csv(csv_path)

        # indices가 제공되면 해당 인덱스들만 사용
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        self.filepaths = self.df["filepath"].tolist()

        # 라벨 컬럼들 추출 (filepath 제외)
        self.label_columns = [col for col in self.df.columns if col != "filepath"]
        self.labels = self.df[self.label_columns].values

        # Prefer the pre-resized 224x224 dataset; fall back to the full-resolution copy.
        ssd_root = str(data_path("DATASET_EXTERNAL_EVAL_224", dataset_name))
        original_root = str(data_path("DATASET", dataset_name))

        if os.path.exists(ssd_root):
            self.root = ssd_root
            log.info(f"Using SSD path for {dataset_name}: {self.root}")
        else:
            self.root = original_root
            log.info(f"Using original path for {dataset_name}: {self.root}")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.filepaths[idx])

        # 이미지 로딩
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # zero padding하여 정사각형으로 만들고 target_size로 리사이즈
        img = pad_and_resize_image(img, target_size=self.target_size)

        return {
            "img": img,
            "filepath": self.filepaths[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def calculate_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, dataset_name: str
) -> Tuple[float, float]:
    """AUROC와 AUPRC 계산"""
    if dataset_name in MULTILABEL_DATASETS:
        # Multi-label: 각 클래스별로 계산 후 평균
        auroc_scores = []
        auprc_scores = []
        skipped_classes = []

        for i in range(y_true.shape[1]):
            unique_values = np.unique(y_true[:, i])
            if len(unique_values) <= 1:
                log.warning(
                    f"Class {i} has only one unique value ({unique_values[0]}), skipping metrics calculation"
                )
                skipped_classes.append(i)
                continue

            try:
                auroc = roc_auc_score(y_true[:, i], y_probs[:, i])
                auprc = average_precision_score(y_true[:, i], y_probs[:, i])
                auroc_scores.append(auroc)
                auprc_scores.append(auprc)
            except ValueError as e:
                log.warning(f"Failed to calculate metrics for class {i}: {e}")
                skipped_classes.append(i)
                continue

        if not auroc_scores:
            raise ValueError(f"No valid classes found for multi-label dataset {dataset_name}")

        log.info(
            f"Skipped {len(skipped_classes)} classes with insufficient data: {skipped_classes}"
        )
        log.info(f"Calculated metrics for {len(auroc_scores)} classes")

        return np.mean(auroc_scores), np.mean(auprc_scores)
    else:
        # Multi-class: macro average (always >= 2 classes)
        # y_true는 one-hot encoding 형태: (n_samples, n_classes)
        # y_probs는 probability 형태: (n_samples, n_classes)

        # Multi-class AUROC/AUPRC 계산
        # y_true는 이미 one-hot encoding이므로 그대로 사용
        auroc = roc_auc_score(y_true, y_probs, average="macro")
        auprc = average_precision_score(y_true, y_probs, average="macro")

        return auroc, auprc


def worker_init_fn(worker_id):
    """DataLoader worker의 재현성을 위한 초기화 함수"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def collate_fn(batch):
    """배치 데이터를 정리하는 함수"""
    # 이미지를 torch.Tensor로 변환한 후 stack
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


@rank_zero_only
@torch.inference_mode()
def get_probabilities_and_labels(
    model: LightningModule, dataloader: DataLoader, dataset_name: str
) -> Tuple[List, List, np.ndarray]:
    """모델로부터 확률값과 레이블 추출"""
    model.eval()
    all_probs = []
    all_filepaths = []
    all_labels = []

    for batch in tqdm(dataloader, desc=f"Getting probabilities and labels for {dataset_name}"):
        if not isinstance(batch, dict) or "img" not in batch:
            raise ValueError(
                f"Invalid batch format for dataset {dataset_name}. Expected dict with 'img' key, got {type(batch)}"
            )

        filepaths = batch["filepath"]
        images = batch["img"]
        labels = batch["label"]

        if not filepaths or images.size(0) == 0:
            raise ValueError(f"Empty batch for dataset {dataset_name}")

        # 모델이 GPU에 있는지 확인하고 GPU로 이동
        device = next(model.parameters()).device
        images = images.to(device)

        # 모델 추론 - 새로운 TransferLearningModule의 forward 메서드 호출
        if hasattr(model, "forward"):
            outputs = model.forward(images, split="test")
        else:
            outputs = model(images, split="test")

        # Multi-label/multi-class에 따른 확률값 변환
        if dataset_name in MULTILABEL_DATASETS:
            # Multi-label: sigmoid for each class independently
            probs = torch.sigmoid(outputs)
        else:
            # Multi-class: softmax across classes (always >= 2 classes)
            probs = F.softmax(outputs, dim=1)

        # CPU로 이동 후 즉시 GPU 텐서 정리
        all_probs.extend(probs.cpu().tolist())
        all_filepaths.extend(filepaths)
        all_labels.extend(labels.cpu().numpy())

        # 배치 처리 후 GPU 텐서 명시적으로 삭제
        del images, outputs, probs, labels

    if not all_probs:
        raise RuntimeError(
            f"No probabilities extracted for dataset {dataset_name}. Check dataloader and model."
        )

    return all_probs, all_filepaths, np.array(all_labels)


@rank_zero_only
def evaluate_model_on_test_set(
    model: LightningModule,
    train_dataset_name: str,
    target_size: int,
    collate_fn,
    worker_init_fn,
    cfg: DictConfig,
    run: int,
    auroc_scores: list,
    auprc_scores: list,
    output_dir: str,
) -> None:
    """모델을 테스트 데이터셋에서 평가하고 결과를 저장"""
    # 해당 데이터셋의 테스트 데이터로 평가
    test_csv_path = TEST_DATASET[train_dataset_name]

    # 테스트 데이터셋 생성
    test_dataset = CustomDataset(test_csv_path, train_dataset_name, target_size=target_size)

    # 테스트 데이터셋 크기 로깅
    log.info(
        f"Evaluating {train_dataset_name} for run {run + 1} - Test dataset: {len(test_dataset)} samples"
    )

    # 데이터로더 생성 (DistributedSampler 없이 전체 데이터셋 사용)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=False,
        prefetch_factor=1,
        worker_init_fn=worker_init_fn,  # 재현성을 위한 worker 초기화
        # sampler=None (기본값) - 전체 데이터셋 사용
    )

    # 확률값 추출, 메트릭 계산, 파일 저장 수행
    model.eval()
    probs_list, filepaths, labels_list = get_probabilities_and_labels(
        model, test_loader, train_dataset_name
    )

    # probs_list가 비어있는 경우 에러 발생
    if not probs_list or len(probs_list) == 0:
        raise RuntimeError(
            f"probs_list is empty for dataset {train_dataset_name} run {run + 1}. "
            f"This might be due to: "
            f"1) Empty test dataset (len={len(test_dataset)}), "
            f"2) No valid batches in dataloader, "
            f"3) All image loading failed, or "
            f"4) Model inference failed for all samples."
        )

    if not filepaths or len(filepaths) == 0:
        raise RuntimeError(f"filepaths is empty for dataset {train_dataset_name} run {run + 1}")

    if labels_list is None or len(labels_list) == 0:
        raise RuntimeError(f"labels_list is empty for dataset {train_dataset_name} run {run + 1}")

    # metrics 계산을 위한 numpy 배열 변환
    y_probs = np.array(probs_list)
    if isinstance(labels_list, np.ndarray):
        y_true = labels_list
    else:
        y_true = np.array(labels_list)

    # AUROC와 AUPRC 계산
    auroc, auprc = calculate_metrics(y_true, y_probs, train_dataset_name)
    auroc_scores.append(auroc)
    auprc_scores.append(auprc)

    log.info(f"Run {run + 1} AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

    # 확률값을 평면화하여 저장
    flattened_probs = []
    for prob in probs_list:
        if isinstance(prob, list):
            flattened_probs.append(prob)
        else:
            flattened_probs.append(prob.tolist() if hasattr(prob, "tolist") else prob)

    # CSV 파일로 저장 (seed별로)
    # 데이터프레임 초기화
    df_data = {"filepath": filepaths}

    # 확률값 리스트를 개별 컬럼으로 분리
    first_prob = flattened_probs[0]
    if isinstance(first_prob, list):
        # 각 클래스별로 컬럼 생성 (class0, class1, ...)
        num_classes = len(first_prob)
        for class_idx in range(num_classes):
            class_probs = [
                (prob[class_idx] if isinstance(prob, list) and len(prob) > class_idx else 0.0)
                for prob in flattened_probs
            ]
            df_data[f"class{class_idx}"] = class_probs
    else:
        # 단일 확률값인 경우
        df_data["prob"] = flattened_probs

    df = pd.DataFrame(df_data)

    # 파일 저장 (seed별로)
    output_path = os.path.join(
        output_dir,
        f"{cfg.model_name}_{train_dataset_name}_probs_seed{run}.csv",
    )
    df.to_csv(output_path, index=False)
    log.info(f"Saved probabilities to {output_path}")

    # DataLoader와 데이터셋 정리
    del test_loader, test_dataset

    # 명시적으로 가비지 컬렉션
    gc.collect()

    # 모델을 CPU로 이동하여 GPU 메모리 해제
    try:
        model.to("cpu")
        log.info("Moved model to CPU after evaluation")
    except Exception as e:
        log.warning(f"Failed to move model to CPU: {e}")

    # GPU 캐시 정리 (여러 번 호출하여 확실히 정리)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # 메모리 상태 로깅
        dev = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(dev) / 1024**2
        reserved = torch.cuda.memory_reserved(dev) / 1024**2
        log.info(f"GPU {dev} memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


@rank_zero_only
def save_dataset_metrics(
    all_metrics_data: List[Dict[str, Any]],
    train_dataset_name: str,
    auroc_scores: List[float],
    auprc_scores: List[float],
) -> None:
    """
    데이터셋별 메트릭을 계산하고 리스트에 추가합니다.

    Args:
        all_metrics_data: 전체 메트릭 데이터를 저장할 리스트
        train_dataset_name: 학습 데이터셋 이름
        auroc_scores: AUROC 점수 리스트
        auprc_scores: AUPRC 점수 리스트
    """
    auroc_mean = np.mean(auroc_scores)
    auroc_std = np.std(auroc_scores)
    auprc_mean = np.mean(auprc_scores)
    auprc_std = np.std(auprc_scores)

    # 전체 메트릭 리스트에 추가
    all_metrics_data.append(
        {
            "dataset": train_dataset_name,
            "auroc_mean": auroc_mean,
            "auroc_std": auroc_std,
            "auprc_mean": auprc_mean,
            "auprc_std": auprc_std,
        }
    )
    log.info(
        f"Dataset {train_dataset_name} - AUROC: {auroc_mean:.4f}±{auroc_std:.4f}, "
        f"AUPRC: {auprc_mean:.4f}±{auprc_std:.4f}"
    )


@rank_zero_only
def save_all_metrics_and_config(
    all_metrics_data: List[Dict[str, Any]], output_dir: str, cfg: DictConfig
) -> None:
    """
    모든 데이터셋의 메트릭과 설정을 저장합니다.

    Args:
        all_metrics_data: 전체 메트릭 데이터 리스트
        output_dir: 출력 디렉토리 경로
        cfg: Hydra 설정
    """
    if not all_metrics_data:
        raise RuntimeError("No metrics collected from any dataset")

    # 메트릭을 CSV로 저장
    metrics_df = pd.DataFrame(all_metrics_data)
    metrics_path = os.path.join(output_dir, f"{cfg.model_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"Saved all metrics to {metrics_path}")

    # cfg를 yaml 파일로 저장
    cfg_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_path)
    log.info(f"Saved config to {cfg_path}")


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # 재현성을 위한 전역 seed 설정
    base_seed = cfg.get("seed", 42)
    log.info(f"Using base seed: {base_seed}")

    # 재현성을 위한 초기 시드 설정 (fold split을 위해)
    L.seed_everything(base_seed, workers=True)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    # DDP 환경에서 rank 확인
    rank = 0
    world_size = 1
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    # Hydra 설정에서 타임스탬프가 포함된 output_dir을 사용
    output_dir = cfg.paths.output_dir

    if rank == 0:
        log.info(f"Using output directory with timestamp: {output_dir}")

    # 전체 메트릭 수집을 위한 리스트
    all_metrics_data = []

    # 각 데이터셋별로 train-test 페어 맞춰서 학습 및 평가
    for train_dataset_name, train_csv_path in TRAIN_DATASET.items():
        log.info(f"Processing dataset: {train_dataset_name}")

        # 각 데이터셋별로 metrics 수집을 위한 리스트
        auroc_scores = []
        auprc_scores = []

        # ==================== 데이터셋별로 한 번만 5-fold split 수행 ====================
        # 이미지 크기 설정 (기본값: 224)
        target_size = cfg.model.get("target_size", 224)

        # 전체 데이터셋 로드
        full_dataset = CustomDataset(train_csv_path, train_dataset_name, target_size=target_size)
        num_classes = full_dataset.labels.shape[1]
        dataset_size = len(full_dataset)

        log.info(f"  Full dataset loaded: {len(full_dataset)} samples, {num_classes} classes")

        # 클래스 분포 확인
        if train_dataset_name in MULTILABEL_DATASETS:
            # Multi-label: 각 클래스별 양성 샘플 수 확인
            class_positive_counts = np.sum(full_dataset.labels, axis=0)
            log.info("  Multi-label class distribution (positive samples per class):")
            for i, count in enumerate(class_positive_counts):
                percentage = (count / len(full_dataset)) * 100
                log.info(f"    Class {i}: {int(count)} / {len(full_dataset)} ({percentage:.1f}%)")

            # 최소 양성 샘플 수 확인
            min_class_idx = np.argmin(class_positive_counts)
            min_count = class_positive_counts[min_class_idx]

            n_splits = 5
            if min_count < n_splits:
                log.warning(
                    f"  ⚠️ WARNING: Class {min_class_idx} has only {int(min_count)} positive samples, "
                    f"which is less than n_splits={n_splits}. Some folds may not contain this class!"
                )
        else:
            # Multi-class: 각 클래스별 샘플 수 확인
            y_classes = np.argmax(full_dataset.labels, axis=1)
            unique_classes, class_counts = np.unique(y_classes, return_counts=True)

            log.info("  Multi-class distribution:")
            for cls, count in zip(unique_classes, class_counts):
                percentage = (count / len(full_dataset)) * 100
                log.info(f"    Class {cls}: {count} / {len(full_dataset)} ({percentage:.1f}%)")

            # 최소 샘플 수 확인
            min_count = np.min(class_counts)

            n_splits = 5
            if min_count < n_splits:
                min_class = unique_classes[np.argmin(class_counts)]
                log.warning(
                    f"  ⚠️ WARNING: Class {min_class} has only {min_count} samples, "
                    f"which is less than n_splits={n_splits}. Some folds may not contain this class!"
                )

        # MultilabelStratifiedKFold를 사용한 5-fold split (multi-label & multi-class 모두 지원)
        n_splits = 5
        skf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
        fold_indices = list(skf.split(np.arange(len(full_dataset.labels)), full_dataset.labels))
        log.info(
            f"Created 5 stratified folds using MultilabelStratifiedKFold with base_seed={base_seed}"
        )

        # 각 fold 정보 로깅 및 클래스 분포 확인
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            log.info(
                f"  Fold {fold_idx + 1}: Train={len(train_idx)} samples, Val={len(val_idx)} samples"
            )

            # Validation set의 클래스 분포 확인
            if train_dataset_name in MULTILABEL_DATASETS:
                # Multi-label: validation set의 각 클래스별 양성 샘플 수
                val_labels = full_dataset.labels[val_idx]
                val_positive_counts = np.sum(val_labels, axis=0)

                log.info("    Validation set class distribution (positive samples):")
                for class_idx, count in enumerate(val_positive_counts):
                    percentage = (count / len(val_idx)) * 100
                    log.info(
                        f"      Class {class_idx}: {int(count)} / {len(val_idx)} ({percentage:.1f}%)"
                    )

                classes_with_no_positive = np.where(val_positive_counts == 0)[0]
                if len(classes_with_no_positive) > 0:
                    log.warning(
                        f"    ⚠️ WARNING: Classes with NO positive samples: {classes_with_no_positive.tolist()}"
                    )
            else:
                # Multi-class: validation set의 클래스 분포
                val_labels = full_dataset.labels[val_idx]
                val_classes = np.argmax(val_labels, axis=1)
                unique_val_classes, val_class_counts = np.unique(val_classes, return_counts=True)

                log.info("    Validation set class distribution:")
                # 모든 클래스에 대해 출력 (없는 클래스는 0으로 표시)
                class_count_dict = dict(zip(unique_val_classes, val_class_counts))
                for class_idx in range(num_classes):
                    count = class_count_dict.get(class_idx, 0)
                    percentage = (count / len(val_idx)) * 100 if count > 0 else 0.0
                    status = "✓" if count > 0 else "⚠️"
                    log.info(
                        f"      {status} Class {class_idx}: {count} / {len(val_idx)} ({percentage:.1f}%)"
                    )

                missing_classes = set(range(num_classes)) - set(unique_val_classes)
                if missing_classes:
                    log.warning(f"    ⚠️ WARNING: MISSING classes: {sorted(missing_classes)}")
        # ==========================================================================

        # 5번의 학습 실행 (각 fold를 사용)
        for run in range(5):
            # 재현성을 위한 seed 설정 (각 run마다 다른 seed 사용)
            run_seed = base_seed + run
            log.info(
                f"Starting run {run + 1}/5 for {train_dataset_name} (fold {run + 1}, seed {run_seed})"
            )

            # 시드 설정 - 모든 랜덤 생성기에 동일한 seed 적용
            L.seed_everything(run_seed, workers=True)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(run_seed)
                torch.cuda.manual_seed_all(run_seed)

            # 추가적인 재현성 보장을 위한 설정
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # 인코더 생성
            encoder = hydra.utils.instantiate(cfg.model.encoder)

            # 해당 fold의 train/val indices 사용
            train_indices, val_indices = fold_indices[run]

            # Train과 validation 데이터셋 생성
            train_dataset = CustomDataset(
                train_csv_path, train_dataset_name, train_indices.tolist(), target_size=target_size
            )
            val_dataset = CustomDataset(
                train_csv_path, train_dataset_name, val_indices.tolist(), target_size=target_size
            )

            # 데이터셋 크기 로깅
            log.info(f"Dataset sizes for {train_dataset_name} run {run + 1} (fold {run + 1}):")
            log.info(f"  Total dataset: {dataset_size} samples")
            log.info(
                f"  Train dataset: {len(train_dataset)} samples ({len(train_dataset)/dataset_size*100:.1f}%)"
            )
            log.info(
                f"  Validation dataset: {len(val_dataset)} samples ({len(val_dataset)/dataset_size*100:.1f}%)"
            )

            # TransferLearningModule 필수 파라미터 확인 및 추출
            if "transfer_type" not in cfg.model:
                raise ValueError("transfer_type must be specified in cfg.model")
            if "pool_type" not in cfg.model:
                raise ValueError("pool_type must be specified in cfg.model")

            model = TransferLearningModule(
                encoder=encoder,
                num_classes=num_classes,
                transfer_type=cfg.model.transfer_type,
                pool_type=cfg.model.pool_type,
                learning_rate=cfg.model.get("learning_rate", 1e-3),
                weight_decay=cfg.model.get("weight_decay", 1e-4),
                warmup_epochs=cfg.model.get("warmup_epochs", 20),
                is_multilabel=train_dataset_name in MULTILABEL_DATASETS,
                target_size=target_size,
            )

            callbacks = instantiate_callbacks(cfg.get("callbacks"))
            logger = instantiate_loggers(cfg.get("logger"))
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer, callbacks=callbacks, logger=logger
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn,
                persistent_workers=True,
                prefetch_factor=1,
                worker_init_fn=worker_init_fn,  # 재현성을 위한 worker 초기화
            )

            # Validation dataloader 생성
            val_loader = DataLoader(
                val_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn,
                persistent_workers=True,
                prefetch_factor=1,
                worker_init_fn=worker_init_fn,  # 재현성을 위한 worker 초기화
            )

            model.max_epochs = cfg.trainer.max_epochs
            model.steps_per_epoch = len(train_loader)

            # 학습률 스케줄러 설정
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

            # 학습
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=cfg.get("ckpt_path"),
            )

            # 최고 체크포인트 로드
            log.info(f"All callbacks: {[type(cb).__name__ for cb in trainer.callbacks]}")

            # ModelCheckpoint 콜백 찾기
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

            # 체크포인트 파일 존재 확인
            if ckpt_path and os.path.exists(ckpt_path):
                log.info(f"Loading best checkpoint: {ckpt_path}")
                model = TransferLearningModule.load_from_checkpoint(
                    ckpt_path,
                    encoder=encoder,
                    num_classes=num_classes,
                    transfer_type=cfg.model.transfer_type,
                    pool_type=cfg.model.pool_type,
                    learning_rate=cfg.model.get("learning_rate", 1e-3),
                    weight_decay=cfg.model.get("weight_decay", 1e-4),
                    warmup_epochs=cfg.model.get("warmup_epochs", 20),
                    is_multilabel=train_dataset_name in MULTILABEL_DATASETS,
                    target_size=target_size,
                )
                log.info("Successfully loaded checkpoint")
            else:
                log.warning(f"Checkpoint file not found: {ckpt_path}")

                # 체크포인트 디렉토리 확인
                checkpoint_dir = os.path.dirname(ckpt_path) if ckpt_path else None
                if checkpoint_dir:
                    log.info(f"Checking checkpoint directory: {checkpoint_dir}")
                    if os.path.exists(checkpoint_dir):
                        log.info(
                            f"Directory exists. Files in directory: {os.listdir(checkpoint_dir)}"
                        )
                    else:
                        log.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")

                # trainer.checkpoint_callback 속성들 확인(읽기만)
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
                # 현재 모델이 이미 최적의 가중치를 가지고 있음

            # 평가 전 메모리 상태 로깅
            if torch.cuda.is_available() and rank == 0:
                log.info(
                    f"GPU memory before evaluation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                )

            # 모델을 테스트 데이터셋에서 평가 (rank 0에서만 실행됨)
            evaluate_model_on_test_set(
                model=model,
                train_dataset_name=train_dataset_name,
                target_size=target_size,
                collate_fn=collate_fn,
                worker_init_fn=worker_init_fn,
                cfg=cfg,
                run=run,
                auroc_scores=auroc_scores,
                auprc_scores=auprc_scores,
                output_dir=output_dir,
            )

            # 평가 후 메모리 상태 로깅
            if torch.cuda.is_available() and rank == 0:
                log.info(
                    f"GPU memory after evaluation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                )

            # ===============================
            # 여기부터 "안전한 최소 정리" 적용
            # ===============================
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

            # 정리 후 메모리 상태 로깅
            if torch.cuda.is_available() and rank == 0:
                log.info(
                    f"GPU memory after cleanup: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                )

        # 해당 데이터셋의 모든 run 완료 후 full_dataset 정리
        del full_dataset

        # 해당 데이터셋의 모든 run 완료 후 메트릭 수집 (rank 0에서만)
        save_dataset_metrics(all_metrics_data, train_dataset_name, auroc_scores, auprc_scores)

    # 모든 데이터셋 완료 후 통합 메트릭 CSV와 config 저장 (rank 0에서만)
    save_all_metrics_and_config(all_metrics_data, output_dir, cfg)

    # 최종 GPU 메모리 정리 및 상태 로깅
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
    """
    학습 완료 후 checkpoints와 logs 디렉토리를 삭제합니다.

    Args:
        output_dir: 타임스탬프가 포함된 출력 디렉토리 경로
    """
    try:
        # 체크포인트 디렉토리 삭제
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            log.info(f"Deleting checkpoint directory: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
            log.info("Checkpoint directory deleted successfully")
        else:
            log.info(f"Checkpoint directory not found: {checkpoint_dir}")

        # 로그 디렉토리 삭제
        log_dir = os.path.join(output_dir, "logs")
        if os.path.exists(log_dir):
            log.info(f"Deleting log directory: {log_dir}")
            shutil.rmtree(log_dir)
            log.info("Log directory deleted successfully")
        else:
            log.info(f"Log directory not found: {log_dir}")

        # Lightning logs 디렉토리도 확인 (일반적으로 lightning_logs)
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
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    extras(cfg)

    try:
        # train the model
        train_results, _ = train(cfg)
        output_dir = train_results.get("output_dir")

        # 학습 완료 후 checkpoints와 logs 삭제
        if output_dir:
            log.info("Training completed successfully. Starting cleanup...")
            cleanup_checkpoints_and_logs(output_dir)
            log.info("Cleanup completed successfully.")
        else:
            log.warning("output_dir not found in train results, skipping cleanup")

    except Exception as e:
        log.error(f"Training failed with error: {e}")
        # 에러가 발생해도 cleanup은 시도
        log.info("Attempting cleanup despite training failure...")
        try:
            log.warning("Cannot perform cleanup without output_dir information")
        except Exception as cleanup_error:
            log.error(f"Cleanup also failed: {cleanup_error}")
        raise


if __name__ == "__main__":
    main()
