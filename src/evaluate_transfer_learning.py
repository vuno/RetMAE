import glob
import os
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import pandas as pd
import rootutils
import torch
import torchmetrics as tm
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchmetrics.aggregation import CatMetric

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


class ProbabilityCollectorCallback(Callback):
    """각 데이터 샘플에 대한 확률과 레이블을 수집하는 콜백 (CatMetric 사용)"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.filepath_to_data = {}  # filepath -> {prob, label} 딕셔너리
        # 리스트로 직접 수집 (CatMetric 대신)
        self.all_probs = []
        self.all_labels = []
        self.all_filepaths = []

    def on_test_start(self, trainer, pl_module):
        """테스트 시작 시 호출"""
        self.filepath_to_data = {}
        self.all_probs = []
        self.all_labels = []
        self.all_filepaths = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """테스트 배치 종료 시 확률과 레이블을 수집 (CatMetric 사용)"""
        try:
            # batch가 딕셔너리인지 확인
            if isinstance(batch, dict) and "img" in batch and "label" in batch:
                img, label = batch["img"], batch["label"]

                # 모델의 예측값을 확률로 변환
                with torch.no_grad():
                    preds = pl_module(img, mode="test")
                    if isinstance(preds, tuple):
                        preds = sum(preds) / len(preds)
                    probs = torch.sigmoid(preds)

                # filepath 기반으로 데이터 저장 (filepath 무조건 존재 가정)
                assert (
                    "filepath" in batch
                ), f"'filepath' key not found in batch. Available keys: {list(batch.keys())}"

                filepaths = batch["filepath"]
                if not isinstance(filepaths, list):
                    filepaths = [filepaths]

                # 리스트로 직접 데이터 수집 (순서 유지)
                self.all_probs.append(probs.cpu())
                self.all_labels.append(label.cpu())
                self.all_filepaths.extend(filepaths)  # filepaths는 이미 리스트

        except Exception as e:  # noqa: BLE001
            print(f"Error in ProbabilityCollectorCallback: {e}")
            print(f"Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}")
            print(f"Batch type: {type(batch)}")
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if hasattr(value, "shape"):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: type={type(value)}, value={value}")
            raise

    def on_test_end(self, trainer, pl_module):
        """테스트 종료 시 리스트 기반 결과 집계"""
        try:
            # 모든 worker의 데이터를 gather
            if trainer.world_size > 1:
                # 모든 worker의 리스트들을 수집
                all_worker_probs = [None] * trainer.world_size
                all_worker_labels = [None] * trainer.world_size
                all_worker_filepaths = [None] * trainer.world_size

                torch.distributed.all_gather_object(all_worker_probs, self.all_probs)
                torch.distributed.all_gather_object(all_worker_labels, self.all_labels)
                torch.distributed.all_gather_object(all_worker_filepaths, self.all_filepaths)

                # 모든 worker의 데이터를 하나의 리스트로 합치기
                combined_probs = []
                combined_labels = []
                combined_filepaths = []

                for worker_probs, worker_labels, worker_filepaths in zip(
                    all_worker_probs, all_worker_labels, all_worker_filepaths
                ):
                    if worker_probs is not None:
                        combined_probs.extend(worker_probs)
                    if worker_labels is not None:
                        combined_labels.extend(worker_labels)
                    if worker_filepaths is not None:
                        combined_filepaths.extend(worker_filepaths)

                print(f"Multi-GPU sync: Combined data from {trainer.world_size} workers")
                print(f"Total samples: {len(combined_filepaths)}")
            else:
                # 단일 GPU인 경우
                combined_probs = self.all_probs
                combined_labels = self.all_labels
                combined_filepaths = self.all_filepaths

            # 빈 결과 체크
            if len(combined_filepaths) == 0:
                raise RuntimeError(
                    "No test data collected. Check if test dataloader is working properly."
                )

            # 길이 일치 확인
            if len(combined_probs) != len(combined_labels) or len(combined_probs) != len(
                combined_filepaths
            ):
                raise ValueError(
                    f"Length mismatch: probs={len(combined_probs)}, labels={len(combined_labels)}, filepaths={len(combined_filepaths)}"
                )

            # 리스트를 텐서로 변환하여 딕셔너리 생성 (순서 보장)
            for i, (prob_tensor, label_tensor, filepath) in enumerate(
                zip(combined_probs, combined_labels, combined_filepaths)
            ):
                prob = prob_tensor.item()
                label = label_tensor.item()
                self.filepath_to_data[filepath] = {"prob": prob, "label": label}

            print(f"Collected {len(self.filepath_to_data)} samples using list-based gathering")

        except Exception as e:
            print(f"Error in list-based data collection: {e}")
            raise RuntimeError(f"Failed to collect test data using list gathering: {e}") from e


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """전이학습 모델을 평가하고 확률을 계산합니다."""

    print("\n" + "=" * 80)
    print("STARTING TRANSFER LEARNING EVALUATION")
    print("=" * 80)
    print(f"Configuration: {cfg.get('experiment', 'default')}")
    print(f"Checkpoints dir: {cfg.get('checkpoints_dir', 'None (will train)')}")
    print(f"Seed: {cfg.get('seed', 'default')}")
    print("=" * 80)

    # instantiate datamodule
    print("Setting up datamodule...")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    datamodule.setup("test")
    print("Datamodule setup completed.")

    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=cfg.seed)

    # 결과 저장용 딕셔너리
    all_probabilities = defaultdict(lambda: defaultdict(dict))  # dataset -> fold -> data

    # datamodule.data_train의 모든 데이터셋을 합쳐 하나로 만듭니다.
    train_dataset = ConcatDataset(list(datamodule.data_train.values()))

    # 체크포인트 폴더가 제공된 경우 학습을 스킵
    if cfg.get("checkpoints_dir"):
        log.info(f"Using checkpoints from: {cfg.checkpoints_dir}")
        checkpoint_files = glob.glob(os.path.join(cfg.checkpoints_dir, "*.ckpt"))
        if len(checkpoint_files) != K:
            raise ValueError(f"Expected {K} checkpoint files, found {len(checkpoint_files)}")
        checkpoint_files.sort()  # 파일명으로 정렬
    else:
        checkpoint_files = [None] * K

    # 테스트 데이터 로더를 한 번만 생성하여 모든 fold에서 일관성 보장
    print("Creating test dataloaders with fixed seed for consistency...")
    L.seed_everything(cfg.seed, workers=True)  # 테스트 데이터 일관성 보장
    test_dataloaders = datamodule.test_dataloader()
    print(f"Test dataloaders created for datasets: {list(test_dataloaders.keys())}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/10")
        print(f"{'='*60}")
        # 학습/검증을 위한 seed 설정 (workers=False로 설정하여 테스트 데이터에 영향 없음)
        L.seed_everything(cfg.seed + fold, workers=False)

        # Subset을 이용해 train/val 데이터 분리 (체크포인트 사용 여부와 관계없이)
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        # DataLoader 생성
        train_loader = DataLoader(
            train_subset,
            batch_size=cfg.data.train_batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=datamodule.collate_fn,
            persistent_workers=cfg.data.persistent_workers,
            prefetch_factor=cfg.data.prefetch_factor,
            worker_init_fn=datamodule.worker_init_fn,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg.data.val_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=datamodule.collate_fn,
            persistent_workers=cfg.data.persistent_workers,
            prefetch_factor=cfg.data.prefetch_factor,
            worker_init_fn=datamodule.worker_init_fn,
        )

        # 모델·콜백·로거·트레이너 인스턴스화
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        callbacks = instantiate_callbacks(cfg.get("callbacks"))
        logger = instantiate_loggers(cfg.get("logger"))
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
        model.max_epochs = cfg.trainer.max_epochs
        model.steps_per_epoch = len(train_loader)

        # 체크포인트가 있는 경우 해당 체크포인트 사용
        if cfg.get("checkpoints_dir"):
            ckpt_path = checkpoint_files[fold]
            print(f"Using checkpoint for fold {fold}: {ckpt_path}")
            log.info(f"Using checkpoint for fold {fold}: {ckpt_path}")
        else:
            # 학습 수행
            print(f"Starting training for fold {fold}...")
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=cfg.get("ckpt_path"),
            )
            ckpt_path = trainer.checkpoint_callback.best_model_path
            print(f"Training completed for fold {fold}. Best checkpoint: {ckpt_path}")

        # 각 테스트 데이터셋에 대해 평가 (미리 생성된 일관된 테스트 데이터 로더 사용)
        for test_dataset_name, test_data_loader in test_dataloaders.items():
            print(f"\n--- Testing {test_dataset_name} on fold {fold} ---")
            log.info(f"Starting evaluation for {test_dataset_name} on fold {fold}!")

            # 확률 수집을 위한 콜백 생성
            prob_callback = ProbabilityCollectorCallback(test_dataset_name)

            # 테스트 실행
            if ckpt_path == "" or ckpt_path is None:
                print("Warning: Best ckpt not found! Using current weights for testing...")
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None

            # 콜백을 트레이너에 추가
            trainer.callbacks.append(prob_callback)

            # 체크포인트가 있는 경우 ckpt_path를 None으로 설정하여 trainer.test에서 처리
            if ckpt_path and os.path.exists(ckpt_path):
                trainer.test(model=model, dataloaders=test_data_loader, ckpt_path=ckpt_path)
            else:
                trainer.test(model=model, dataloaders=test_data_loader)

            # 콜백에서 트레이너 제거
            if prob_callback in trainer.callbacks:
                trainer.callbacks.remove(prob_callback)

            # 수집된 확률과 레이블 저장 (filepath 기반, CatMetric 사용)
            if not prob_callback.filepath_to_data:
                raise RuntimeError(
                    f"No data collected for {test_dataset_name} fold {fold}. CatMetric failed to collect test data."
                )

            all_probabilities[test_dataset_name][fold][
                "filepath_data"
            ] = prob_callback.filepath_to_data
            print(
                f"Saved {len(prob_callback.filepath_to_data)} samples for {test_dataset_name} fold {fold}"
            )

    # 결과 저장
    print(f"\nAll {K} folds completed. Saving results...")
    save_results(cfg, all_probabilities)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    # 명시적으로 GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")

    # 프로세스 종료를 위한 명시적 처리
    print(f"Process {os.getpid()} completed successfully")

    return {}, {}


def save_results(cfg: DictConfig, all_probabilities: Dict):
    """결과를 CSV 파일로 저장합니다."""

    print("\n" + "=" * 50)
    print("SAVING RESULTS")
    print("=" * 50)

    # 출력 디렉토리 확인 및 생성
    output_dir = cfg.paths.output_dir
    print(f"Output directory: {output_dir}")
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        print(f"Output directory already exists: {output_dir}")

    # 1. 각 데이터 샘플별 평균 확률 저장 및 AUROC/AUPRC 계산
    print("1. Saving sample-wise probabilities and computing metrics...")

    individual_metrics = []

    for dataset_name, fold_data in all_probabilities.items():
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Number of folds: {len(fold_data)}")

        # 모든 fold의 filepath_data 수집
        all_filepath_data = {}

        for fold in sorted(fold_data.keys()):
            fold_filepath_data = fold_data[fold]["filepath_data"]
            print(f"Fold {fold}: {len(fold_filepath_data)} samples")

            for filepath, data in fold_filepath_data.items():
                # 첫 번째 fold에서만 초기화
                if filepath not in all_filepath_data:
                    all_filepath_data[filepath] = {"probs": [], "label": data["label"]}
                else:
                    # 레이블은 모든 fold에서 동일해야 함
                    assert (
                        all_filepath_data[filepath]["label"] == data["label"]
                    ), f"Filepath {filepath}의 레이블이 fold마다 다름: {all_filepath_data[filepath]['label']} vs {data['label']}"

                all_filepath_data[filepath]["probs"].append(data["prob"])

        # 각 filepath별로 10-fold 평균 확률 계산
        filepaths = []
        mean_probs = []
        labels = []

        for filepath, data in all_filepath_data.items():
            assert len(data["probs"]) > 0, f"Filepath {filepath}에 대한 확률 데이터가 없습니다"

            filepaths.append(filepath)
            mean_probs.append(np.mean(data["probs"]))
            labels.append(data["label"])

        mean_probs = np.array(mean_probs)
        labels = np.array(labels)

        print(f"Processed {len(filepaths)} unique filepaths")
        print(f"Mean probs shape: {mean_probs.shape}, labels shape: {labels.shape}")
        print(f"Mean probs range: [{mean_probs.min():.4f}, {mean_probs.max():.4f}]")
        print(f"Labels unique: {np.unique(labels)}")

        # CSV 저장
        df = pd.DataFrame({"filepath": filepaths, "probability": mean_probs, "label": labels})

        filename = f"{dataset_name}.csv"
        filepath = os.path.join(output_dir, filename)

        try:
            df.to_csv(filepath, index=False)
            print(f"✓ Saved probabilities to {filepath}")
            log.info(f"Saved probabilities to {filepath}")

            # AUROC, AUPRC 계산
            if len(mean_probs) > 0:
                prob_tensor = torch.tensor(mean_probs)
                label_tensor = torch.tensor(labels)

                auroc = tm.functional.auroc(prob_tensor, label_tensor.long(), task="binary")
                auprc = tm.functional.average_precision(
                    prob_tensor, label_tensor.long(), task="binary"
                )

                print(f"✓ AUROC: {auroc.item():.4f}, AUPRC: {auprc.item():.4f}")

                individual_metrics.append(
                    {
                        "dataset": dataset_name,
                        "auroc": auroc.item(),
                        "auprc": auprc.item(),
                        "samples": len(mean_probs),
                    }
                )

        except Exception as e:
            print(f"✗ Failed to save probabilities to {filepath}: {e}")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"DataFrame head:\n{df.head()}")
            log.error(f"Failed to save probabilities to {filepath}: {e}")
            raise

    # 2. 개별 메트릭 저장
    print("\n2. Saving individual metrics...")
    print(f"Number of individual metrics: {len(individual_metrics)}")

    if individual_metrics:
        individual_df = pd.DataFrame(individual_metrics)
        individual_filepath = os.path.join(output_dir, "individual_metrics.csv")

        print(f"Individual metrics DataFrame shape: {individual_df.shape}")
        print(f"Individual metrics columns: {individual_df.columns.tolist()}")

        try:
            individual_df.to_csv(individual_filepath, index=False)
            print(f"✓ Saved individual metrics to {individual_filepath}")
            log.info(f"Saved individual metrics to {individual_filepath}")
        except Exception as e:
            print(f"✗ Failed to save individual metrics to {individual_filepath}: {e}")
            print(f"Individual metrics DataFrame shape: {individual_df.shape}")
            print(f"Individual metrics DataFrame columns: {individual_df.columns.tolist()}")
            print(f"Individual metrics DataFrame head:\n{individual_df.head()}")
            log.error(f"Failed to save individual metrics to {individual_filepath}: {e}")
            raise
    else:
        print("⚠ No individual metrics to save!")

    print("=" * 50)
    print("RESULTS SAVED SUCCESSFULLY")
    print("=" * 50)


@hydra.main(version_base="1.3", config_path="../configs", config_name="transfer_learning.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    extras(cfg)

    # evaluate the model
    metric_dict, _ = evaluate(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
