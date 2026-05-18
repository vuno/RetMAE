import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import pandas as pd
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Subset

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


def _nested_subsample_map(
    full_idx: np.ndarray, ratios: list[float], seed: int
) -> dict[str, list[int]]:
    """
    Generate nested index sets for few-shot learning using a single random permutation.
    Guarantees that 1% ⊂ 5% ⊂ 10% (i.e., smaller subsets are prefixes of larger ones).
    Returns a dict with keys like '1%', '5%', '10%'.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(full_idx)  # Random order without duplicates
    n = len(perm)

    out: dict[str, list[int]] = {}
    for r in sorted(set(ratios)):  # Ascending order: 0.01, 0.05, 0.10
        k = max(1, int(math.ceil(r * n)))
        # Select the first k elements (prefix) to guarantee nesting
        idx = np.sort(perm[:k])
        out[f"{int(r * 100)}%"] = idx.tolist()
    return out


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main training loop for few-shot transfer learning with cross-validation.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Metric dictionary and additional outputs.
    """
    # Instantiate datamodule and set up data
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    datamodule.setup("test")

    # Few-shot settings
    ratios: list[float] = list(
        getattr(getattr(cfg, "eval", {}), "fewshot_ratios", [0.01, 0.05, 0.10])
    )
    repeats: int = int(getattr(getattr(cfg, "eval", {}), "fewshot_repeats", 5))

    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=cfg.seed)

    # Results: fold -> { test_dataset_name -> { col_name -> value } }
    res: Dict[int, Dict[str, Dict[str, float]]] = {fold: defaultdict(dict) for fold in range(K)}

    # Concatenate all training datasets
    train_dataset = ConcatDataset(list(datamodule.data_train.values()))

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        L.seed_everything(int(cfg.seed) + fold, workers=True)

        # Fixed validation set
        val_subset = Subset(train_dataset, val_idx)
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

        # For each repeat, generate a single permutation for all ratios (guarantees nesting)
        for rep in range(1, repeats + 1):
            rep_seed = int(cfg.seed) * 10_000 + fold * 100 + rep
            nested_map = _nested_subsample_map(np.array(train_idx), ratios, seed=rep_seed)

            # Iterate over ratios (recommended to keep sorted: 1%, 5%, 10%)
            for r in ratios:
                pct_label = f"{int(r * 100)}%"
                sub_train_idx = nested_map[pct_label]
                train_subset = Subset(train_dataset, sub_train_idx)

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

                # Model, callbacks, logger, trainer instantiation
                model: LightningModule = hydra.utils.instantiate(cfg.model)
                callbacks = instantiate_callbacks(cfg.get("callbacks"))
                logger = instantiate_loggers(cfg.get("logger"))
                trainer: Trainer = hydra.utils.instantiate(
                    cfg.trainer, callbacks=callbacks, logger=logger
                )

                # (Optional) Inject meta information into the model
                model.max_epochs = cfg.trainer.max_epochs
                model.steps_per_epoch = len(train_loader)
                model.fewshot_fraction = r
                model.fewshot_repeat = rep
                model.cv_fold = fold

                # Training
                trainer.fit(
                    model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=cfg.get("ckpt_path"),
                )

                # Testing
                ckpt_cb = getattr(trainer, "checkpoint_callback", None)
                best_ckpt = ckpt_cb.best_model_path if ckpt_cb is not None else ""
                for test_dataset_name, test_loader in datamodule.test_dataloader().items():
                    log.info(
                        f"[fold={fold}] Testing on {test_dataset_name} (r={pct_label}, rep={rep})"
                    )
                    ckpt_use = best_ckpt if best_ckpt else None
                    trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_use)
                    test_metrics = trainer.callback_metrics

                    # Logging columns: "1%_auroc_1", "1%_auprc_1", ...
                    auroc_key = f"test/{test_dataset_name}/auroc"
                    auprc_key = f"test/{test_dataset_name}/auprc"
                    if auroc_key in test_metrics:
                        res[fold][test_dataset_name][f"{pct_label}_auroc_{rep}"] = float(
                            test_metrics[auroc_key].item()
                        )
                    if auprc_key in test_metrics:
                        res[fold][test_dataset_name][f"{pct_label}_auprc_{rep}"] = float(
                            test_metrics[auprc_key].item()
                        )

        # Save per-fold CSV
        rows = []
        for ds_name, cols in res[fold].items():
            row = {"test_dataset": ds_name}

            def _col_sort_key(c: str):
                # Example: "1%_auroc_3"
                parts = c.replace("%", "").split("_")
                pct = int(parts[0])
                metric = parts[1]
                rep = int(parts[2])
                metric_rank = 0 if metric == "auroc" else 1  # auroc first
                return (pct, metric_rank, rep)

            for k in sorted(cols.keys(), key=_col_sort_key):
                row[k] = cols[k]
            rows.append(row)

        res_dataframe = pd.DataFrame(rows)
        out_path = os.path.join(
            cfg.paths.output_dir, f"{cfg.task_name}-{cfg.model_name}-{fold}.csv"
        )
        res_dataframe.to_csv(out_path, index=False)
        log.info(f"Saved fold results to: {out_path}")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Hydra entry point for transfer learning experiment.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        Optional[float]: Optimized metric value, if available.
    """
    extras(cfg)
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
