"""Shared helpers used across the downstream entry points.

Today this only contains :func:`safe_release` because the three entry points
(`finetune.py`, `finetune_seg.py`, `finetune_data_efficient.py`) duplicate this
exact routine — extracting it removes a copy/paste source of bugs without
collapsing the entry points themselves (which differ in dataset shape, metric
set, and KFold strategy).
"""

from __future__ import annotations

import gc
from typing import Tuple

import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def safe_release(
    *,
    trainer=None,
    model=None,
    extra_to_del: Tuple = (),
    empty_cuda_cache: bool = True,
) -> None:
    """Detach references and release GPU memory without poking trainer internals.

    Used between cross-validation folds to avoid OOM on long sweeps.

    Args:
        trainer: Lightning Trainer instance to drop the reference to.
        model: LightningModule instance that should first be moved to CPU.
        extra_to_del: Additional objects (dataloaders, callbacks, ...) to delete.
        empty_cuda_cache: Whether to call ``torch.cuda.empty_cache()`` afterwards.
    """
    # 1) Move the model back to CPU before dropping the reference.
    try:
        if model is not None:
            try:
                model.to("cpu")
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Failed to move model to CPU: {exc}")
            finally:
                del model
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Error during model cleanup: {exc}")

    # 2) Drop any extra references the caller wants to clean up.
    for obj in extra_to_del:
        try:
            del obj
        except Exception:  # noqa: BLE001
            log.warning("Failed to delete object")

    # 3) Drop the trainer reference (avoids touching private hooks).
    try:
        if trainer is not None:
            del trainer
    except Exception:  # noqa: BLE001
        log.warning(f"Failed to delete trainer: {trainer}")

    # 4) Clear the CUDA cache if requested and run GC.
    if empty_cuda_cache and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Failed to empty CUDA cache: {exc}")
    gc.collect()
