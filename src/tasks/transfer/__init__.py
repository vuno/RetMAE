"""Shared utilities for downstream transfer-learning entry points (`finetune.py`,
`finetune_seg.py`, `finetune_data_efficient.py`).
"""

from src.tasks.transfer._common import safe_release

__all__ = ["safe_release"]
