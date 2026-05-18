"""Helpers for resolving filesystem paths from environment variables.

The repository is configured so that all dataset and checkpoint locations are read
from `DATA_ROOT` / `CKPT_ROOT` (see `.env.example`). Yaml configs interpolate them
via `${paths.data_root}` / `${paths.ckpt_root}`. Python code that has historically
hard-coded paths should now go through these helpers so we have a single source of
truth.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["data_root", "ckpt_root", "data_path", "ckpt_path"]


def data_root() -> Path:
    """Root directory containing pre-training and downstream datasets."""
    return Path(os.environ.get("DATA_ROOT", "."))


def ckpt_root() -> Path:
    """Root directory containing pre-trained baseline checkpoints."""
    return Path(os.environ.get("CKPT_ROOT", "."))


def data_path(*parts: str | os.PathLike) -> Path:
    """Compose `DATA_ROOT / parts...`."""
    return data_root().joinpath(*map(str, parts))


def ckpt_path(*parts: str | os.PathLike) -> Path:
    """Compose `CKPT_ROOT / parts...`."""
    return ckpt_root().joinpath(*map(str, parts))
