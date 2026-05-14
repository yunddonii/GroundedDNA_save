"""Append-mode CSV logger for per-epoch training/evaluation metrics.

Each call to ``log(row_dict)`` appends one row. Missing keys are written as
empty cells so a single fixed schema can mix per-epoch losses with
intermittent retrieval metrics (only filled every N epochs).

Designed to be opened mid-training (e.g. ``tail -f metrics.csv`` or load it
in pandas/Excel) so we can watch optimization progress without waiting for
training to finish.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def _coerce_scalar(v: Any) -> Any:
    """Convert numpy / torch scalars to plain Python types so csv.writer
    produces clean numeric strings (no `np.float64(...)` wrappers)."""
    if v is None:
        return ""
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if torch.is_tensor(v):
        if v.dim() == 0:
            return v.item()
        return v.detach().cpu().tolist()
    return v


class EpochCSVLogger:
    """Tiny per-epoch CSV writer (append mode).

    Args:
        path:        output .csv path. The header is written if the file
                     does not exist yet.
        fieldnames:  fixed list of column names (epoch + per-phase losses
                     + eval metrics).

    Example:
        >>> logger = EpochCSVLogger("metrics.csv",
        ...                         ["epoch", "train_loss", "val_loss", "eval_mAP"])
        >>> logger.log({"epoch": 0, "train_loss": 0.91, "val_loss": 0.95})  # eval_mAP empty
        >>> logger.log({"epoch": 1, "train_loss": 0.74, "val_loss": 0.83,
        ...             "eval_mAP": 0.42})
    """

    def __init__(self, path: str, fieldnames: Iterable[str]) -> None:
        self.path: str = str(path)
        self.fieldnames: List[str] = list(fieldnames)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """Append one row. Missing keys → empty string."""
        cleaned: Dict[str, Any] = {
            k: _coerce_scalar(row.get(k, "")) for k in self.fieldnames
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(cleaned)
