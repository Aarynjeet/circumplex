"""Small shared helpers: seeding, devices, run directories, git hashes."""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and every CUDA device).

    cuDNN autotuning is turned off so convolutions/RNN kernels are picked
    deterministically; full bitwise determinism on GPU is not guaranteed
    (some CUDA kernels, e.g. atomics in embedding backward, are nondeterministic).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit_hash() -> str:
    """Return the current git commit (with ``-dirty`` if there are local edits)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


class RunDir:
    """Owns ``runs/<name>/`` and the standard files every run writes.

    Files: ``config.yaml`` (resolved config + git hash), ``metrics.json``,
    ``curve.csv`` (one row per evaluation), plus whatever artifacts the run adds.
    """

    def __init__(self, runs_dir: str | Path, name: str) -> None:
        self.path = Path(runs_dir) / name
        self.path.mkdir(parents=True, exist_ok=True)
        self._curve_fields: list[str] | None = None

    def save_config(self, config: dict[str, Any]) -> None:
        """Write the resolved config with the git commit and torch/CUDA versions."""
        meta = {
            "git_commit": git_commit_hash(),
            "torch_version": str(torch.__version__),
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        with (self.path / "config.yaml").open("w") as f:
            yaml.safe_dump({**config, "_meta": meta}, f, sort_keys=False)

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        """Write ``metrics.json`` (numpy types are converted)."""
        with (self.path / "metrics.json").open("w") as f:
            json.dump(to_jsonable(metrics), f, indent=2)

    def log_curve(self, row: dict[str, Any]) -> None:
        """Append one row to ``curve.csv`` (header written on first call)."""
        file = self.path / "curve.csv"
        if self._curve_fields is None:
            self._curve_fields = list(row)
            with file.open("w", newline="") as f:
                csv.DictWriter(f, self._curve_fields).writeheader()
        with file.open("a", newline="") as f:
            csv.DictWriter(f, self._curve_fields).writerow(row)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays so ``json.dump`` accepts them."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager that records elapsed wall-clock seconds in ``out["seconds"]``."""
    out: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - start
