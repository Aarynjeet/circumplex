"""Typed, YAML-driven experiment configuration.

Every run is described by one YAML file. It is parsed into nested dataclasses so
that typos in keys fail loudly instead of being silently ignored, and it can be
overridden from the command line with ``--set train.lr=1e-5``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Where the data comes from and how it is preprocessed."""

    source: str = "auto"  # "auto" (HF, then GitHub fallback), "hf", or "github"
    cache_dir: str = "data/cache"
    max_len: int = 64  # tokens; GoEmotions comments are short
    subsample: int | None = None  # keep only N train examples (smoke tests)
    eval_subsample: int | None = None  # keep only N dev/test examples (smoke tests)
    min_freq: int = 2  # BiLSTM vocabulary: drop rarer words


@dataclass
class LoraConfig:
    """LoRA adapter settings (only used when ``enabled``)."""

    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["query", "value"])


@dataclass
class ModelConfig:
    """Which model to build."""

    type: str = "transformer"  # "bilstm" or "transformer"
    name: str = "roberta-base"  # HF id, or "tiny-random" for offline smoke tests
    dropout: float = 0.1
    # BiLSTM only
    emb_dim: int = 300
    hidden_dim: int = 256
    num_layers: int = 1
    pooling: str = "attention"  # "attention" or "max"
    glove_path: str | None = None  # optional GloVe .txt to initialise embeddings
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class TrainConfig:
    """Optimisation settings for the shared training loop."""

    epochs: int = 4
    batch_size: int = 32
    eval_batch_size: int = 128
    grad_accum_steps: int = 1
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    amp: bool = True  # only takes effect on CUDA
    patience: int = 2  # early stopping, in evaluations without dev macro-F1 gain
    evals_per_epoch: int = 1
    max_steps: int | None = None  # hard cap on optimiser steps (smoke tests)
    loss: str = "bce"  # "bce" or "focal"
    pos_weight: str | None = None  # None or "inverse_freq"
    pos_weight_cap: float = 30.0
    focal_gamma: float = 2.0
    num_workers: int = 0


@dataclass
class OutputConfig:
    """What to write to ``runs/<name>/``."""

    runs_dir: str = "runs"
    save_checkpoint: bool = True
    n_bootstrap: int = 1000


@dataclass
class ExperimentConfig:
    """Top-level config: one of these fully specifies a run."""

    name: str = "unnamed"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Free-form section for non-neural runs (e.g. TF-IDF hyperparameters).
    baseline: dict[str, Any] = field(default_factory=dict)
    # Free-form section for the RQ5 transfer experiment (see transfer.py).
    transfer: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain nested dicts (for YAML dumping)."""
        return dataclasses.asdict(self)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build dataclass ``cls`` from ``data``, rejecting unknown keys."""
    if not dataclasses.is_dataclass(cls):
        return data
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        sub = _SECTION_TYPES.get((cls, key))
        if sub and isinstance(value, dict):
            kwargs[key] = _from_dict(sub, value)
        else:
            kwargs[key] = _coerce(value, str(known[key].type))
    return cls(**kwargs)


def _coerce(value: Any, type_name: str) -> Any:
    """Fix YAML 1.1 quirks: ``1e-4`` (no dot) loads as a string, not a float."""
    if isinstance(value, str) and type_name.split(" |")[0] in {"float", "int"}:
        return float(value) if type_name.startswith("float") else int(value)
    if isinstance(value, int) and not isinstance(value, bool) and type_name.startswith("float"):
        return float(value)
    return value


_SECTION_TYPES: dict[tuple[type, str], type] = {
    (ExperimentConfig, "data"): DataConfig,
    (ExperimentConfig, "model"): ModelConfig,
    (ExperimentConfig, "train"): TrainConfig,
    (ExperimentConfig, "output"): OutputConfig,
    (ModelConfig, "lora"): LoraConfig,
}


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Rebuild a config from a dict (e.g. a saved ``config.yaml``; ``_meta`` is ignored)."""
    return _from_dict(ExperimentConfig, {k: v for k, v in data.items() if k != "_meta"})


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``data["a"]["b"] = value`` for ``dotted == "a.b"``."""
    *parents, leaf = dotted.split(".")
    node = data
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load a YAML config and apply ``key.sub=value`` overrides.

    A config may contain ``base: other.yaml`` to inherit from another file
    (resolved relative to the including file); keys in the child win.

    Args:
        path: Path to the YAML file.
        overrides: Strings like ``"train.lr=1e-5"``; values are parsed as YAML,
            so ``true``, ``3``, ``null`` and lists work.

    Returns:
        The fully resolved :class:`ExperimentConfig`.
    """
    raw = _load_yaml_with_base(Path(path))
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        _set_dotted(raw, key.strip(), yaml.safe_load(value))
    return _from_dict(ExperimentConfig, raw)


def _deep_merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    if base is None:
        return raw
    return _deep_merge(_load_yaml_with_base(path.parent / base), raw)
