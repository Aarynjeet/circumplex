import pytest

from circumplex.config import load_config


def test_base_inheritance_and_overrides():
    cfg = load_config("configs/transformer_lora.yaml", ["train.lr=1e-4", "model.lora.r=4"])
    assert cfg.name == "roberta_lora"
    assert cfg.model.name == "roberta-base"  # inherited from transformer_full.yaml
    assert cfg.model.lora.enabled and cfg.model.lora.r == 4
    assert cfg.train.lr == pytest.approx(1e-4)
    assert cfg.data.max_len == 64  # inherited from base.yaml


def test_unknown_key_fails_loudly(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("train:\n  learning_rate: 0.1\n")
    with pytest.raises(KeyError):
        load_config(p)


def test_every_config_parses():
    from pathlib import Path

    for path in Path("configs").rglob("*.yaml"):
        load_config(path)
