"""Offline end-to-end checks with tiny models: checkpoint round trip and IG completeness."""

import numpy as np
import pytest
import torch

from circumplex.analysis.attribution import integrated_gradients, to_html
from circumplex.config import load_config
from circumplex.models.transformer import TransformerClassifier, build_encoder, train_tiny_tokenizer
from circumplex.train import load_trained, run_experiment

TEXTS = [
    "I love this so much!",
    "This is awful and I hate it.",
    "Thanks a lot [NAME]",
    "what? why?",
] * 25


def test_integrated_gradients_completeness():
    tok = train_tiny_tokenizer(TEXTS, vocab_size=200)
    torch.manual_seed(0)
    model = TransformerClassifier(build_encoder("tiny-random", vocab_size=len(tok)), 28).eval()
    r = integrated_gradients(model, tok, "I love this so much!", "love", steps=200)
    assert len(r["tokens"]) == len(r["attributions"])
    # completeness: sum of attributions ~= F(x) - F(baseline)
    assert r["completeness_gap"] < 0.05 * max(1.0, abs(r["logit"] - r["baseline_logit"]))
    assert "<span" in to_html(r)


@pytest.mark.slow
@pytest.mark.parametrize("cfg_path", ["configs/smoke.yaml", "configs/smoke_lora.yaml"])
def test_train_and_reload_reproduces_predictions(tmp_path, cfg_path):
    """Needs the GoEmotions TSVs (downloaded on first use)."""
    cfg = load_config(
        cfg_path, [f"output.runs_dir={tmp_path}", "output.n_bootstrap=10", "train.max_steps=3"]
    )
    metrics = run_experiment(cfg)
    assert 0.0 <= metrics["tuned"]["macro_f1"] <= 1.0
    run = tmp_path / cfg.name
    for f in (
        "config.yaml",
        "curve.csv",
        "metrics.json",
        "predictions.npz",
        "errors.jsonl",
        "best.pt",
    ):
        assert (run / f).exists(), f
    model, tok, cfg2 = load_trained(run, torch.device("cpu"))
    from circumplex.data import load_goemotions
    from circumplex.evaluate import predict_proba
    from circumplex.train import make_loaders

    splits = load_goemotions(cfg2.data.source, cfg2.data.cache_dir)
    test = {"test": splits["test"].subset(cfg2.data.eval_subsample, cfg2.seed)}
    probs = predict_proba(model, make_loaders(test, tok, cfg2)["test"], torch.device("cpu"))
    np.testing.assert_allclose(probs, np.load(run / "predictions.npz")["test_probs"], atol=1e-5)
