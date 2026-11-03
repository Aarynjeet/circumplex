"""Classical baseline: TF-IDF features + one-vs-rest logistic regression (scikit-learn).

One independent binary logistic regression per label, all sharing the same
sparse TF-IDF features (word 1-2 grams + character 2-5 grams). It runs on CPU
in a few minutes and goes through exactly the same evaluation as the neural models.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from circumplex.config import ExperimentConfig, load_config
from circumplex.data import load_goemotions
from circumplex.evaluate import full_report, most_confident_errors, write_jsonl
from circumplex.utils import RunDir, set_seed, timer

log = logging.getLogger(__name__)


class TfidfLogReg:
    """TF-IDF (word + char n-grams) followed by one logistic regression per label."""

    def __init__(self, params: dict[str, Any], seed: int = 0) -> None:
        p = params
        self.word = TfidfVectorizer(
            ngram_range=tuple(p.get("word_ngrams", (1, 2))),
            max_features=p.get("max_features_word", 100_000),
            min_df=p.get("min_df", 2),
            sublinear_tf=p.get("sublinear_tf", True),
            lowercase=True,
        )
        self.char = (
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=tuple(p["char_ngrams"]),
                max_features=p.get("max_features_char", 100_000),
                min_df=p.get("min_df", 2),
                sublinear_tf=p.get("sublinear_tf", True),
            )
            if p.get("char_ngrams")
            else None
        )
        self.clf = OneVsRestClassifier(
            LogisticRegression(
                C=p.get("C", 4.0),
                class_weight=p.get("class_weight"),
                solver="liblinear",
                max_iter=1000,
                random_state=seed,
            ),
            n_jobs=-1,
        )

    def _features(self, texts: list[str], fit: bool = False) -> sp.csr_matrix:
        f = (lambda v: v.fit_transform) if fit else (lambda v: v.transform)
        mats = [f(self.word)(texts)]
        if self.char is not None:
            mats.append(f(self.char)(texts))
        return sp.hstack(mats).tocsr()

    def fit(self, texts: list[str], y: np.ndarray) -> TfidfLogReg:
        """Fit vectorisers and the 28 classifiers on training texts."""
        self.clf.fit(self._features(texts, fit=True), y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return (N, 28) probabilities."""
        return self.clf.predict_proba(self._features(texts))

    @property
    def n_parameters(self) -> int:
        """Number of learned weights (one coefficient per feature per label, plus biases)."""
        return int(sum(e.coef_.size + e.intercept_.size for e in self.clf.estimators_))


def run_baseline(cfg: ExperimentConfig) -> dict[str, Any]:
    """Fit on train, tune thresholds on dev, evaluate on test; write ``runs/<name>/``."""
    set_seed(cfg.seed)
    run = RunDir(cfg.output.runs_dir, cfg.name)
    run.save_config(cfg.to_dict())
    splits = load_goemotions(cfg.data.source, cfg.data.cache_dir)
    train = splits["train"].subset(cfg.data.subsample, cfg.seed)
    dev = splits["dev"].subset(cfg.data.eval_subsample, cfg.seed)
    test = splits["test"].subset(cfg.data.eval_subsample, cfg.seed)

    model = TfidfLogReg(cfg.baseline, cfg.seed)
    with timer() as t:
        model.fit(train.texts, train.y)
    dev_probs, test_probs = model.predict_proba(dev.texts), model.predict_proba(test.texts)
    np.savez_compressed(
        run.path / "predictions.npz",
        dev_probs=dev_probs,
        dev_y=dev.y,
        test_probs=test_probs,
        test_y=test.y,
        test_ids=np.array(test.ids),
        dev_ids=np.array(dev.ids),
    )
    report = full_report(dev.y, dev_probs, test.y, test_probs, cfg.output.n_bootstrap, cfg.seed)
    thresholds = np.array(list(report["thresholds"].values()))
    write_jsonl(
        most_confident_errors(test.texts, test.y, test_probs, thresholds), run.path / "errors.jsonl"
    )
    metrics = {
        "run": cfg.name,
        "model_type": "tfidf_lr",
        "params": {"total": model.n_parameters, "trainable": model.n_parameters},
        "train_seconds": t["seconds"],
        "device": "cpu",
        **report,
    }
    run.save_metrics(metrics)
    log.info(
        "test macro-F1: %.4f @0.5, %.4f tuned (%.0fs)",
        report["fixed_0.5"]["macro_f1"],
        report["tuned"]["macro_f1"],
        t["seconds"],
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m circumplex.baselines --config configs/tfidf_lr.yaml``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_baseline(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
