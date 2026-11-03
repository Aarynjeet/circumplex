"""Regenerate every figure and statistic in the README from whatever exists in ``runs/``.

Each section runs only if its inputs exist (and says so if they do not), so this
works on day one (data-only figures) and again once the Kaggle outputs are back.

Usage::

    python scripts/run_analysis.py --main-run roberta_full_s42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from circumplex import plots
from circumplex.analysis import confusion, cooccurrence, probing, report
from circumplex.data import class_counts, load_goemotions, load_raw_annotations
from circumplex.evaluate import reliability_curve
from circumplex.labels import EMOTIONS, EMOTIONS_27, load_group_mapping
from circumplex.utils import to_jsonable
from circumplex.vad import VAD_TABLE, load_emotion_vad, va_matrix

log = logging.getLogger("analysis")


def ordering(vad: pd.DataFrame | None) -> list[str]:
    """Emotion order for heatmaps: by lexicon valence if available, else by Ekman group."""
    if vad is not None:
        return vad.sort_values("valence")["emotion"].tolist()
    return [
        e for members in load_group_mapping("ekman").values() for e in members if e != "neutral"
    ]


def reorder(M27: np.ndarray, order: list[str]) -> np.ndarray:
    idx = [EMOTIONS_27.index(e) for e in order]
    return M27[np.ix_(idx, idx)]


def main() -> None:  # noqa: C901 (a linear script of independent sections)
    """Run every analysis section whose inputs are available."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--figures", default="figures")
    ap.add_argument(
        "--main-run", default="roberta_full_s42", help="model analysed for RQ2-RQ4 and RQ3"
    )
    ap.add_argument("--source", default="auto")
    ap.add_argument("--n-perm", type=int, default=10_000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs_dir, fig = Path(args.runs), Path(args.figures)
    out_dir = runs_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"status": {}}

    # ------------------------------------------------------------------ data (always)
    splits = load_goemotions(args.source)
    counts = class_counts(splits["train"])
    counts.to_csv(out_dir / "train_class_counts.csv", index=False)
    plots.class_distribution(
        counts, fig / "class_distribution.png", "GoEmotions train set: label frequency (long tail)"
    )
    results["data"] = {
        "sizes": {k: len(v) for k, v in splits.items()},
        "max_to_min_ratio": float(counts["count"].max() / counts["count"].min()),
        "train_label_cardinality": float(splits["train"].y.sum(1).mean()),
        "frac_multi_label_train": float((splits["train"].y.sum(1) > 1).mean()),
    }

    vad = load_emotion_vad() if VAD_TABLE.exists() else None
    results["status"]["vad"] = "available" if vad is not None else "MISSING: run `make vad`"
    order = ordering(vad)
    order_note = (
        "rows/cols sorted by valence" if vad is not None else "rows/cols grouped by Ekman category"
    )
    keep27 = [EMOTIONS.index(e) for e in EMOTIONS_27]

    # Human co-occurrence (RQ4, human side): train set
    H_npmi = cooccurrence.npmi(splits["train"].y, min_count=5)
    plots.heatmap(
        reorder(H_npmi[np.ix_(keep27, keep27)], order),
        order,
        fig / "rq4_human_npmi.png",
        f"Human co-labelling (NPMI, train; {order_note})",
        diverging=True,
        vmax=1.0,
        cbar_label="NPMI",
    )
    results["rq4_human_top_pairs"] = cooccurrence.top_pairs(
        H_npmi[np.ix_(keep27, keep27)], 15, EMOTIONS_27
    ).to_dict("records")

    # Human disagreement (RQ2, human side): raters vs consensus on the test split
    test = splits["test"]
    gold = pd.DataFrame(test.y.astype(int), columns=EMOTIONS).assign(id=test.ids)
    raw = load_raw_annotations()
    H_conf = confusion.human_confusion(raw, gold)
    np.save(out_dir / "human_confusion.npy", H_conf)
    H27 = confusion.to_27(H_conf)
    plots.heatmap(
        reorder(H27, order),
        order,
        fig / "rq2_human_confusion.png",
        f"Raters vs consensus: soft confusion (test; {order_note})",
        cbar_label="P(rater says col, misses row)",
    )
    results["rq2_human_top_pairs"] = (
        confusion.pair_table(H27, np.zeros_like(H27))
        .head(15)[["a", "b", "confusion"]]
        .to_dict("records")
    )

    if vad is not None:
        dist = confusion.pairwise_distance(va_matrix(vad))
        results["rq2_human_vs_distance"] = confusion.mantel_test(
            confusion.symmetrize(H27), dist, args.n_perm
        )
        pairs_h = confusion.pair_table(H27, dist)
        s = results["rq2_human_vs_distance"]
        plots.confusion_vs_distance(
            pairs_h,
            fig / "rq2_human_confusion_vs_distance.png",
            "Human raters: confusion vs valence-arousal distance",
            f"Spearman ρ = {s['rho']:.2f}, permutation p = {s['p_value']:.4f}",
        )

    # ------------------------------------------------------------------ models (RQ1)
    run_dirs = report.find_runs(runs_dir)
    runs = [report.load_run(d) for d in run_dirs]
    results["status"]["runs_found"] = [r["name"] for r in runs]
    if runs:
        table = report.results_table(runs)
        table.to_csv(out_dir / "results_table.csv", index=False)
        (out_dir / "results_table.md").write_text(report.markdown_table(table) + "\n")
        results["results_table"] = table.to_dict("records")
        names = {r["name"] for r in runs}
        candidates = [
            (args.main_run, "tfidf_lr"),
            (args.main_run, "bilstm"),
            (args.main_run, "roberta_lora_s42"),
            ("bilstm", "tfidf_lr"),
        ]
        results["pairwise_tests"] = report.pairwise_tests(
            runs, [p for p in candidates if set(p) <= names], args.n_perm
        )

        shown = [
            r
            for r in runs
            if r["name"] in ("tfidf_lr", "bilstm", args.main_run, "roberta_lora_s42")
        ]
        plots.reliability_diagrams(
            {r["name"]: reliability_curve(r["test_y"], r["test_probs"]) for r in shown},
            fig / "calibration.png",
            {r["name"]: r["metrics"]["test_ece"] for r in shown},
        )
        trio = [r for r in runs if r["name"] in ("tfidf_lr", "bilstm", args.main_run)]
        f1 = pd.DataFrame(
            {
                r["name"]: {e: v["f1"] for e, v in r["metrics"]["tuned"]["per_class"].items()}
                for r in trio
            }
        )
        f1 = f1.loc[counts.sort_values("count")["emotion"]]  # rarest at the bottom
        plots.per_class_f1(
            f1,
            fig / "rq1_per_class_f1.png",
            "Per-emotion test F1 (rows: rare at bottom, frequent at top)",
        )
        results["rq1_per_class_f1"] = f1.to_dict()

    main = next((r for r in runs if r["name"] == args.main_run), None)
    results["status"]["main_run"] = "available" if main else f"MISSING: {args.main_run}"
    if main is not None:
        # ------------------------------------------------------------------ RQ2 (model)
        C = confusion.soft_confusion(main["test_y"], main["test_pred"])
        np.save(out_dir / "model_confusion.npy", C)
        C27 = confusion.to_27(C)
        plots.heatmap(
            reorder(C27, order),
            order,
            fig / "rq2_model_confusion.png",
            f"{args.main_run}: soft confusion (test; {order_note})",
            cbar_label="P(model says col, misses row)",
        )
        results["rq2_model_vs_human"] = confusion.mantel_test(
            confusion.symmetrize(C27), confusion.symmetrize(H27), args.n_perm, alternative="greater"
        )
        if vad is not None:
            dist = confusion.pairwise_distance(va_matrix(vad))
            results["rq2_model_vs_distance"] = confusion.mantel_test(
                confusion.symmetrize(C27), dist, args.n_perm
            )
            pairs_m = confusion.pair_table(C27, dist)
            pairs_m.to_csv(out_dir / "rq2_model_pairs.csv", index=False)
            s = results["rq2_model_vs_distance"]
            plots.confusion_vs_distance(
                pairs_m,
                fig / "rq2_model_confusion_vs_distance.png",
                f"{args.main_run}: confusion vs valence-arousal distance",
                f"Spearman ρ = {s['rho']:.2f}, permutation p = {s['p_value']:.4f}",
            )

        # ------------------------------------------------------------------ RQ4 (model)
        results["rq4_human_vs_model"] = cooccurrence.compare_human_model(
            main["test_y"], main["test_pred"]
        )
        M_npmi = cooccurrence.npmi(main["test_pred"], min_count=5)
        plots.heatmap(
            reorder(M_npmi[np.ix_(keep27, keep27)], order),
            order,
            fig / "rq4_model_npmi.png",
            f"{args.main_run}: co-predicted labels (NPMI, test; {order_note})",
            diverging=True,
            vmax=1.0,
            cbar_label="NPMI",
        )
        r4 = results["rq4_human_vs_model"]
        card = pd.DataFrame(
            {
                "human labels": r4["cardinality_human_dist"],
                "model predictions": r4["cardinality_model_dist"],
            }
        ).fillna(0)
        card = card.loc[sorted(card.index, key=int)]
        plots.grouped_bars(
            card,
            fig / "rq4_cardinality.png",
            "Labels per test example: humans vs model",
            "number of labels",
            "fraction of test examples",
        )

    # ------------------------------------------------------------------ RQ3 (probing)
    emb = {
        v: runs_dir / args.main_run / f"embeddings_{v}_{{}}.npz"
        for v in ("finetuned", "base", "random")
    }
    have = {
        v: p
        for v, p in emb.items()
        if Path(str(p).format("dev")).exists() and Path(str(p).format("test")).exists()
    }
    results["status"]["embeddings"] = sorted(have) or "MISSING: run scripts/extract_embeddings.py"
    if have and vad is not None:
        va = va_matrix(vad)
        layer_tables, rq3 = {}, {}
        for v, p in have.items():
            dev, tst = np.load(str(p).format("dev")), np.load(str(p).format("test"))
            for pool in ("cls", "mean"):
                lt = probing.layerwise_probe(dev[pool], dev["y"], tst[pool], tst["y"], va)
                layer_tables[f"{v}_{pool}"] = lt
                lt.to_csv(out_dir / f"rq3_layers_{v}_{pool}.csv", index=False)
            last = tst["cls"].shape[0] - 1
            means = probing.emotion_means(tst["mean"][last].astype(np.float32), tst["y"])
            rq3[v] = {
                "last_layer_cls": layer_tables[f"{v}_cls"].iloc[-1].to_dict(),
                "last_layer_mean": layer_tables[f"{v}_mean"].iloc[-1].to_dict(),
                "pca": probing.pca_alignment(means, va, n_perm=args.n_perm),
            }
            pca = rq3[v]["pca"]
            plots.emotion_map(
                np.array(pca["coords"])[:, :2],
                EMOTIONS_27,
                va[:, 0],
                fig / f"rq3_pca_{v}.png",
                f"{v} encoder: PCA of the 27 emotion-mean embeddings",
                f"PC1 ({100 * pca['explained_variance_ratio'][0]:.0f}% var)",
                f"PC2 ({100 * pca['explained_variance_ratio'][1]:.0f}% var)",
            )
        if main is not None:  # control: probe the fine-tuned model's own 28 probabilities
            pd_dev = np.load(
                run_dirs[[d.name for d in run_dirs].index(args.main_run)] / "predictions.npz"
            )
            t_dev, m_dev = probing.example_va_targets(pd_dev["dev_y"], va)
            t_tst, m_tst = probing.example_va_targets(pd_dev["test_y"], va)
            rq3["probe_on_output_probs"] = probing.ridge_probe(
                pd_dev["dev_probs"][m_dev], t_dev, pd_dev["test_probs"][m_tst], t_tst
            )
        results["rq3"] = rq3
        for dim in ("valence", "arousal"):
            cols = [k for k in ("finetuned_cls", "base_cls", "random_cls") if k in layer_tables]
            df = pd.DataFrame(
                {
                    "layer": layer_tables[cols[0]]["layer"],
                    **{k: layer_tables[k][f"{dim}_r2"] for k in cols},
                }
            )
            plots.lines(
                df,
                "layer",
                cols,
                fig / f"rq3_layerwise_{dim}.png",
                f"Linear probe R² for {dim}, by layer ([CLS] vector)",
                "layer (0 = embeddings)",
                f"held-out R² ({dim})",
                labels=[c.replace("_cls", "") for c in cols],
            )

    # ------------------------------------------------------------------ RQ5 (transfer)
    tr = runs_dir / "transfer" / "results.csv"
    results["status"]["transfer"] = (
        "available" if tr.exists() else "MISSING: run configs/transfer.yaml"
    )
    if tr.exists():
        from circumplex.transfer import paired_seed_differences

        df = pd.read_csv(tr)
        g = df.groupby(["fraction", "init"])["test_macro_f1"].agg(["mean", "std"]).unstack("init")
        wide = pd.DataFrame({"fraction": g.index})
        for init in ("vanilla", "emotion"):
            if ("mean", init) in g:
                wide[init] = g[("mean", init)].to_numpy()
                wide[f"{init}_std"] = g[("std", init)].fillna(0).to_numpy()
        inits = [i for i in ("vanilla", "emotion") if i in wide]
        plots.lines(
            wide,
            "fraction",
            inits,
            fig / "rq5_learning_curves.png",
            "Thinking-trap detection: learning curves (mean ± std over seeds)",
            "fraction of training data",
            "test macro-F1",
            err=[f"{i}_std" for i in inits],
            labels=["vanilla roberta-base", "GoEmotions-tuned"][: len(inits)],
            logx=True,
        )
        if set(inits) == {"vanilla", "emotion"}:
            results["rq5_paired_differences"] = paired_seed_differences(df).to_dict("records")
        results["rq5_curve"] = wide.to_dict("records")

    with (out_dir / "analysis.json").open("w") as f:
        json.dump(to_jsonable(results), f, indent=2, default=str)
    log.info("status: %s", results["status"])
    log.info("wrote %s and figures in %s", out_dir / "analysis.json", fig)


if __name__ == "__main__":
    main()
