"""Render Integrated-Gradients token highlights for curated examples to HTML.

Examples come from two places: a fixed list of test-set comments (by id), and
the most confident errors in ``runs/<run>/errors.jsonl``. For each one we
attribute the top predicted label (and, for errors, the gold label too).

Usage::

    python scripts/attributions.py --run runs/roberta_full_s42 --out figures/attributions.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from circumplex.analysis.attribution import integrated_gradients, to_html
from circumplex.data import load_goemotions
from circumplex.labels import EMOTIONS
from circumplex.train import load_trained


def main() -> None:
    """Pick examples, run IG, write one HTML page."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="figures/attributions.html")
    ap.add_argument("--n-correct", type=int, default=6)
    ap.add_argument("--n-errors", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    run = Path(args.run)
    model, tok, cfg = load_trained(run)
    device = next(model.parameters()).device
    test = load_goemotions(cfg.data.source, cfg.data.cache_dir)["test"]
    preds = np.load(run / "predictions.npz")
    probs = preds["test_probs"]

    # Correct, confident, single-label examples spread over different emotions.
    y = test.y
    top = probs.argmax(1)
    chosen, seen = [], set()
    for i in np.argsort(-probs.max(1)):
        lab = int(top[i])
        if y[i].sum() == 1 and y[i, lab] == 1 and lab not in seen and EMOTIONS[lab] != "neutral":
            chosen.append((test.texts[i], EMOTIONS[lab], "correct"))
            seen.add(lab)
        if len(chosen) >= args.n_correct:
            break
    errors = [json.loads(line) for line in (run / "errors.jsonl").read_text().splitlines()][
        : args.n_errors
    ]
    for e in errors:
        chosen.append((e["text"], e["wrong_label"], f"error (gold: {', '.join(e['gold'])})"))
        chosen.append((e["text"], e["gold"][0], "same error, attributed to the gold label"))

    parts = [
        "<html><head><meta charset='utf-8'><title>Token attributions</title></head>",
        "<body style='font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;line-height:1.7'>",
        "<h2>Integrated Gradients token attributions</h2>",
        "<p>Green pushes the model toward the named emotion, red pushes away. "
        "Baseline: pad embeddings; 50 interpolation steps.</p>",
    ]
    for text, label, kind in chosen:
        with torch.enable_grad():
            r = integrated_gradients(
                model, tok, text, label, steps=args.steps, max_len=cfg.data.max_len, device=device
            )
        parts.append(
            f"<div><small>{kind}; completeness gap {r['completeness_gap']:.3f}</small>{to_html(r)}</div>"
        )
    parts.append("</body></html>")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(parts))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
