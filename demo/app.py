"""Gradio demo: emotions, a valence-arousal plot, and token attributions for any text.

Runs on CPU. Point ``CIRCUMPLEX_MODEL_DIR`` at a run directory containing
``config.yaml``, ``best.pt``, ``metrics.json`` and ``tokenizer/`` (default:
``demo/model``). See demo/README.md for Hugging Face Spaces deployment.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
# Repo checkout: demo/app.py -> repo root. Hugging Face Space: app.py sits at the root.
ROOT = HERE if (HERE / "src").exists() else HERE.parent
sys.path.insert(0, str(ROOT / "src"))  # works without pip install

from circumplex.analysis.attribution import clean_token, integrated_gradients  # noqa: E402
from circumplex.labels import EMOTIONS, EMOTIONS_27  # noqa: E402
from circumplex.train import load_trained  # noqa: E402
from circumplex.vad import VAD_TABLE, load_emotion_vad, va_matrix  # noqa: E402

MODEL_DIR = Path(os.environ.get("CIRCUMPLEX_MODEL_DIR", HERE / "model"))
DISCLAIMER = (
    "**This is a research demo, not a mental-health or diagnostic tool.** It guesses which "
    "of 27 emotion labels Reddit annotators might apply to a short English comment. It is often "
    "wrong, misses sarcasm and context, and must not be used to make decisions about anyone."
)

model, tokenizer, cfg = load_trained(MODEL_DIR, torch.device("cpu"))
with (MODEL_DIR / "metrics.json").open() as f:
    THRESHOLDS = np.array(list(json.load(f)["thresholds"].values()))
VA = va_matrix(load_emotion_vad()) if VAD_TABLE.exists() else None


@torch.no_grad()
def predict(text: str) -> np.ndarray:
    """Sigmoid probabilities (28,) for one text."""
    enc = tokenizer(text, truncation=True, max_length=cfg.data.max_len, return_tensors="pt")
    return torch.sigmoid(model(enc["input_ids"], enc["attention_mask"]))[0].numpy()


def va_figure(probs: np.ndarray) -> plt.Figure | None:
    """The 27 emotions placed by lexicon valence/arousal, plus the input's weighted position."""
    if VA is None:
        return None
    w = probs[[EMOTIONS.index(e) for e in EMOTIONS_27]]
    point = (w[:, None] * VA).sum(0) / max(w.sum(), 1e-9)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(VA[:, 0], VA[:, 1], s=20 + 300 * w, color="#2a78d6", alpha=0.35, edgecolor="none")
    for (v, a), name in zip(VA, EMOTIONS_27, strict=True):
        ax.annotate(
            name, (v, a), fontsize=7, color="#52514e", xytext=(3, 2), textcoords="offset points"
        )
    ax.scatter(*point, s=160, marker="*", color="#eb6834", zorder=3, label="this text")
    ax.set_xlabel("valence (NRC VAD)")
    ax.set_ylabel("arousal (NRC VAD)")
    ax.set_title("Where the predicted emotions sit (probability-weighted)", fontsize=10, loc="left")
    ax.legend(frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def analyse(text: str):
    """Gradio callback: label scores, predicted set, VA plot, token attributions."""
    text = (text or "").strip()
    if not text:
        return {}, "", None, []
    probs = predict(text)
    on = [EMOTIONS[j] for j in np.flatnonzero(probs >= THRESHOLDS)]
    top = int(np.argmax(np.where(np.arange(28) == EMOTIONS.index("neutral"), -1, probs)))
    with torch.enable_grad():
        ig = integrated_gradients(model, tokenizer, text, top, steps=32, max_len=cfg.data.max_len)
    scale = max(abs(a) for a in ig["attributions"]) or 1.0
    spans = [
        (clean_token(t), round(a / scale, 3))
        for t, a in zip(ig["tokens"], ig["attributions"], strict=True)
        if t not in {"<s>", "</s>", "[CLS]", "[SEP]"}
    ]
    labels = {EMOTIONS[j]: float(probs[j]) for j in np.argsort(-probs)[:8]}
    summary = (
        f"**Predicted labels** (per-class thresholds tuned on dev): {', '.join(on) or 'none'}\n\n"
    )
    summary += f"Attributions below explain **{EMOTIONS[top]}** (the top non-neutral emotion)."
    return labels, summary, va_figure(probs), spans


with gr.Blocks(title="Circumplex: emotion detection demo") as demo:
    gr.Markdown("# Circumplex\nMulti-label emotion detection trained on GoEmotions.")
    gr.Markdown(DISCLAIMER)
    inp = gr.Textbox(label="Text", lines=3, placeholder="e.g. Thanks so much, this made my day!")
    btn = gr.Button("Analyse", variant="primary")
    with gr.Row():
        out_labels = gr.Label(label="Top emotions (probability)", num_top_classes=8)
        out_plot = gr.Plot(label="Valence / arousal")
    out_md = gr.Markdown()
    out_attr = gr.HighlightedText(
        label="Token attributions (Integrated Gradients)", combine_adjacent=False
    )
    gr.Examples(
        [
            "Thanks so much, this made my day!",
            "I can't believe they cancelled it again. Unreal.",
            "Wait, what? How does that even work?",
            "I'm so sorry for your loss.",
        ],
        inputs=inp,
    )
    btn.click(analyse, inp, [out_labels, out_md, out_plot, out_attr])
    inp.submit(analyse, inp, [out_labels, out_md, out_plot, out_attr])

if __name__ == "__main__":
    demo.launch()
