"""Print one example's journey through a model: text -> ids -> tensors -> logits -> labels.

Used to fill in the worked traces in LEARNING.md with real numbers.

Usage::

    python scripts/trace_example.py --run runs/bilstm
    python scripts/trace_example.py --run runs/roberta_full_s42
    python scripts/trace_example.py --untrained-bilstm   # offline: real vocab, random weights
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from circumplex.data import Vocab, load_goemotions, simple_tokenize
from circumplex.labels import EMOTIONS
from circumplex.models.bilstm import BiLSTMClassifier
from circumplex.train import load_trained

DEFAULT_TEXT = "Thank you so much, this made my day!"


def fmt(t: torch.Tensor, k: int = 6) -> str:
    v = t.detach().flatten()[:k].tolist()
    return "[" + ", ".join(f"{x:.4f}" for x in v) + (", ...]" if t.numel() > k else "]")


@torch.no_grad()
def trace_bilstm(model: BiLSTMClassifier, vocab: Vocab, text: str, thresholds: np.ndarray) -> None:
    """Walk through BiLSTMClassifier.forward step by step, printing shapes and values."""
    model.eval()
    toks = simple_tokenize(text)
    ids = vocab.encode(text, 64)
    print("tokens        :", toks)
    print("token ids     :", ids)
    x_ids = torch.tensor([ids])
    mask = torch.ones_like(x_ids)
    print("input_ids     :", tuple(x_ids.shape))
    emb = model.embedding(x_ids)
    print("embeddings    :", tuple(emb.shape), "first token, first dims:", fmt(emb[0, 0]))
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        emb, mask.sum(1), batch_first=True, enforce_sorted=False
    )
    out, _ = model.lstm(packed)
    h, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    print("LSTM output   :", tuple(h.shape), "(forward and backward halves concatenated)")
    if model.attn is not None:
        pooled, w = model.attn(h, mask)
        print("attn weights  :", {t: round(float(a), 3) for t, a in zip(toks, w[0], strict=False)})
    else:
        pooled = h.max(1).values
    print("pooled        :", tuple(pooled.shape), fmt(pooled))
    _finish(model.head(pooled), thresholds)


@torch.no_grad()
def trace_transformer(model, tok, text: str, thresholds: np.ndarray) -> None:
    """Walk through TransformerClassifier.forward, printing shapes and values."""
    enc = tok(text, return_tensors="pt")
    print("tokens        :", tok.convert_ids_to_tokens(enc["input_ids"][0]))
    print("token ids     :", enc["input_ids"][0].tolist())
    emb = model.encoder.get_input_embeddings()(enc["input_ids"])
    print("word embeds   :", tuple(emb.shape), fmt(emb[0, 0]))
    out = model.encoder(**enc, output_hidden_states=True)
    print("hidden states :", len(out.hidden_states), "x", tuple(out.hidden_states[-1].shape))
    cls = out.last_hidden_state[:, 0]
    print("[CLS]/<s> vec :", tuple(cls.shape), fmt(cls))
    _finish(model.head(cls), thresholds)


def _finish(logits: torch.Tensor, thresholds: np.ndarray) -> None:
    probs = torch.sigmoid(logits)[0].numpy()
    print("logits        :", tuple(logits.shape))
    print(
        "  all 28      :", {e: round(float(v), 2) for e, v in zip(EMOTIONS, logits[0], strict=True)}
    )
    print(
        "sigmoid top-5 :", {EMOTIONS[j]: round(float(probs[j]), 4) for j in np.argsort(-probs)[:5]}
    )
    on = np.flatnonzero(probs >= thresholds)
    print("thresholds    :", {EMOTIONS[j]: float(thresholds[j]) for j in np.argsort(-probs)[:5]})
    print(
        f"predicted     : {len(on)} of {len(probs)} labels at or above their threshold:",
        [EMOTIONS[j] for j in on],
    )


def main() -> None:
    """Trace a trained run or an untrained BiLSTM on the real vocabulary."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run")
    ap.add_argument("--untrained-bilstm", action="store_true")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    args = ap.parse_args()
    if args.untrained_bilstm:
        torch.manual_seed(0)
        vocab = Vocab.build(load_goemotions("github")["train"].texts, min_freq=2)
        print(
            f"(untrained BiLSTM, seed 0; vocabulary of {len(vocab):,} words from the real train set)"
        )
        trace_bilstm(BiLSTMClassifier(len(vocab)), vocab, args.text, np.full(28, 0.5))
        return
    run = Path(args.run)
    model, tok, cfg = load_trained(run, torch.device("cpu"))
    thresholds = np.array(
        list(json.loads((run / "metrics.json").read_text())["thresholds"].values())
    )
    print(f"run: {run.name}  text: {args.text!r}")
    if cfg.model.type == "bilstm":
        trace_bilstm(model, tok, args.text, thresholds)
    else:
        trace_transformer(model, tok, args.text, thresholds)


if __name__ == "__main__":
    main()
