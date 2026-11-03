"""Integrated Gradients for the transformer, implemented directly in PyTorch.

IG (Sundararajan et al., 2017) attributes the logit of class c to each input
token. With input embeddings x, a baseline x' (here: the embedding of the pad
token at every position) and m interpolation steps::

    IG_i = (x_i - x'_i) * mean_{k=1..m} dF_c(x' + (k/m)(x - x')) / dx_i

summed over the embedding dimension to give one score per token. The
"completeness" property says sum_i IG_i ~= F_c(x) - F_c(x'), which
:func:`integrated_gradients` reports as a sanity check.
"""

from __future__ import annotations

import html

import numpy as np
import torch

from circumplex.labels import EMOTIONS


def integrated_gradients(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    target: int | str,
    steps: int = 50,
    max_len: int = 64,
    device: torch.device | None = None,
) -> dict:
    """Token attributions for ``text`` toward label ``target`` (index or name).

    Returns:
        ``tokens``, ``attributions`` (one float per token), the target
        ``logit``, the ``baseline_logit`` and ``completeness_gap``
        (|sum IG - (F(x) - F(x'))|; small means the Riemann sum is accurate).
    """
    device = device or next(model.parameters()).device
    c = EMOTIONS.index(target) if isinstance(target, str) else target
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    emb_layer = model.encoder.get_input_embeddings()
    model.eval()
    with torch.no_grad():
        x = emb_layer(ids)  # (1, T, D)
        baseline_ids = torch.full_like(ids, tokenizer.pad_token_id)
        baseline_ids[0, 0], baseline_ids[0, -1] = ids[0, 0], ids[0, -1]  # keep <s> ... </s>
        x0 = emb_layer(baseline_ids)  # (1, T, D)

    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device)  # right Riemann sum
    path = x0 + alphas[:, None, None] * (x - x0)  # (steps, T, D)
    path.requires_grad_(True)
    logits = model(None, mask.expand(steps, -1), inputs_embeds=path)[:, c]  # (steps,)
    (grads,) = torch.autograd.grad(logits.sum(), path)  # (steps, T, D)
    ig = ((x - x0)[0] * grads.mean(0)).sum(-1)  # (T,)

    with torch.no_grad():
        f_x = model(None, mask, inputs_embeds=x)[0, c].item()
        f_x0 = model(None, mask, inputs_embeds=x0)[0, c].item()
    attributions = ig.detach().cpu().numpy()
    return {
        "tokens": tokenizer.convert_ids_to_tokens(ids[0].tolist()),
        "attributions": attributions.tolist(),
        "target": EMOTIONS[c],
        "logit": f_x,
        "baseline_logit": f_x0,
        "completeness_gap": float(abs(attributions.sum() - (f_x - f_x0))),
    }


def clean_token(tok: str) -> str:
    """Make BPE/WordPiece tokens readable (RoBERTa marks word starts with 'Ġ')."""
    return tok.replace("Ġ", " ").replace("Ċ", " ").replace("##", "")


def to_html(result: dict) -> str:
    """Render attributions as inline HTML: green = pushes toward the label, red = away."""
    attr = np.asarray(result["attributions"])
    scale = np.abs(attr).max() or 1.0
    spans = []
    for tok, a in zip(result["tokens"], attr, strict=True):
        if tok in {"<s>", "</s>", "[CLS]", "[SEP]", "<pad>", "[PAD]"}:
            continue
        alpha = abs(a) / scale
        color = f"rgba(22,163,74,{alpha:.2f})" if a >= 0 else f"rgba(220,38,38,{alpha:.2f})"
        spans.append(
            f'<span style="background:{color};padding:1px 2px;border-radius:3px">{html.escape(clean_token(tok))}</span>'
        )
    head = f"<b>{html.escape(result['target'])}</b> (logit {result['logit']:.2f}): "
    return f"<p>{head}{''.join(spans)}</p>"
