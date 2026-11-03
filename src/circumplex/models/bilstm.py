"""A bidirectional LSTM classifier written from scratch.

Shape legend used in comments: B = batch size, T = padded sequence length,
E = embedding dim, H = LSTM hidden size per direction, L = number of labels (28).
Example numbers assume B=32, T=64, E=300, H=256, L=28.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Learned attention over time steps: a weighted average of the LSTM outputs.

    Each time step gets a score ``w . tanh(W h_t)``; padding positions get
    -inf so softmax gives them weight 0.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool ``h`` (B, T, D) with ``mask`` (B, T) -> pooled (B, D), weights (B, T)."""
        scores = self.score(torch.tanh(self.proj(h))).squeeze(-1)  # (32, 64)
        scores = scores.masked_fill(mask == 0, float("-inf"))  # (32, 64), pads -> -inf
        weights = torch.softmax(scores, dim=-1)  # (32, 64), each row sums to 1
        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(
            1
        )  # (32, 1, 64)@(32, 64, 512) -> (32, 512)
        return pooled, weights


class BiLSTMClassifier(nn.Module):
    """Embedding -> BiLSTM -> attention (or max) pooling -> dropout -> linear (28 logits)."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int = 28,
        emb_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        pooling: str = "attention",
        pad_idx: int = 0,
        pretrained_embeddings: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if pooling not in {"attention", "max"}:
            raise ValueError(f"pooling must be 'attention' or 'max', got {pooling!r}")
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(pretrained_embeddings))
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = AttentionPooling(2 * hidden_dim) if pooling == "attention" else None
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_dim, num_labels)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the pooled sentence vector, shape (B, 2H)."""
        x = self.embedding(input_ids)  # (32, 64) -> (32, 64, 300)
        lengths = attention_mask.sum(dim=1).cpu()  # (32,) true lengths, e.g. [17, 9, 64, ...]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )  # packing lets the LSTM skip pad positions entirely
        out, _ = self.lstm(packed)  # packed sequence of (sum(lengths), 512)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=input_ids.size(1)
        )  # (32, 64, 512): forward (256) and backward (256) states concatenated
        if self.attn is not None:
            pooled, _ = self.attn(h, attention_mask)  # (32, 512)
        else:
            h = h.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))  # (32, 64, 512)
            pooled = h.max(dim=1).values  # (32, 512): max over time per feature
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, num_labels); apply sigmoid for probabilities."""
        pooled = self.encode(input_ids, attention_mask)  # (32, 512)
        return self.head(self.dropout(pooled))  # (32, 512) @ (512, 28) + b -> (32, 28)


def load_glove(path: str, itos: list[str], dim: int, seed: int = 0) -> tuple[np.ndarray, float]:
    """Build an embedding matrix from a GloVe ``.txt`` file.

    Words not in GloVe get small random vectors; <pad> (index 0) gets zeros.

    Returns:
        The (vocab_size, dim) matrix and the fraction of vocabulary covered.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, size=(len(itos), dim)).astype(np.float32)
    matrix[0] = 0.0
    index = {w: i for i, w in enumerate(itos)}
    hits = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            word, *values = line.rstrip().split(" ")
            if word in index and len(values) == dim:
                matrix[index[word]] = np.asarray(values, dtype=np.float32)
                hits += 1
    return matrix, hits / len(itos)
