"""Pretrained transformer encoder + a hand-written multi-label head, with optional LoRA.

The encoder comes from Hugging Face ``transformers`` (weights and tokenizer only;
no ``Trainer``). The classification head is ours, so the same ``[CLS]`` vector
is used for prediction, embedding extraction (RQ3) and attribution.

Shape legend: B = batch, T = tokens, D = hidden size (768 for roberta-base), L = 28.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertConfig,
    BertModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from circumplex.config import LoraConfig

TINY = "tiny-random"


class TransformerClassifier(nn.Module):
    """``encoder -> hidden state of the first token ([CLS] / <s>) -> dropout -> linear``."""

    def __init__(self, encoder: nn.Module, num_labels: int = 28, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_labels)

    def encode(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the first-token vector of the last layer, shape (B, D)."""
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds
        )
        return out.last_hidden_state[:, 0]  # (32, 64, 768) -> (32, 768)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits (B, num_labels). Pass ``inputs_embeds`` instead of ids for IG."""
        pooled = self.encode(input_ids, attention_mask, inputs_embeds)  # (32, 768)
        return self.head(self.dropout(pooled))  # (32, 768) @ (768, 28) -> (32, 28)

    @torch.no_grad()
    def all_layer_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-layer pooled features for probing.

        Returns:
            ``cls`` and ``mean`` tensors of shape (n_layers + 1, B, D): the first
            token and the mask-averaged token vectors at every layer, where
            layer 0 is the embedding output.
        """
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        hs = torch.stack(out.hidden_states)  # (13, 32, 64, 768) for a 12-layer model
        cls = hs[:, :, 0]  # (13, 32, 768)
        m = attention_mask[None, :, :, None].to(hs.dtype)  # (1, 32, 64, 1)
        mean = (hs * m).sum(2) / m.sum(2).clamp(min=1)  # (13, 32, 768)
        return cls, mean


def train_tiny_tokenizer(texts: Sequence[str], vocab_size: int = 2000) -> PreTrainedTokenizerFast:
    """Train a small WordPiece tokenizer on ``texts`` (offline; for smoke tests)."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(
        texts, trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special)
    )
    from tokenizers.processors import TemplateProcessing

    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def build_tokenizer(name: str, texts: Sequence[str] | None = None) -> PreTrainedTokenizerBase:
    """Load a pretrained tokenizer, or train a tiny one if ``name == "tiny-random"``.

    ``name`` may also be a local directory saved with ``save_pretrained``.
    """
    if name == TINY:
        if texts is None:
            raise ValueError("tiny-random tokenizer needs training texts")
        return train_tiny_tokenizer(texts)
    return AutoTokenizer.from_pretrained(name)


def build_encoder(name: str, vocab_size: int | None = None, random_init: bool = False) -> nn.Module:
    """Build the encoder.

    Args:
        name: HF model id, or ``"tiny-random"`` for a 2-layer, 32-dim BERT.
        vocab_size: Needed for ``tiny-random`` (size of the trained tokenizer).
        random_init: Same architecture as ``name`` but randomly initialised
            (the RQ3 control: what structure exists without any pretraining?).
    """
    if name == TINY:
        cfg = BertConfig(
            vocab_size=vocab_size or 2000,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=256,
        )
        return BertModel(cfg, add_pooling_layer=False)
    if random_init:
        return AutoModel.from_config(AutoConfig.from_pretrained(name), add_pooling_layer=False)
    return AutoModel.from_pretrained(name, add_pooling_layer=False)


def apply_lora(encoder: nn.Module, cfg: LoraConfig) -> nn.Module:
    """Freeze the encoder and insert trainable low-rank adapters (``peft``).

    For each targeted ``nn.Linear`` with weight W (d_out x d_in), LoRA adds
    ``(alpha / r) * B @ A`` with A: (r x d_in), B: (d_out x r), B initialised to 0,
    so training starts exactly from the pretrained model.
    """
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
    )
    return get_peft_model(encoder, peft_cfg)


def save_tokenizer(tokenizer: PreTrainedTokenizerBase, run_dir: Path) -> None:
    """Save the tokenizer next to the checkpoint so the run is self-contained."""
    tokenizer.save_pretrained(str(run_dir / "tokenizer"))
