import torch

from circumplex.config import LoraConfig
from circumplex.models.bilstm import BiLSTMClassifier
from circumplex.models.transformer import TransformerClassifier, apply_lora, build_encoder


def _batch(lengths, T, vocab=50):
    ids = torch.zeros(len(lengths), T, dtype=torch.long)
    mask = torch.zeros(len(lengths), T, dtype=torch.long)
    for i, n in enumerate(lengths):
        ids[i, :n] = torch.randint(2, vocab, (n,))
        mask[i, :n] = 1
    return ids, mask


def test_bilstm_output_shape_both_poolings():
    ids, mask = _batch([5, 3, 8, 1], T=8)
    for pooling in ("attention", "max"):
        model = BiLSTMClassifier(vocab_size=50, emb_dim=16, hidden_dim=12, pooling=pooling)
        assert model(ids, mask).shape == (4, 28)
        assert model.encode(ids, mask).shape == (4, 24)  # 2 directions x 12


def test_bilstm_ignores_padding():
    """Adding extra pad positions must not change the output (packing + masking work)."""
    model = BiLSTMClassifier(vocab_size=50, emb_dim=16, hidden_dim=12).eval()
    ids, mask = _batch([4], T=4)
    ids_padded = torch.cat([ids, torch.zeros(1, 6, dtype=torch.long)], 1)
    mask_padded = torch.cat([mask, torch.zeros(1, 6, dtype=torch.long)], 1)
    torch.testing.assert_close(model(ids, mask), model(ids_padded, mask_padded))


def test_transformer_forward_and_lora_param_count():
    enc = build_encoder("tiny-random", vocab_size=100)
    model = TransformerClassifier(enc, 28)
    ids, mask = _batch([5, 7], T=7, vocab=100)
    assert model(ids, mask).shape == (2, 28)

    lora = TransformerClassifier(
        apply_lora(build_encoder("tiny-random", vocab_size=100), LoraConfig(r=4)), 28
    )
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    # per adapted 32x32 matrix: A (4x32) + B (32x4) = 256; 2 layers x {query, value} = 4 matrices
    head = 32 * 28 + 28
    assert trainable == 4 * 256 + head
