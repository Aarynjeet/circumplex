import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from circumplex.config import TrainConfig
from circumplex.data import EncodedDataset, make_collate
from circumplex.models.bilstm import BiLSTMClassifier
from circumplex.train import FocalLoss, build_loss, compute_pos_weight, fit, linear_warmup_decay


def _tiny_loader(n=8, T=6, vocab=30, seed=0):
    rng = np.random.default_rng(seed)
    ids = [rng.integers(2, vocab, size=rng.integers(2, T + 1)).tolist() for _ in range(n)]
    Y = np.zeros((n, 28), dtype=np.float32)
    for i in range(n):
        Y[i, rng.choice(28, size=rng.integers(1, 3), replace=False)] = 1
    return DataLoader(EncodedDataset(ids, Y), batch_size=n, collate_fn=make_collate(0))


def test_loop_overfits_a_batch_of_8():
    """If the loop can't memorise 8 examples, something in it is broken."""
    loader = _tiny_loader()
    model = BiLSTMClassifier(vocab_size=30, emb_dim=32, hidden_dim=32, dropout=0.0)
    cfg = TrainConfig(
        epochs=300, lr=1e-2, warmup_ratio=0.0, weight_decay=0.0, amp=False, grad_accum_steps=1
    )
    loss_fn = nn.BCEWithLogitsLoss()
    batch = next(iter(loader))
    with torch.no_grad():
        start = loss_fn(model(batch["input_ids"], batch["attention_mask"]), batch["labels"]).item()
    fit(model, loader, None, loss_fn, cfg, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        end = loss_fn(model(batch["input_ids"], batch["attention_mask"]), batch["labels"]).item()
    assert start > 0.5
    assert end < 0.02, f"loss only went from {start:.3f} to {end:.3f}"


def test_grad_accumulation_matches_full_batch():
    """2 micro-batches of 4 with accumulation == 1 batch of 8 (same single update)."""
    ids, Y = [[2, 3, 4]] * 4 + [[5, 6]] * 4, np.eye(28, dtype=np.float32)[:8]
    ds = EncodedDataset(ids, Y)
    results = []
    for bs, accum in ((8, 1), (4, 2)):
        torch.manual_seed(0)
        model = BiLSTMClassifier(vocab_size=10, emb_dim=8, hidden_dim=8, dropout=0.0)
        loader = DataLoader(ds, batch_size=bs, collate_fn=make_collate(0))
        cfg = TrainConfig(
            epochs=1, lr=0.1, warmup_ratio=0.0, amp=False, grad_accum_steps=accum, max_grad_norm=1e9
        )
        fit(model, loader, None, nn.BCEWithLogitsLoss(), cfg, torch.device("cpu"))
        results.append(model.head.weight.detach().clone())
    torch.testing.assert_close(results[0], results[1], rtol=1e-4, atol=1e-5)


def test_warmup_then_linear_decay():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = linear_warmup_decay(opt, warmup_steps=2, total_steps=6)
    lrs = []
    for _ in range(6):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs == pytest.approx([0.5, 1.0, 1.0, 0.75, 0.5, 0.25])


def test_pos_weight_is_neg_over_pos_capped():
    y = np.zeros((10, 28), dtype=np.float32)
    y[:5, 0] = 1  # 5 pos / 5 neg -> 1
    y[:1, 1] = 1  # 1 pos / 9 neg -> 9
    pw = compute_pos_weight(y, cap=30.0)
    assert pw[0] == pytest.approx(1.0) and pw[1] == pytest.approx(9.0)
    assert pw[2] == pytest.approx(10.0)  # no positives: 10 / max(0, 1)
    assert compute_pos_weight(y, cap=5.0)[1] == pytest.approx(5.0)


def test_focal_with_gamma_zero_equals_bce():
    logits, y = torch.randn(4, 28), (torch.rand(4, 28) > 0.8).float()
    torch.testing.assert_close(FocalLoss(gamma=0.0)(logits, y), nn.BCEWithLogitsLoss()(logits, y))
    assert FocalLoss(gamma=2.0)(logits, y) < nn.BCEWithLogitsLoss()(logits, y)


def test_build_loss_options():
    y = np.eye(28, dtype=np.float32)
    assert isinstance(build_loss(TrainConfig(loss="bce"), y), nn.BCEWithLogitsLoss)
    assert isinstance(
        build_loss(TrainConfig(loss="focal", pos_weight="inverse_freq"), y), FocalLoss
    )
    with pytest.raises(ValueError):
        build_loss(TrainConfig(loss="mse"), y)
