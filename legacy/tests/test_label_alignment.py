from __future__ import annotations

import torch

from aether.model import AetherConfig, AetherForCausalLM


def _ring_batch(vocab: int, total: int, seq_len: int) -> torch.Tensor:
    starts = torch.randint(0, vocab, (total, 1))
    pos = starts + torch.arange(seq_len).view(1, -1)
    return pos % vocab


def test_model_learns_one_ahead_not_two_ahead():
    """On a deterministic successor ring (i -> i+1), the trained model must
    predict the 1-ahead token, not the 2-ahead token.

    Regression guard for the 2026-08-09 label double-shift bug: the data loader
    pre-shifted labels AND model.forward shifted again, turning pretraining into
    a 2-ahead objective. With the old code this test's 1-ahead accuracy was ~0%.
    """
    torch.manual_seed(0)
    vocab = 64
    cfg = AetherConfig(
        vocab_size=vocab,
        d_model=64,
        d_inner=128,
        n_layers=2,
        n_heads=2,
        d_k=32,
        max_seq_len=64,
        tie_embeddings=True,
        mod_routing_points=(),
        mod_keep_ratios=(),
    )
    model = AetherForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    seq_len = 16
    steps = 220
    for _ in range(steps):
        ids = _ring_batch(vocab, total=4, seq_len=seq_len)
        out = model(ids, labels=ids.clone())
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    singles = torch.arange(vocab).view(-1, 1)
    with torch.no_grad():
        logits = model(singles)["logits"][:, -1, :]
    pred = logits.argmax(dim=-1)
    gold = (torch.arange(vocab) + 1) % vocab
    acc1 = (pred == gold).float().mean().item()
    acc2 = (pred == (torch.arange(vocab) + 2) % vocab).float().mean().item()
    assert acc1 > 0.9, f"expected 1-ahead accuracy > 0.9, got {acc1:.3f}"
    assert acc2 < 0.3, f"model still predicts 2-ahead (acc2={acc2:.3f})"
