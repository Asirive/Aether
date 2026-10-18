from __future__ import annotations

import torch
import torch.nn.functional as F

from aether.model import AetherConfig, AetherForCausalLM


def test_chunked_causal_loss_matches_full_logits_loss():
    """The memory-saving chunked loss must equal the full-logits loss exactly.

    Regression guard: flattening batch and sequence together pairs the last
    token of one row with the first token of the next, which the full-logits
    path (`logits[..., :-1]` vs `labels[..., 1:]`) never does.
    """
    torch.manual_seed(0)
    config = AetherConfig(
        vocab_size=128,
        d_model=32,
        d_inner=64,
        n_layers=2,
        n_heads=4,
        d_k=8,
        max_seq_len=64,
        mod_routing_points=(),
        mod_keep_ratios=(),
    )
    model = AetherForCausalLM(config).to(torch.float64).eval()

    hidden = torch.randn(3, 37, config.d_model, dtype=torch.float64)
    labels = torch.randint(0, config.vocab_size, (3, 37))
    labels[1, 10:15] = -100

    with torch.no_grad():
        logits = model._compute_logits(hidden)
        reference = F.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        for chunk in (8, 16, 512):
            got = model._chunked_causal_loss(hidden, labels, chunk_tokens=chunk)
            assert torch.allclose(got, reference, atol=1e-10), (
                f"chunk={chunk}: chunked={got.item():.10f} full={reference.item():.10f}"
            )


def test_chunked_causal_loss_handles_short_and_all_masked():
    torch.manual_seed(1)
    config = AetherConfig(
        vocab_size=64,
        d_model=32,
        d_inner=64,
        n_layers=2,
        n_heads=4,
        d_k=8,
        max_seq_len=32,
        mod_routing_points=(),
        mod_keep_ratios=(),
    )
    model = AetherForCausalLM(config).to(torch.float64).eval()

    hidden = torch.randn(1, 1, config.d_model, dtype=torch.float64)
    labels = torch.zeros(1, 1, dtype=torch.long)
    out = model._chunked_causal_loss(hidden, labels)
    assert torch.isfinite(out) and out.item() == 0.0

    hidden = torch.randn(2, 9, config.d_model, dtype=torch.float64)
    masked = torch.full((2, 9), -100, dtype=torch.long)
    out = model._chunked_causal_loss(hidden, masked)
    assert torch.isfinite(out) and out.item() == 0.0


def test_aether_overfit_single_sequence():
    """A properly-functioning LM should be able to memorize ONE sequence."""
    torch.manual_seed(0)
    config = AetherConfig(
        vocab_size=64,
        d_model=128,
        d_inner=256,
        n_layers=4,
        n_heads=4,
        d_k=32,
        max_seq_len=128,
        mod_routing_points=(2, 4),
        mod_keep_ratios=(0.5, 1.0),
        polymamba_orders=2,
        polymamba_state_dim=8,
    )
    model = AetherForCausalLM(config).to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    seq = torch.randint(0, config.vocab_size, (1, 16))
    seq = seq.to(torch.long)

    initial_loss = model(seq, labels=seq)["loss"].item()
    target = max(0.5, initial_loss * 0.05)
    for step in range(150):
        out = model(seq, labels=seq)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    final_loss = model(seq, labels=seq)["loss"].item()
    assert final_loss < target, (
        f"Failed to overfit single sequence: init {initial_loss:.3f} → final {final_loss:.3f}, target < {target:.3f}"
    )


def test_aether_full_model_causality():
    """THE regression test: changing future tokens must not change past logits."""
    torch.manual_seed(0)
    config = AetherConfig(
        vocab_size=128,
        d_model=64,
        d_inner=128,
        n_layers=4,
        n_heads=4,
        d_k=16,
        max_seq_len=64,
        mod_routing_points=(2,),
        mod_keep_ratios=(1.0,),
    )
    model = AetherForCausalLM(config).eval()
    ids = torch.randint(0, 128, (1, 48))
    with torch.no_grad():
        out1 = model(ids)["logits"]
    for cut in (1, 7, 16, 24, 33, 47):
        ids2 = ids.clone()
        ids2[:, cut:] = torch.randint(0, 128, (1, 48 - cut))
        with torch.no_grad():
            out2 = model(ids2)["logits"]
        diff = (out1[:, :cut] - out2[:, :cut]).abs().max().item()
        assert diff < 2e-4, f"cut={cut}: past logits changed by {diff} — FUTURE LEAK"


def test_aether_weight_init_distributions():
    """Linear weights should init with roughly the standard deviation predicted by init scheme."""
    torch.manual_seed(42)
    config = AetherConfig(
        vocab_size=128,
        d_model=64,
        d_inner=256,
        n_layers=4,
        n_heads=4,
        d_k=16,
        max_seq_len=64,
    )
    model = AetherForCausalLM(config)

    linear_stds = []
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() == 2 and "norm" not in name.lower():
            linear_stds.append(p.std().item())

    assert all(s > 1e-4 for s in linear_stds), f"Some linear weights init to zero: {linear_stds}"


def test_aether_tied_embeddings_save_params():
    """Tied embeddings should produce fewer params than untied."""
    tied = AetherConfig(tie_embeddings=True)
    untied = AetherConfig(tie_embeddings=False)
    tied_model = AetherForCausalLM(tied)
    untied_model = AetherForCausalLM(untied)
    diff = untied_model.num_parameters() - tied_model.num_parameters()
    assert diff >= tied.vocab_size * tied.d_model, (
        f"Tied should save {tied.vocab_size * tied.d_model} params, saved {diff}"
    )


def test_aether_reproducibility_under_seed():
    """Same seed + same input → identical output."""
    config = AetherConfig(
        vocab_size=128, d_model=64, d_inner=128, n_layers=4, n_heads=4, d_k=16, max_seq_len=64,
    )
    torch.manual_seed(123)
    m1 = AetherForCausalLM(config)
    torch.manual_seed(123)
    m2 = AetherForCausalLM(config)

    x = torch.randint(0, config.vocab_size, (2, 32))
    torch.manual_seed(99)
    with torch.no_grad():
        y1 = m1(x)["logits"]
    torch.manual_seed(99)
    with torch.no_grad():
        y2 = m2(x)["logits"]
    err = (y1 - y2).abs().max().item()
    assert err == 0, f"Same seed → different output: err {err}"


def test_aether_loss_only_uses_masked_tokens():
    """Standard CE with ignore_index=-100 should treat those tokens as ignored."""
    torch.manual_seed(0)
    config = AetherConfig(
        vocab_size=64, d_model=64, d_inner=128, n_layers=4, n_heads=4, d_k=16, max_seq_len=64,
    )
    model = AetherForCausalLM(config)
    x = torch.randint(0, 64, (1, 8))
    labels_full = x.clone()
    labels_masked = x.clone()
    labels_masked[0, -3:] = -100
    with torch.no_grad():
        out_full = model(x, labels=labels_full)["loss"].item()
        out_masked = model(x, labels=labels_masked)["loss"].item()
    assert abs(out_full - out_masked) > 1e-3, (
        f"Masking labels should change loss significantly: full {out_full:.4f} vs masked {out_masked:.4f}"
    )


def test_aether_param_grad_all_reachable():
    """Every parameter should receive gradient after a single training step."""
    torch.manual_seed(0)
    config = AetherConfig(
        vocab_size=64, d_model=64, d_inner=128, n_layers=4, n_heads=4, d_k=16, max_seq_len=64,
        mod_routing_points=(1, 3),
        mod_keep_ratios=(0.5, 1.0),
    )
    model = AetherForCausalLM(config)
    x = torch.randint(0, 64, (2, 16))
    out = model(x, labels=x)
    out["loss"].backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"
            assert p.grad.abs().sum() > 0, f"{name} grad is all zero"


def test_aether_rmsnorm_weight_init_ones():
    """RMSNorm weight should initialize to ones (identity transform)."""
    torch.manual_seed(0)
    config = AetherConfig(d_model=64, d_inner=128, n_layers=4, n_heads=4, d_k=16, max_seq_len=64)
    model = AetherForCausalLM(config)
    for name, p in model.named_parameters():
        if "norm" in name.lower() and "weight" in name and p.dim() == 1:
            assert torch.allclose(p, torch.ones_like(p), atol=1e-6), f"{name} not init to ones"


def test_aether_init_logits_for_gate_and_mask():
    """Auxiliary param logits (gate, mask_logits, mix_gate) should init small."""
    torch.manual_seed(0)
    config = AetherConfig(d_model=64, d_inner=128, n_layers=4, n_heads=4, d_k=16, max_seq_len=64)
    model = AetherForCausalLM(config)
    for name, p in model.named_parameters():
        if "gate.weight" in name and p.dim() == 2:
            assert p.std() < 0.05, f"{name} std {p.std()} should be small (~0.01)"
        if "mask_logits" in name:
            assert 0.005 < p.std() < 0.05, (
                f"{name} std {p.std()} should be ~0.02 (docs/architecture.md:59, AGENTS.md)"
            )
        if "mix_gate" in name:
            assert torch.allclose(p, torch.zeros_like(p), atol=1e-6), f"{name} should init to 0"


def test_aether_forward_loss_gradient_all_finite():
    """Under bf16, all gradients should remain finite (no NaN/Inf)."""
    torch.manual_seed(7)
    config = AetherConfig(
        vocab_size=128, d_model=128, d_inner=256, n_layers=4, n_heads=4, d_k=32, max_seq_len=128,
    )
    model = AetherForCausalLM(config).to(torch.bfloat16)
    x = torch.randint(0, 128, (2, 64))
    out = model(x, labels=x)
    out["loss"].backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"


def test_aether_pretrain_sanity_decreases_loss_50_steps():
    """50 steps of bf16 training on synthetic data should decrease loss."""
    torch.manual_seed(0)
    config = AetherConfig(
        vocab_size=256,
        d_model=128,
        d_inner=256,
        n_layers=4,
        n_heads=4,
        d_k=32,
        max_seq_len=128,
        mod_routing_points=(2, 4),
        mod_keep_ratios=(0.5, 1.0),
        polymamba_orders=2,
        polymamba_state_dim=8,
    )
    model = AetherForCausalLM(config).to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    seq = torch.randint(0, 256, (4, 64))
    initial_loss = model(seq, labels=seq)["task_loss"].item()
    for _ in range(50):
        out = model(seq, labels=seq)
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    final_loss = model(seq, labels=seq)["task_loss"].item()
    assert final_loss < initial_loss * 0.95, (
        f"Loss did not decrease enough: {initial_loss:.3f} → {final_loss:.3f}"
    )
