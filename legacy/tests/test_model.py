from __future__ import annotations

import torch

from aether.model import AetherConfig, AetherForCausalLM


def test_aether_forward_shape():
    config = AetherConfig(
        vocab_size=1024,
        d_model=64,
        d_inner=128,
        n_layers=4,
        n_heads=4,
        d_k=16,
        max_seq_len=64,
        mod_routing_points=(2, 4),
        mod_keep_ratios=(0.5, 1.0),
    )
    model = AetherForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    out = model(input_ids)
    assert out["logits"].shape == (2, 32, config.vocab_size)


def test_aether_loss_computation():
    config = AetherConfig(
        vocab_size=1024,
        d_model=64,
        d_inner=128,
        n_layers=4,
        n_heads=4,
        d_k=16,
        max_seq_len=64,
        mod_routing_points=(2, 4),
        mod_keep_ratios=(0.5, 1.0),
    )
    model = AetherForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    assert "loss" in out
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() > 0.0


def test_aether_loss_decreases():
    config = AetherConfig(
        vocab_size=1024,
        d_model=64,
        d_inner=128,
        n_layers=4,
        n_heads=4,
        d_k=16,
        max_seq_len=64,
        mod_routing_points=(2, 4),
        mod_keep_ratios=(0.5, 1.0),
    )
    model = AetherForCausalLM(config)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    labels = input_ids.clone()

    initial_loss = model(input_ids, labels=labels)["loss"].item()
    for _ in range(20):
        out = model(input_ids, labels=labels)
        optim.zero_grad()
        out["loss"].backward()
        optim.step()
    final_loss = model(input_ids, labels=labels)["loss"].item()
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} → {final_loss}"


def test_aether_param_count_within_budget():
    config = AetherConfig()
    n = _approx_full_param_count(config)
    assert 100_000_000 < n < 220_000_000, (
        f"Param count {n:,} outside 100M-220M range"
    )


def _approx_full_param_count(config):
    embedding = config.vocab_size * config.d_model
    mask_logits = config.n_heads * 32
    mix_gate = config.n_heads
    attn = 4 * config.d_model * config.d_model
    ffn = 3 * config.d_model * config.d_inner
    polymamba_in_out = 3 * config.d_model * config.d_inner
    polymamba_gates = config.polymamba_orders * config.d_model
    polymamba_state = config.polymamba_orders * (
        config.polymamba_state_dim
        + 2 * config.polymamba_state_dim * config.d_inner
        + config.d_inner
    )
    polymamba = polymamba_in_out + polymamba_gates + polymamba_state
    fsa_block = attn + ffn
    pm_block = polymamba + ffn
    n_fsa = config.n_layers // 2
    n_pm = config.n_layers - n_fsa
    head = 0 if config.tie_embeddings else config.vocab_size * config.d_model
    return (
        embedding
        + n_fsa * fsa_block
        + n_pm * pm_block
        + mask_logits
        + mix_gate
        + head
    )


def test_mod_unselected_tokens_bypass_block_span():
    torch.manual_seed(0)
    config = AetherConfig(
        vocab_size=256,
        d_model=64,
        d_inner=128,
        n_layers=2,
        n_heads=4,
        d_k=16,
        max_seq_len=64,
        mod_routing_points=(0,),
        mod_keep_ratios=(0.5,),
    )
    model = AetherForCausalLM(config).eval()
    ids = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        out = model(ids)
        gate = model._last_router_state["mod_mask_0"]
        emb = model.token_embedding(ids)
        expected_hidden = model.norm(emb)
        expected_logits = model._compute_logits(expected_hidden)
    unselected = (gate == 0).nonzero(as_tuple=True)[0].tolist()
    assert len(unselected) > 0, "test requires at least one unselected token"
    diff = (out["logits"][0, unselected] - expected_logits[0, unselected]).abs().max().item()
    assert diff < 1e-4, (
        f"unselected tokens changed across gated span: diff {diff} (zero-poisoning regression)"
    )


def test_mod_keep_all_equals_no_router():
    torch.manual_seed(0)
    config = AetherConfig(
        vocab_size=256, d_model=64, d_inner=128, n_layers=2, n_heads=4, d_k=16,
        max_seq_len=64, mod_routing_points=(0,), mod_keep_ratios=(1.0,),
    )
    model = AetherForCausalLM(config).eval()
    ids = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        out = model(ids)
        gate = model._last_router_state["mod_mask_0"]
    assert (gate == 1.0).all()


def test_aether_checkpoint_roundtrip(tmp_path):
    config = AetherConfig(
        vocab_size=1024,
        d_model=64,
        d_inner=128,
        n_layers=4,
        n_heads=4,
        d_k=16,
        max_seq_len=64,
    )
    model = AetherForCausalLM(config)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    from aether.utils.checkpoint import save_checkpoint, load_checkpoint

    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optim, step=10, path=path)
    state = load_checkpoint(path, model=model, optimizer=optim)
    assert state["step"] == 10
