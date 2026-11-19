from __future__ import annotations

from pathlib import Path

import torch
import yaml

from aether.components.attention import CausalSelfAttention
from aether.model import AetherConfig, AetherForCausalLM

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _tiny(**kw) -> AetherConfig:
    base = dict(
        vocab_size=97, d_model=64, d_inner=128, n_layers=4, n_heads=4, d_k=16,
        max_seq_len=64, mod_routing_points=(2,), mod_keep_ratios=(0.5,), arch="vanilla",
    )
    base.update(kw)
    return AetherConfig(**base)


def test_vanilla_blocks_are_all_attention_and_no_router():
    model = AetherForCausalLM(_tiny())
    assert all(isinstance(b.mixer, CausalSelfAttention) for b in model.blocks)
    assert model.mod_router is None
    assert model.config.mod_routing_points == ()
    assert not any("mod_router" in n for n, _ in model.named_parameters())


def test_vanilla_aux_losses_are_zero():
    torch.manual_seed(0)
    model = AetherForCausalLM(_tiny())
    ids = torch.randint(0, 97, (2, 32))
    out = model(ids, labels=ids)
    for v in out["aux_losses"].values():
        assert float(v) == 0.0
    assert torch.allclose(out["loss"], out["task_loss"])


def test_attention_is_strictly_causal():
    """Changing tokens at positions > t must not move outputs at positions <= t."""
    torch.manual_seed(0)
    model = AetherForCausalLM(_tiny()).double().eval()
    ids = torch.randint(0, 97, (1, 40))
    cut = 23
    ids2 = ids.clone()
    ids2[0, cut + 1 :] = torch.randint(0, 97, (40 - cut - 1,))
    with torch.no_grad():
        a = model(ids)["logits"][0, : cut + 1]
        b = model(ids2)["logits"][0, : cut + 1]
    assert (a - b).abs().max().item() < 1e-10


def test_rope_preserves_norm_and_is_relative():
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=32, n_heads=2, d_k=16, max_seq_len=64).double()
    attn.rope_cos = attn.rope_cos.double()
    attn.rope_sin = attn.rope_sin.double()
    x = torch.randn(1, 2, 64, 16, dtype=torch.float64)
    y = attn._rope(x)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-10)
    # q_i . k_j depends only on i - j when q and k are the same vector at every position.
    q = torch.randn(16, dtype=torch.float64).expand(1, 1, 64, 16)
    k = torch.randn(16, dtype=torch.float64).expand(1, 1, 64, 16)
    rq, rk = attn._rope(q.clone())[0, 0], attn._rope(k.clone())[0, 0]
    assert torch.allclose(rq[10] @ rk[3], rq[40] @ rk[33], atol=1e-10)


def test_vanilla_learns_successor_task():
    """Next-token = (token + 1) mod V. Also guards against the B23 double-shift bug."""
    torch.manual_seed(0)
    model = AetherForCausalLM(_tiny(vocab_size=16, n_layers=2))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        start = torch.randint(0, 16, (8, 1))
        ids = (start + torch.arange(32)) % 16
        loss = model(ids, labels=ids)["task_loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_baseline_config_differs_from_aether_only_where_intended():
    aether = yaml.safe_load((CONFIGS / "aether_200m.yaml").read_text())
    base = yaml.safe_load((CONFIGS / "baseline_200m.yaml").read_text())
    assert base["model"].pop("arch") == "vanilla"
    assert base["mod"]["routing_points"] == [] and base["mod"]["keep_ratios"] == []
    for d in (aether, base):
        d["mod"].pop("routing_points"), d["mod"].pop("keep_ratios"), d["output"].pop("dir")
    assert aether == base


def test_aether_arch_unchanged_by_default():
    cfg = _tiny(arch="aether")
    model = AetherForCausalLM(cfg)
    assert not any(isinstance(b.mixer, CausalSelfAttention) for b in model.blocks)
    assert model.mod_router is not None
