"""Component-attribution ablation for AETHER (verify round 1, angle: is PolyMamba2
actually contributing, or is it dead weight?).

Trains tiny variants of AetherForCausalLM on small synthetic sequence-memory tasks and
compares loss curves, parameter counts and mixer output scale.  CPU only, tiny tensors.

Tasks (baked in, no downloads)
------------------------------
  lag       (primary): x_0..x_{L-1} iid uniform over V.  The model must output x_{t-k}
            at position t (k>=1), i.e. a pure delayed copy.  Dense, deterministic target:
            a *memoryless* position-wise mixer is at chance (ln V) by construction, a
            causal conv with kernel > k can solve it exactly, and a linear recurrence can
            implement a delay line.  Standard content-based attention with NO positional
            encoding cannot select a fixed offset and is also at chance; the same attention
            WITH RoPE can.  (AETHER itself has no positional encoding anywhere --
            `rope_base` is a config field that is never read -- so the position-sensitive
            behaviour of each mixer is exactly what this task measures.)
  induction (cross-check): one random block of n=L//repeats tokens repeated `repeats`
            times; target = next token.  Content-based induction, position-free, so it is
            the fair task for FSA vs attention.  Block 0 is unpredictable (floor ln V),
            later blocks are copies.

Arms
----
    aether        full AetherForCausalLM (PolyMamba2 even / FSA odd / MoD on)
    nomod         full model, MoD routing disabled entirely (gate == 1)
    linear        PolyMamba2 (even) -> param-matched gated linear (no recurrence)
    conv          PolyMamba2 (even) -> param-matched causal depthwise conv (cheap memory)
    attn          FSA (odd) -> standard causal MHSA, no positional encoding
    attn_rope     FSA (odd) -> standard causal MHSA + RoPE
    allattn       both mixer families -> causal MHSA (no PE)  [control transformer]
    allattn_rope  both mixer families -> causal MHSA + RoPE   [reference transformer]
    matched       linear + attn swaps with each replacement's OUTPUT SCALE matched to the
                  component it replaces (fair-comparison control)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

from aether.components.fsa import FreqSparseAttention
from aether.components.polymamba import PolyMamba2
from aether.model import AetherConfig, AetherForCausalLM


# --------------------------------------------------------------------------------------
# Replacement mixers
# --------------------------------------------------------------------------------------
class _AuxZeroMixin:
    """AetherBlock.forward() calls either balance_loss() or orthogonality_loss()."""

    def balance_loss(self) -> torch.Tensor:
        return torch.zeros(())

    def orthogonality_loss(self) -> torch.Tensor:
        return getattr(self, "_zero", torch.zeros(()))


class GatedLinearMixer(_AuxZeroMixin, nn.Module):
    """Position-wise gated linear: y = W_o(silu(W_g x) * W_v x). No sequence mixing."""

    def __init__(self, d_model: int, d_inner: int, scale: float = 1.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, g = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(g) * v) * self.scale


class CausalConvMixer(_AuxZeroMixin, nn.Module):
    """Gated depthwise causal conv1d + linear out: a cheap mixer WITH memory."""

    def __init__(self, d_model: int, d_inner: int, kernel: int = 8, scale: float = 1.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.dwconv = nn.Conv1d(d_inner, d_inner, kernel, groups=d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.kernel = kernel
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, g = self.in_proj(x).chunk(2, dim=-1)
        y = F.silu(g) * v
        y = F.pad(y.transpose(1, 2), (self.kernel - 1, 0))
        y = self.dwconv(y)[..., : x.size(1)]
        return self.out_proj(y.transpose(1, 2)) * self.scale


def _rope_tables(d_k: int, seq_len: int, base: float, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                       # (L, d_k/2)
    cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
    return cos, sin


def _apply_rope(x, cos, sin):
    # x: (B, H, L, d_k); rotate consecutive pairs
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class CausalSelfAttention(_AuxZeroMixin, nn.Module):
    """Standard multi-head causal attention (scaled dot-product, tril mask)."""

    def __init__(self, d_model: int, n_heads: int, d_k: int, scale: float = 1.0,
                 rope: bool = False, max_seq_len: int = 512, rope_base: float = 10_000.0) -> None:
        super().__init__()
        assert d_model == n_heads * d_k
        self.n_heads, self.d_k, self.rope = n_heads, d_k, rope
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_o = nn.Linear(n_heads * d_k, d_model, bias=False)
        self.scale = scale
        self._mask: torch.Tensor | None = None
        self._rope: tuple | None = None
        if rope:
            cos, sin = _rope_tables(d_k, max_seq_len, rope_base, torch.device("cpu"), torch.float32)
            self.register_buffer("_cos", cos, persistent=False)
            self.register_buffer("_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q = self.w_q(x).view(b, l, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, l, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, l, self.n_heads, self.d_k).transpose(1, 2)
        if self.rope:
            cos, sin = self._cos[:l].to(x.dtype), self._sin[:l].to(x.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self._mask is None or self._mask.shape[-1] != l or self._mask.device != x.device:
            self._mask = torch.tril(torch.ones(l, l, dtype=torch.bool, device=x.device))
        s = s.float().masked_fill(~self._mask, -1e9)
        a = F.softmax(s, dim=-1).to(x.dtype)
        o = (a @ v).transpose(1, 2).contiguous().view(b, l, -1)
        return self.w_o(o) * self.scale


# --------------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------------
def _arm_touches(arm: str, which: str) -> bool:
    if which == "even":
        return arm in ("linear", "conv", "matched", "allattn", "allattn_rope")
    return arm in ("attn", "attn_rope", "matched", "allattn", "allattn_rope")


def param_matched_inner(d_model: int, target_params: int) -> int:
    """Width h for GatedLinearMixer (3*d_model*h params) ~= target."""
    return max(8, int(round(target_params / (3.0 * d_model))))


def build_model(arm: str, base_cfg: AetherConfig, seed: int, scale_map: dict | None,
                conv_kernel: int):
    torch.manual_seed(seed)
    cfg = AetherConfig(**{k: getattr(base_cfg, k) for k in (
        "vocab_size", "d_model", "d_inner", "n_layers", "n_heads", "d_k", "max_seq_len",
        "rope_base", "tie_embeddings", "fsa_topk_ratio", "fsa_ortho_loss_weight",
        "polymamba_orders", "polymamba_state_dim", "polymamba_kbal_loss_weight",
        "mod_routing_points", "mod_keep_ratios", "mod_loss_weight")})
    if arm == "nomod":
        cfg.mod_routing_points = ()
        cfg.mod_keep_ratios = ()
        cfg.mod_loss_weight = 0.0

    model = AetherForCausalLM(cfg)
    scales = scale_map or {}

    if _arm_touches(arm, "even"):
        ref = PolyMamba2(cfg.d_model, cfg.d_inner, cfg.polymamba_orders, cfg.polymamba_state_dim)
        target = sum(p.numel() for p in ref.parameters())
        h = param_matched_inner(cfg.d_model, target)
        for blk in model.blocks:
            if isinstance(blk.mixer, PolyMamba2):
                if arm == "conv":
                    nm: nn.Module = CausalConvMixer(cfg.d_model, cfg.d_inner, kernel=conv_kernel)
                elif arm in ("allattn", "allattn_rope"):
                    nm = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.d_k,
                                             rope=(arm == "allattn_rope"),
                                             max_seq_len=cfg.max_seq_len)
                else:
                    nm = GatedLinearMixer(cfg.d_model, h)
                nm.scale = scales.get("even", 1.0)
                blk.mixer = nm

    if _arm_touches(arm, "odd"):
        for blk in model.blocks:
            if isinstance(blk.mixer, FreqSparseAttention):
                nm = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.d_k,
                                         scale=scales.get("odd", 1.0),
                                         rope=(arm in ("attn_rope", "allattn_rope")),
                                         max_seq_len=cfg.max_seq_len)
                blk.mixer = nm

    for blk in model.blocks:
        blk.mixer._zero = torch.zeros(())
    return model


# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------
def make_lag_data(vocab: int, n_seq: int, seq_len: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n_seq, seq_len), generator=g)


def make_induction_data(vocab: int, n_seq: int, seq_len: int, repeats: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = seq_len // repeats
    return torch.randint(0, vocab, (n_seq, n), generator=g).repeat(1, repeats)


def task_ce(model, ids: torch.Tensor, task: str, lag: int, repeats: int):
    """Return (mean_loss, per_position_ce (B, P)) where P = supervised positions."""
    logits = model(ids)["logits"].float()
    if task == "lag":
        pred = logits[:, lag:, :]
        tgt = ids[:, : ids.size(1) - lag]
    else:
        pred = logits[:, :-1, :]
        tgt = ids[:, 1:]
    ce = F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="none")
    return ce.mean(), ce.view(tgt.size(0), tgt.size(1)), tgt


def region_metrics(model, ids, task, lag, repeats) -> dict:
    with torch.no_grad():
        mean, ce, tgt = task_ce(model, ids, task, lag, repeats)
        res = {"eval_loss": float(mean.item())}
        if task == "lag":
            p = ce.size(1)
            res["eval_recent"] = float(ce[:, : p // 2].mean().item())      # early positions
            res["eval_late"] = float(ce[:, p // 2:].mean().item())
        else:
            n = ids.size(1) // repeats
            res["eval_first_half"] = float(ce[:, : n - 1].mean().item())
            res["eval_copy"] = float(ce[:, n - 1:].mean().item())
    return res


# --------------------------------------------------------------------------------------
# Train / eval one run
# --------------------------------------------------------------------------------------
def run_one(arm, seed, args, train_data, eval_data, outdir: Path) -> dict:
    base_cfg = AetherConfig(
        vocab_size=args.vocab, d_model=args.d_model, d_inner=args.d_inner,
        n_layers=args.layers, n_heads=args.heads, d_k=args.dk,
        max_seq_len=args.seq_len, fsa_topk_ratio=0.25,
        polymamba_orders=2, polymamba_state_dim=args.state_dim,
        mod_routing_points=(1, 3), mod_keep_ratios=(0.5, 0.7),
        mod_loss_weight=0.01, fsa_ortho_loss_weight=0.001, polymamba_kbal_loss_weight=0.01,
    )

    scale_map = None
    if arm == "matched":
        probe = torch.randn(args.batch, args.seq_len, args.d_model,
                            generator=torch.Generator().manual_seed(1234))
        probe = probe / probe.pow(2).mean(-1, keepdim=True).sqrt()
        torch.manual_seed(seed)
        base = AetherForCausalLM(base_cfg)
        pm = next(b.mixer for b in base.blocks if isinstance(b.mixer, PolyMamba2))
        fs = next(b.mixer for b in base.blocks if isinstance(b.mixer, FreqSparseAttention))
        with torch.no_grad():
            ref_even = float(pm(probe).float().std())
            ref_odd = float(fs(probe).float().std())
        ref = PolyMamba2(base_cfg.d_model, base_cfg.d_inner, base_cfg.polymamba_orders,
                         base_cfg.polymamba_state_dim)
        h = param_matched_inner(base_cfg.d_model, sum(p.numel() for p in ref.parameters()))
        torch.manual_seed(seed)
        lin = GatedLinearMixer(base_cfg.d_model, h)
        torch.manual_seed(seed)
        att = CausalSelfAttention(base_cfg.d_model, base_cfg.n_heads, base_cfg.d_k)
        with torch.no_grad():
            scale_map = {"even": ref_even / max(float(lin(probe).float().std()), 1e-8),
                         "odd": ref_odd / max(float(att(probe).float().std()), 1e-8)}

    model = build_model(arm, base_cfg, seed, scale_map, args.conv_kernel)
    n_params = model.num_parameters()

    # realised mixer output RMS (diagnostic for "matched output scale")
    with torch.no_grad():
        probe = torch.randn(4, args.seq_len, args.d_model, generator=torch.Generator().manual_seed(5))
        probe = probe / probe.pow(2).mean(-1, keepdim=True).sqrt()
        rms = {}
        for nm in ("even", "odd"):
            blk = next((b for b in model.blocks
                        if (isinstance(b.mixer, PolyMamba2) and nm == "even")
                        or (isinstance(b.mixer, FreqSparseAttention) and nm == "odd")), None)
            rms[nm] = None if blk is None else round(float(blk.mixer(probe).float().std()), 4)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    curve = []
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        ids = train_data[step]
        loss, _, _ = task_ce(model, ids, args.task, args.lag, args.repeats)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            r = region_metrics(model, eval_data, args.task, args.lag, args.repeats)
            r.update(step=step, train_loss=float(loss.item()))
            curve.append(r)
            print(f"  [{arm} s{seed}] step {step:5d} eval {r['eval_loss']:.4f} "
                  f"train {loss.item():.4f}", flush=True)
            model.train()

    final = region_metrics(model, eval_data, args.task, args.lag, args.repeats)
    res = {
        "arm": arm, "seed": seed, "task": args.task, "params": n_params,
        "mixer_rms": rms, "scale_map": scale_map,
        "final_train_loss": sum(c["train_loss"] for c in curve[-6:]) / len(curve[-6:]),
        **{f"final_{k}": v for k, v in final.items()},
        "curve": curve, "seconds": round(time.time() - t0, 1),
    }
    (outdir / f"{args.task}_{arm}_seed{seed}.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["lag", "induction"], default="lag")
    p.add_argument("--lag", type=int, default=6)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--vocab", type=int, default=16)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-inner", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dk", type=int, default=16)
    p.add_argument("--state-dim", type=int, default=16)
    p.add_argument("--conv-kernel", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--arms", type=str, nargs="+",
                   default=["aether", "nomod", "linear", "conv", "attn", "attn_rope", "allattn_rope"])
    p.add_argument("--outdir", type=str, default="ablations/results/05_component_attribution")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.task == "lag":
        mk = lambda seed: make_lag_data(args.vocab, args.steps * args.batch, args.seq_len, seed)
    else:
        mk = lambda seed: make_induction_data(args.vocab, args.steps * args.batch, args.seq_len,
                                              args.repeats, seed)
    train_data = mk(777).view(args.steps, args.batch, args.seq_len)
    eval_data = (make_lag_data(args.vocab, 32, args.seq_len, 888) if args.task == "lag"
                 else make_induction_data(args.vocab, 32, args.seq_len, args.repeats, 888))

    results = []
    for arm in args.arms:
        for seed in args.seeds:
            results.append(run_one(arm, seed, args, train_data, eval_data, outdir))

    (outdir / f"summary_{args.task}.json").write_text(json.dumps(results, indent=2))
    print(f"\n=== {args.task.upper()} SUMMARY (mean over seeds) ===")
    keys = [k for k in results[0] if k.startswith("final_eval")]
    print(f"{'arm':13s} {'params':>9s} " + " ".join(f"{k[6:]:>13s}" for k in keys) + f" {'spread':>7s}")
    for arm in args.arms:
        rs = [r for r in results if r["arm"] == arm]
        vals = [sum(r[k] for r in rs) / len(rs) for k in keys]
        spread = max(r["final_eval_loss"] for r in rs) - min(r["final_eval_loss"] for r in rs)
        print(f"{arm:13s} {rs[0]['params']:>9,} " + " ".join(f"{v:>13.4f}" for v in vals)
              + f" {spread:>7.4f}")


if __name__ == "__main__":
    main()
