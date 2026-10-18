"""Scale-bound instrument: is every activation/weight norm bounded by construction?

Trains a tiny AETHER (d_model=64, 4 layers, tiny vocab) on cheap learnable synthetic
data for ~400 steps on CPU and logs, every step:

  * per-layer residual-stream RMS            (res_rms_i)
  * block-local forward gain out/in           (block_gain_i)
  * ||token_embedding.weight||_F              (embed_weight_norm)  [tied head]
  * logit RMS and max|logit|                  (logit_rms / logit_absmax)
  * every RMSNorm gain norm                   (norm1_i / norm2_i / final / route_norm)
  * each mixer's output RMS                   (mixer_rms_i)
  * FSA mask_logits norm, route_proj norm, grad norm

CPU-only, single thread, < 3 minutes.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn as nn

torch.set_num_threads(1)

from aether.model import AetherConfig, AetherForCausalLM
from aether.training.optim import build_optimizer

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------- config
VOCAB = 64
SEQ = 32
BATCH = int(os.environ.get("SCALE_BATCH", "8"))
D_MODEL = 64
D_INNER = 192
N_LAYERS = 4
HEADS = 4
D_K = 16
STEPS = int(os.environ.get("SCALE_STEPS", "400"))
LR = 5e-3
WD = 0.1
EMBED_LR_SCALE = 0.1   # post-15-Sep config
WARMUP = 50
SEED = 0


def rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())


def build_cfg() -> AetherConfig:
    return AetherConfig(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        d_inner=D_INNER,
        n_layers=N_LAYERS,
        n_heads=HEADS,
        d_k=D_K,
        max_seq_len=SEQ + 2,
        tie_embeddings=True,
        fsa_topk_ratio=0.25,
        fsa_ortho_loss_weight=0.01,
        polymamba_orders=2,
        polymamba_state_dim=8,
        polymamba_kbal_loss_weight=0.01,
        mod_routing_points=(1, 3),
        mod_keep_ratios=(0.5, 0.85),
        mod_loss_weight=0.01,
    )


# ------------------------------------------------- learnable synthetic data
def make_transition() -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    t = torch.randint(0, VOCAB, (VOCAB,), generator=g)
    return t


def sample_batch(trans: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Hard-but-learnable trigram task: next = (7*a + 13*b + 3) % V with 20% noise.

    A tiny model cannot memorise the 64x64 map in 400 steps, so the loss keeps
    falling throughout the run -- the regime in which scale drift is visible.
    """
    ids = torch.empty(BATCH, SEQ + 2, dtype=torch.long)
    ids[:, 0] = torch.randint(0, VOCAB, (BATCH,), generator=gen)
    ids[:, 1] = torch.randint(0, VOCAB, (BATCH,), generator=gen)
    for t in range(SEQ):
        a = ids[:, t]
        b = ids[:, t + 1]
        nxt = (a * 7 + b * 13 + 3) % VOCAB
        rand = torch.randint(0, VOCAB, (BATCH,), generator=gen)
        flip = torch.rand(BATCH, generator=gen) < 0.20
        ids[:, t + 2] = torch.where(flip, rand, nxt)
    return ids


# ---------------------------------------------------------------- hooks
class Recorder:
    def __init__(self, model: AetherForCausalLM) -> None:
        self.model = model
        self.reset()
        self.handles = []
        self._register()

    def reset(self) -> None:
        self.block_in: dict[int, torch.Tensor] = {}
        self.block_out: dict[int, torch.Tensor] = {}
        self.mixer_out: dict[int, torch.Tensor] = {}
        self.gate: dict[int, torch.Tensor] = {}
        self.embed_out: torch.Tensor | None = None
        self.norm_in: torch.Tensor | None = None
        self._current_gate: torch.Tensor | None = None

    def _register(self) -> None:
        m = self.model

        def emb_hook(_mod, _inp, out):
            self.embed_out = out

        self.handles.append(m.token_embedding.register_forward_hook(emb_hook))

        def router_hook(_mod, _inp, out):
            # out = (x, loss, state, gate)
            g = out[3]
            if g is not None:
                self._current_gate = g

        self.handles.append(m.mod_router.register_forward_hook(router_hook))

        def norm_hook(_mod, inp, _out):
            self.norm_in = inp[0]

        self.handles.append(m.norm.register_forward_hook(norm_hook))

        def make_block_hook(i):
            def hook(_mod, inp, out):
                self.block_in[i] = inp[0]
                self.block_out[i] = out[0]
                self.gate[i] = self._current_gate
            return hook

        for i, blk in enumerate(m.blocks):
            self.handles.append(blk.register_forward_hook(make_block_hook(i)))
            self.handles.append(
                blk.mixer.register_forward_hook(
                    lambda _m, _i, o, i=i: self.mixer_out.__setitem__(i, o)
                )
            )

    def record(self, logits: torch.Tensor, loss: float, task_loss: float) -> dict:
        m = self.model
        rec: dict[str, float] = {"loss": loss, "task_loss": task_loss}

        # residual stream RMS through the stack
        rec["res_rms_in"] = rms(self.embed_out) if self.embed_out is not None else float("nan")
        prev = self.embed_out
        for i in range(N_LAYERS):
            g = self.gate[i]
            if g is not None:
                res = g * self.block_out[i] + (1.0 - g) * self.block_in[i]
            else:
                res = self.block_out[i]
            rec[f"res_rms_{i}"] = rms(res)
            rec[f"block_gain_{i}"] = rms(self.block_out[i]) / max(rms(self.block_in[i]), 1e-12)
            rec[f"mixer_rms_{i}"] = rms(self.mixer_out[i])
            prev = res
        rec["_reconstruction_err"] = abs(float(prev.detach().float().sub(self.norm_in.detach().float()).abs().max()))

        # tied embedding / head
        rec["embed_weight_norm"] = float(m.token_embedding.weight.detach().float().norm())
        rec["logit_rms"] = rms(logits)
        rec["logit_absmax"] = float(logits.detach().float().abs().max())

        # RMSNorm gains
        for i, blk in enumerate(m.blocks):
            rec[f"norm1_{i}"] = float(blk.norm1.weight.detach().float().norm())
            rec[f"norm2_{i}"] = float(blk.norm2.weight.detach().float().norm())
        rec["norm_final"] = float(m.norm.weight.detach().float().norm())
        rec["route_norm"] = float(m.mod_router.route_norm_weight.detach().float().norm())
        rec["route_proj"] = float(m.mod_router.route_proj.weight.detach().float().norm())

        # FSA mask logits norm
        fsa_norms = []
        for blk in m.blocks:
            if hasattr(blk.mixer, "mask_logits"):
                fsa_norms.append(float(blk.mixer.mask_logits.detach().float().norm()))
        rec["fsa_mask_logits"] = sum(fsa_norms)
        return rec


# ---------------------------------------------------------------- main
def main() -> None:
    torch.manual_seed(SEED)
    cfg = build_cfg()
    model = AetherForCausalLM(cfg).to(DEVICE)
    # give CC-FSA a real cross-chunk path at this short seq_len (chunk 16 -> 2 chunks)
    from aether.components.fsa import FreqSparseAttention
    for blk in model.blocks:
        if isinstance(blk.mixer, FreqSparseAttention):
            blk.mixer.chunk_size = 16
            blk.mixer.keep_bins = max(1, int(16 * cfg.fsa_topk_ratio))
            blk.mixer.mask_logits = nn.Parameter(torch.randn(HEADS, 16) * 1.0)
            blk.mixer._cross_mask_cache.clear()
            blk.mixer._intra_mask_cache.clear()

    ablate = os.environ.get("SCALE_ABLATE", "")
    if ablate == "route_decay":
        import aether.training.optim as _opt
        _opt._NO_DECAY_SUBSTRINGS = tuple(
            s for s in _opt._NO_DECAY_SUBSTRINGS if s != "route_proj"
        )
        print("ABLATION: route_proj moved into the weight-decay group")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"tiny model params: {n_params:,}")

    opt = build_optimizer(model, lr=LR, weight_decay=WD, fused=False, embed_lr_scale=EMBED_LR_SCALE)
    for g in opt.param_groups:
        print(f"  param group: n={len(g['params'])} wd={g['weight_decay']} lr={g['lr']:.2e}")

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / WARMUP if s < WARMUP else 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, STEPS - WARMUP))),
    )

    rec = Recorder(model)
    trans = make_transition()
    gen = torch.Generator().manual_seed(SEED)

    model.train()
    rows: list[dict] = []
    for step in range(STEPS + 1):
        ids = sample_batch(trans, gen)
        inp = ids[:, :-1].contiguous()
        lab = ids[:, 1:].contiguous()

        out = model(inp, labels=lab, return_logits=True)
        loss = out["loss"]
        logits = out["logits"]

        row = rec.record(logits, float(loss.detach()), float(out["task_loss"].detach()))
        row["step"] = step
        row["lr"] = opt.param_groups[0]["lr"]

        if step < STEPS:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()), float("inf")))
            row["grad_norm"] = gnorm
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            opt.step()
            sched.step()
        rows.append(row)

    # -------------------------------------------------- trend analysis
    recon_err = max(r["_reconstruction_err"] for r in rows if "_reconstruction_err" in r)
    print(f"\nresidual-reconstruction check: max|reconstructed - true pre-final-norm residual| = {recon_err:.3e}")

    keys = [
        k for k in rows[0]
        if k not in ("step", "lr") and not k.startswith("_")
        and isinstance(rows[0][k], float)
        and all(k in r and math.isfinite(r[k]) for r in rows)
    ]

    def slope_last(series, w=100):
        y = series[-w:]
        n = len(y)
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(y) / n
        num = sum((x - mx) * (v - my) for x, v in zip(xs, y))
        den = sum((x - mx) ** 2 for x in xs)
        return num / den

    def mono_frac(series):
        d = [series[i + 1] - series[i] for i in range(len(series) - 1)]
        return sum(1 for x in d if x > 0) / max(1, len(d))

    print(f"\n================ TREND SUMMARY (step 0 -> {STEPS}) ================")
    print("'still climbing' = slope over the last 100 steps is positive AND the last-100 value "
          "is above the mid-run value.")
    print(f"{'quantity':>18} {'start':>10} {'mid(200)':>10} {'end':>10} {'x(end/start)':>12} {'mono%':>6} "
          f"{'slope/50':>11} {'still_climb':>11}")
    summary = []
    for k in keys:
        s = [r[k] for r in rows]
        start, mid, end = s[0], s[len(s) // 2], s[-1]
        growth = end / start if start != 0 else float("inf")
        mono = mono_frac(s) * 100
        sl = slope_last(s) * 50
        climbing = (sl > 0) and (s[-1] > s[-100])
        summary.append((k, start, mid, end, growth, mono, sl, climbing))
        print(f"{k:>18} {start:>10.4g} {mid:>10.4g} {end:>10.4g} {growth:>12.2f} {mono:>6.1f} "
              f"{sl:>11.2e} {str(climbing):>11}")

    print("\nRanked by growth factor (end/start):")
    for k, start, mid, end, growth, mono, sl, climbing in sorted(
        summary, key=lambda t: t[4], reverse=True
    )[:14]:
        print(f"  {k:>18}  x{growth:>10.2f}   mono {mono:>5.1f}%   "
              f"{'CLIMBING' if climbing else 'saturating/declining'}")

    # checkpoint table
    steps_to_show = list(range(0, STEPS + 1, max(1, STEPS // 8)))
    print("\n================ CHECKPOINT TABLE (selected quantities) ================")
    show_keys = [
        "loss", "res_rms_0", "res_rms_1", "res_rms_2", "res_rms_3",
        "block_gain_0", "block_gain_1", "block_gain_2", "block_gain_3",
        "mixer_rms_0", "mixer_rms_1", "mixer_rms_2", "mixer_rms_3",
        "embed_weight_norm", "logit_rms", "logit_absmax",
        "norm_final", "route_norm", "route_proj", "fsa_mask_logits", "grad_norm",
    ]
    hdr = "step  " + "  ".join(f"{k[:9]:>9}" for k in show_keys)
    print(hdr)
    for st in steps_to_show:
        r = rows[st]
        line = f"{st:>4}  " + "  ".join(
            (f"{r[k]:>9.3g}" if k in r and math.isfinite(r[k]) else f"{'--':>9}") for k in show_keys
        )
        print(line)

    import json
    outdir = os.path.dirname(os.path.abspath(__file__))
    tag = os.environ.get("SCALE_TAG", f"s{STEPS}_b{BATCH}")
    path = os.path.join(outdir, f"scale_bounds_rows_{tag}.json")
    with open(path, "w") as f:
        json.dump(rows, f)
    print(f"\nraw per-step rows -> {path}")


if __name__ == "__main__":
    main()
