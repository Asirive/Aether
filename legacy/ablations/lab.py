"""Component lab: factorial ablations of AETHER's components on CPU.

One model skeleton (pre-norm blocks, SwiGLU FFN, RMSNorm, tied embeddings, the same
init as `aether.model`) whose three architectural choices are swapped independently:

    odd blocks   : attn  (causal MHSA + RoPE)      | fsa  (CC-FSA, original)  | v2 variants
    even blocks  : attn  (causal MHSA + RoPE)      | pm   (PolyMamba-2, orig) | v2 variants
    depth routing: none                            | mod  (MoD-GDR, original) | v2 variants

`attn/attn/none` is the vanilla baseline; `fsa/pm/mod` is AETHER as shipped. Training
mirrors the real recipe (AdamW 0.9/0.95, wd 0.1 with the repo's no-decay groups, linear
warmup + cosine to 10%, grad clip 1.0, aux-loss weights and warmup ramps, FSA temperature
and Gumbel schedules from `aether.training.curriculum`).

Data: byte-level Python standard-library source, split train/val *by file*. Primary
metric: held-out loss in bits per byte. Also reported: a causality probe (does changing
future tokens move past logits?), throughput, and synthetic recall tasks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import sysconfig
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from aether.components.attention import CausalSelfAttention  # noqa: E402
from aether.components.fsa import FreqSparseAttention  # noqa: E402
from aether.components.mod_router import MoDRouter  # noqa: E402
from aether.components.polymamba import PolyMamba2  # noqa: E402
from aether.training.curriculum import fsa_gumbel_noise, fsa_temperature, linear_ramp_weight  # noqa: E402
from aether.training.optim import build_scheduler, _split_param_groups  # noqa: E402

EOS = 256
VOCAB = 257
USE_AMP = True


def AMP():
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=USE_AMP)

# Registry of mixers / routers. v2 components register themselves from aether.components.v2.
MIXERS: dict[str, callable] = {}
ROUTERS: dict[str, callable] = {}


def register_mixer(name):
    def deco(fn):
        MIXERS[name] = fn
        return fn
    return deco


def register_router(name):
    def deco(fn):
        ROUTERS[name] = fn
        return fn
    return deco


@dataclass
class Dims:
    d_model: int = 256
    d_inner: int = 768          # FFN width (and PolyMamba inner width, as in Aether)
    n_layers: int = 8
    n_heads: int = 4
    d_k: int = 64
    seq_len: int = 256


@register_mixer("attn")
def _attn(d: Dims, layer: int):
    return CausalSelfAttention(d.d_model, d.n_heads, d.d_k, max_seq_len=d.seq_len)


@register_mixer("fsa")
def _fsa(d: Dims, layer: int):
    return FreqSparseAttention(d.d_model, d.n_heads, d.d_k, max_seq_len=d.seq_len, topk_ratio=0.25)


@register_mixer("pm")
def _pm(d: Dims, layer: int):
    return PolyMamba2(d.d_model, d.d_inner, n_orders=2, state_dim=16)


from aether.components.v2 import (  # noqa: E402
    CausalDepthRouter, ChunkCompressedAttention, SelectiveSSM, matched_ssm_inner,
)


@register_mixer("fsa2")
def _fsa2(d: Dims, layer: int):
    return ChunkCompressedAttention(d.d_model, d.n_heads, d.d_k, max_seq_len=d.seq_len)


@register_mixer("pm2")
def _pm2(d: Dims, layer: int):
    # Parameter-matched to one attention mixer (4 d^2).
    return SelectiveSSM(d.d_model, matched_ssm_inner(d.d_model))


@register_mixer("pm2nc")
def _pm2nc(d: Dims, layer: int):
    """Ablation: selective SSM without the short causal conv (kernel 1 = per-channel affine)."""
    return SelectiveSSM(d.d_model, matched_ssm_inner(d.d_model, conv_k=1), conv_k=1)


@register_mixer("pm2fdt")
def _pm2fdt(d: Dims, layer: int):
    """Ablation: step size dt is a learned per-head constant (not input-dependent)."""
    return SelectiveSSM(d.d_model, matched_ssm_inner(d.d_model), selective_dt=False)


class GatedConvMixer(nn.Module):
    """Gated depthwise causal conv (k=4) + projection, param-matched to attention. Baseline aid."""

    def __init__(self, d_model: int, kernel: int = 4) -> None:
        super().__init__()
        inner = round(4 * d_model * d_model / (3 * d_model + kernel + 1))
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, kernel, groups=inner, bias=True)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.k = kernel

    def forward(self, x):
        v, g = self.in_proj(x).chunk(2, dim=-1)
        v = self.conv(F.pad(v.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2)
        return self.out_proj(F.silu(g) * v)


class SlidingWindowAttention(CausalSelfAttention):
    """Causal attention restricted to the last `window` positions (RoPE), as in LFM3's SWA layers."""

    def __init__(self, d_model, n_heads, d_k, max_seq_len, window):
        super().__init__(d_model, n_heads, d_k, max_seq_len=max_seq_len)
        i = torch.arange(max_seq_len)
        mask = (i.view(-1, 1) >= i.view(1, -1)) & (i.view(-1, 1) - i.view(1, -1) < window)
        self.register_buffer("win_mask", mask, persistent=False)

    def forward(self, x):
        b, L, _ = x.shape

        def heads(t):
            return t.view(b, L, self.n_heads, self.d_k).transpose(1, 2)

        q = self._rope(heads(self.w_q(x)))
        k = self._rope(heads(self.w_k(x)))
        v = heads(self.w_v(x))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.win_mask[:L, :L])
        return self.w_o(out.transpose(1, 2).reshape(b, L, self.n_heads * self.d_k))


@register_mixer("swa")
def _swa(d: Dims, layer: int):
    # LFM3 runs SWA well below its 32K context; at our 256-token context use a 64-token window.
    return SlidingWindowAttention(d.d_model, d.n_heads, d.d_k, d.seq_len, window=64)


@register_mixer("conv")
def _conv(d: Dims, layer: int):
    return GatedConvMixer(d.d_model)


@register_router("mod2")
def _mod2(d: Dims):
    pts = tuple(p for p in (2, 4, 6) if p < d.n_layers)
    return CausalDepthRouter(d.d_model, pts, (0.50, 0.70, 1.00)[: len(pts)])


def _aux(m: nn.Module) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if isinstance(m, PolyMamba2):
        return m.balance_loss(), None
    if isinstance(m, FreqSparseAttention):
        return None, m.orthogonality_loss()
    return None, None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_inner: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_inner, bias=False)
        self.w_up = nn.Linear(d_model, d_inner, bias=False)
        self.w_down = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, mixer: nn.Module, d: Dims) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d.d_model)
        self.norm2 = RMSNorm(d.d_model)
        self.mixer = mixer
        self.ffn = SwiGLU(d.d_model, d.d_inner)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


# ---------------------------------------------------------------------------------------
# Original MoD-GDR, wired exactly as AetherForCausalLM.forward does: one shared router,
# called at each routing point; the returned gate then blends EVERY following block
# (it persists until the next routing point overwrites it).
# ---------------------------------------------------------------------------------------
class MoDv1(nn.Module):
    def __init__(self, d: Dims) -> None:
        super().__init__()
        pts = [p for p in (2, 4, 6) if p < d.n_layers]
        self.points = pts
        self.router = MoDRouter(d.d_model, tuple(pts), (0.50, 0.70, 1.00)[: len(pts)])
        self.weight = 0.01

    def run(self, blocks, x):
        loss = x.new_zeros(())
        gate = None
        for i, blk in enumerate(blocks):
            if i in self.points:
                _, rl, _, gate = self.router(x, layer_id=i)
                loss = loss + rl
            if gate is not None:
                x = gate * blk(x) + (1.0 - gate) * x
            else:
                x = blk(x)
        return x, loss

    def set_temperature(self, t):  # router temperature is fixed at 1.0 after PAF-9
        pass


@register_router("mod")
def _mod(d: Dims):
    return MoDv1(d)


@dataclass
class Spec:
    """'odd/even/route', or 'P:m0-m1-...-m{L-1}/route' for an explicit per-layer pattern."""
    odd: str = "attn"
    even: str = "attn"
    route: str = "none"
    pattern: tuple[str, ...] | None = None

    @property
    def name(self) -> str:
        if self.pattern is not None:
            return "P:" + "-".join(self.pattern) + f"/{self.route}"
        return f"{self.odd}/{self.even}/{self.route}"

    def mixer_for(self, layer: int) -> str:
        if self.pattern is not None:
            return self.pattern[layer]
        return self.even if layer % 2 == 0 else self.odd

    @classmethod
    def parse(cls, s: str) -> "Spec":
        if s.startswith("P:"):
            pat, r = s[2:].rsplit("/", 1)
            return cls(route=r, pattern=tuple(pat.split("-")))
        o, e, r = s.split("/")
        return cls(o, e, r)


class LabLM(nn.Module):
    def __init__(self, spec: Spec, d: Dims) -> None:
        super().__init__()
        self.spec, self.d = spec, d
        self.token_embedding = nn.Embedding(VOCAB, d.d_model)
        blocks = []
        for i in range(d.n_layers):
            kind = spec.mixer_for(i)
            blocks.append(Block(MIXERS[kind](d, i), d))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(d.d_model)
        if spec.pattern is not None and len(spec.pattern) != d.n_layers:
            raise ValueError(f"pattern has {len(spec.pattern)} layers, model has {d.n_layers}")
        self.router = ROUTERS[spec.route](d) if spec.route != "none" else None
        self._init_weights()

    def _init_weights(self) -> None:
        # Same rule as AetherForCausalLM._init_weights.
        std = 0.02
        residual_out = ("out_proj", "w_down", "w_o")
        block_std = std / math.sqrt(2.0 * self.d.n_layers)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)
        for blk in self.blocks:
            for name, mod in blk.named_modules():
                if isinstance(mod, nn.Linear):
                    if getattr(mod, "_no_reinit", False):
                        continue
                    s = block_std if name.endswith(residual_out) else std / (mod.weight.shape[1] ** 0.5)
                    nn.init.normal_(mod.weight, mean=0.0, std=s)
        for blk in self.blocks:
            if hasattr(blk.mixer, "post_init"):
                blk.mixer.post_init()

    def set_knobs(self, step: int, total: int) -> dict[str, float]:
        t = fsa_temperature(step, total, 0.10)
        g = fsa_gumbel_noise(step, total, 0.10)
        for blk in self.blocks:
            if isinstance(blk.mixer, FreqSparseAttention):
                blk.mixer._temperature, blk.mixer._gumbel_noise = t, g
        return {
            "fsa": 0.01 * linear_ramp_weight(step, total, 0.10),
            "pm": 0.01 * linear_ramp_weight(step, total, 0.05),
            "mod": 0.01 * linear_ramp_weight(step, total, 0.05),
        }

    def forward(self, ids, labels=None, aux_w: dict | None = None):
        x = self.token_embedding(ids)
        if self.router is not None:
            x, route_loss = self.router.run(self.blocks, x)
        else:
            for blk in self.blocks:
                x = blk(x)
            route_loss = x.new_zeros(())
        self._route_loss = route_loss
        x = self.norm(x)
        logits = F.linear(x, self.token_embedding.weight)
        if labels is None:
            return logits
        task = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB).float(), labels[:, 1:].reshape(-1))
        return task, task + self.aux_loss(aux_w)

    def aux_loss(self, aux_w: dict | None = None) -> torch.Tensor:
        """Weighted aux losses from the most recent forward pass."""
        aux_w = aux_w or {"fsa": 0.0, "pm": 0.0, "mod": 0.0}
        aux = self._route_loss * aux_w["mod"]
        for blk in self.blocks:
            kb, orth = _aux(blk.mixer)
            if kb is not None:
                aux = aux + aux_w["pm"] * kb
            if orth is not None:
                aux = aux + aux_w["fsa"] * orth
            if hasattr(blk.mixer, "aux_loss"):
                aux = aux + blk.mixer.aux_loss()
        if self.router is not None and hasattr(self.router, "aux_loss"):
            aux = aux + self.router.aux_loss()
        return aux

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------
_CORPUS: dict = {}


def corpus(max_bytes: int = 24_000_000, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    key = (max_bytes, seed)
    if key in _CORPUS:
        return _CORPUS[key]
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in stdlib.rglob("*.py")
                   if "site-packages" not in p.parts and "dist-packages" not in p.parts)
    random.Random(seed).shuffle(files)
    train, val = bytearray(), bytearray()
    for i, f in enumerate(files):
        if len(train) + len(val) >= max_bytes:
            break
        try:
            b = f.read_bytes()
        except OSError:
            continue
        (val if i % 10 == 0 else train).extend(b + b"\x00")
    def to_t(b: bytearray) -> torch.Tensor:
        t = torch.tensor(list(b), dtype=torch.long)
        return t.masked_fill_(t == 0, EOS)
    _CORPUS[key] = (to_t(train), to_t(val))
    return _CORPUS[key]


def batches(data: torch.Tensor, bsz: int, seq_len: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = data.numel() - seq_len - 1
    while True:
        starts = torch.randint(0, n, (bsz,), generator=g)
        yield torch.stack([data[s : s + seq_len] for s in starts.tolist()])


@torch.no_grad()
def evaluate(model: LabLM, val: torch.Tensor, seq_len: int, n_windows: int = 96, bsz: int = 16) -> float:
    model.eval()
    stride = (val.numel() - seq_len - 1) // n_windows
    wins = torch.stack([val[i * stride : i * stride + seq_len] for i in range(n_windows)])
    tot = 0.0
    for i in range(0, n_windows, bsz):
        with AMP():
            task, _ = model(wins[i : i + bsz], wins[i : i + bsz])
        tot += float(task) * wins[i : i + bsz].size(0)
    model.train()
    return tot / n_windows / math.log(2)  # bits per byte


@torch.no_grad()
def causal_evaluate(model: LabLM, val: torch.Tensor, seq_len: int, n_windows: int = 48,
                    per_window: int = 12, seed: int = 123) -> float:
    """Loss at sampled positions using ONLY the prefix up to each position (bits/byte).

    This is how the model behaves when generating. For a strictly causal model it equals
    the full-sequence loss at the same positions; for a model whose forward pass leaks
    future information (e.g. sequence-level top-k routing) it removes the leak.
    Returns (prefix-only loss, full-sequence loss at the same positions), so the two are
    directly comparable; any gap between them is leak, not position-sampling noise.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    stride = (val.numel() - seq_len - 1) // n_windows
    tot, full, n = 0.0, 0.0, 0
    for i in range(n_windows):
        win = val[i * stride : i * stride + seq_len]
        pos = torch.randint(8, seq_len - 1, (per_window,), generator=g).tolist()
        with AMP():
            all_logits = model(win.unsqueeze(0))[0].float()
        for t in pos:
            with AMP():
                logits = model(win[: t + 1].unsqueeze(0))[0, -1].float()
            tot += float(F.cross_entropy(logits.unsqueeze(0), win[t + 1].view(1)))
            full += float(F.cross_entropy(all_logits[t].unsqueeze(0), win[t + 1].view(1)))
            n += 1
    model.train()
    return tot / n / math.log(2), full / n / math.log(2)


@torch.no_grad()
def causality_leak(model: LabLM, seq_len: int, seed: int = 0) -> float:
    """Max |change| in logits at positions <= t when only tokens > t are changed."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256, (2, seq_len), generator=g)
    cut = seq_len // 2
    ids2 = ids.clone()
    ids2[:, cut + 1 :] = torch.randint(0, 256, (2, seq_len - cut - 1), generator=g)
    a = model(ids)[:, : cut + 1].float()   # fp32 here: the probe needs exact zeros
    b = model(ids2)[:, : cut + 1].float()
    model.train()
    return float((a - b).abs().max())


@torch.no_grad()
def copy_probe(model: LabLM, val: torch.Tensor, span: int = 64, reps: int = 2, n: int = 32, seed: int = 7) -> dict:
    """In-context retrieval: a held-out text span repeated. Loss (bits/byte) on each copy.

    copy0 is ordinary LM loss on the span. A model that can look up what followed the same
    context earlier drives later copies toward 0; `copy_gain` = copy0 - copy2 measures that.
    Uses in-distribution text (random bytes are out of distribution for a code-trained LM).
    Needs no task-specific training, so it runs on every LM checkpoint.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, val.numel() - span, (n,), generator=g).tolist()
    base = torch.stack([val[i : i + span] for i in starts])
    ids = base.repeat(1, reps + 1)
    with AMP():
        logits = model(ids).float()
    lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1),
                         reduction="none").view(n, -1) / math.log(2)
    model.train()
    out = {}
    for r in range(0, reps + 1):
        out[f"copy{r}_bits"] = round(float(lp[:, max(r * span, 8) : (r + 1) * span - 1].mean()), 4)
    out["copy_gain"] = round(out["copy0_bits"] - out[f"copy{reps}_bits"], 4)
    return out


def append_result(out: Path, r: dict) -> None:
    """Append one result under an exclusive lock: parallel jobs share one results file."""
    import fcntl
    with open(str(out) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rs = json.loads(out.read_text()) if out.exists() else []
        rs.append(r)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rs, indent=1))
        tmp.replace(out)


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------
@dataclass
class RunCfg:
    steps: int = 800
    bsz: int = 16
    lr: float = 2e-3
    warmup_frac: float = 0.05
    seed: int = 0
    eval_every: int = 200
    dims: Dims = field(default_factory=Dims)


def _gn_stats(g: list[float]) -> dict:
    if not g:
        return {}
    s = sorted(g)
    return {"median": round(s[len(s) // 2], 3), "p99": round(s[int(0.99 * (len(s) - 1))], 3),
            "max": round(s[-1], 3), "n_over_5": sum(x > 5 for x in g), "n_over_20": sum(x > 20 for x in g),
            "argmax_step": int(g.index(max(g)))}


def train(spec: Spec, rc: RunCfg, log=print, sink: list | None = None) -> dict:
    torch.manual_seed(rc.seed)
    random.seed(rc.seed)
    d = rc.dims
    model = LabLM(spec, d)
    groups = _split_param_groups(model, weight_decay=0.1)
    opt = torch.optim.AdamW(groups, lr=rc.lr, betas=(0.9, 0.95))
    sched = build_scheduler(opt, rc.steps, warmup_steps=max(1, int(rc.warmup_frac * rc.steps)), min_lr_ratio=0.1)
    tr, va = corpus()
    it = batches(tr, rc.bsz, d.seq_len, seed=1000 + rc.seed)  # same batches for every arm
    curve, gmax, nonfinite = [], 0.0, 0
    gns: list[float] = []
    warm_steps = max(1, int(rc.warmup_frac * rc.steps))
    t0 = time.time()
    ema = None
    for step in range(rc.steps):
        aux_w = model.set_knobs(step, rc.steps)
        ids = next(it)
        with AMP():
            task, loss = model(ids, ids, aux_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(gn):
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
        else:
            gmax = max(gmax, gn)
            gns.append(gn)
            opt.step()
        sched.step()
        task = task.detach()
        ema = float(task) if ema is None else 0.95 * ema + 0.05 * float(task)
        if (step + 1) % rc.eval_every == 0 or step + 1 == rc.steps:
            v = evaluate(model, va, d.seq_len)
            curve.append((step + 1, round(ema / math.log(2), 4), round(v, 4), round(gn, 3)))
            log(f"  [{spec.name} s{rc.seed}] step {step+1} train_bpb={ema/math.log(2):.4f} val_bpb={v:.4f} gn={gn:.2f}")
    wall = time.time() - t0
    if sink is not None:
        sink.append(model)
    return {
        "spec": spec.name, "seed": rc.seed, "params": model.n_params(),
        "val_bpb": curve[-1][2], "curve": curve, "max_grad_norm": round(gmax, 3),
        "nonfinite_steps": nonfinite, "leak": causality_leak(model, d.seq_len),
        "gn_after_warmup": _gn_stats(gns[warm_steps:]),
        **copy_probe(model, va),
        **dict(zip(("causal_val_bpb", "paired_full_bpb"),
                   (round(v, 4) for v in causal_evaluate(model, va, d.seq_len)))),
        "tok_per_s": round(rc.steps * rc.bsz * d.seq_len / wall), "wall_s": round(wall, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True, help="comma list of odd/even/route, e.g. attn/attn/none,fsa/pm/mod")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-inner", type=int, default=768)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    args = ap.parse_args()
    global USE_AMP
    USE_AMP = not args.fp32
    torch.set_num_threads(args.threads)
    dims = Dims(d_model=args.d_model, d_inner=args.d_inner, n_layers=args.layers,
                n_heads=args.d_model // 64, d_k=64, seq_len=args.seq_len)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else []
    done = {(r["spec"], r["seed"]) for r in results}
    for seed in [int(s) for s in args.seeds.split(",")]:
        for s in args.specs.split(","):
            spec = Spec.parse(s)
            if (spec.name, seed) in done:
                continue
            rc = RunCfg(steps=args.steps, bsz=args.bsz, lr=args.lr, seed=seed, dims=dims)
            r = train(spec, rc, log=lambda m: print(m, flush=True))
            r["dims"] = asdict(dims)
            r["run"] = {"steps": args.steps, "bsz": args.bsz, "lr": args.lr, "amp_bf16": USE_AMP}
            print(f"RESULT {r['spec']} seed={seed} val_bpb={r['val_bpb']} params={r['params']:,} "
                  f"causal_bpb={r['causal_val_bpb']} leak={r['leak']:.2e} tok/s={r['tok_per_s']} gmax={r['max_grad_norm']} copy_gain={r['copy_gain']}", flush=True)
            append_result(out, r)


if __name__ == "__main__":
    main()
