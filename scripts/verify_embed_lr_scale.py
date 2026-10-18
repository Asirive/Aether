"""Controlled CPU experiment: does `optim.embed_lr_scale` slow early learning?

Tiny `AetherForCausalLM` (d_model=64, d_inner=128, n_layers=4, n_heads=4, d_k=16,
vocab=512, tied embeddings) trained on a deterministic repeating cycle of 16 distinct
tokens: the successor function is exactly learnable, so CE falls from ln(512)=6.24
toward 0. Two or more copies share the SAME init and the SAME batches; the ONLY
difference is `optim.embed_lr_scale`. This isolates the tied-embedding param group
created by `aether/training/optim.py::_split_param_groups`.

CPU only, one thread, <1 GB, ~50 s per 400-step arm.

    python scripts/verify_embed_lr_scale.py --steps 400 --lr 1e-4 \
        --scales 1.0,0.5,0.3,0.1 --seeds 0,1,2 --out embed_scale.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aether.model import AetherConfig, AetherForCausalLM  # noqa: E402
from aether.training.optim import build_optimizer  # noqa: E402

VOCAB = 512
PERIOD = 16
SEQ = 32
BATCH = 4


def make_cfg() -> AetherConfig:
    return AetherConfig(
        vocab_size=VOCAB, d_model=64, d_inner=128, n_layers=4, n_heads=4, d_k=16,
        max_seq_len=SEQ, rope_base=10000.0, tie_embeddings=True, fsa_topk_ratio=0.25,
        fsa_ortho_loss_weight=0.001, polymamba_orders=2, polymamba_state_dim=16,
        polymamba_kbal_loss_weight=0.01, mod_routing_points=(2,), mod_keep_ratios=(0.85,),
        mod_loss_weight=0.01,
    )


def make_data(seed: int, steps: int, n_eval: int = 16) -> tuple[list[torch.Tensor], torch.Tensor]:
    g = torch.Generator().manual_seed(seed + 7777)
    cycle = torch.randperm(VOCAB, generator=g)[:PERIOD]
    total = SEQ * (steps + n_eval + 4) + 16
    stream = cycle.repeat(total // PERIOD + 2)[:total]

    def sample(n: int) -> torch.Tensor:
        offs = torch.randint(0, len(stream) - SEQ - 1, (n,), generator=g)
        return torch.stack([stream[o:o + SEQ] for o in offs]).long()

    return [sample(BATCH) for _ in range(steps)], sample(BATCH)


def run(scale, seed, steps, lr, log_every, base_state, batches, eval_batch) -> dict:
    model = AetherForCausalLM(make_cfg())
    model.load_state_dict(base_state)
    model.train()
    opt = build_optimizer(model, lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
                          fused=False, embed_lr_scale=scale)
    curve: dict[int, float] = {}
    gradnorms: dict[int, float] = {}
    emb_grads: dict[int, float] = {}
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        ids = batches[step - 1]
        out = model(input_ids=ids, labels=ids)
        out["loss"].backward()
        pre = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], 1.0).item())
        emb_grads[step] = float(model.token_embedding.weight.grad.norm().item())
        opt.step()
        opt.zero_grad(set_to_none=True)
        gradnorms[step] = pre
        if step == 1 or step % log_every == 0:
            curve[step] = float(out["task_loss"].item())
    wall = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        eval_loss = float(model(input_ids=eval_batch, labels=eval_batch)["task_loss"].item())
        logit_rms = float(model(input_ids=eval_batch, labels=None)["logits"].float().std().item())
    w = model.token_embedding.weight.detach()
    return {
        "scale": scale, "seed": seed, "lr": lr, "steps": steps,
        "curve": curve, "gradnorm": gradnorms, "emb_grad": emb_grads,
        "eval_task_loss": eval_loss, "logit_rms": logit_rms,
        "emb_std": float(w.std().item()), "emb_fro": float(w.norm().item()),
        "wall_s": wall,
    }


def first_at_or_below(curve: dict, target: float) -> float | None:
    prev = None
    for s in sorted(int(x) for x in curve):
        v = curve[str(s)] if str(s) in curve else curve[s]
        if v <= target:
            if prev is None:
                return float(s)
            ps, pv = prev
            return ps + (pv - target) / (pv - v) * (s - ps) if pv != v else float(s)
        prev = (s, v)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scales", type=str, default="1.0,0.5,0.3,0.1")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    scales = [float(s) for s in args.scales.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        base = AetherForCausalLM(make_cfg())
        sd = copy.deepcopy(base.state_dict())
        batches, eval_batch = make_data(seed, args.steps)
        print(f"--- seed {seed}: params={sum(p.numel() for p in base.parameters())}")
        for scale in scales:
            r = run(scale, seed, args.steps, args.lr, args.log_every, sd, batches, eval_batch)
            results.append(r)
            print(f"  scale={scale:<4} eval_ce={r['eval_task_loss']:.4f} "
                  f"logit_rms={r['logit_rms']:.3f} wall={r['wall_s']:.1f}s")

    print(f"\n{'step':>5}" + "".join(f"{'s='+str(s):>10}" for s in scales))
    for st in sorted(results[0]["curve"]):
        row = []
        for s in scales:
            vs = [r["curve"][st] for r in results if r["scale"] == s]
            row.append(sum(vs) / len(vs))
        print(f"{st:5d}" + "".join(f"{v:10.4f}" for v in row))

    print("\nsteps-of-progress-lost (scale=0.1 vs 1.0)")
    for seed in seeds:
        b = next(r for r in results if r["scale"] == scales[0] and r["seed"] == seed)
        w = next(r for r in results if r["scale"] == scales[-1] and r["seed"] == seed)
        t = first_at_or_below(b["curve"], w["curve"][args.steps])
        if t:
            print(f"  seed {seed}: {scales[-1]}x @ {args.steps} == {scales[0]}x @ step "
                  f"{t:.0f} -> lost {args.steps-t:.0f} steps ({100*(args.steps-t)/args.steps:.0f}%)")

    Path(args.out).write_text(json.dumps(results))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
