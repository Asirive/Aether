"""Multi-query associative recall (MQAR) probe for lab architectures.

Sequence (length 2P): P key/value pairs  k1 v1 k2 v2 ... kP vP, then the same P keys in a
random order, each followed by its value:  k_a v_a k_b v_b ...  Loss and accuracy are measured
ONLY on the values in the query half, which can only be predicted by retrieving the
matching pair from up to 2P tokens earlier. Keys are drawn from 0..127 (distinct within a
sequence), values from 128..255. Chance accuracy is 1/128.

This is the task class where pure recurrent models are known to struggle and attention
excels, so it checks that an architecture's language-modelling gains do not come from
giving up exact retrieval.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import lab  # noqa: E402
from lab import AMP, Dims, LabLM, Spec  # noqa: E402


def make_batch(bsz: int, pairs: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.argsort(torch.rand(bsz, 128, generator=g), dim=1)[:, :pairs]
    vals = torch.randint(128, 256, (bsz, pairs), generator=g)
    ctx = torch.stack([keys, vals], dim=-1).view(bsz, 2 * pairs)
    order = torch.argsort(torch.rand(bsz, pairs, generator=g), dim=1)
    qk = keys.gather(1, order)
    qv = vals.gather(1, order)
    qry = torch.stack([qk, qv], dim=-1).view(bsz, 2 * pairs)
    ids = torch.cat([ctx, qry], dim=1)
    # target mask: positions whose NEXT token is a query value
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 2 * pairs : -1 : 2] = True
    return ids, mask


def loss_acc(model, ids, mask):
    logits = model(ids)[:, :-1].float()
    tgt = ids[:, 1:]
    m = mask[:, :-1]
    lg, tg = logits[m], tgt[m]
    return F.cross_entropy(lg, tg), (lg.argmax(-1) == tg).float().mean()


def run(spec: str, steps: int, pairs: int, seed: int, lr: float, dims: Dims, bsz: int) -> dict:
    torch.manual_seed(seed)
    model = LabLM(Spec.parse(spec), dims)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))))
    g = torch.Generator().manual_seed(1000 + seed)
    t0 = time.time()
    curve = []
    for step in range(steps):
        aux_w = model.set_knobs(step, steps)
        ids, mask = make_batch(bsz, pairs, g)
        with AMP():
            loss, acc = loss_acc(model, ids, mask)
            aux = model.aux_loss(aux_w)
        opt.zero_grad(set_to_none=True)
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 250 == 0:
            curve.append((step + 1, round(float(loss), 4), round(float(acc), 4)))
            print(f"  [{spec} s{seed}] step {step+1} loss={float(loss):.3f} acc={float(acc):.3f}", flush=True)
    model.eval()
    ge = torch.Generator().manual_seed(999)
    accs = []
    with torch.no_grad():
        for _ in range(8):
            ids, mask = make_batch(64, pairs, ge)
            with AMP():
                _, acc = loss_acc(model, ids, mask)
            accs.append(float(acc))
    return {"spec": spec, "seed": seed, "pairs": pairs, "steps": steps, "test_acc": sum(accs) / len(accs),
            "curve": curve, "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    dims = Dims(d_model=args.d_model, d_inner=3 * args.d_model, n_layers=args.layers,
                n_heads=args.d_model // 64, d_k=64, seq_len=4 * args.pairs)
    out = Path(args.out)
    res = json.loads(out.read_text()) if out.exists() else []
    for seed in map(int, args.seeds.split(",")):
        for spec in args.specs.split(","):
            r = run(spec, args.steps, args.pairs, seed, args.lr, dims, args.bsz)
            print(f"RECALL {spec} seed={seed} acc={r['test_acc']:.4f}", flush=True)
            res.append(r)
            out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
