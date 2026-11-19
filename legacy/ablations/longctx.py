"""Length generalization: train at 256 bytes of context, evaluate at up to 2048.

A live assistant sees far longer contexts than any affordable training length. This
measures bpb by position bucket on 2048-byte held-out windows for models trained only on
256-byte windows. Lower bpb at later positions means the model uses the extra context;
a rise means it breaks beyond its training length.

    python ablations/longctx.py --specs attn/attn/none,... --out ablations/results/17_longctx/l.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402

BUCKETS = ((0, 256), (256, 512), (512, 1024), (1024, 2048))


@torch.no_grad()
def bucket_bpb(model, val: torch.Tensor, length: int, n_windows: int = 24, bsz: int = 4) -> dict:
    model.eval()
    stride = (val.numel() - length - 1) // n_windows
    wins = torch.stack([val[i * stride: i * stride + length] for i in range(n_windows)])
    tot = torch.zeros(length - 1)
    for i in range(0, n_windows, bsz):
        w = wins[i:i + bsz]
        logits = model(w).float()
        tot += F.cross_entropy(logits[:, :-1].reshape(-1, lab.VOCAB), w[:, 1:].reshape(-1),
                               reduction="none").view(w.size(0), -1).sum(0)
    per = tot / n_windows / math.log(2)
    return {f"{a}-{b}": round(float(per[a:min(b, length) - 1].mean()), 4) for a, b in BUCKETS if a < length - 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--eval-len", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else []
    _, va = lab.corpus()
    for s in args.specs.split(","):
        spec = lab.Spec.parse(s)
        if any(r["spec"] == spec.name and r["seed"] == args.seed for r in results):
            continue
        train_dims = lab.Dims(d_model=256, d_inner=768, n_layers=8, n_heads=4, d_k=64, seq_len=256)
        sink: list = []
        lab.USE_AMP = True
        lab.train(spec, lab.RunCfg(steps=args.steps, bsz=16, lr=2e-3, seed=args.seed, dims=train_dims),
                  log=lambda m: print(m, flush=True), sink=sink)
        long_dims = lab.Dims(d_model=256, d_inner=768, n_layers=8, n_heads=4, d_k=64, seq_len=args.eval_len)
        torch.manual_seed(0)
        model = lab.LabLM(spec, long_dims)
        model.load_state_dict(sink[0].state_dict())
        lab.USE_AMP = False
        r = {"spec": spec.name, "seed": args.seed, "eval_len": args.eval_len,
             "buckets": bucket_bpb(model, va, args.eval_len)}
        print(f"LONGCTX {spec.name} " + " ".join(f"{k}={v}" for k, v in r["buckets"].items()), flush=True)
        results.append(r)
        out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
