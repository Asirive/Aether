"""Post-training quantization sensitivity: does the v2 hybrid lose more than a transformer?

Trains each design in the lab setting, then evaluates held-out bpb (fp32 eval) under:
  fp16      all weights and activations in float16 (incl. the SSM state)
  w8        int8 weights, symmetric per output channel (Linear, conv, embedding)
  w4g32     int4 weights, groups of 32 along the input dim (typical on-device CPU/GPU format)
  w8a8      w8 plus static per-tensor int8 inputs to every Linear, scales calibrated on
            8 training batches (the usual NPU setting)

    python ablations/quant.py --specs attn/attn/none,P:conv-pm2-conv-attn-conv-pm2-conv-attn/none --out ablations/results/16_quant/q.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402


def _fq(w: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    q = 2 ** (bits - 1) - 1
    s = w.abs().amax(dim=dim, keepdim=True).clamp_min(1e-12) / q
    return (w / s).round().clamp(-q, q) * s


def quant_weights(model: nn.Module, bits: int, group: int | None = None) -> None:
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Embedding, nn.Conv1d)):
                w = m.weight
                if group and w.dim() == 2 and w.shape[1] % group == 0:
                    g = w.view(w.shape[0], -1, group)
                    w.copy_(_fq(g, bits, -1).view_as(w))
                else:
                    w.copy_(_fq(w, bits, tuple(range(1, w.dim())) if w.dim() > 1 else 0))


def add_act_quant(model: nn.Module, calib: list[torch.Tensor], bits: int = 8) -> None:
    lin = [m for m in model.modules() if isinstance(m, nn.Linear)]
    amax = {id(m): 0.0 for m in lin}

    def rec(m, inp):
        amax[id(m)] = max(amax[id(m)], float(inp[0].detach().abs().max()))

    hs = [m.register_forward_pre_hook(rec) for m in lin]
    with torch.no_grad():
        for ids in calib:
            model(ids, ids)
    for h in hs:
        h.remove()
    q = 2 ** (bits - 1) - 1
    for m in lin:
        s = max(amax[id(m)], 1e-8) / q
        m.register_forward_pre_hook(lambda mod, inp, s=s: ((inp[0] / s).round().clamp(-q, q) * s,))


def evaluate_fp(model, val, seq_len, half=False) -> float:
    if half:
        model = copy.deepcopy(model).half()
        orig = lab.LabLM.forward

        def fwd(self, ids, labels=None, aux_w=None):
            task, loss = orig(self, ids, labels, aux_w)
            return task.float(), loss
        model.forward = fwd.__get__(model)
    return lab.evaluate(model, val, seq_len)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-inner", type=int, default=768)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dims = lab.Dims(d_model=args.d_model, d_inner=args.d_inner, n_layers=8,
                    n_heads=args.d_model // 64, d_k=64, seq_len=256)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else []
    tr, va = lab.corpus()
    calib_it = lab.batches(tr, 16, 256, seed=999)
    calib = [next(calib_it) for _ in range(8)]
    for s in args.specs.split(","):
        spec = lab.Spec.parse(s)
        if any(r["spec"] == spec.name and r["seed"] == args.seed for r in results):
            continue
        sink: list = []
        lab.USE_AMP = True
        lab.train(spec, lab.RunCfg(steps=args.steps, bsz=16, lr=2e-3, seed=args.seed, dims=dims),
                  log=lambda m: print(m, flush=True), sink=sink)
        base = sink[0].eval()
        lab.USE_AMP = False
        r = {"spec": spec.name, "seed": args.seed, "fp32": round(evaluate_fp(base, va, 256), 4)}
        r["fp16"] = round(evaluate_fp(base, va, 256, half=True), 4)
        for name, bits, group in (("w8", 8, None), ("w4g32", 4, 32)):
            m = copy.deepcopy(base)
            quant_weights(m, bits, group)
            r[name] = round(lab.evaluate(m, va, 256), 4)
        m = copy.deepcopy(base)
        quant_weights(m, 8)
        add_act_quant(m, calib)
        r["w8a8"] = round(lab.evaluate(m, va, 256), 4)
        print("QUANT " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)
        results.append(r)
        out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
