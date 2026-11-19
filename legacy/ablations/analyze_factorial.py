"""Factorial analysis of lab results: main effects and interactions of three components.

Usage: python ablations/analyze_factorial.py RESULTS_DIR --odd fsa --even pm --route mod

Each run is coded on three binary factors: F (odd mixer = the given FSA variant instead of
attention), P (even mixer = the given SSM variant instead of attention), M (depth router
on). Effects are in bits/byte on held-out data (negative = component HELPS):
  main effect of F   = mean(F on) - mean(F off)
  F x P interaction  = half the difference between F's effect with P on and with P off;
                       negative means the two components help each other (synergy),
                       positive means they get in each other's way.
Uses the strictly-causal metric (`causal_val_bpb`) by default, because the v1 MoD router
leaks future tokens into the full-sequence forward pass.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics as st
from pathlib import Path


_RESULT = re.compile(r"^RESULT (\S+) seed=(\d+) val_bpb=([\d.]+) params=([\d,]+) causal_bpb=([\d.]+) "
                     r"leak=(\S+) tok/s=(\d+) gmax=([\d.]+)")


def load(d: Path) -> list[dict]:
    """JSON results, plus any run that only reached the log.

    Early queue workers wrote one shared JSON without a lock, so concurrent runs could
    overwrite each other's entries; the RESULT line in the log is always complete.
    """
    out = []
    for f in sorted(d.glob("*.json")):
        out.extend(json.loads(f.read_text()))
    have = {(r["spec"], r["seed"]) for r in out}
    for f in sorted(d.glob("*.log")):
        for line in f.read_text().splitlines():
            m = _RESULT.match(line)
            if not m or (m[1], int(m[2])) in have:
                continue
            have.add((m[1], int(m[2])))
            out.append({"spec": m[1], "seed": int(m[2]), "val_bpb": float(m[3]),
                        "params": int(m[4].replace(",", "")), "causal_val_bpb": float(m[5]),
                        "leak": float(m[6]), "tok_per_s": int(m[7]), "max_grad_norm": float(m[8]),
                        "from_log": True})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--odd", default="fsa")
    ap.add_argument("--even", default="pm")
    ap.add_argument("--route", default="mod")
    ap.add_argument("--metric", default="causal_val_bpb")
    args = ap.parse_args()
    runs = [r for d in args.dirs for r in load(Path(d))]

    cells: dict[tuple[int, int, int], list[float]] = {}
    other = {}
    for r in runs:
        o, e, m = r["spec"].split("/")
        if o not in ("attn", args.odd) or e not in ("attn", args.even) or m not in ("none", args.route):
            other.setdefault(r["spec"], []).append(r)
            continue
        key = (int(o != "attn"), int(e != "attn"), int(m != "none"))
        cells.setdefault(key, []).append(r[args.metric])

    names = {0: "F", 1: "P", 2: "M"}
    print(f"metric = {args.metric} (bits/byte, lower is better)\n")
    print(f"{'cell':22s} {'n':>2s} {'mean':>8s} {'sd':>7s}  vs baseline")
    base = st.mean(cells.get((0, 0, 0), [float('nan')]))
    for key in sorted(cells, key=lambda k: (sum(k), k)):
        v = cells[key]
        label = "+".join(names[i] for i in range(3) if key[i]) or "baseline (attn)"
        sd = st.stdev(v) if len(v) > 1 else float("nan")
        print(f"{label:22s} {len(v):2d} {st.mean(v):8.4f} {sd:7.4f}  {st.mean(v) - base:+.4f}")

    # Seed noise: pooled within-cell SD
    sds = [st.stdev(v) for v in cells.values() if len(v) > 1]
    if sds:
        print(f"\npooled seed SD per cell: {st.mean(sds):.4f}")

    if len(cells) == 8:
        mean = {k: st.mean(v) for k, v in cells.items()}
        print("\nEffects (negative = helps / synergy):")
        for r in (1, 2, 3):
            for combo in itertools.combinations(range(3), r):
                eff = 0.0
                for key, val in mean.items():
                    sign = 1
                    for i in combo:
                        sign *= 1 if key[i] else -1
                    eff += sign * val
                eff /= 4  # standard 2^3 contrast / 2^(k-1)
                label = " x ".join(names[i] for i in combo)
                kind = "main effect" if r == 1 else "interaction"
                print(f"  {label:10s} {kind:12s} {eff:+.4f}")
        print("\nConditional effects (component added on top of a context):")
        for i in range(3):
            for ctx in itertools.product((0, 1), repeat=2):
                others = [j for j in range(3) if j != i]
                k0 = [0, 0, 0]
                for j, c in zip(others, ctx):
                    k0[j] = c
                k1 = list(k0)
                k1[i] = 1
                ctx_label = "+".join(names[j] for j, c in zip(others, ctx) if c) or "baseline"
                print(f"  add {names[i]} to {ctx_label:10s}: {mean[tuple(k1)] - mean[tuple(k0)]:+.4f}")
    if other:
        print("\nOther runs:")
        for spec, rs in sorted(other.items()):
            v = [r[args.metric] for r in rs]
            print(f"  {spec:24s} n={len(v)} mean={st.mean(v):.4f}  vs baseline {st.mean(v) - base:+.4f}")
    print("\nCausality leak (max |delta logit| from future tokens), stability:")
    for r in sorted(runs, key=lambda r: (r["spec"], r["seed"])):
        print(f"  {r['spec']:24s} s{r['seed']} leak={r['leak']:.1e} full_bpb={r['val_bpb']:.4f} "
              f"causal_bpb={r['causal_val_bpb']:.4f} gmax={r['max_grad_norm']:.2f} "
              f"params={r['params']:,} tok/s={r['tok_per_s']}")


if __name__ == "__main__":
    main()
