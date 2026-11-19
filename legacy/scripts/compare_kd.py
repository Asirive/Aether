"""Side-by-side summary of distillation runs written by scripts/distill.py.

    python scripts/compare_kd.py runs/kd_aether runs/kd_vanilla [--hours-for 1e9]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run: Path) -> tuple[dict, list[dict], list[dict]]:
    meta = json.loads((run / "meta.json").read_text())
    train, evals = [], []
    for line in (run / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        (evals.append(r["eval"]) if "eval" in r else train.append(r))
    return meta, train, evals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--hours-for", type=float, default=1e9, help="token budget for the time estimate")
    args = ap.parse_args()
    print(f"{'run':28s} {'params':>8s} {'step':>6s} {'tok/s':>7s} {'mem GB':>6s} {'student bpb':>11s} "
          f"{'teacher bpb':>11s} {'copy gain':>9s} {'h per budget':>12s}")
    for run in args.runs:
        meta, train, evals = load(run)
        t = train[-1] if train else {}
        e = evals[-1] if evals else {}
        tps = t.get("tok_per_s") or 0
        hours = args.hours_for / tps / 3600 if tps else float("nan")
        print(f"{run.name:28s} {meta['params'] / 1e6:7.1f}M {t.get('step', 0):6d} {tps:7d} "
              f"{t.get('mem_gb', float('nan')):6.1f} {e.get('student_bpb', float('nan')):11.4f} "
              f"{e.get('teacher_bpb', float('nan')):11.4f} {e.get('copy_gain', float('nan')):9.3f} {hours:12.1f}")
    if all(len(load(r)[2]) > 1 for r in args.runs):
        print("\nEval curves (student bpb by step):")
        curves = {r.name: {e["step"]: e["student_bpb"] for e in load(r)[2]} for r in args.runs}
        steps = sorted(set().union(*[c.keys() for c in curves.values()]))
        print("step    " + "  ".join(f"{n[:16]:>16s}" for n in curves))
        for s in steps:
            print(f"{s:<7d} " + "  ".join(f"{curves[n].get(s, float('nan')):16.4f}" for n in curves))


if __name__ == "__main__":
    main()
