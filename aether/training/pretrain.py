from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aether.training.trainer import PretrainConfig, PretrainTrainer
from aether.utils.device import describe_device, select_device


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for name in ("rolling.pt", "interrupted.pt", "final.pt"):
        p = output_dir / name
        if p.exists():
            try:
                step = int(torch.load(p, map_location="cpu", weights_only=False).get("step", 0))
            except Exception:
                step = 0
            candidates.append((step, p))
    if candidates:
        return max(candidates, key=lambda t: t[0])[1]
    latest_txt = output_dir / "latest.txt"
    if latest_txt.exists():
        target = output_dir / latest_txt.read_text().strip()
        if target.exists():
            return target
    step_files = sorted(output_dir.glob("step_*.pt"))
    return step_files[-1] if step_files else None


def main() -> None:
    parser = argparse.ArgumentParser(description="AETHER-200M pretrain")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint .pt, or 'auto' to find latest in output dir")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-tui", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    cfg = PretrainConfig.from_yaml(args.config)
    if args.output_dir is not None:
        cfg.output["dir"] = args.output_dir
    if args.resume == "auto":
        output_dir = Path(cfg.output["dir"])
        found = _find_latest_checkpoint(output_dir)
        if found:
            args.resume = str(found)
            print(f"[pretrain] auto-resume from {found}")
        else:
            args.resume = None
            print(f"[pretrain] no checkpoint found in {output_dir}, starting fresh")
    if args.steps is not None:
        cfg.training["total_steps"] = args.steps
    if args.total_steps is not None:
        cfg.training["total_steps"] = args.total_steps
    if args.micro_batch is not None:
        cfg.training["micro_batch"] = args.micro_batch
    if args.seq_len is not None:
        cfg.training["seq_len_phases"] = [[0.0, 1.0, args.seq_len]]
    if args.smoke:
        cfg.training["total_steps"] = max(20, min(cfg.training.get("total_steps", 100), 50))
        cfg.eval["every_steps"] = 25
        cfg.output["save_every_steps"] = max(100, cfg.training["total_steps"])
    if args.no_tui:
        cfg.tui["enabled"] = False

    if args.print_config:
        print(json.dumps(cfg.__dict__, indent=2, default=str))
        return

    device = select_device()
    print(f"[pretrain] device={describe_device(device)}")
    print(f"[pretrain] total_steps={cfg.training['total_steps']} "
          f"micro_batch={cfg.training['micro_batch']} "
          f"effective_batch_tokens={cfg.training['effective_batch_tokens']}")

    trainer = PretrainTrainer(cfg, resume_path=args.resume)
    print(f"[pretrain] model params={trainer.num_parameters:,}")
    print("[pretrain] master=fp32 compute=bf16  optimizer=torch.optim.AdamW(fused=True) (PAF-3: no bnb ROCm Windows wheel)")
    trainer.run()


if __name__ == "__main__":
    main()