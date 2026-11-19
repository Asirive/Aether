from __future__ import annotations

"""Offline, CPU-sized Aether-vs-vanilla comparison through the real trainer.

Runs `PretrainTrainer` (same optimizer, schedule, spike guard, grad clipping,
aux-loss ramping, gradient checkpointing, bf16 autocast) for both
`configs/aether_200m.yaml` and `configs/baseline_200m.yaml`, with three
substitutions so it runs on a machine with no GPU and no Hugging Face access:

  * tokenizer: byte-level (vocab 257, eos=256) instead of the Llama-3 tokenizer
  * data: Python standard-library source files instead of FineWeb-Edu, split
    train/val by file so validation text is never seen in training
  * size: a shrunken model (default d_model=256, 8 layers) and a short run

Everything not listed in the CLI overrides comes from the YAML, so both arms
share exactly one recipe. This is a pipeline check and a small-scale signal,
not evidence about 200M-scale behaviour.

    python scripts/cpu_compare.py --steps 600 --out runs/cpu_compare
"""

import argparse
import json
import math
import random
import sys
import sysconfig
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

EOS = 256
VOCAB = 257


class ByteTokenizer:
    eos_token_id = EOS
    pad_token_id = EOS
    eos_token = "<eos>"
    pad_token = "<eos>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def __call__(self, text, add_special_tokens: bool = False, **_):
        return {"input_ids": self.encode(text)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


def build_corpus(max_bytes: int, seed: int = 0) -> tuple[list[int], list[int]]:
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in stdlib.rglob("*.py") if "site-packages" not in p.parts and "test" not in p.parts)
    random.Random(seed).shuffle(files)
    train: list[int] = []
    val: list[int] = []
    for i, f in enumerate(files):
        target = val if i % 10 == 0 else train
        if len(train) + len(val) >= max_bytes:
            break
        try:
            target.extend(f.read_bytes())
        except OSError:
            continue
        target.append(EOS)
    return train, val


def patch_trainer(train_tokens: list[int]) -> None:
    import aether.training.trainer as tr

    class CorpusStream:
        def __init__(self, tokenizer, seq_len, seed=42, **_):
            self.seq_len = seq_len
            self.rng = random.Random(seed)
            self.data = torch.tensor(train_tokens, dtype=torch.long)

        def set_seq_len(self, seq_len):
            self.seq_len = seq_len

        def next_batch(self, micro_batch):
            n = self.data.numel() - self.seq_len - 1
            starts = [self.rng.randint(0, n) for _ in range(micro_batch)]
            ids = torch.stack([self.data[s : s + self.seq_len] for s in starts])
            # Same convention as FineWebEduStream: labels == input_ids, the model shifts.
            return {"input_ids": ids, "labels": ids.clone()}

    tr.FineWebEduStream = CorpusStream
    tr.build_tokenizer = lambda *_a, **_k: ByteTokenizer()


@torch.no_grad()
def val_loss(model, val_tokens: list[int], seq_len: int, n_windows: int = 64) -> float:
    model.eval()
    data = torch.tensor(val_tokens, dtype=torch.long)
    stride = max(1, (data.numel() - seq_len - 1) // n_windows)
    total, count = 0.0, 0
    for i in range(n_windows):
        ids = data[i * stride : i * stride + seq_len].unsqueeze(0)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(ids, labels=ids)
        total += float(out["task_loss"]) * (seq_len - 1)
        count += seq_len - 1
    model.train()
    return total / count


def parse_log(path: Path) -> dict:
    steps, losses, gnorms, skips = [], [], [], 0
    for line in path.read_text(errors="replace").splitlines():
        if "[spike-guard] skipped" in line:
            skips += 1
        # Trainer format: "[HH:MM:SS] step N loss=... grad_norm=... ..."
        parts = line.split()
        if len(parts) > 2 and parts[1] == "step" and parts[2].isdigit():
            fields = dict(kv.split("=", 1) for kv in parts if "=" in kv)
            try:
                steps.append(int(parts[2]))
                losses.append(float(fields["loss"]))
                gnorms.append(float(fields.get("grad_norm", "nan")))
            except (KeyError, ValueError):
                continue
    return {"steps": steps, "loss": losses, "grad_norm": gnorms, "skipped_updates": skips}


def run_arm(name: str, config: str, args, train_tokens, val_tokens) -> dict:
    from aether.training.trainer import PretrainConfig, PretrainTrainer

    cfg = PretrainConfig.from_yaml(REPO_ROOT / config)
    m = cfg.model
    m.update(vocab_size=VOCAB, d_model=args.d_model, d_inner=args.d_inner, n_layers=args.layers,
             n_heads=args.d_model // 64, d_k=64, max_seq_len=args.seq_len)
    if m.get("arch", "aether") == "aether":
        pts = [p for p in (2, 4, 6) if p < args.layers]
        cfg.mod["routing_points"] = pts
        cfg.mod["keep_ratios"] = [0.50, 0.70, 1.00][: len(pts)]
    cfg.training.update(
        total_steps=args.steps,
        micro_batch=args.micro_batch,
        effective_batch_tokens=args.micro_batch * args.seq_len * args.accum,
        seq_len_phases=[[0.0, 1.0, args.seq_len]],
    )
    cfg.optim.update(lr=args.lr, warmup_steps=max(1, args.steps // 20))
    cfg.eval.update(every_steps=0, canary_every_steps=10**9)
    cfg.logging["log_every_steps"] = args.log_every
    cfg.tui["enabled"] = False
    out_dir = Path(args.out) / name
    if (out_dir / "train.log").exists():
        (out_dir / "train.log").unlink()
    cfg.output.update(dir=str(out_dir), save_every_steps=0)

    torch.manual_seed(args.seed)
    t0 = time.time()
    trainer = PretrainTrainer(cfg, resume_path=None)
    n_params = trainer.num_parameters
    trainer.run()
    wall = time.time() - t0
    v = val_loss(trainer.model, val_tokens, args.seq_len)
    log = parse_log(out_dir / "train.log")
    result = {
        "arm": name, "params": n_params, "wall_sec": round(wall, 1),
        "tokens_per_sec": round(args.steps * args.micro_batch * args.seq_len * args.accum / wall),
        "final_train_loss": log["loss"][-1] if log["loss"] else None,
        "val_loss": round(v, 4), "val_bits_per_byte": round(v / math.log(2), 4),
        "max_grad_norm": max((g for g in log["grad_norm"] if math.isfinite(g)), default=None),
        "skipped_updates": log["skipped_updates"],
        "curve": list(zip(log["steps"], log["loss"], log["grad_norm"])),
    }
    print(f"[{name}] params={n_params:,} val_loss={v:.4f} ({result['val_bits_per_byte']} bpb) "
          f"wall={wall:.0f}s skipped={log['skipped_updates']}", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-inner", type=int, default=768)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--corpus-bytes", type=int, default=20_000_000)
    p.add_argument("--arms", default="baseline,aether")
    p.add_argument("--out", default="runs/cpu_compare")
    args = p.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    train_tokens, val_tokens = build_corpus(args.corpus_bytes)
    print(f"[corpus] train={len(train_tokens):,} bytes  val={len(val_tokens):,} bytes", flush=True)
    patch_trainer(train_tokens)

    configs = {"baseline": "configs/baseline_200m.yaml", "aether": "configs/aether_200m.yaml"}
    results = [run_arm(a, configs[a], args, train_tokens, val_tokens) for a in args.arms.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_seed{args.seed}.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != "curve"} for r in results], indent=2))


if __name__ == "__main__":
    main()
