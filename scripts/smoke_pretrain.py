from __future__ import annotations

"""5-step smoke test of the AETHER pretrain pipeline.

Uses a tiny in-memory corpus (no HF download, no tokenization bottleneck) so the
data path, forward/backward, optimizer step, gradient checkpointing, and file
logging can all be exercised end-to-end on the real GPU. File logs go to
`runs/_smoke/train.log` so progress is visible even if the TUI breaks.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


_SMALL_POOL = (
    "The quick brown fox jumps over the lazy dog. "
    "Once upon a time in a small village by the sea there lived a fisherman. "
    "Albert Einstein developed the theory of relativity which changed physics. "
    "The capital of France is Paris, known for the Eiffel Tower and great cuisine. "
    "Photosynthesis is the process by which plants convert sunlight into energy. "
    "Machine learning models are trained on large datasets to learn patterns. "
    "DNA is the molecule that carries genetic information in living organisms. "
    "Climate change is one of the most pressing challenges facing humanity today."
)


def _patch_fake_stream():
    import random

    import aether.training.data as _data
    from aether.training.data import FineWebEduStream as _RealStream

    class _FakeStream(_RealStream):
        def __init__(self, tokenizer, seq_len, split="train", streaming=True, cache_dir=None, seed=42):
            self.tokenizer = tokenizer
            self.seq_len = seq_len
            self.seed = seed
            self.eos_token_id = getattr(tokenizer, "eos_token_id", 2) or 2
            self._buffer = []
            self._iter = None
            self._dataset = None
            self._rng = random.Random(seed)
            self._doc_count = 0
            for w in (_SMALL_POOL * 200).split():
                self._buffer.extend(tokenizer.encode(w, add_special_tokens=False))
                self._buffer.append(self.eos_token_id)
            self._doc_count = len((_SMALL_POOL * 200).split())

        def set_seq_len(self, seq_len):
            self.seq_len = seq_len

        def next_batch(self, micro_batch):
            seq_len = self.seq_len
            chunk_len = seq_len + 1
            chunks = []
            for _ in range(micro_batch):
                start = self._rng.randint(0, max(0, len(self._buffer) - chunk_len))
                chunk = self._buffer[start : start + chunk_len]
                if len(chunk) != chunk_len:
                    chunk = self._buffer[:chunk_len]
                chunks.append(chunk)
            import torch as _t
            input_ids = _t.tensor([c[:-1] for c in chunks], dtype=_t.long)
            return {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
            }

    _data.FineWebEduStream = _FakeStream
    import aether.training.trainer as _t_module
    _t_module.FineWebEduStream = _FakeStream
    import aether.training
    aether.training.FineWebEduStream = _FakeStream
    return _FakeStream


def main():
    parser = argparse.ArgumentParser(description="AETHER pretrain smoke test")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--config", type=str, default="configs/aether_200m.yaml")
    parser.add_argument("--output", type=str, default="runs/_smoke")
    args = parser.parse_args()

    _patch_fake_stream()

    import torch
    from aether.training.trainer import PretrainConfig, PretrainTrainer
    from aether.utils.device import describe_device, select_device

    cfg = PretrainConfig.from_yaml(args.config)
    cfg.training["total_steps"] = args.steps
    cfg.training["effective_batch_tokens"] = args.micro_batch * args.seq_len * max(1, args.steps)
    cfg.training["micro_batch"] = args.micro_batch
    cfg.training["seq_len_phases"] = [[0.0, 1.0, args.seq_len]]
    cfg.eval["every_steps"] = 0
    cfg.output["save_every_steps"] = 0
    cfg.output["dir"] = args.output
    cfg.tui["enabled"] = False
    cfg.logging["log_every_steps"] = 1

    print(f"[smoke] device: {describe_device(select_device())}")
    print(f"[smoke] config: seq_len={args.seq_len} micro_batch={args.micro_batch} steps={args.steps}")
    t0 = time.time()
    trainer = PretrainTrainer(cfg, resume_path=None)
    print(f"[smoke] params={trainer.num_parameters:,}")
    trainer.run()
    print(f"[smoke] DONE in {time.time()-t0:.2f}s")
    print(f"[smoke] log file: {Path(args.output) / 'train.log'}")


if __name__ == "__main__":
    main()