from __future__ import annotations

import threading
from typing import Iterator

import torch

DEFAULT_EOS_TOKEN_ID = 2


class FineWebEduStream:
    """Streaming FineWeb-Edu with persistent token buffer + background producer.

    A background thread opens the HF streaming iterable and pushes raw doc texts
    into a queue. The main thread (training loop) only does tokenization + buffer
    management, so the GPU is never starved waiting for the next token.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        split: str = "train",
        streaming: bool = True,
        cache_dir: str | None = None,
        seed: int = 42,
        eos_token_id: int | None = None,
        producer_queue_size: int = 32,
        producer_workers: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.seed = seed
        self.eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(tokenizer, "eos_token_id", DEFAULT_EOS_TOKEN_ID)
        )
        if self.eos_token_id is None:
            self.eos_token_id = DEFAULT_EOS_TOKEN_ID

        from datasets import load_dataset

        self._dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
        )

        import queue
        self._doc_queue: queue.Queue = queue.Queue(maxsize=producer_queue_size)
        self._stop = threading.Event()
        self._producer = threading.Thread(
            target=self._producer_run,
            name="fineweb-edu-producer",
            daemon=True,
        )
        self._producer.start()

        self._buffer: list[int] = []
        self._rng_seed = seed
        self._doc_count = 0

    def set_seq_len(self, seq_len: int) -> None:
        self.seq_len = seq_len

    def _producer_run(self) -> None:
        try:
            for example in self._dataset:
                if self._stop.is_set():
                    break
                text = example.get("text", "")
                if not text:
                    continue
                self._doc_queue.put(text)
        except Exception:
            pass
        finally:
            self._doc_queue.put(None)

    def _consume_one_doc(self) -> int:
        import queue as _q
        try:
            text = self._doc_queue.get(timeout=60)
        except _q.Empty:
            return 0
        if text is None:
            return 0
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self._buffer.extend(ids)
        self._buffer.append(self.eos_token_id)
        self._doc_count += 1
        return len(ids) + 1

    def _fill_buffer(self, target: int) -> None:
        while len(self._buffer) < target:
            added = self._consume_one_doc()
            if added == 0:
                break

    def next_batch(self, micro_batch: int) -> dict:
        seq_len = self.seq_len
        chunk_len = seq_len + 1
        max_jitter = 64

        self._fill_buffer(micro_batch * (chunk_len + max_jitter))

        import random
        rng = random.Random(self._rng_seed + self._doc_count)

        chunks: list[list[int]] = []
        for i in range(micro_batch):
            remaining = micro_batch - i
            self._fill_buffer(remaining * (chunk_len + max_jitter))
            upper = min(max_jitter, max(0, len(self._buffer) - (remaining * (chunk_len + max_jitter))))
            jitter = rng.randint(0, upper)
            start = jitter
            end = start + chunk_len
            chunk = self._buffer[start:end]
            if len(chunk) != chunk_len:
                chunk = self._buffer[:chunk_len]
                del self._buffer[:chunk_len]
                if len(chunk) < chunk_len:
                    chunk = chunk + [self.eos_token_id] * (chunk_len - len(chunk))
            else:
                del self._buffer[:end]
            chunks.append(chunk)

        input_ids = torch.tensor([c[:-1] for c in chunks], dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield self.next_batch(micro_batch=1)

    def close(self) -> None:
        self._stop.set()
        try:
            self._producer.join(timeout=2.0)
        except Exception:
            pass