from __future__ import annotations

import queue as _q
import random as _r

import torch

from aether.training.data import FineWebEduStream


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        for word in text.split():
            out.extend([hash(word) % 30000 + 10 for _ in word[:5]])
        return out


_DOCS = [
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi",
    "the quick brown fox jumps over the lazy dog and runs into the forest",
    "one two three four five six seven eight nine ten eleven twelve thirteen",
]


def _new_stream(seq_len: int = 16) -> FineWebEduStream:
    stream = FineWebEduStream.__new__(FineWebEduStream)
    stream.tokenizer = _FakeTok()
    stream.seq_len = seq_len
    stream.eos_token_id = 2
    stream._buffer = []
    stream._rng_seed = 0
    stream._doc_count = 0
    stream.streaming = True
    stream._doc_queue = _q.Queue()
    for _ in range(100):
        for d in _DOCS:
            stream._doc_queue.put(d)
    stream._stop = type("S", (), {"is_set": lambda self: False, "set": lambda self: None})()
    stream._producer = type("T", (), {"join": lambda self, *a, **k: None, "start": lambda self: None})()
    return stream


def test_fineweb_stream_persistent_buffer_no_loss():
    """next_batch must return same shape every call; buffer must persist between calls."""
    stream = _new_stream(seq_len=16)

    b1 = stream.next_batch(micro_batch=2)
    b2 = stream.next_batch(micro_batch=2)
    b3 = stream.next_batch(micro_batch=2)

    assert b1["input_ids"].shape == (2, 16)
    assert b2["input_ids"].shape == (2, 16)
    assert b3["input_ids"].shape == (2, 16)
    assert b1["labels"].shape == b1["input_ids"].shape


def test_fineweb_stream_seq_len_changes_via_setter():
    stream = _new_stream(seq_len=16)
    b = stream.next_batch(micro_batch=1)
    assert b["input_ids"].shape == (1, 16)

    stream.set_seq_len(8)
    b2 = stream.next_batch(micro_batch=1)
    assert b2["input_ids"].shape == (1, 8)


def test_fineweb_stream_buffer_survives_across_seq_len_change():
    stream = _new_stream(seq_len=16)
    _ = stream.next_batch(micro_batch=1)
    buffer_after = len(stream._buffer)
    stream.set_seq_len(8)
    _ = stream.next_batch(micro_batch=1)
    assert len(stream._buffer) <= buffer_after + 100


def test_fineweb_stream_labels_match_hf_convention():
    """labels must equal input_ids (HF convention).

    The model's forward applies the shift (logits[:-1] vs labels[1:]) to build
    the next-token objective. If the loader pre-shifts labels (labels = tokens[1:]),
    training silently becomes a 2-ahead objective and the model can never learn
    next-token prediction. Regression guard for the 2026-08-09 double-shift bug.
    """
    stream = _new_stream(seq_len=16)
    for _ in range(5):
        b = stream.next_batch(micro_batch=3)
        assert torch.equal(b["labels"], b["input_ids"]), (
            "data loader must use HF convention labels == input_ids; "
            "a pre-shift here double-shifts against model.forward"
        )