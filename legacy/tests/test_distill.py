from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    "distill", Path(__file__).resolve().parent.parent / "scripts" / "distill.py")
distill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distill)


class _Student:
    def __init__(self, w):
        self.token_embedding = torch.nn.Embedding.from_pretrained(w, freeze=False)
        self.lm_head = None


def test_chunked_kd_loss_matches_direct_computation():
    torch.manual_seed(0)
    V, D, b, L = 50, 16, 2, 13
    w = torch.randn(V, D)
    tw = torch.randn(V, D)
    h = torch.randn(b, L, D)
    th = torch.randn(b, L, D)
    ids = torch.randint(0, V, (b, L))
    with torch.no_grad():
        ce, kl = distill.kd_loss(_Student(w), h, th, tw, ids, 0.5, 2.0, chunk=5)
    s = (h[:, :-1] @ w.T).reshape(-1, V)
    t = (th[:, :-1] @ tw.T).reshape(-1, V)
    ref_ce = F.cross_entropy(s, ids[:, 1:].reshape(-1))
    ref_kl = F.kl_div(F.log_softmax(s / 2, -1), F.log_softmax(t / 2, -1), log_target=True,
                      reduction="batchmean") * 4
    assert torch.allclose(ce, ref_ce, atol=1e-5)
    assert torch.allclose(kl, ref_kl, atol=1e-5)


def test_kd_loss_gradient_flows_through_checkpointed_chunks():
    torch.manual_seed(0)
    st = _Student(torch.randn(30, 8))
    h = torch.randn(1, 9, 8, requires_grad=True)
    ce, kl = distill.kd_loss(st, h, torch.randn(1, 9, 8), torch.randn(30, 8),
                             torch.randint(0, 30, (1, 9)), 0.9, 1.0, chunk=3)
    (ce + kl).backward()
    assert h.grad is not None and torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0


def _run(tmp, student, steps):
    return distill.main(["--mock", "--student", student, "--d-model", "64", "--d-inner", "128",
                         "--layers", "4", "--seq-len", "128", "--micro-bsz", "2", "--grad-accum", "1",
                         "--tokens", "1e5", "--warmup-steps", "2", "--eval-rows", "2", "--eval-docs", "20",
                         "--eval-every", "3", "--ckpt-every", "3", "--max-steps", str(steps),
                         "--out", str(tmp)])


def test_distill_end_to_end_and_resume(tmp_path):
    first = _run(tmp_path, "aether_v2", 3)
    assert first["step"] == 3 and all(k in first for k in ("student_bpb", "teacher_bpb", "copy_gain"))
    second = _run(tmp_path, "aether_v2", 6)
    assert second["step"] == 6
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert sum('"eval"' in ln for ln in lines) == 2
    assert json.loads((tmp_path / "meta.json").read_text())["params"] > 0


def test_distill_vanilla_student_runs(tmp_path):
    assert _run(tmp_path, "vanilla", 2)["step"] == 2
