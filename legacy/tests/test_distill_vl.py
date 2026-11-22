from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest
import torch

pytest.importorskip("PIL")

_spec = importlib.util.spec_from_file_location(
    "distill_vl", Path(__file__).resolve().parent.parent / "scripts" / "distill_vl.py")
distill_vl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distill_vl)


def test_relation_labels_follow_the_dominant_axis():
    a = (("red", "square"), 10, 50)
    b = (("blue", "circle"), 60, 55)
    assert distill_vl.relation(a, b) == "left of"
    assert distill_vl.relation(b, a) == "right of"
    c = (("green", "triangle"), 12, 5)
    assert distill_vl.relation(c, a) == "above"
    assert distill_vl.relation(a, c) == "below"


def test_scene_objects_are_distinct_and_separated():
    rng = random.Random(0)
    for _ in range(20):
        img, objs = distill_vl.draw_scene(rng, 64)
        assert img.size == (64, 64) and len({o[0] for o in objs}) == 3


@pytest.mark.parametrize("student", ["aether_v2", "vanilla"])
def test_mock_vision_stage_trains_and_resumes(tmp_path, student):
    argv = ["--mock", "--student", student, "--d-model", "64", "--d-inner", "128", "--layers", "4",
            "--enc-dim", "32", "--grad-accum", "2", "--steps", "4", "--eval-every", "4", "--probe-n", "4",
            "--ckpt-every", "2", "--log-every", "2", "--out", str(tmp_path)]
    r = distill_vl.main(argv)
    assert r["step"] == 4 and torch.isfinite(torch.tensor(r["ce"]))
    assert 0.0 <= r["spatial_acc"] <= 1.0
    r2 = distill_vl.main(argv[:argv.index("--steps") + 1] + ["6"] + argv[argv.index("--steps") + 2:])
    assert r2["step"] == 6


def test_image_tokens_reach_the_text_positions():
    torch.manual_seed(0)
    args = distill_vl.argparse.Namespace(student="aether_v2", pattern="hybrid_attn", d_model=64, d_inner=128,
                                         layers=4, seq_len=256)
    lm = distill_vl.distill.build_student(args, 256)
    st = distill_vl.VLStudent(distill_vl.PatchEncoder(64, 8, 32), lm, train_encoder=True)
    ids = torch.randint(0, 256, (1, 10))
    pix = torch.randn(1, 3, 64, 64)
    h1, n = st.hidden(pix, ids)
    h2, _ = st.hidden(pix + 1.0, ids)
    assert n == 64 and (h1[0, n:] - h2[0, n:]).abs().max() > 1e-4
