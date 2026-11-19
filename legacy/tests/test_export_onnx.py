from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

_spec = importlib.util.spec_from_file_location(
    "export_onnx", Path(__file__).resolve().parent.parent / "scripts" / "export_onnx.py")
export_onnx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_onnx)


@pytest.mark.parametrize("student", ["aether_v2", "vanilla"])
def test_static_onnx_step_matches_pytorch(tmp_path, student):
    r = export_onnx.main(["--student", student, "--vocab", "500", "--d-model", "128", "--d-inner", "256",
                          "--layers", "4", "--max-ctx", "32", "--check-tokens", "8", "--bench-tokens", "8",
                          "--out", str(tmp_path)])
    assert r["static_vs_forward_max_abs"] < 1e-4
    assert r["ort_vs_torch_max_abs"] < 1e-4
    assert r["onnx_dynamic_dims"] == 0
    assert "Einsum" not in r["onnx_ops"] and "Loop" not in r["onnx_ops"]
