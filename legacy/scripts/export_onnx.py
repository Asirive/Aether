"""Export one decode step of an Aether v2 or vanilla student to ONNX with static shapes.

NPU runtimes (Qualcomm QNN via ONNX Runtime) need fixed tensor shapes and no data-dependent
control flow. The wrapper here keeps every state at a fixed size: conv and SSM layers carry
their constant-size recurrent state, and attention layers carry a KV buffer of max_ctx slots
that is written with a mask (no concat, no growing tensors).

    python scripts/export_onnx.py --student aether_v2 --out runs/onnx/v2 [--int8]

Checks: the static step matches the training forward pass, the ONNX graph matches PyTorch,
and it reports the operator set, state size and per-token latency in ONNX Runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aether.components.attention import CausalSelfAttention  # noqa: E402
from aether.model import AetherConfig, AetherForCausalLM  # noqa: E402

def _rms_norm(x, normalized_shape, weight=None, eps=None):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + (1e-6 if eps is None else eps))
    return y if weight is None else y * weight


F.rms_norm = _rms_norm

PATTERNS = {"hybrid_attn": ("conv", "ssm", "conv", "attn"), "ssm3_attn": ("ssm", "ssm", "ssm", "attn")}


def build(args) -> AetherForCausalLM:
    pattern = None
    if args.student == "aether_v2":
        unit = PATTERNS[args.pattern]
        pattern = unit * (args.layers // len(unit))
    cfg = AetherConfig(
        vocab_size=args.vocab, d_model=args.d_model, d_inner=args.d_inner, n_layers=args.layers,
        n_heads=args.d_model // 64, d_k=64, max_seq_len=args.max_ctx, arch=args.student,
        layer_pattern=pattern, mod_routing_points=(), mod_keep_ratios=(),
        ssm_state_dim=64, ssm_head_dim=64, fsa2_chunk_size=64, fsa2_n_summary=8,
    )
    return AetherForCausalLM(cfg).eval()


def _rope_at(attn: CausalSelfAttention, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    cos = attn.rope_cos.index_select(0, pos)
    sin = attn.rope_sin.index_select(0, pos)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class StaticDecoder(nn.Module):
    """forward(token[1], pos[1], *states) -> (logits[1, V], *new_states); all shapes fixed."""

    def __init__(self, model: AetherForCausalLM, max_ctx: int) -> None:
        super().__init__()
        self.model, self.max_ctx = model, max_ctx
        self.kinds = []
        for b in model.blocks:
            m = b.mixer
            if isinstance(m, CausalSelfAttention):
                self.kinds.append(("attn", 2))
            elif isinstance(m.init_state(1), tuple):
                self.kinds.append(("ssm", 2))
            else:
                self.kinds.append(("conv", 1))

    def init_states(self) -> list[torch.Tensor]:
        out = []
        for b, (kind, _) in zip(self.model.blocks, self.kinds):
            m = b.mixer
            if kind == "attn":
                out += [torch.zeros(1, m.n_heads, self.max_ctx, m.d_k)] * 2
            elif kind == "ssm":
                out += list(m.init_state(1))
            else:
                out.append(m.init_state(1))
        return out

    def _attn(self, m: CausalSelfAttention, x, pos, kc, vc):
        q = _rope_at(m, m.w_q(x).view(1, m.n_heads, 1, m.d_k), pos)
        k = _rope_at(m, m.w_k(x).view(1, m.n_heads, 1, m.d_k), pos)
        v = m.w_v(x).view(1, m.n_heads, 1, m.d_k)
        slot = (torch.arange(self.max_ctx) == pos).view(1, 1, self.max_ctx, 1)
        kc = torch.where(slot, k, kc)
        vc = torch.where(slot, v, vc)
        mask = torch.where(torch.arange(self.max_ctx) <= pos, 0.0, float("-inf")).view(1, 1, 1, -1)
        out = F.scaled_dot_product_attention(q, kc, vc, attn_mask=mask)
        return m.w_o(out.reshape(1, m.n_heads * m.d_k)), kc, vc

    def forward(self, token: torch.Tensor, pos: torch.Tensor, *states: torch.Tensor):
        x = self.model.token_embedding(token)
        new, i = [], 0
        for b, (kind, n) in zip(self.model.blocks, self.kinds):
            st = states[i:i + n]
            i += n
            h = b.norm1(x)
            if kind == "attn":
                mix, kc, vc = self._attn(b.mixer, h, pos, *st)
                new += [kc, vc]
            elif kind == "ssm":
                mix, (cb, hs) = b.mixer.step(h, tuple(st))
                new += [cb, hs]
            else:
                mix, cb = b.mixer.step(h, st[0])
                new.append(cb)
            x = x + mix
            x = x + b.ffn(b.norm2(x))
        return (self.model._compute_logits(self.model.norm(x)), *new)


def state_names(dec: StaticDecoder) -> list[str]:
    names = []
    for li, (kind, n) in enumerate(dec.kinds):
        names += [f"s{li}_{kind}_{j}" for j in range(n)]
    return names


@torch.no_grad()
def check_static_vs_forward(dec: StaticDecoder, ids: torch.Tensor) -> float:
    ref = dec.model(ids.unsqueeze(0))["logits"][0]
    states = dec.init_states()
    worst = 0.0
    for t in range(ids.numel()):
        logits, *states = dec(ids[t:t + 1], torch.tensor([t]), *states)
        worst = max(worst, (logits[0] - ref[t]).abs().max().item())
    return worst


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", choices=["aether_v2", "vanilla"], default="aether_v2")
    ap.add_argument("--pattern", choices=sorted(PATTERNS), default="hybrid_attn")
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--d-inner", type=int, default=3072)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--check-tokens", type=int, default=24)
    ap.add_argument("--bench-tokens", type=int, default=32)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--int8", action="store_true", help="also quantize weights to INT8 and compare")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    model = build(args)
    dec = StaticDecoder(model, args.max_ctx).eval()
    ids = torch.randint(0, args.vocab, (args.check_tokens,))
    res = {"student": args.student, "params": sum(p.numel() for p in model.parameters()),
           "max_ctx": args.max_ctx, "kinds": dict(Counter(k for k, _ in dec.kinds))}
    res["static_vs_forward_max_abs"] = check_static_vs_forward(dec, ids)

    states = dec.init_states()
    res["state_bytes_fp32"] = sum(s.numel() * 4 for s in states)
    res["state_bytes_fp32_no_attn"] = sum(s.numel() * 4 for s, nm in zip(states, state_names(dec)) if "attn" not in nm)
    names = state_names(dec)
    onnx_path = args.out / "step.onnx"
    torch.onnx.export(
        dec, (ids[:1], torch.tensor([0]), *states), str(onnx_path), dynamo=False, opset_version=17,
        input_names=["token", "pos", *names], output_names=["logits", *[n + "_out" for n in names]],
        external_data=True)

    import onnx
    import onnxruntime as ort

    g = onnx.load(str(onnx_path), load_external_data=False)
    res["onnx_ops"] = dict(sorted(Counter(n.op_type for n in g.graph.node).items()))
    res["onnx_dynamic_dims"] = sum(
        1 for v in list(g.graph.input) + list(g.graph.output)
        for d in v.type.tensor_type.shape.dim if not d.HasField("dim_value"))

    def run(path: Path) -> tuple[float, list[float]]:
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        st = [s.numpy() for s in dec.init_states()]
        pst = dec.init_states()
        worst, lat = 0.0, []
        n = max(args.check_tokens, args.bench_tokens)
        seq = torch.randint(0, args.vocab, (n,))
        for t in range(n):
            feed = {"token": seq[t:t + 1].numpy(), "pos": torch.tensor([t]).numpy(), **dict(zip(names, st))}
            t0 = time.perf_counter()
            out = sess.run(None, feed)
            lat.append(time.perf_counter() - t0)
            st = out[1:]
            if t < args.check_tokens:
                with torch.no_grad():
                    ref, *pst = dec(seq[t:t + 1], torch.tensor([t]), *pst)
                worst = max(worst, float(abs(out[0] - ref.numpy()).max()))
        return worst, lat[2:]

    worst, lat = run(onnx_path)
    res["ort_vs_torch_max_abs"] = worst
    res["ort_ms_per_token"] = 1000 * sum(lat) / len(lat)
    (args.out / "report.json").write_text(json.dumps(res, indent=2))
    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q_path = args.out / "step_int8.onnx"
        quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8, use_external_data_format=True)
        qworst, qlat = run(q_path)
        res["int8_vs_fp32_max_abs"] = qworst
        res["int8_ms_per_token"] = 1000 * sum(qlat) / len(qlat)
    (args.out / "report.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
