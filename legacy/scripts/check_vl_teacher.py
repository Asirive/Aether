"""Check a vision-language teacher and a vision encoder before the vision stage (run on Vast).

    python scripts/check_vl_teacher.py --teacher Qwen/Qwen3-VL-4B-Instruct --teacher Qwen/Qwen3.5-2B

For each teacher: loads it, reports vocabulary size, image-token handling, memory, the
alignment the vision stage relies on (image placeholder followed by the text tokens), and
its accuracy on the synthetic spatial probe (chance 25%). Then loads the student encoder
and reports its token count and width. Every check prints PASS/FAIL and carries on.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    "distill_vl", Path(__file__).resolve().parent / "distill_vl.py")
vl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vl)


@torch.no_grad()
def teacher_probe(model, pipe, n: int, device) -> float:
    rng = random.Random(12345)
    correct = 0
    for _ in range(n):
        img, prompt, caption, rel = vl.spatial_example(rng, pipe.args.teacher_image_size)
        scores = []
        for r in vl.RELATIONS:
            inp, p = pipe.teacher_batch(img, prompt, caption(r))
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**inp).logits[0, p:-1].float()
            scores.append(-F.cross_entropy(logits, inp["input_ids"][0, p + 1:], reduction="sum").item())
        correct += vl.RELATIONS[max(range(4), key=lambda i: scores[i])] == rel
    return correct / n


def check_teacher(name: str, args, device) -> dict:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    res: dict = {"teacher": name}
    try:
        t0 = time.time()
        proc = AutoProcessor.from_pretrained(name)
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
        res["load_s"] = round(time.time() - t0, 1)
        res["params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
        res["vocab"] = model.get_output_embeddings().weight.size(0)
        res["tokenizer_len"] = len(proc.tokenizer)
        res["image_token_id"] = getattr(model.config, "image_token_id", None)
        if device == "cuda":
            res["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        print(f"PASS load {name}: {json.dumps(res)}", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"FAIL load {name}", flush=True)
        return res
    pipe = vl.Pipeline(args, None, model, proc, proc.tokenizer)
    try:
        img, prompt, caption, rel = vl.spatial_example(random.Random(0), args.teacher_image_size)
        inp, p = pipe.teacher_batch(img, prompt, caption(rel))
        tail = proc.tokenizer.decode(inp["input_ids"][0, p + 1:])
        res["image_tokens"] = int((inp["input_ids"][0] == (res["image_token_id"] or -1)).sum())
        res["text_after_image"] = tail[-120:]
        hid = model.model(**{k: v.to(device) for k, v in inp.items()}).last_hidden_state
        res["hidden_dim"] = hid.size(-1)
        ok = caption(rel) in tail and prompt in tail
        print(f"{'PASS' if ok else 'FAIL'} alignment: text after the image = {tail[-120:]!r}", flush=True)
        res["alignment_ok"] = ok
    except Exception:
        traceback.print_exc()
        print("FAIL alignment (teacher.model(...).last_hidden_state or image token)", flush=True)
    try:
        res["spatial_probe_acc"] = teacher_probe(model, pipe, args.probe_n, device)
        print(f"PASS probe: teacher spatial accuracy {res['spatial_probe_acc']:.2f} (chance 0.25)", flush=True)
    except Exception:
        traceback.print_exc()
        print("FAIL probe", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="append", default=[])
    ap.add_argument("--encoder", default="google/siglip2-base-patch16-224")
    ap.add_argument("--teacher-image-size", type=int, default=448)
    ap.add_argument("--probe-n", type=int, default=50)
    ap.add_argument("--out", default="runs/vl_teacher_check.json")
    args = ap.parse_args()
    args.mock = False
    teachers = args.teacher or ["Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3.5-2B"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = [check_teacher(t, args, device) for t in teachers]
    try:
        enc = vl.HFEncoder(args.encoder).to(device)
        img, *_ = vl.spatial_example(random.Random(0), 448)
        pix = enc.processor(images=img, return_tensors="pt")["pixel_values"].to(device)
        feats = enc(pix)
        results.append({"encoder": args.encoder, "tokens": feats.size(1), "dim": feats.size(2),
                        "params_m": round(sum(p.numel() for p in enc.parameters()) / 1e6, 1)})
        print(f"PASS encoder: {json.dumps(results[-1])}", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"FAIL encoder {args.encoder}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
