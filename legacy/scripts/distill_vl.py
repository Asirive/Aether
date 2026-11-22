"""Vision stage: distil a vision-language teacher into image encoder + projector + Aether.

    camera -> vision encoder (pretrained, attention) -> projector -> Aether (language)

The student sees [image tokens, text tokens]; the teacher sees its own chat-formatted
image + text. Both predict the same text tokens, and the student matches the teacher's
full next-token distribution on every text position (plus a little plain CE), the same
loss as scripts/distill.py. The vision encoder is frozen by default; the projector and
Aether train.

    python scripts/distill_vl.py --student aether_v2 --teacher Qwen/Qwen3-VL-4B-Instruct \
        --encoder google/siglip2-base-patch16-224 --dataset <hf id> --out runs/vl_aether

`--init` loads an Aether checkpoint from scripts/distill.py (text stage) first; its
vocabulary must match the teacher's.

Spatial probe (logged at every eval): synthetic scenes of three coloured shapes; the student
scores "the A is <left of|right of|above|below> the B" for all four relations and must rank
the true one first (chance 25%). It needs the named objects located in the image tokens,
so it tests grounding, not language priors.

`--mock` uses a small trainable patch encoder, a byte tokenizer, no teacher and synthetic
scenes only, so the whole pipeline runs on a CPU with no network.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("distill", ROOT / "scripts" / "distill.py")
distill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distill)

COLORS = {"red": (220, 40, 40), "green": (40, 180, 60), "blue": (40, 80, 230), "yellow": (230, 210, 40),
          "purple": (150, 60, 190), "white": (245, 245, 245)}
SHAPES = ("square", "circle", "triangle")
RELATIONS = ("left of", "right of", "above", "below")


def draw_scene(rng: random.Random, size: int):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (20, 20, 20))
    d = ImageDraw.Draw(img)
    names = rng.sample([(c, s) for c in COLORS for s in SHAPES], 3)
    r = size // 10
    objs = []
    while len(objs) < 3:
        x, y = rng.randint(r, size - r - 1), rng.randint(r, size - r - 1)
        if all(abs(x - ox) > 2 * r or abs(y - oy) > 2 * r for _, ox, oy in objs):
            objs.append((names[len(objs)], x, y))
    for (color, shape), x, y in objs:
        box, fill = (x - r, y - r, x + r, y + r), COLORS[color]
        if shape == "square":
            d.rectangle(box, fill=fill)
        elif shape == "circle":
            d.ellipse(box, fill=fill)
        else:
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=fill)
    return img, objs


def relation(a, b) -> str:
    (_, ax, ay), (_, bx, by) = a, b
    if abs(ax - bx) >= abs(ay - by):
        return "left of" if ax < bx else "right of"
    return "above" if ay < by else "below"


TASK = {"name": "pair"}


def spatial_example(rng: random.Random, size: int):
    img, objs = draw_scene(rng, size)
    if TASK["name"] == "single":
        from PIL import ImageDraw

        img = img.copy()
        ImageDraw.Draw(img).rectangle((0, 0, size, size), fill=(20, 20, 20))
        (color, shape), x, y = objs[0]
        r = size // 10
        ImageDraw.Draw(img).rectangle((x - r, y - r, x + r, y + r), fill=COLORS[color])
        rel = "left of" if x < size / 2 else "right of"
        return img, "Is the square on the left or the right?", lambda r_: f"{r_}.", rel
    a, b = rng.sample(objs, 2)
    rel = relation(a, b)
    name = lambda o: f"{o[0][0]} {o[0][1]}"
    prompt = f"Where is the {name(a)} relative to the {name(b)}?"
    return img, prompt + " Answer with left of, right of, above or below.", lambda r: f"{r}.", rel


class PatchEncoder(nn.Module):
    def __init__(self, size: int, patch: int, dim: int, layers: int = 2):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)
        self.pos = nn.Parameter(torch.randn(1, (size // patch) ** 2, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, max(1, dim // 64), 4 * dim, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out_dim = dim

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = self.proj(pixels).flatten(2).transpose(1, 2) + self.pos
        return self.enc(x)


class HFEncoder(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        m = AutoModel.from_pretrained(name, dtype=torch.bfloat16)
        self.model = getattr(m, "vision_model", m)
        self.processor = AutoImageProcessor.from_pretrained(name)
        cfg = getattr(m.config, "vision_config", m.config)
        self.out_dim = cfg.hidden_size

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixels.to(self.model.dtype)).last_hidden_state.float()


class Projector(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VLStudent(nn.Module):
    def __init__(self, encoder: nn.Module, lm, train_encoder: bool):
        super().__init__()
        self.encoder, self.lm = encoder, lm
        self.proj = Projector(encoder.out_dim, lm.config.d_model)
        self.encoder.requires_grad_(train_encoder)

    def image_tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(pixels))

    def hidden(self, pixels: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, int]:
        v = self.image_tokens(pixels)
        x = torch.cat([v, self.lm.token_embedding(ids).to(v.dtype)], dim=1)
        return self.lm.hidden_from_embeds(x), v.size(1)

    def head(self) -> torch.Tensor:
        return self.lm.token_embedding.weight if self.lm.lm_head is None else self.lm.lm_head.weight


def mock_pixels(img, size: int) -> torch.Tensor:
    t = torch.frombuffer(bytearray(img.resize((size, size)).tobytes()), dtype=torch.uint8)
    return (t.view(size, size, 3).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0)


class Pipeline:
    """Turns (image, prompt, answer) into student pixels + text ids and teacher inputs."""

    def __init__(self, args, student: VLStudent, teacher, processor, tok):
        self.args, self.student, self.teacher, self.processor, self.tok = args, student, teacher, processor, tok

    def pixels(self, img) -> torch.Tensor:
        if self.args.mock:
            return mock_pixels(img, self.args.image_size)
        return self.student.encoder.processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"]

    def text_ids(self, prompt: str, answer: str) -> torch.Tensor:
        ids = self.tok([f"{prompt}\n{answer}"], add_special_tokens=False)["input_ids"][0]
        return torch.tensor([ids + [self.tok.eos_token_id]])

    def teacher_batch(self, img, prompt: str, answer: str):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False)
        img = img.convert("RGB").resize((self.args.teacher_image_size,) * 2)
        inp = self.processor(text=[text], images=[img], return_tensors="pt")
        ids = inp["input_ids"][0]
        img_tok = getattr(self.teacher.config, "image_token_id", None)
        if img_tok is None:
            img_tok = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        p = int((ids == img_tok).nonzero().max())
        return inp, p

    def student_ids(self, img, prompt: str, answer: str) -> torch.Tensor:
        if self.teacher is None:
            return self.text_ids(prompt, answer)
        inp, p = self.teacher_batch(img, prompt, answer)
        return inp["input_ids"][:, p + 1:]

    def answer_start(self, ids: torch.Tensor, prompt: str, answer: str) -> int:
        if not self.args.answer_only:
            return 0
        if self.teacher is None:
            return len(self.tok([f"{prompt}\n"], add_special_tokens=False)["input_ids"][0])
        a = self.tok([answer], add_special_tokens=False)["input_ids"][0]
        seq = ids[0].tolist()
        for i in range(len(seq) - len(a), -1, -1):
            if seq[i:i + len(a)] == a:
                return i
        return 0

    def example(self, img, prompt: str, answer: str, device):
        if self.teacher is None:
            ids = self.text_ids(prompt, answer)
            return self.pixels(img).to(device), ids.to(device), None, self.answer_start(ids, prompt, answer)
        inp, p = self.teacher_batch(img, prompt, answer)
        inp = {k: v.to(device) for k, v in inp.items()}
        ids = inp["input_ids"][:, p + 1:]
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out = self.teacher.model(**inp)
        th = out.last_hidden_state[:, p:-1]
        return self.pixels(img).to(device), ids, th, self.answer_start(ids, prompt, answer)


def kd_text_loss(student: VLStudent, h, n_img: int, ids, th, t_head_w, kd_weight: float, temp: float,
                 chunk: int, start: int = 0, n_tokens: int = 0):
    end = ids.size(1) if n_tokens <= 0 else min(ids.size(1), start + n_tokens)
    hs = h[0, n_img - 1 + start:n_img - 1 + end]
    tgt = ids[0, start:end]
    th = th[:, start:end] if th is not None else None
    w = student.head()
    ce_tot, kl_tot = hs.new_zeros((), dtype=torch.float32), hs.new_zeros((), dtype=torch.float32)
    for s in range(0, hs.size(0), chunk):
        e = min(s + chunk, hs.size(0))
        tl = F.linear(th[0, s:e].to(t_head_w.dtype), t_head_w) if th is not None else torch.empty(0)
        ce, kl = distill._chunk_loss(hs[s:e], w, tl, tgt[s:e], kd_weight if th is not None else 0.0, temp)
        ce_tot, kl_tot = ce_tot + ce, kl_tot + kl
    n = hs.size(0)
    return ce_tot / n, kl_tot / n


@torch.no_grad()
def spatial_probe(student: VLStudent, pipe: Pipeline, n: int, device, amp) -> dict:
    student.eval()
    rng = random.Random(12345)
    correct, margin = 0, 0.0
    for _ in range(n):
        img, prompt, caption, rel = spatial_example(rng, max(pipe.args.image_size, 64) if pipe.args.mock
                                                    else pipe.args.teacher_image_size)
        pix = pipe.pixels(img).to(device)
        scores = []
        for r in RELATIONS:
            ids = pipe.student_ids(img, prompt, caption(r)).to(device)
            with amp():
                h, n_img = student.hidden(pix, ids)
            logits = F.linear(h[0, n_img - 1:-1].float(), student.head().float())
            scores.append(-F.cross_entropy(logits, ids[0], reduction="sum").item())
        best = max(range(4), key=lambda i: scores[i])
        correct += RELATIONS[best] == rel
        others = [s for r, s in zip(RELATIONS, scores) if r != rel]
        margin += scores[RELATIONS.index(rel)] - max(others)
    student.train()
    return {"spatial_acc": correct / n, "spatial_margin_nats": margin / n}


def data_stream(args, skip: int, seed: int):
    if args.mock or not args.dataset:
        rng = random.Random(seed)
        for i in range(10**12):
            img, prompt, caption, rel = spatial_example(rng, max(args.image_size, 64))
            if i >= skip:
                yield img, prompt, caption(rel)
        return
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config or None, split=args.split, streaming=True).skip(skip)
    rng = random.Random(seed)
    for row in ds:
        ans = row[args.answer_col]
        if isinstance(ans, list):
            ans = ans[0]["answer"] if ans and isinstance(ans[0], dict) else (ans[0] if ans else "")
        if args.synthetic_frac and rng.random() < args.synthetic_frac:
            img, prompt, caption, rel = spatial_example(rng, args.teacher_image_size)
            yield img, prompt, caption(rel)
        yield row[args.image_col], (row[args.prompt_col] if args.prompt_col else args.prompt), str(ans)


def build(args, device):
    teacher = processor = None
    if args.mock:
        tok = distill.ByteTokenizer()
        vocab = 256
        encoder = PatchEncoder(args.image_size, args.patch, args.enc_dim)
        t_head_w = None
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(args.teacher)
        teacher = AutoModelForImageTextToText.from_pretrained(args.teacher, dtype=torch.bfloat16)
        teacher.to(device).eval().requires_grad_(False)
        tok = processor.tokenizer
        t_head_w = teacher.get_output_embeddings().weight
        vocab = t_head_w.size(0)
        encoder = HFEncoder(args.encoder)
    lm = distill.build_student(args, vocab)
    if args.init:
        state = torch.load(args.init, map_location="cpu", weights_only=False)
        lm.load_state_dict(state.get("model", state))
    student = VLStudent(encoder, lm, args.train_encoder or args.mock).to(device)
    return student, teacher, processor, tok, t_head_w


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", choices=["aether_v2", "vanilla"], required=True)
    ap.add_argument("--pattern", choices=sorted(distill.PATTERNS), default="hybrid_attn")
    ap.add_argument("--teacher", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--encoder", default="google/siglip2-base-patch16-224")
    ap.add_argument("--train-encoder", action="store_true")
    ap.add_argument("--init", default="", help="Aether text-stage checkpoint (ckpt.pt from distill.py)")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--dataset-config", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--prompt-col", default="question")
    ap.add_argument("--answer-col", default="answer")
    ap.add_argument("--prompt", default="Describe the scene for a blind person, with where things are.")
    ap.add_argument("--task", choices=["pair", "single"], default="pair",
                    help="pair: relation between two named objects; single: one square, left or right")
    ap.add_argument("--answer-only", action="store_true", help="score only the answer tokens")
    ap.add_argument("--answer-tokens", type=int, default=0,
                    help="with --answer-only, score only the first N answer tokens (0 = all)")
    ap.add_argument("--synthetic-frac", type=float, default=0.1, help="mix in synthetic spatial scenes")
    ap.add_argument("--teacher-image-size", type=int, default=448)
    ap.add_argument("--image-size", type=int, default=64, help="mock encoder input size")
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--enc-dim", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--d-inner", type=int, default=3072)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--kd-weight", type=float, default=0.9)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--loss-chunk", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--probe-n", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    TASK["name"] = args.task

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = lambda: torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    student, teacher, processor, tok, t_head_w = build(args, device)
    pipe = Pipeline(args, student, teacher, processor, tok)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    step, seen = 0, 0
    ckpt = out / "ckpt.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        student.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, seen = st["step"], st["seen"]
        print(f"resumed at step {step}", flush=True)
    stream = data_stream(args, seen, args.seed)
    meta = {"args": vars(args), "device": device,
            "params_lm": sum(p.numel() for p in student.lm.parameters()),
            "params_proj": sum(p.numel() for p in student.proj.parameters()),
            "params_encoder": sum(p.numel() for p in student.encoder.parameters())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    log = open(out / "log.jsonl", "a")
    last: dict = {}
    t0 = time.time()
    while step < args.steps:
        lr = distill.lr_at(step, args.steps, args.warmup_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        ce_acc = kl_acc = 0.0
        for _ in range(args.grad_accum):
            img, prompt, answer = next(stream)
            seen += 1
            pix, ids, th, start = pipe.example(img, prompt, answer, device)
            with amp():
                h, n_img = student.hidden(pix, ids)
                ce, kl = kd_text_loss(student, h, n_img, ids, th, t_head_w, args.kd_weight, args.temp,
                                      args.loss_chunk, start, args.answer_tokens)
                loss = ce if th is None else (1 - args.kd_weight) * ce + args.kd_weight * kl
            (loss / args.grad_accum).backward()
            ce_acc += float(ce.detach()) / args.grad_accum
            kl_acc += float(kl.detach()) / args.grad_accum
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if math.isfinite(gn):
            opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0 or step == args.steps:
            last = {"step": step, "lr": lr, "ce": ce_acc, "kl": kl_acc, "grad_norm": gn,
                    "examples_per_s": round(step * args.grad_accum / (time.time() - t0), 2)}
            print(json.dumps(last), flush=True)
            log.write(json.dumps(last) + "\n")
            log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            ev = spatial_probe(student, pipe, args.probe_n, device, amp)
            ev["step"] = step
            print("EVAL " + json.dumps(ev), flush=True)
            log.write(json.dumps({"eval": ev}) + "\n")
            log.flush()
            last.update(ev)
        if step % args.ckpt_every == 0 or step == args.steps:
            tmp = out / "ckpt.tmp"
            torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "step": step, "seen": seen}, tmp)
            os.replace(tmp, ckpt)
    log.close()
    return last


if __name__ == "__main__":
    main()
