"""Knowledge distillation from a Hugging Face teacher into an Aether student.

The rehearsal: distil the same teacher into two parameter-matched students that differ
only in their sequence mixers (aether_v2 hybrid vs vanilla attention), with the same data,
tokens, optimiser and schedule, and compare held-out loss, retrieval and speed.

    python scripts/distill.py --student aether_v2 --out runs/kd_aether
    python scripts/distill.py --student vanilla   --out runs/kd_vanilla

Loss per token: (1 - kd_weight) * CE(labels) + kd_weight * T^2 * KL(teacher_T || student_T)
over the full vocabulary. Logits are never materialised for the whole sequence: each chunk
of positions is projected, scored and discarded, and recomputed in backward (checkpointed).

Restartable: `--out/ckpt.pt` is rewritten atomically every `--ckpt-every` steps and loaded
on start, including the data position, so a preempted instance just reruns the command.

`--mock` swaps the teacher for a tiny random Qwen3 and the data for local text, so the full
pipeline runs on a CPU with no network (used by tests/test_distill.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aether.model import AetherConfig, AetherForCausalLM  # noqa: E402

PATTERNS = {
    "hybrid": ("conv", "ssm", "conv", "fsa2"),
    "ssm3": ("ssm", "ssm", "ssm", "fsa2"),
    "hybrid_attn": ("conv", "ssm", "conv", "attn"),
    "ssm3_attn": ("ssm", "ssm", "ssm", "attn"),
}


# ---------------------------------------------------------------------------------------
# Teacher, tokenizer, data
# ---------------------------------------------------------------------------------------
class ByteTokenizer:
    """Stand-in tokenizer for --mock runs (no network)."""

    eos_token_id = 0

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [list(t.encode("utf-8")) for t in texts]}

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def load_teacher(args, device):
    if args.mock:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(1234)
        cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                          max_position_embeddings=4096, tie_word_embeddings=True)
        teacher, tok = Qwen3ForCausalLM(cfg), ByteTokenizer()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.teacher)
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16)
    teacher.to(device).eval().requires_grad_(False)
    return teacher, tok


def doc_stream(args, skip_docs: int):
    """Yields raw documents (strings). The first `--eval-docs` are reserved for evaluation."""
    if args.mock:
        import sysconfig

        files = sorted(Path(sysconfig.get_paths()["stdlib"]).rglob("*.py"))
        texts = (f.read_text(errors="ignore") for f in files)
        for i, t in enumerate(texts):
            if i >= skip_docs and t.strip():
                yield t
        return
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config or None, split=args.split, streaming=True)
    ds = ds.skip(skip_docs)
    for row in ds:
        yield row[args.text_field]


class Packer:
    """Tokenises documents and packs them, EOS-separated, into fixed-length rows."""

    def __init__(self, tok, docs, seq_len: int, eos: int):
        self.tok, self.docs, self.seq_len, self.eos = tok, docs, seq_len, eos
        self.buf: list[int] = []
        self.docs_used = 0

    def rows(self, n: int) -> torch.Tensor:
        need = n * self.seq_len
        while len(self.buf) < need:
            batch = [next(self.docs) for _ in range(16)]
            self.docs_used += len(batch)
            for ids in self.tok(batch, add_special_tokens=False)["input_ids"]:
                self.buf.extend(ids + [self.eos])
        out, self.buf = self.buf[:need], self.buf[need:]
        return torch.tensor(out, dtype=torch.long).view(n, self.seq_len)


# ---------------------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------------------
def build_student(args, vocab: int) -> AetherForCausalLM:
    pattern = None
    if args.student == "aether_v2":
        unit = PATTERNS[args.pattern]
        if args.layers % len(unit):
            raise SystemExit(f"--layers must be a multiple of {len(unit)} for pattern {args.pattern}")
        pattern = unit * (args.layers // len(unit))
    cfg = AetherConfig(
        vocab_size=vocab, d_model=args.d_model, d_inner=args.d_inner, n_layers=args.layers,
        n_heads=args.d_model // 64, d_k=64, max_seq_len=args.seq_len, arch=args.student,
        layer_pattern=pattern, mod_routing_points=(), mod_keep_ratios=(),
        ssm_state_dim=64, ssm_head_dim=64, fsa2_chunk_size=64, fsa2_n_summary=8,
    )
    return AetherForCausalLM(cfg)


# ---------------------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------------------
def _chunk_loss(h, w, t_logits, targets, kd_weight: float, temp: float):
    s_logits = F.linear(h, w).float()
    ce = F.cross_entropy(s_logits, targets, reduction="sum")
    if kd_weight == 0.0:
        return ce, ce.new_zeros(())
    t_logp = F.log_softmax(t_logits.float() / temp, dim=-1)
    s_logp = F.log_softmax(s_logits / temp, dim=-1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum() * temp * temp
    return ce, kl


def kd_loss(student, h, t_hidden, t_head_w, ids, kd_weight, temp, chunk):
    """Sum over positions of CE and KL, chunked and checkpointed; returns per-token means."""
    w = student.token_embedding.weight if student.lm_head is None else student.lm_head.weight
    b, L, _ = h.shape
    ce_tot, kl_tot, n = h.new_zeros((), dtype=torch.float32), h.new_zeros((), dtype=torch.float32), 0
    hs, tgt = h[:, :-1].reshape(-1, h.size(-1)), ids[:, 1:].reshape(-1)
    th = t_hidden[:, :-1].reshape(-1, t_hidden.size(-1)) if t_hidden is not None else None
    for s in range(0, hs.size(0), chunk):
        e = min(s + chunk, hs.size(0))
        with torch.no_grad():
            tl = F.linear(th[s:e], t_head_w) if th is not None else torch.empty(0)
        if torch.is_grad_enabled():
            ce, kl = checkpoint(_chunk_loss, hs[s:e], w, tl, tgt[s:e], kd_weight, temp, use_reentrant=False)
        else:
            ce, kl = _chunk_loss(hs[s:e], w, tl, tgt[s:e], kd_weight, temp)
        ce_tot, kl_tot, n = ce_tot + ce, kl_tot + kl, n + (e - s)
    return ce_tot / n, kl_tot / n


@torch.no_grad()
def teacher_hidden(teacher, ids):
    out = teacher.model(input_ids=ids)
    return out.last_hidden_state


# ---------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(student, teacher, t_head_w, eval_rows, byte_len, args, device, amp):
    student.eval()
    res = {"student_nll": 0.0, "teacher_nll": 0.0, "bytes": 0.0, "tokens": 0}
    for i in range(0, eval_rows.size(0), args.micro_bsz):
        ids = eval_rows[i : i + args.micro_bsz].to(device)
        with amp():
            h = student.hidden(ids)
            ce, _ = kd_loss(student, h, None, None, ids, 0.0, 1.0, args.loss_chunk)
            th = teacher_hidden(teacher, ids)
            t_logits = F.linear(th[:, :-1], t_head_w).float()
        n = ids[:, 1:].numel()
        res["student_nll"] += float(ce) * n
        res["teacher_nll"] += float(F.cross_entropy(t_logits.reshape(-1, t_logits.size(-1)),
                                                    ids[:, 1:].reshape(-1), reduction="sum"))
        res["bytes"] += float(byte_len[ids[:, 1:].cpu()].sum())
        res["tokens"] += n
    out = {
        "student_loss": res["student_nll"] / res["tokens"],
        "teacher_loss": res["teacher_nll"] / res["tokens"],
        "student_bpb": res["student_nll"] / math.log(2) / res["bytes"],
        "teacher_bpb": res["teacher_nll"] / math.log(2) / res["bytes"],
    }
    out.update(copy_probe(student, eval_rows, args, device, amp))
    student.train()
    return out


@torch.no_grad()
def copy_probe(student, eval_rows, args, device, amp, span: int = 128, n: int = 8) -> dict:
    """Held-out span repeated 3x; loss on each copy. Retrieval drives copies 1-2 toward 0."""
    span = min(span, args.seq_len // 3)
    n = min(n, eval_rows.size(0))
    ids = eval_rows[:n, :span].repeat(1, 3).to(device)
    with amp():
        logits = F.linear(student.hidden(ids), student.token_embedding.weight
                          if student.lm_head is None else student.lm_head.weight).float()
    lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1),
                         reduction="none").view(n, -1)
    c = [float(lp[:, max(k * span, 8) : (k + 1) * span - 1].mean()) for k in range(3)]
    return {"copy0_loss": c[0], "copy2_loss": c[2], "copy_gain": c[0] - c[2]}


# ---------------------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------------------
def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", choices=["aether_v2", "vanilla"], required=True)
    ap.add_argument("--pattern", choices=sorted(PATTERNS), default="hybrid")
    ap.add_argument("--teacher", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--dataset-config", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--d-inner", type=int, default=3072)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-bsz", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--tokens", type=float, default=1e9)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--kd-weight", type=float, default=0.9)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--loss-chunk", type=int, default=2048)
    ap.add_argument("--eval-docs", type=int, default=400)
    ap.add_argument("--eval-rows", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--act-ckpt", action="store_true", help="recompute each block in backward (less memory)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (testing)")
    ap.add_argument("--mock", action="store_true", help="tiny random teacher + local text (CPU test)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = (lambda: torch.autocast(device, dtype=torch.bfloat16))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    teacher, tok = load_teacher(args, device)
    t_head_w = teacher.get_output_embeddings().weight
    vocab = t_head_w.size(0)
    eos = tok.eos_token_id
    byte_len = torch.tensor([len(tok.decode([i]).encode("utf-8")) for i in range(vocab)], dtype=torch.float32)

    torch.manual_seed(0)
    student = build_student(args, vocab).to(device)
    n_params = student.num_parameters()
    tokens_per_step = args.micro_bsz * args.grad_accum * args.seq_len
    total_steps = max(1, int(args.tokens // tokens_per_step))

    decay, no_decay = [], []
    for name, p in student.named_parameters():
        (no_decay if p.ndim < 2 or "token_embedding" in name or name.endswith(("dt_bias", "A_log", "D")) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=device == "cuda")

    eval_packer = Packer(tok, doc_stream(args, 0), args.seq_len, eos)
    eval_rows = eval_packer.rows(args.eval_rows)
    if eval_packer.docs_used > args.eval_docs:
        raise SystemExit(f"eval set used {eval_packer.docs_used} docs > --eval-docs {args.eval_docs}; raise it")
    step, docs_skip = 0, args.eval_docs
    ckpt = out / "ckpt.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        student.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, docs_skip = state["step"], state["docs_skip"]
        print(f"resumed at step {step} (docs consumed {docs_skip})", flush=True)
    packer = Packer(tok, doc_stream(args, docs_skip), args.seq_len, eos)

    def hidden_ckpt(ids):
        x = student.token_embedding(ids)
        for blk in student.blocks:
            x = checkpoint(lambda t, blk=blk: blk(t)[0], x, use_reentrant=False)
        return student.norm(x)

    fwd = hidden_ckpt if args.act_ckpt else student.hidden
    fwd = torch.compile(fwd) if args.compile else fwd
    meta = {"args": vars(args), "params": n_params, "tokens_per_step": tokens_per_step,
            "total_steps": total_steps, "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    log = open(out / "log.jsonl", "a")

    def save():
        tmp = out / "ckpt.tmp"
        torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "step": step,
                    "docs_skip": docs_skip + packer.docs_used}, tmp)
        os.replace(tmp, ckpt)

    last = {}
    t0, tok_seen = time.time(), 0
    stop = min(total_steps, args.max_steps) if args.max_steps else total_steps
    while step < stop:
        lr = lr_at(step, total_steps, args.warmup_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        ce_acc = kl_acc = 0.0
        for _ in range(args.grad_accum):
            ids = packer.rows(args.micro_bsz).to(device, non_blocking=True)
            with amp():
                th = teacher_hidden(teacher, ids) if args.kd_weight > 0 else None
                h = fwd(ids)
                ce, kl = kd_loss(student, h, th, t_head_w, ids, args.kd_weight, args.temp, args.loss_chunk)
                loss = (1 - args.kd_weight) * ce + args.kd_weight * kl
            (loss / args.grad_accum).backward()
            ce_acc += float(ce.detach()) / args.grad_accum
            kl_acc += float(kl.detach()) / args.grad_accum
        gn = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
        if math.isfinite(gn):
            opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tok_seen += tokens_per_step
        if step % args.log_every == 0 or step == stop:
            dt = time.time() - t0
            last = {"step": step, "lr": lr, "ce": ce_acc, "kl": kl_acc, "grad_norm": gn,
                    "tok_per_s": round(tok_seen / dt), "elapsed_h": round(dt / 3600, 3)}
            if device == "cuda":
                last["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            print(json.dumps(last), flush=True)
            log.write(json.dumps(last) + "\n")
            log.flush()
        if step % args.eval_every == 0 or step == stop:
            ev = evaluate(student, teacher, t_head_w, eval_rows, byte_len, args, device, amp)
            ev["step"] = step
            print("EVAL " + json.dumps(ev), flush=True)
            log.write(json.dumps({"eval": ev}) + "\n")
            log.flush()
            last.update(ev)
        if step % args.ckpt_every == 0 or step == stop:
            save()
    log.close()
    return last


if __name__ == "__main__":
    main()
