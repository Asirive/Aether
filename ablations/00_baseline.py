from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.utils.device import select_device
from aether.utils.numerics import to_fp32


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = to_fp32(x)
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(x.dtype) * self.weight


class StandardAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.w_qkv = nn.Linear(d_model, 3 * n_heads * d_k, bias=False)
        self.w_o = nn.Linear(n_heads * d_k, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.w_qkv(x).view(bsz, seq_len, 3, self.n_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores.float(), dim=-1).to(x.dtype)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.w_o(out)


class StandardBlock(nn.Module):
    def __init__(self, d_model: int, d_inner: int, n_heads: int, d_k: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = StandardAttention(d_model, n_heads, d_k)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_inner, bias=False),
            nn.GELU(),
            nn.Linear(d_inner, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class StandardTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32_000,
        d_model: int = 384,
        d_inner: int = 1024,
        n_layers: int = 6,
        n_heads: int = 6,
        d_k: int = 64,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [StandardBlock(d_model, d_inner, n_heads, d_k) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = F.linear(x, self.token_embedding.weight)
        result = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="ablations/results/00_baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    torch.manual_seed(42)

    model = StandardTransformer(max_seq_len=args.seq_len).to(device).to(torch.bfloat16)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    log: list[dict] = []
    model.train()
    for step in range(args.steps):
        input_ids = torch.randint(0, 32000, (args.batch, args.seq_len), device=device)
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 100 == 0:
            log.append({"step": step, "loss": loss.item()})
            print(f"step {step}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    with (output_dir / "training_log.jsonl").open("w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")

    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, 32000, (args.batch, args.seq_len), device=device)
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
    metrics = {
        "eval_loss": out["loss"].item(),
        "perplexity": float(torch.exp(out["loss"]).item()),
        "step_count": args.steps,
        "name": "baseline",
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"DONE: {metrics}")


if __name__ == "__main__":
    main()
