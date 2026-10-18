from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.components.fsa import FreqSparseAttention
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


class StandardMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_inner: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.in_proj(x)
        x_val, x_gate = x_proj.chunk(2, dim=-1)
        x_inner = F.silu(x_gate) * x_val
        return self.out_proj(x_inner)


class FSABlock(nn.Module):
    def __init__(self, d_model: int, d_inner: int, n_heads: int, d_k: int, max_seq_len: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.fsa = FreqSparseAttention(d_model, n_heads, d_k, max_seq_len=max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn = StandardMambaBlock(d_model, d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.fsa(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FSAModel(nn.Module):
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
            [FSABlock(d_model, d_inner, n_heads, d_k, max_seq_len) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.token_embedding(input_ids)
        ortho_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
            ortho_loss = ortho_loss + block.fsa.orthogonality_loss()
        x = self.norm(x)
        logits = F.linear(x, self.token_embedding.weight)
        result: dict = {"logits": logits, "ortho_loss": ortho_loss}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = task_loss + 0.001 * ortho_loss
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="ablations/results/01_fsa_only")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    torch.manual_seed(42)

    model = FSAModel(max_seq_len=args.seq_len).to(device).to(torch.bfloat16)
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
            log.append({"step": step, "loss": loss.item(), "ortho": out["ortho_loss"].item()})
            print(f"step {step}: loss={loss.item():.4f} ortho={out['ortho_loss'].item():.4f}")

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
        "name": "fsa_only",
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"DONE: {metrics}")


if __name__ == "__main__":
    main()
