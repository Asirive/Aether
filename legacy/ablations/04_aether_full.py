from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aether.model import AetherConfig, AetherForCausalLM
from aether.utils.device import select_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="ablations/results/04_aether_full")
    parser.add_argument("--proxy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    torch.manual_seed(42)

    if args.proxy:
        config = AetherConfig(
            d_model=384,
            d_inner=1024,
            n_layers=6,
            n_heads=6,
            max_seq_len=args.seq_len,
            mod_routing_points=(2, 4, 6),
            mod_keep_ratios=(0.50, 0.70, 1.00),
        )
    else:
        config = AetherConfig()

    model = AetherForCausalLM(config).to(device).to(torch.bfloat16)
    n_params = model.num_parameters()
    print(f"Model has {n_params:,} parameters")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    log: list[dict] = []
    model.train()
    for step in range(args.steps):
        input_ids = torch.randint(0, config.vocab_size, (args.batch, args.seq_len), device=device)
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 100 == 0:
            aux = out.get("aux_losses", {})
            log.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "task_loss": out["task_loss"].item(),
                    "mod": aux.get("mod", torch.zeros(())).item() if isinstance(aux.get("mod"), torch.Tensor) else 0.0,
                }
            )
            print(f"step {step}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    with (output_dir / "training_log.jsonl").open("w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")

    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, config.vocab_size, (args.batch, args.seq_len), device=device)
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
    metrics = {
        "eval_loss": out["loss"].item(),
        "perplexity": float(torch.exp(out["loss"]).item()),
        "step_count": args.steps,
        "params": n_params,
        "name": "aether_full",
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"DONE: {metrics}")


if __name__ == "__main__":
    main()
