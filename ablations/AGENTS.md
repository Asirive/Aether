# ablations/ — Ablation Protocol

## Goal

Validate that each of the 3 components (CC-FSA, PolyMamba-2, MoD-GDR) trains stably and does
not regress vs a baseline.

## Protocol

All ablation scripts (`ablations/00_*` through `04_*`) train on **random tokens**
(`torch.randint`). They are **SMOKE TESTS ONLY**: they verify that each variant trains without
NaN, that loss decreases, and that aux losses stay bounded. They do NOT measure component
quality (random tokens have no learnable structure; perplexity differences reflect
capacity-to-memorize, not component contribution).

For real component ablations, use `configs/aether_25m_proxy.yaml` with the actual
FineWeb-Edu stream (the trainer's data path) and the 25M proxy model — not these scripts.

## Variants

| Script | Components | Purpose |
|--------|------------|---------|
| `00_baseline.py` | Standard transformer | Reference quality floor |
| `01_fsa_only.py` | +FSA in odd blocks | Isolate FSA contribution |
| `02_polymamba_only.py` | +PolyMamba-2 in even blocks | Isolate PolyMamba-2 contribution |
| `03_mod_only.py` | +MoD-GDR routing | Isolate MoD-GDR contribution |
| `04_aether_full.py` | All 3 components | Full model |

## Execution

```bash
for i in 00 01 02 03 04; do
    python ablations/${i}_*.py --steps 12500 --output ablations/results/${i}
done
```

Each run: 12,500 steps × ~0.3 sec/step ≈ **1 hour per ablation**. **5 hours total.**

## Results Format

Each script writes to `ablations/results/{name}/`:
- `final_model.pt`: model checkpoint
- `metrics.json`: `{"eval_loss": X, "perplexity": Y, "step_count": Z}`
- `training_log.jsonl`: step-by-step loss + aux loss components

## Analysis

After all 5 ablations complete, compare:
- `eval_loss` of each variant
- Component contribution = `(aether_full_loss - baseline_loss) - Σ(individual_losses - baseline_loss)`
- Positive = components interfere constructively; negative = components conflict

If any individual component *hurts* compared to baseline, investigate:
- Is the auxiliary loss weight too high?
- Is the component collapsing (e.g., MoD always-routes-to-same-tokens)?
- Is the cold-start initialization broken?

## Critical Checks

- **Baseline trains successfully**: If 00_baseline doesn't converge, the test setup is broken.
- **Each variant trains**: If 01/02/03/04 NaN, that's a component bug — fix before scaling up.
- **Loss decreases monotonically over first 1000 steps**: If not, LR or init is wrong.
- **Aux losses stay bounded**: `L_ortho < 10`, `L_kbal < 0.5`, `L_mbd < 1.0` throughout training.

## Reporting

Final ablation table format (paste into paper):
```
| Variant | Eval Loss | Perplexity | Δ vs Baseline |
|---------|-----------|------------|---------------|
| Baseline | X | Y | — |
| +FSA only | X | Y | +Z |
| +PolyMamba-2 only | X | Y | +Z |
| +MoD only | X | Y | +Z |
| +All (AETHER-Lite) | X | Y | +Z |
```
