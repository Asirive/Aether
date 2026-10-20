# Benchmark Targets

## Goal

**AETHER-200M-Instruct should match or exceed SmolLM2-360M-Instruct on standard benchmarks at ~55% the parameters.**

### Honest expectation note (31 Jul 2026)

SmolLM2-360M was pretrained on **4 trillion tokens**. AETHER-200M will train on **6.55B tokens**
(100K steps x 65,536), roughly **600x fewer**. Realistically this places AETHER-200M closer to
the SmolLM2-135M class than the 360M class on knowledge-intensive benchmarks. The Floor tier
below is the honest expectation; Target/Stretch require either far more compute or distillation
from a larger model. The paper is publishable if the ablation story is clean and the gap is
well-diagnosed.

## Reference Models

### SmolLM2-360M-Instruct
- Params: 360M
- Training tokens: 4T (SmolLM2-360M base), then SFT/DPO
- Benchmarks (from SmolLM2 paper):
  - MMLU: 31.6
  - HellaSwag: 48.6
  - ARC-C: 36.0
  - PIQA: 72.4
  - WinoGrande: 60.4
  - GSM8K (5-shot): 14.4
  - HumanEval: 12.2
  - TruthfulQA: 39.0

### SmolLM2-135M-Instruct
- Params: 135M
- MMLU: 28.5
- HellaSwag: 41.7
- ARC-C: 30.2
- GSM8K: 5.1

### MobileLLM-350M
- Params: 350M
- MMLU: ~30
- HellaSwag: ~48
- ARC-C: ~35

## AETHER-200M Targets

| Tier | Strategy | Outcome |
|------|----------|---------|
| **Floor** | Just trains stably, hits floor numbers | Match SmolLM2-135M-Instruct |
| **Target** | Components contribute as designed | Match SmolLM2-360M-Instruct on most metrics |
| **Stretch** | All ablations show positive delta | Beat SmolLM2-360M-Instruct by 2–5 points avg |

### Target Benchmark Numbers

| Benchmark | Floor | Target | Stretch |
|-----------|-------|--------|---------|
| MMLU (5-shot) | 28 | 32 | 36 |
| HellaSwag | 42 | 50 | 55 |
| ARC-Challenge | 30 | 38 | 42 |
| PIQA | 70 | 75 | 78 |
| WinoGrande | 58 | 62 | 65 |
| GSM8K (5-shot) | 5 | 14 | 18 |
| HumanEval | 5 | 12 | 16 |
| TruthfulQA | 35 | 40 | 45 |
| IFEval | 25 | 40 | 50 |

## Eval Methodology

### Tools
- **lm-eval-harness** (EleutherAI) — standard
- **lighteval** (HuggingFace) — alternative, used by SmolLM2
- Pick ONE for consistency with prior work; lm-eval-harness preferred for transparency

### Few-shot Counts
Always use the same few-shot count as the reference model:
- MMLU: 5-shot
- HellaSwag: 10-shot (some papers use 0-shot; document explicitly)
- ARC-C: 25-shot
- PIQA: 5-shot
- WinoGrande: 5-shot
- GSM8K: 5-shot
- HumanEval: pass@1 (0-shot)
- TruthfulQA: 0-shot MC2
- IFEval: 0-shot

### Prompt Format
- Use the model's chat template for IFEval (instruct model)
- Use raw completions for base benchmarks (MMLU, ARC, etc.)
- Document any deviations from lm-eval-harness defaults

### Reporting
- Report all numbers in a single table with few-shot count and eval tool version
- Include baseline numbers (SmolLM2 family, MobileLLM family) for context
- Run eval 3 times with different seeds; report mean ± std
- For pass@1 on HumanEval: report average over k=1 samples (single greedy generation)

## Stretch Goals (Beyond Benchmark Numbers)

1. **Long-context eval:** needle-in-haystack at 4K and 8K context
2. **Reasoning eval:** MMLU-Pro (if feasible at 200M)
3. **Throughput:** tokens/sec at inference (target: ≥ 50 tok/sec on RX 7600 bf16)
4. **Memory at inference:** GPU VRAM at L=4096 (target: < 4GB for 200M)
5. **Latency:** time-to-first-token, inter-token latency

## Failure Mode Reporting

If AETHER-200M underperforms targets, document:
1. Which component failed (FSA / PolyMamba-2 / MoD)
2. Whether failure is from training instability, undertraining, or genuine architectural limitation
3. Specific numbers from ablation runs
4. Comparison to baseline (same recipe, no novel components)

The paper is publishable even if AETHER-200M underperforms targets, IF the ablation story is clean (each component contributes positively) and the failure is well-diagnosed.
