# Aether component study and the v2 architecture (CPU lab results)

Every number here comes from `ablations/lab.py`: ~6.9M-parameter 8-layer models, 800 steps
x 16 x 256 bytes of Python source, bf16, same data order for every arm, held-out bits per
byte (lower is better). Seed-to-seed SD within a cell is about 0.007-0.01 bpb. Raw results
are in `ablations/results/` (06 = v1 factorial, 07 = v2 factorial, 08 = layer patterns,
10 = LFM3-style comparison, 11 = after the SSM init fix, 12 = copy probe).

These are small, short runs. They rank architectures reliably at this scale; they do not
prove anything at 1B parameters. That is what the GPU rehearsal (`docs/vast_rehearsal.md`) is for.

## 1. Original Aether (v1) components, 2^3 factorial

Factors: F = CC-FSA instead of attention in odd layers, P = PolyMamba-2 instead of attention
in even layers, M = MoD-GDR router. Metric: strictly causal (prefix-only) bpb.

| cell | bpb | vs transformer |
|---|---|---|
| transformer | 2.161 | |
| +P (PolyMamba-2) | 2.123 | -0.038 (but 22% more parameters) |
| +F (CC-FSA) | 2.227 | +0.066 |
| +M (MoD-GDR) | 2.401 | +0.240 |
| F+P+M (= Aether v1) | 2.413 | +0.252 |

* No v1 component helped by more than noise, and F and M hurt. F and P interfere
  (adding F on top of P: +0.31).
* **MoD-GDR is not causal**: its sequence-level top-k lets future tokens change past
  logits by up to 3.2 (measured). Its training loss is optimistic; generation is worse.
* Root causes found in code: PolyMamba-2 is a time-invariant SSM with a 16-dim state shared
  by all channels and a 0.95 decay cap (memory ~20 tokens); CC-FSA zeroes the DC bin by
  mean-centring before the DCT, uses no positional encoding and a fixed gate between paths.

## 2. Aether v2 components (each fixes a v1 defect), 2^3 factorial

F2 = chunk-compressed attention (`fsa2`), P2 = selective SSM (Mamba-2 / SSD form, `pm2`),
M2 = causal per-token depth router (`mod2`). All three verified causal (leak = 0).

| effect | bpb |
|---|---|
| P2 main effect | **-0.477** |
| F2 main effect | +0.061 |
| M2 main effect | +0.090 |
| add F2 on top of P2 | -0.020 (synergy: F2 hurts alone, helps with P2) |
| add M2 on top of P2 | -0.012 (neutral) |

The selective SSM is the component that matters. The router gives nothing and needs
data-dependent control flow, which NPUs handle badly, so v2 drops it.

## 3. Layer patterns (after the SSM init fix), 2 seeds each

| 8-layer pattern | bpb (mean) | CPU tok/s |
|---|---|---|
| transformer (all attention) | 2.038 | 3,250 |
| LFM3-style: conv, conv, SWA, attn (x2) | 1.820 | 3,600 |
| all SSM | 1.753 (1 seed, pre-fix) | 2,060 |
| SSM, SSM, SSM, fsa2 (x2) | 1.678 | 2,150 |
| SSM, SSM, SSM, attn (x2) | 1.660 | 2,190 |
| conv, SSM, conv, fsa2 (x2) | 1.680 | 2,660-3,050 |
| **conv, SSM, conv, attn (x2)** | **1.677** | **3,050** |
| LFM3 layout with our SSM instead of conv | 1.683 (1 seed) | 2,470 |

* Any pattern with SSMs plus one attention-type layer in four beats the transformer by
  ~0.36 bpb (18%) and LFM3-style by ~0.14 bpb (8%), far outside seed noise.
* Half the SSM layers can be cheap gated short convs with no quality loss and ~1.4x speed.
* Same-slot comparison: SSM beats short conv by 0.10 bpb (attn/pm2 1.73 vs attn/conv 1.83).
  What beats LFM3 is the SSM, not the layout.
* Compressed vs full attention in the 1-in-4 slot: within noise (-0.018 and -0.003 for full).
  Full attention is the simpler, better-supported op on an NPU; fsa2 is kept as the option
  for long contexts where the KV cache matters.
* All-SSM is worse than SSM + some attention.

## 4. Fixes made along the way

* **SSM step-0 gradient spike (88 vs 3 for attention)**: the shared init gave the SSM input
  projection std 0.02/sqrt(d), so the output gate was ~0 and the gated RMSNorm amplified its
  gradient. Mamba-style init (1/sqrt(3 d_in)) cut it to 18 and improved bpb by ~0.02-0.07.
* **Strict-metric "gap"**: the prefix-only metric sampled different positions from the full
  metric. Now both are reported at the same positions; for v2 they match exactly.
* **Lost results**: parallel jobs overwrote a shared JSON. Writes are locked now and the
  analyzer also reads every RESULT line from the logs.

## 5. What the CPU lab cannot answer

* **In-context retrieval.** At this scale no model, including pure attention, learns to copy
  a repeated span (copy gain -0.14 to -0.16 for every model), and the MQAR task does not train
  within the CPU budget. Whether the SSM-heavy design gives up retrieval is the first thing
  the GPU rehearsal measures (`copy_gain` in `scripts/distill.py`).
* **Scale.** 7M parameters and 800 steps. Effects can shrink or flip at 0.5-1B.
* **Real speed.** The SSM scan is plain PyTorch; the NPU/GPU numbers need the target hardware.

## Recommended v2 for the rehearsal and the Snapdragon target

`conv, SSM, conv, attn` repeated (28 layers at 0.54B), no router: equal-best quality,
fastest of the SSM designs, constant-memory decode in 21 of 28 layers, and only standard
ops (matmul, depthwise conv, elementwise, softmax attention). Run in `scripts/distill.py`
with `--pattern hybrid_attn`.
