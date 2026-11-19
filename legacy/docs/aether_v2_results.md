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
* Compressed vs full attention in the 1-in-4 slot: within noise on bpb (-0.018 and -0.003
  for full), but full attention wins decisively on exact retrieval (section 5) and is the
  better-supported op on an NPU.
* All-SSM is worse than SSM + some attention.

## 4. Fixes made along the way

* **SSM step-0 gradient spike (88 vs 3 for attention)**: the shared init gave the SSM input
  projection std 0.02/sqrt(d), so the output gate was ~0 and the gated RMSNorm amplified its
  gradient. Mamba-style init (1/sqrt(3 d_in)) cut it to 18 and improved bpb by ~0.02-0.07.
* **Strict-metric "gap"**: the prefix-only metric sampled different positions from the full
  metric. Now both are reported at the same positions; for v2 they match exactly.
* **Lost results**: parallel jobs overwrote a shared JSON. Writes are locked now and the
  analyzer also reads every RESULT line from the logs.

## 5. In-context retrieval (copy task)

The language-model runs are too short for any model, attention included, to learn copying
(copy gain -0.14 to -0.16 everywhere), and MQAR does not train within the CPU budget. So
retrieval was tested directly: `ablations/recall.py --task copy` trains 4-layer models to
reproduce a span of 128 random tokens after seeing it once (1000 steps; results/14_copy).

| 4-layer design | exact-copy accuracy |
|---|---|
| attention x4 | 100% |
| **conv, SSM, conv, attention (v2)** | **100% / 100% (seed 0 / 1)** |
| LFM3-style: conv, conv, SWA, attention | 3.6% / 15.7% (seed 0 / 1) |
| conv, SSM, conv, compressed attention | 3.4% / 1.7% (seed 0 / 1) |
| SSM, SSM, SSM, compressed attention | 0.8% |
| SSM x4 | 0.4% |
| conv x4 (control) | 0.4% |

* The v2 design keeps attention's exact retrieval: one full-attention layer in four suffices.
* Compressed attention cannot copy exactly from beyond its raw window (summaries are lossy),
  and SSMs alone cannot store 128 random tokens. This is why v2 uses full attention in the
  1-in-4 slot, not fsa2, even though the two tie on bits per byte.
* The LFM3-style stack also has one full-attention layer but did not learn the task in 1000
  steps (3.6% and 15.7% over two seeds).

## 6. Scale-up check (2.2x parameters, 4x steps)

d_model 384, SwiGLU 1152, 8 layers (~15.4M params), 3200 steps (results/13_scale, seed 0;
seed 1 queued).

| 8-layer design | bpb, all positions | bpb, strict prefix-only | copy gain (bits) |
|---|---|---|---|
| transformer | 1.601 | 1.627 | -0.197 |
| LFM3-style | 1.477 | 1.599 | -0.119 |
| **conv, SSM, conv, attn (v2)** | **1.305** | **1.505** | **+0.192** |

* v2 still leads both: -0.30 bpb (-19%) vs the transformer and -0.17 vs LFM3 on all
  positions; -0.12 and -0.09 on the strict metric (which samples fewer, earlier positions).
* The strict-metric margin over the transformer is smaller than at 7M (-0.12 vs -0.36), so
  the advantage is not guaranteed to hold at 0.5B; that is the rehearsal's job.
* v2 is the only design whose copy gain turned positive: with longer training it started
  using the context to predict repeated text (repeat costs 1.22 bits/byte vs 1.41 the
  first time). The transformer and LFM3 had not learned this yet at the same step.
* Max gradient norm: transformer 83, v2 17, LFM3 13.

## 7. Deployment check at the real size (0.54B, static-shape ONNX)

`scripts/export_onnx.py` exports one decode step with every shape fixed, as the Snapdragon
NPU runtime (QNN through ONNX Runtime) requires: conv/SSM layers carry their constant-size
state, attention layers a 2048-slot KV buffer written by mask. Untrained weights, ONNX
Runtime on 1 x86 core with other jobs running, so only the ratio matters.

| 0.54B student | ONNX matches PyTorch | dynamic dims | decode state (fp32, 2k ctx) | ms per token |
|---|---|---|---|---|
| vanilla transformer (28 attn) | 1e-5 | 0 | 470 MB | 729 |
| **v2 (14 conv, 7 SSM, 7 attn)** | 5e-6 | 0 | **120 MB** | **254** |

* Both export cleanly with only standard ops (MatMul, Mul/Add, Softmax, Where, Sigmoid,
  Exp, Softplus, ReduceMean/Sum); no Einsum, loops or dynamic shapes.
* The 2.9x latency gap here is mostly the cost of moving the KV state in and out each step;
  on compute alone (per-token multiply-adds) v2 saves 13% at 2k context, 35% at 8k and 58%
  at 32k, because only 7 of 28 layers read a KV cache.
* KV memory per token of context: 229 KB (transformer) vs 57 KB (v2), fp32. At 32k context
  that is 7.5 GB vs 1.9 GB (halve both for fp16), which decides what fits next to the
  vision and audio models on a 16-32 GB board.
* Not tested here: INT8 accuracy on trained weights and the QNN backend itself (needs the
  Q8B and Qualcomm's SDK).

## 8. What the CPU lab cannot answer
* **Scale.** 15M parameters and 3200 steps at most. Effects can shrink or flip at 0.5-1B.
* **Real speed.** The SSM scan is plain PyTorch; the NPU/GPU numbers need the target hardware.

## Recommended v2 for the rehearsal and the Snapdragon target

`conv, SSM, conv, attn` repeated (28 layers at 0.54B), no router: equal-best quality,
exact retrieval as good as a transformer, fastest of the SSM designs, constant-memory decode in 21 of 28 layers, and only standard
ops (matmul, depthwise conv, elementwise, softmax attention). Run in `scripts/distill.py`
with `--pattern hybrid_attn`.
