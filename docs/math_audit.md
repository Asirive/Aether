# 10-Agent Architecture Audit (Frozen)

**Date:** 22 July 2026
**Status:** FROZEN — do not re-litigate without strong new evidence

## Background

AETHER went through an iterative design process. The original design had 6 "novel" components. A 10-agent parallel audit was run to verify math, novelty, feasibility, stability, and coherence. This document records the findings so future work doesn't repeat the analysis or fall for the same overclaims.

## Agent Mandates

1. **Math/FLOPs auditor** — verify every FLOP count, memory calc
2. **Novelty auditor** — confirm/dispute each "novel" claim against literature
3. **DirectML feasibility auditor** — check op coverage on AMD stack
4. **Stability auditor** — gradient flow, init, NaN/Inf risks
5. **Compute efficiency comparator** — AETHER vs standard transformer
6. **SOTA comparator** — what is actually SOTA at 1B in 2026
7. **Memory budget auditor** — does AETHER fit in 8GB
8. **Hyperbolic correctness auditor** — GDR math + TT-Mem addressing
9. **TT-Mem correctness auditor** — tensor-train math + novelty
10. **Coherence critic** — does the architecture hold together as a whole

## CRITICAL Findings

### C1. AR-FFN arithmetic was wrong
Original claim: "AR-FFN = 4 GFLOPs per block". With stated rank options {d_ff/4, d_ff/2, 3d_ff/4, d_ff}, the minimum is `2·L·d·(d_ff/4) = 8.05 GFLOPs`. **4 GFLOPs is mathematically impossible** without additional low-rank factorization.

**Lesson:** Always recompute FLOPs from first principles. Don't trust the math until you have.

### C2. Standard transformer baseline was 5–10× wrong
Original claim: "Standard 1B = 480 GFLOPs/token fwd+bwd". Correct value at L=4096, d=1536, 24L, d_ff=2560 is **9.74 TFLOPs/sequence fwd+bwd** = **2.38 GFLOPs/token fwd+bwd**. Original was off by **20×** depending on convention used.

**Lesson:** Always state units explicitly (per-seq vs per-token, MACs vs FLOPs).

### C3. MoD exit decision is non-differentiable
Original GDR design: `if min_geodesic_distance < θ: exit else continue`. Step function has zero gradient. With hyperbolic distance, gradients vanish near boundary. Without a soft relaxation, the router never learns a useful policy.

**Lesson:** Any "if < threshold" routing decision needs a straight-through estimator or soft relaxation.

### C4. FSA cold-start collapses all heads
At init, DCT of random embeddings is roughly flat. Top-k picks the lowest few bins for every head. Without per-head orthogonal pre-rotation, all FSA masks collapse to identical frequency bands. The component becomes a static mask.

**Lesson:** Per-head symmetry breaking at init is mandatory for any head-specific learned structure.

### C5. PolyMamba-2 A matrices can have spectral radius > 1
Tridiagonal Jacobi and Legendre A matrices can easily have eigenvalues with |λ| > 1 if not properly normalized. SSM `h_t = A h_{t-1} + B x_t` diverges exponentially. NaN within a few hundred steps.

**Lesson:** Hurwitz-stable init is mandatory: `A = I - softplus(dt)·M` with `dt_init = 0.001`.

### C6. TT-Mem was not actually a scalar TT
Plan claimed: "8 cores of size 8×16×8 = 8192 entries". For a scalar TT, boundary ranks must satisfy `r_0 = r_K = 1`. With all ranks equal to 8, the result is an 8×8 matrix (or a Tensor Ring with trace), not a scalar. The math doesn't give what the design claimed.

**Lesson:** TT decomposition has strict rank-vector requirements. Don't hand-wave the boundary conditions.

### C7. TT-Mem bf16 numerical collapse
K=8 sequential matmuls in bf16 compound errors. Worst-case condition number with r=8: `κ_total ≤ 8^8 = 1.6e7`. bf16 mantissa is 7 bits. Worst-case error: 1.6e7 × 1e-2 = **1.6e5**. Catastrophic.

**Lesson:** Sequential deep matmul chains in bf16 need fp32 accumulation, OR reduced intermediate rank.

### C8. GDR hyperbolic distance has vanishing gradients near boundary
`d/dx arccosh(x) = 1/√(x²-1)`. At x=1 (boundary), derivative → ∞. Near boundary, arguments explode and gradients vanish. Tokens/anchors near boundary become frozen.

**Lesson:** Hyperbolic geometry has numerical pathologies that aren't worth engineering around for 200M-scale benefits.

### C9. DirectML bf16 permanently rejected
Microsoft closed Issue #688 (`torch_directml: Invalid or unsupported data type BFloat16`) as "Not planned" on Jul 23, 2025. torch-directml is effectively abandoned. All open issues were closed "Not planned" on the same date.

**Lesson:** Do not attempt AETHER on DirectML. Use ROCm or ZLUDA.

## MAJOR Findings

### M1. AETHER-1B target is obsolete
Qwen2.5-1.5B was SOTA in Sep 2024. Qwen3-0.6B (April 2025) matches it at 40% parameters. Alibaba proved the 67% target is wrong.

### M2. 1B from scratch on RX 7600 is infeasible in 1 week
Compute gap: 5,000–15,000×. SOTA-competitive 1B requires 5–15T tokens. RX 7600 in 1 week ≈ 1–3T. **Downsize to 200M.**

### M3. 18 TFLOPS sustained on RX 7600 is optimistic
Realistic sustained throughput on RX 7600 ROCm: 8–12 TFLOPS. Plan's 18 TFLOPS is 2× optimistic.

### M4. TT-Mem was a duplicate of Tensor Memory
arXiv:2605.27686 (May 2026) — Swain et al. propose Tensor Memory, a tensor-structured external memory for transformers with continuous sampling. The architectural slot (tensor-decomposed external memory for an LLM) is freshly published.

### M5. HPE was a duplicate of Wiola/SRPE
arXiv:2607.01394 (1 Jul 2026) — Chowdhury et al. propose Spiral Rotary Positional Encoding, a 3D helical manifold position encoding for transformers. The novelty surface (3D helix for RoPE) is taken.

### M6. AR-FFN overclaimed
Minima (Feb 2026) and LoRA-MoE (prior 6 months) cover the "low-rank FFN body" slot. "TRULY NEW" claim is not defensible.

### M7. 200B tokens on RX 7600 is months-long
Standard 1B at 200B tokens: ~309–551 days on RX 7600. AETHER's 8× speedup brings it to ~37–66 days. Still not a 1-week training run.

### M8. Memory budget was loose in the wrong directions
Plan claimed 9.0 GB total at 1B params. Correct: 8.8–9.4 GB at 1B, but with omissions (DirectML overhead, PyTorch runtime ~0.5GB) and overestimates (TT-Mem was 100× overestimated at 0.3 GB; actual <3 MB).

### M9. PolyMamba-2 param count was 2× too high
Plan said ~19M per block; actual ~9.8M. The full 1B model was actually 0.73B, not 1B.

### M10. "Möbius-transform soft addressing" was false
The TT-Mem addressing formula `softmax(q̃^T G_k / √r)` is standard Euclidean softmax attention. No Möbius operation occurs. The hyperbolic framing was marketing.

### M11. DirectML had no path for parallel scan
SSM parallel scan requires either custom CUDA kernels or Mamba-3 chunked SSD. DirectML has neither. Would have given 5–10× slowdown vs CUDA.

## Resolutions

| Original Claim | Status | Resolution |
|----------------|--------|------------|
| 6 novel components | Cut to 3 | Drop HPE, AR-FFN, TT-Mem |
| 1B parameters | Downsize to 200M | Match SmolLM2-360M class |
| DirectML backend | Switch to ROCm | **HIP SDK 7.1.1 on Windows 11 native** (WSL2 ROCm has no `/dev/kfd` for gfx1102; HIP SDK bypasses WSL2 entirely and officially supports RX 7600) |
| Match Qwen2.5-1.5B | New target | Match SmolLM2-360M-Instruct at 55% params |
| 200B tokens | Cut to 30B | Realistic for 1-week budget |
| 18 TFLOPS throughput | Revise to 8–12 | Realistic RX 7600 sustained |
| Hyperbolic GDR | Cut | Use Euclidean MoD top-k |
| TT-Mem / "Möbius" addressing | Cut | Replace with bigger FFN if needed |

## What Survived

- **FSA** — defensible novel recombination (FNet × AFNO × NSA)
- **PolyMamba-2** — defensible novel recombination (Liquid-S4 × CTDG-SSM) at K=2
- **MoD-GDR** — established MoD technique, no novelty claim

These 3 components form the publishable subset. The paper title is "AETHER: A 200M Language Model with Frequency-Sparse Attention, Multi-Order SSM, and Depth Routing."

## What Future Agents Should NOT Do

1. Don't try to reintroduce the dropped components. The literature now has Wiola, Tensor Memory, and Minima covering those slots.
2. Don't trust FLOP math without re-deriving. Several numbers were wrong in the original plan.
3. Don't use hyperbolic geometry for routing at this scale. The math is correct but the benefit isn't there.
4. Don't claim novelty for FSA or PolyMamba-2 without acknowledging prior art.
5. Don't try to scale beyond 200M without re-checking the memory budget.

## Audit Trail

Full audit agents run on 22 July 2026. No re-litigation permitted without strong new evidence (e.g., a paper that explicitly contradicts the dropped claims, a numerical issue that invalidates a finding, etc.).

---

## Post-Audit Findings (23 July 2026)

#### PAF-0: Audit assumed WSL2 ROCm was viable. It is not.

The 10-agent audit assumed ROCm on WSL2 Ubuntu would work for the RX 7600 with `HSA_OVERRIDE_GFX_VERSION=11.0.0`. **This assumption was wrong** for gfx1102 (Navi 33, RX 7600). The fix in this post-audit finding supersedes the original recommendation.

The remaining PAFs are:

### PAF-1: WSL2 ROCm path blocked, HIP SDK on Windows native is the fix

After the audit, attempting to install ROCm on WSL2 Ubuntu 24.04 with the official AMD apt repo (`rocm 7.2 noble`) confirmed:

- `rocm-dev 7.2.0` installs successfully to `/opt/rocm`
- BUT `rocminfo` fails with `hsa_init Failed, possibly no supported GPU devices`
- Root cause: WSL2 kernel does not load the `amdgpu` module and does not expose `/dev/kfd` to the user namespace
- WSL2 only provides `/dev/dxg` (DirectX passthrough), which ROCm cannot use

**Discovered alternate path: AMD HIP SDK 7.1.1 (Feb 2026 release) has official native Windows support for RX 7600 (gfx1102).**

Per https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html, the AMD-supported GPUs for HIP SDK include `RX 7600 (RDNA3, gfx1102) ✅`. The HIP SDK ships:
- AMD GPU driver (replaces existing Adrenalin)
- HIP runtime
- HIP Libraries
- AMD ROCm Debugger

TheRock build system (https://github.com/ROCm/TheRock, ROCm 7.14.0 released Jul 16 2026) explicitly tracks gfx1102 as `Build Passing` + `Sanity Tested` on Windows native. Native Windows PyTorch wheels are built and tested via AMD's CI.

**Updated decision:** AETHER-200M runs on **Windows 11 native with HIP SDK 7.1.1**, not WSL2. The ROCm-on-WSL2 path is dead for gfx1102.

This finding **invalidates** the original audit's recommendation to use `HSA_OVERRIDE_GFX_VERSION=11.0.0` env var in WSL2 (the env var is harmless but irrelevant without KFD).
