# aether/ — Implementation Context

## Module Ownership

| File | Owns |
|------|------|
| `aether/model.py` | `AetherForCausalLM`, `AetherConfig`, block assembly |
| `aether/components/fsa.py` | `FreqSparseAttention` |
| `aether/components/polymamba.py` | `PolyMamba2` |
| `aether/components/mod_router.py` | `MoDRouter` |
| `aether/ops/dct.py` | DCT/IDCT with cached matrix |
| `aether/ops/topk_soft.py` | Differentiable top-k |
| `aether/training/data.py` | FineWeb-Edu streaming loader |
| `aether/training/tokenizer.py` | Llama-3 BPE wrapper |
| `aether/training/curriculum.py` | L=1024→2048→4096 schedule |
| `aether/training/optim.py` | `torch.optim.AdamW(fused=True)` factory + decay groups |
| `aether/training/losses.py` | `L_ortho`, `L_kbal`, `L_mbd` |
| `aether/post_train/sft.py` | SFT loop |
| `aether/post_train/dpo.py` | DPO loop |
| `aether/eval/harness.py` | lm-eval-harness runner |
| `aether/utils/numerics.py` | fp32 helpers, NaN guards |
| `aether/utils/checkpoint.py` | HF safetensors save/load |
| `aether/utils/logging.py` | W&B init |

## Numerics Policy

- **bf16**: all linears, FFN activations, embeddings, attention scores (with max-subtraction trick)
- **fp32**: softmax, MoD routing scores, SSM state updates + parallel scan, A matrix construction, DCT/IDCT, RMSNorm statistics, all aux losses
- **No fp32 master weights** (PAF-3): bf16 params + fp32 Adam moments via `torch.optim.AdamW(fused=True)`
- **Gradient checkpointing**: at block level with `use_reentrant=False`; blocks return `(x, kbal, ortho)` tuples so aux losses are valid checkpoint outputs (never call mixer loss methods outside the checkpointed region)

## Init Schemes (one-time, at model construction)

| Param | Init |
|-------|------|
| Linear weights | small normal `N(0, (0.02/√fan_in)²)` (std 0.02/sqrt(fan_in)) |
| Linear bias | zeros (none currently; all linears are bias=False) |
| Embedding | small normal `N(0, 0.02²)` |
| RMSNorm | ones |
| FSA mask logits | `N(0, 0.02²)` — random tie-break gives per-head diversity at init (replaces the old frozen orthogonal pre-rotation) |
| FSA mix gate | zeros (blend = 0.5 cross / 0.5 intra at init) |
| PolyMamba-2 A | `A_log = log(exp(0.001) - 1)`; forward clamps `dt ≥ dt_floor` so spectral radius ≤ `max_radius=0.95` (bounds backward gradient amplification — see B27) |
| PolyMamba-2 B, C | `N(0, 1/d_inner)` |
| PolyMamba-2 D | zeros |
| PolyMamba-2 gate | `N(0, 0.01²)` (near-uniform softmax) |
| MoD router | `N(0, 0.01²)` |

## Test Requirements

Each module must have a unit test in `tests/` BEFORE integration into `model.py`. Required tests:

| Module | Required Tests |
|--------|----------------|
| `ops/dct.py` | invertibility (DCT then IDCT = identity); gradient flow |
| `ops/topk_soft.py` | top-k correctness; soft→hard annealing |
| `components/fsa.py` | forward shape; spectral entropy per head; mask diversity (cosine < 0.9) |
| `components/polymamba.py` | forward shape; spectral radius ≤ `max_radius` (0.95) at every step; gradient flow through A |
| `components/mod_router.py` | top-k correctness; L_mbd bounds; load distribution per layer |
| `model.py` | end-to-end forward; loss decreases over 100 steps; checkpoint save/load roundtrip |
| `utils/numerics.py` | NaN guards trigger correctly; fp32 promotion |

## Code Style

- Type hints on every public function and method
- No comments (per project conventions)
- One class per file in `components/` and `ops/`
- Public API uses snake_case; classes use PascalCase
- Imports at top of file; no inline imports except for type-only imports inside `if TYPE_CHECKING`
- Constants in UPPER_SNAKE_CASE at top of module
