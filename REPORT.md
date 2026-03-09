# autoresearch/mar9 — Experiment Report

**Date**: March 9, 2026
**GPU**: NVIDIA (non-H100, likely RTX-class — no Flash Attention 3, using PyTorch SDPA)
**Dataset**: UAV-Flow (drone navigation logs → text)
**Time budget**: 5 minutes per experiment
**Total experiments**: 66 (1 baseline + 65 modifications)
**Improvements kept**: 3

## Summary

Starting from a baseline val_bpb of **0.9675**, three changes were kept, bringing the final best to **0.9022** — a **6.8% reduction**. All three improvements happened in the first 15 experiments. The remaining 50+ experiments exhaustively searched the hyperparameter and architecture space but found no further gains.

## The Three Wins

### 1. MATRIX_LR: 0.04 → 0.06 (0.9675 → 0.9526)

The Muon optimizer's learning rate for transformer matrix parameters was too conservative. Increasing it by 50% gave a clean 1.5% improvement. This makes sense: with a fixed 5-minute budget, you want to learn as aggressively as stability allows. The sweet spot was sharp — 0.05 was too low (0.921), 0.07 was too high (0.922), and 0.08 clearly overshot (0.962).

### 2. WINDOW_PATTERN: "SSSL" → "L" (0.9526 → 0.9120)

This was the single biggest win — a **4.3%** drop in val_bpb. The original "SSSL" pattern alternates between short (half-context) and long (full-context) sliding window attention. On this GPU without Flash Attention 3, the sliding window implementation falls back to a manually constructed attention mask with `F.scaled_dot_product_attention`. Using all full-context ("L") means every layer gets `is_causal=True` which is much more efficiently handled by PyTorch's optimized SDPA kernels — no mask materialization needed. The quality gain comes from two sources: (a) all layers can attend to the full context, and (b) the faster kernel execution means more training steps in the same 5-minute window.

### 3. softcap: 15 → 30 (0.9120 → 0.9022)

The logit soft-capping (`softcap * tanh(logits / softcap)`) was too aggressive at 15. This squashes logit magnitudes, preventing the model from being confidently correct. Doubling to 30 relaxes this constraint, allowing sharper predictions while still providing gradient-friendly regularization. The optimum was precisely at 30 — values of 20, 25, 35, and 50 were all worse, and removing the softcap entirely (0.911) was also worse. Some capping helps training stability, but too much hampers expressiveness.

## What Didn't Work (and Why)

### Architecture changes (all discarded)

| Change | val_bpb | Why it failed |
|--------|---------|---------------|
| DEPTH 8→10 | 2.303 | Catastrophic. Larger model = slower steps. Only ~20 steps completed in 5 min vs ~400 at depth 8. Massively underfits. |
| DEPTH 8→7 | 0.946 | Fewer layers = less representational capacity, not compensated by extra steps. |
| DEPTH 8→6, width 80 | 0.926 | Same story — depth matters more than width for this task. |
| ASPECT_RATIO 64→80 | 0.955 | Wider model is too slow per step on this GPU. Only 290 steps vs ~400. |
| ASPECT_RATIO 64→48 | 0.943 | Too narrow — insufficient model capacity. |
| HEAD_DIM 128→64 | 0.943 | More heads didn't help. The 4-head configuration with 128-dim heads is well-matched. |
| MLP 4x→6x | 0.943 | Slower steps (25 GB VRAM) outweigh capacity gains. |
| MLP 4x→3x | 0.913 | Close, but capacity loss > speed gain. |
| SwiGLU | 0.948 | With 2/3 hidden size to match params, SwiGLU has less effective capacity than ReLU^2. |
| GELU | 0.913 | Close but ReLU^2 is better — the squaring acts as an implicit sparsity mechanism. |
| ReLU (no square) | 0.928 | Confirms the squaring in ReLU^2 is doing meaningful work. |
| GQA (half KV heads) | 0.916 | Reduced KV capacity hurts more than the speed gain helps. |
| Remove QK norm | 0.986 | QK norm is critical for training stability with this optimizer. |
| Remove x0 residual | 0.911 | The shortcut connection to initial embeddings helps. |
| Remove softcap | 0.911 | Some regularization is necessary. |

**Key insight**: On this compute-constrained GPU, the model is at a Pareto frontier between capacity and throughput. Any change that adds compute (bigger model, wider MLP) produces fewer training steps and worse results. Any change that reduces capacity (fewer layers, narrower) loses quality faster than it gains steps.

### Optimizer/LR changes (all discarded)

| Change | val_bpb | Why it failed |
|--------|---------|---------------|
| MATRIX_LR 0.05 | 0.921 | Too conservative, not enough learning per step. |
| MATRIX_LR 0.07 | 0.922 | Starting to overshoot — the loss landscape is sharply peaked at 0.06. |
| MATRIX_LR 0.08 | 0.962 | Clear overshooting — training becomes unstable. |
| EMBEDDING_LR: 0.3, 0.5, 0.7, 0.8, 1.0 | 0.925–0.948 | 0.6 is optimal. Both lower and higher hurt. |
| UNEMBEDDING_LR: 0.006, 0.01 | 0.936–0.953 | 0.004 is optimal. Higher makes the output layer too volatile. |
| SCALAR_LR: 0.3, 1.0 | 0.916–0.947 | 0.5 is optimal. |
| ADAM_BETAS: (0.7,0.95), (0.8,0.99), (0.9,0.999) | 0.914–0.948 | (0.8, 0.95) is the sweet spot for this training duration. |
| Muon ns_steps 5→3 | 0.951 | Fewer Newton-Schulz iterations = worse orthogonalization. Quality of the Muon update matters more than speed. |
| Muon momentum warmup 300→100 | 0.955 | Rushing to full momentum destabilizes early training. |
| Gradient clipping | 0.940 | The Muon optimizer already handles gradient magnitudes internally via orthogonalization. |

**Key insight**: The original optimizer hyperparameters were already well-tuned. The only LR that was meaningfully off was MATRIX_LR (0.04→0.06). Every other parameter was already near-optimal.

### Schedule/regularization changes (all discarded)

| Change | val_bpb | Why it failed |
|--------|---------|---------------|
| WARMUP_RATIO: 0.05 | 0.924 | Training is already stable from step 0 — warmup wastes time budget. |
| WARMDOWN_RATIO: 0.3, 0.4, 0.6, 0.7 | 0.910–0.918 | 0.5 is the sweet spot — the right balance between learning and settling. |
| FINAL_LR_FRAC: 0.1 | 0.905 | Close (0.905 vs 0.902) but keeping some LR at the end slightly hurts convergence. |
| Cosine decay | 0.931 | Linear warmdown works better — probably because it spends more effective time at moderate LRs. |
| WEIGHT_DECAY: 0, 0.1, 0.15, 0.25, 0.3, 0.4 | 0.915–0.940 | 0.2 is optimal. Too little = overfitting. Too much = underfitting. |
| Constant WD (no decay) | 0.910 | Decaying WD with progress is important — late regularization hurts. |
| Standard WD (not cautious) | 0.940 | Cautious WD (gradient-aligned mask) is clearly better. |
| Label smoothing 0.1 | 1.707 | Catastrophic — eval uses raw cross-entropy, so the model is trained on a mismatched objective. |
| Z-loss 1e-4 | 0.952 | Softcap already serves this role; stacking regularizers hurts. |

### Batch size changes (all discarded)

| Change | val_bpb | Why it failed |
|--------|---------|---------------|
| TOTAL_BATCH 2^20 | 0.989 | Fewer optimizer steps in the time budget. |
| TOTAL_BATCH 2^18 | 0.969 | More steps but noisier gradients. |
| TOTAL_BATCH 2^17 | 0.941 | Even more steps (1585) but gradient quality is too low. |
| DEVICE_BATCH 128 | crash | 34s/step — only ~20 steps possible. GPU can't handle it. |
| DEVICE_BATCH 32 | 0.953 | More grad accum overhead outweighs lower VRAM. |

**Key insight**: TOTAL_BATCH_SIZE = 2^19 (~524K tokens) is at the sweet spot for this GPU's throughput and the training dynamics.

## Observations

1. **Early wins, long plateau**: All improvements were found in experiments 1–15. The remaining 50+ experiments explored extensively but found nothing better. This suggests the configuration quickly converged to a local optimum.

2. **Compute-bound, not idea-bound**: On this GPU (non-H100, ~3-4s per step), the model is firmly at a throughput-quality frontier. Every experiment that adds compute per step (bigger model, wider MLP, more heads) fails because it reduces the number of training steps. This is a fundamentally different regime from H100 where you have enough raw FLOPS to scale up.

3. **SDPA vs FA3 matters**: The biggest single win (WINDOW_PATTERN → "L") was essentially a fix for the non-H100 codepath. The "SSSL" pattern relies on efficient sliding window attention via Flash Attention 3, which isn't available here. With SDPA, full causal attention is the faster path.

4. **ReLU^2 is surprisingly good**: Both GELU and plain ReLU were worse. The squaring operation in ReLU^2 provides implicit sparsity (killing small activations more aggressively) which seems to help with this small model and short training duration.

5. **Softcap sensitivity**: The logit capping value had a surprisingly narrow optimum at exactly 30. Values of 15, 20, 25, 35, and 50 were all worse. This suggests the training dynamics are quite sensitive to how the output distribution is regularized.

6. **The optimizer was already good**: The MuonAdamW optimizer with cautious weight decay, Newton-Schulz orthogonalization, and NorMuon variance reduction is well-engineered. Most optimizer tweaks (betas, gradient clipping, momentum schedule, ns_steps) made things worse. The only gain was a simple LR bump.

## Final Configuration

```
MATRIX_LR = 0.06        # (was 0.04)
WINDOW_PATTERN = "L"    # (was "SSSL")
softcap = 30            # (was 15)
# Everything else unchanged from defaults
```

**Baseline → Best: 0.9675 → 0.9022 (6.8% improvement)**
