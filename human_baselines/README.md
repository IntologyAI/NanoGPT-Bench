# human_baselines_minimal

This package ships one folder per historical
[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
world record from `2025-09-03_FA3` through `2026-01-19_BigramHashEmbedding`,
plus `mean_speedups.tsv` summarizing how much wallclock each record shaves off
the FA3 anchor when re-timed on consistent hardware.

Each `<record>/` directory contains the `train_gpt.py` (and `run.sh` /
`triton_kernels.py` where applicable) for that record.

Every `train_gpt.py` here is identical to the
upstream blob at the commit that introduced that record — with two
intentional exceptions (both single line changes which do not effect functionality of the records but ensure compatibility with the fixed dependency set used for the record timings).

### 1. FlashAttention-3 import path (not a precision change)

For records from `2025-09-29_PolarExpress` onward, we keep the original direct `from flash_attn_interface import flash_attn_varlen_func` form instead of upstream's newer `kernels.get_kernel('varunneal/flash-attention-3')` loader, since the benchmark image installs `flash_attn_3` directly and both paths resolve to the same FA3 kernel.

### 2. Symmetric-matmul `aux_desc.store` cast (precision-related, kept)

In the fused transposed-MLP Triton matmul (inlined into `train_gpt.py` for `2026-01-10_FusedLinearReLUSquare` and `2026-01-16_FusedSoftcappedEntropy`, then promoted to `triton_kernels.py` from `2026-01-18_UnifiedOptimizers` onward), we keep the explicit `c0_post.to(dtype)` / `c1_post.to(dtype)` casts on `aux_desc.store` that upstream has since dropped, because Triton 3.4.0 (the pinned benchmark version) promotes the `tl.maximum(...) * ...` arithmetic to fp32 and requires an explicit downcast to store into the bf16 TMA descriptor.