# V3 — Manual operator fusion (QKV + GELU-tanh)

**File:** `solutions/v3_fused.py`
**Idea:** Apply two hand-rolled fusions that don't require a graph
compiler, then leave the rest of the forward identical to baseline.

## What the code does

1. **`FusedQKVSelfAttention`** keeps the same parameter names as the
   baseline (`q_proj, k_proj, v_proj, out_proj`) so weight copy still
   works, but at forward time it concatenates the three QKV weight
   matrices into one `(3*d_model, d_model)` tensor (cached on first
   call as a buffer) and runs **one** `F.linear` instead of three.
   The rest of the attention is the same as baseline (manual
   matmul, fp32 softmax).
2. **`FusedTransformerBlock`** uses the fused attention module and
   switches GELU from `approximate='none'` (the default, erf-based)
   to `approximate='tanh'`. The tanh approximation is a couple of
   cheap flops (one tanh + one cubic) instead of an `erf` call, and
   PyTorch emits a single Metal kernel for it.
3. **`UserOptimizedTransformer`** builds its own `nn.ModuleList` of
   `FusedTransformerBlock`s (skipping the parent's layer creation)
   and runs the standard `for layer in self.layers` loop in
   `forward`. Parameter names are unchanged → `copy_model_weights`
   works.

## Benchmark result (default config)

```
baseline : median=14.18 ms | mean=13.39 ms | p90=15.63 ms | min=8.35 ms
optimized: median=14.02 ms | mean=13.22 ms | p90=15.48 ms | min=8.05 ms
speedup  : 1.011x
```

Accuracy: `max_abs ≈ 4.9e-4` and `max_rel` is large but stays inside
the `(atol OR rtol)` rule — most failing positions are very near
zero where tiny absolute errors show as huge relative errors, but
abs ≤ 0.002 holds everywhere.

## Why it barely moves the needle

The QKV fusion saves **kernel-launch overhead**: three `Linear(512,
512)` calls collapse into one `Linear(512, 1536)`. Each Linear here
is a 1024×512 → 1024×512 (×3) matmul which is small enough that the
fp32 GEMM is already memory-bandwidth bound on the M4 Pro's
unified-memory GPU; the launch-overhead savings are drowned out by
the actual compute time.

The GELU-tanh switch is a similar story. `erf` and `tanh` on Metal
are both tiny elementwise kernels; the saved flops don't show up in
wall time.

The lesson: at this problem size the baseline forward is **GEMM
dominated**, and the easiest hand fusions don't change the GEMM
shape. V3 is still useful because it confirms the hypothesis: if you
want big speedups you have to attack the **attention** (V1) or write
fused attention-shaped kernels (V5/Triton), not rearrange the GEMM
borders.
