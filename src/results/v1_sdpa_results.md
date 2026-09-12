# V1 — `torch.nn.functional.scaled_dot_product_attention` (SDPA)

**File:** `solutions/v1_sdpa.py`
**Idea:** Replace the hand-rolled QKᵀ → mask → softmax → ·V pipeline
with a single fused `F.scaled_dot_product_attention` call. The kernel
backend is picked automatically by PyTorch (math / mem-efficient /
FlashAttention-style on supported hardware).

## What the code does

A new `SDPASelfAttention` is a drop-in replacement for
`BaselineSelfAttention` (same parameter names: `q_proj, k_proj,
v_proj, out_proj`). Its `forward`:

1. Projects X → Q, K, V with the same three linears as baseline.
2. Splits heads → `(B, H, S, D)`.
3. Builds an additive `attn_mask` when needed (`-inf` for blocked
   positions, `0` for allowed). On the MPS backend SDPA does **not**
   accept `is_causal=True` together with `attn_mask`, so when both are
   needed (causal + padding) we fuse them into one `(1, 1, S, S)`
   tensor.
4. One call to `F.scaled_dot_product_attention(q, k, v, attn_mask=...)`
   (or `is_causal=True` for the pure-causal fast path).
5. Merges heads, runs the output projection, applies the valid-token
   zero-mask.

`UserOptimizedTransformer.__init__` calls `super().__init__` to build
the standard layers, then swaps each block's `attention` for an
`SDPASelfAttention`. Parameter names are unchanged, so the harness's
`copy_model_weights` still works.

## Benchmark result (default config)

```
baseline : median=13.40 ms | mean=20.21 ms | p90=14.98 ms | min=1.86 ms
optimized: median= 1.58 ms | mean= 1.71 ms | p90= 2.05 ms | min=1.36 ms
speedup  : 8.467x
```

Accuracy: `max_abs ≈ 2.1e-6`, well inside `atol=0.002 / rtol=0.02`. All
5 accuracy trials pass.

## Why it is fast

The baseline's explicit attention does:
1. Allocate an `(8, 8, 128, 128) = 1.05 M` fp32 scores tensor.
2. Materialize the causal mask and zero out the upper triangle.
3. Cast scores to fp32 and run `softmax` (writes another 1.05 M).
4. Cast back to the model dtype, then matmul with V.

That is two full S×S fp32 buffers per layer × 6 layers = 12.6 M
elements of temporary memory traffic, plus four matmul/elementwise
kernel launches. SDPA fuses steps 1–4 into a single streaming
kernel: the attention matrix is never materialized, so memory traffic
drops from O(S²) to O(S·D). For our default config (S=128, D=512,
B=8, H=8) the working-set shrinks from 4 MB/layer to ~16 KB/layer.

## Scaling check (extras A and G)

| Setup                              | Median speedup |
| ---------------------------------- | -------------- |
| Default, no causal, fp32           | **8.47×**      |
| seq=256, no causal, fp32           | **15.0×**      |
| seq=512, batch=4, no causal, fp32  | **20.2×**      |
| Default + causal mask              | 8.62×          |
| Default, bf16 (`--atol 0.1`)       | 7.51×          |

Speedup grows super-linearly with `seq_len²` because the baseline
attention memory cost is `O(S²)` per layer, while SDPA stays close
to `O(S)`. Causal masking is essentially free (V1 has a dedicated
`is_causal=True` fast path on the SDPA side). bf16 with a relaxed
tolerance still gives a 7.5× win, so the optimization is robust
across precisions.
