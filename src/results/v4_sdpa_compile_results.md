# V4 — SDPA + `torch.compile` (combined)

**File:** `solutions/v4_sdpa_compile.py`
**Idea:** Combine the two best ideas from V1 and V2: use
`scaled_dot_product_attention` for the attention block, then wrap the
whole forward in `torch.compile`. On a CUDA stack this is almost
always the strongest single config that doesn't require a custom
kernel.

## What the code does

```python
class UserOptimizedTransformer(BaselineTransformer):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            layer.attention = SDPASelfAttention(config.d_model, config.num_heads)
        self._compiled_forward = None

    def _baseline_forward(self, x, valid_token_mask=None):
        return BaselineTransformer.forward(self, x, valid_token_mask)

    def forward(self, x, valid_token_mask=None):
        if self._compiled_forward is None:
            self._compiled_forward = torch.compile(self._baseline_forward, dynamic=False)
        return self._compiled_forward(x, valid_token_mask)
```

Same parameter names as the baseline; the harness's weight copy
fills the SDPA attention's projections the same way as V1.

## Benchmark result (default config)

```
baseline : median=14.32 ms | mean=14.37 ms | p90=18.22 ms | min=9.18 ms
optimized: median=39.63 ms | mean=38.91 ms | p90=45.82 ms | min=24.02 ms
speedup  : 0.361x   ← SLOWER than baseline
```

Accuracy: `max_abs ≈ 3.1e-6`, all 5 trials pass.

## Why it is **slower** even with SDPA underneath

The whole point of V4 was to get V1's attention win and V2's
compiler win simultaneously. We got the SDPA win (the forward graph
does contain the fast `scaled_dot_product_attention` call), but V2's
compile overhead swamps it on this machine.

In V1 the SDPA kernel is called from **eager** Python, which dispatches
straight to the optimized Metal kernel. In V4 the call is wrapped in
Inductor's generated wrapper, which adds:

* a per-call dispatch into the compiled artifact,
* shape / stride / contiguity guards for the `(B, H, S, D)` tensors,
* an autotune miss fallback for the underlying GEMMs (same warning
  as V2).

Net effect: SDPA's ~13× speedup is needed just to recover the
inductor overhead, and it isn't quite enough — we land at 0.36×.

## When V4 would win

On a CUDA GPU, V4 typically beats V1 by another 1.2–1.4× because
Inductor can fuse the surrounding LayerNorm + GELU + residual into
Triton kernels that match SDPA's matmul time. On MPS, where the
Inductor → Metal codegen is still rough, the best current choice is
**V1 (SDPA eager)**, and that's what the report recommends.
