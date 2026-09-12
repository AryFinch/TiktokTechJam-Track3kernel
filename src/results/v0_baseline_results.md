# V0 — Baseline (no optimization)

**File:** `solutions/v0_baseline.py`
**Idea:** Keep the parent `BaselineTransformer.forward()` unchanged. The
optimized class is a thin subclass that only adds documentation.

## What the code does

```python
class UserOptimizedTransformer(BaselineTransformer):
    def forward(self, x, valid_token_mask=None):
        return super().forward(x, valid_token_mask)
```

The attention uses three separate `nn.Linear` layers for Q, K, V, an
explicit fp32 softmax, and a manual `probs @ V`. The block uses
pre-norm + residual + GELU-exact.

## Benchmark result (default config)

```
baseline : median=14.10 ms | mean=13.04 ms | p90=14.84 ms | min=9.43 ms
optimized: median=14.04 ms | mean=13.04 ms | p90=14.88 ms | min=9.24 ms
speedup  : 1.004x
```

`max_abs=0` because both implementations are bit-identical — the
"optimized" class is the baseline itself.

## Why it is what it is

The two numbers are statistically indistinguishable: same code, same
weights, same device, just two module instances. This variant exists
purely as a sanity check that the harness itself isn't biasing the
results and that V1–V4 are actually measuring real optimizations.
