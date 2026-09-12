# V2 — `torch.compile` (TorchInductor)

**File:** `solutions/v2_compile.py`
**Idea:** Wrap the entire forward in `torch.compile`. Inductor
analyzes the Python+FX graph, generates fused Triton (CUDA) /
AOT (Metal) kernels, and removes per-op dispatch overhead.

## What the code does

```python
class UserOptimizedTransformer(BaselineTransformer):
    def __init__(self, config):
        super().__init__(config)
        self._compiled_forward = None

    def _baseline_forward(self, x, valid_token_mask=None):
        return BaselineTransformer.forward(self, x, valid_token_mask)

    def forward(self, x, valid_token_mask=None):
        if self._compiled_forward is None:
            self._compiled_forward = torch.compile(
                self._baseline_forward, dynamic=False
            )
        return self._compiled_forward(x, valid_token_mask)
```

The compiled function is created lazily on the first forward call so
the trace happens **after** `copy_model_weights` and `.to(device,
dtype)`. Warmup absorbs the compilation cost.

## Benchmark result (default config)

```
baseline : median=14.46 ms | mean=13.76 ms | p90=15.75 ms | min=9.55 ms
optimized: median=41.64 ms | mean=41.08 ms | p90=43.67 ms | min=31.20 ms
speedup  : 0.347x   ← SLOWER than baseline
```

Accuracy: `max_abs ≈ 2.9e-6`, all 5 trials pass.

## Why it is **slower** on this machine

`torch.compile` on the MPS backend currently goes through a less
mature codegen path than CUDA. Three things are visible in the
output:

1. The warning
   `W... Not enough SMs to use max_autotune_gemm mode` appears for
   every compile run. Inductor wants to autotune the GEMM kernel
   choice, cannot (MPS exposes one GEMM, no `max_autotune`), and
   falls back to a per-op wrapper that adds dispatch overhead.
2. The baseline forward at this size is dominated by a small number
   of large GEMMs (6 × `Linear(512,2048)` + 6 × `Linear(2048,512)` +
   6 × 4 × `Linear(512,512)`). MPSGraph already picks well-tuned
   Metal kernels for those; Inductor's wrapper layer (Python guard
   checks, tensor meta propagation) costs more than the fusion
   saves.
3. The fp32 softmax, residual adds, and layer-norms are pointwise
   ops. Inductor *can* fuse them with surrounding matmuls on CUDA,
   but on MPS it often ends up emitting one Metal shader per op,
   paying extra launch latency per elementwise.

The same code on a CUDA GPU almost always wins (typically 1.2–1.5×);
on Apple Silicon the inductor codegen for Metal is not yet at parity
with the eager path for fp32 transformer blocks of this size.
