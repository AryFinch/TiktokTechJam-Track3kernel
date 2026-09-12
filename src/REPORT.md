# GPU Transformer inference — what actually helps, what doesn't (Apple M4 Pro / MPS)

This report walks through five copies of the reference benchmark
`torch_transformer_benchmark.py`. Each copy only changes the
`UserOptimizedTransformer` class; the harness, weights, and config
are kept identical so the numbers are directly comparable.

The five copies, and a one-line summary of the result, are:

| # | Solution name | One-line result |
| - | ------------- | --------------- |
| V0 | Baseline (no opt)            | 1.00× — sanity check |
| V1 | `F.scaled_dot_product_attention` | **8.5×** at the default config, **20×** at seq 512 |
| V2 | `torch.compile`              | **0.35×** — slower on MPS, the Inductor codegen is not yet competitive |
| V3 | Manual fusion (QKV + GELU-tanh) | 1.01× — GEMM-shape change too small to matter at this size |
| V4 | SDPA + `torch.compile`       | **0.36×** — Inductor overhead eats the SDPA win |

A more detailed per-variant write-up lives in
`results/v0_baseline_results.md` … `results/v4_sdpa_compile_results.md`,
and the raw numbers are in `results/*.log`. The cross-experiment
summary table is in [`results/summary.md`](results/summary.md).

---

## Test environment

* **Hardware:** Apple Mac (M-series), M4 Pro, 24 GPU cores, unified
  memory.
* **OS:** macOS 15.7.9 (24G830), arm64.
* **Software:** Python 3.13.12, PyTorch 2.13.0 (built with MPS),
  no Triton runtime required (Triton-on-MPS is available but
  unnecessary for the wins we found).
* **Acceleration:** GPU = Metal Performance Shaders via
  `torch.backends.mps`. There is **no CUDA** on this machine, so
  numbers that are quoted from CUDA blog posts will not match.
* **Default config** (used unless stated):
  `batch=8, seq=128, d_model=512, heads=8, ffn=2048, layers=6,
   causal=False, padding=0, dtype=float32, warmup=20, repeats=100,
   rounds=3`.

The harness measures median wall time of one full forward pass, with
20 warmup calls to amortize compilation/caching, and 100 timed
iterations × 3 rounds. CUDA-style `cuda.Event` is replaced by
`time.perf_counter` on MPS, which the harness does automatically.

---

## V0 — Baseline (no optimization)

**Solution name:** `v0_baseline`  (file: `solutions/v0_baseline.py`)

**Description:** Subclass `BaselineTransformer` and let
`forward()` delegate straight to the parent. The whole point is to
have a "do nothing" copy so the other four can be measured against
the exact same code path the harness builds for the reference.

**Steps:**

```python
class UserOptimizedTransformer(BaselineTransformer):
    def forward(self, x, valid_token_mask=None):
        return super().forward(x, valid_token_mask)
```

**Result:** `optimized = 14.04 ms` vs `baseline = 14.10 ms` → **1.00×**.
`max_abs = 0` (literally the same code). Use this number to confirm
the harness isn't adding bias.

---

## V1 — `F.scaled_dot_product_attention`

**Solution name:** `v1_sdpa`  (file: `solutions/v1_sdpa.py`)

**Description:** A new `SDPASelfAttention` keeps the same parameter
names (`q_proj, k_proj, v_proj, out_proj`) so `copy_model_weights`
still works, but its `forward` replaces the explicit
QKᵀ → mask → softmax → ·V pipeline with one call to
`F.scaled_dot_product_attention`. The block (LayerNorm + residual +
FFN) is left untouched, so the comparison isolates exactly the
attention change.

**Steps:**

1. Build the parent layers via `super().__init__`, then swap each
   block's `attention` for `SDPASelfAttention(d_model, num_heads)`.
2. In the new attention, project X to Q, K, V exactly as the baseline
   does.
3. Build an additive `attn_mask` (0 allowed, `-inf` blocked) when
   needed. The MPS SDPA kernel rejects `is_causal=True` together with
   `attn_mask`, so when both are required (causal + padding) we fuse
   them into one `(1, 1, S, S)` tensor.
4. Call `F.scaled_dot_product_attention(q, k, v, attn_mask=...)`
   (or `is_causal=True` for the pure-causal fast path).
5. Merge heads and run the output projection, same as baseline.

**Result:** `optimized = 1.58 ms` vs `baseline = 13.40 ms` → **8.47×**.
Accuracy: `max_abs ≈ 2.1e-6`, every trial passes the
`atol=0.002 / rtol=0.02` rule.

**Why it is fast.** The baseline attention materializes a full
`(B, H, S, S) = (8, 8, 128, 128) = 1.05 M` element fp32 scores
tensor, runs softmax on it, and only then matmuls with V. That is
O(S²) memory traffic per layer × 6 layers = 12.6 M elements of
working set, plus four kernel launches per layer. SDPA fuses all of
that into a single streaming kernel that never holds the full S×S
matrix in memory, dropping the working set to roughly O(S·D). For
seq=128 the speedup is 8.5×; for seq=256 it grows to 15×, and for
seq=512 we measured **20.2×** — the win scales with the attention
matrix size, exactly as the theory predicts.

**When it would fail.** If you needed the attention probabilities as
a separate output (e.g., for attention-rollout interpretability
or differentiable masking), SDPA hides them. Also, on a hardware
backend where the SDPA math-fallback is the only option (some
older ROCm / CPU configurations), it can be slower than the manual
implementation; here the MPS backend picks a fused kernel.

---

## V2 — `torch.compile`

**Solution name:** `v2_compile`  (file: `solutions/v2_compile.py`)

**Description:** Wrap the parent forward in `torch.compile`. The
class lazily creates the compiled function on the first forward
call (so weight copy and `.to(device, dtype)` happen first), then
delegates every subsequent call to the compiled artifact.

**Steps:**

```python
def forward(self, x, valid_token_mask=None):
    if self._compiled_forward is None:
        self._compiled_forward = torch.compile(
            self._baseline_forward, dynamic=False
        )
    return self._compiled_forward(x, valid_token_mask)
```

**Result:** `optimized = 41.64 ms` vs `baseline = 14.46 ms` → **0.35×**.
Slower than the eager baseline by ~3×. Accuracy passes (max_abs ≈
2.9e-6) because the math is the same, only the dispatch is
different.

**Why it is slow on this machine.** Two visible signals:

1. The log line
   `W... Not enough SMs to use max_autotune_gemm mode`
   fires on every run. Inductor wants to autotune GEMM kernel
   choice, MPS exposes one GEMM, so it falls back to a per-op
   wrapper.
2. The per-op wrapper adds Python-side guard checks, meta-tensor
   propagation, and a per-call dispatch into the compiled artifact
   that MPSGraph (the eager path) doesn't have.

For this 6-layer fp32 transformer the baseline forward is dominated
by a small number of large GEMMs (6 × `Linear(512,2048)`, 6 ×
`Linear(2048,512)`, 6 × 4 × `Linear(512,512)`). MPSGraph already
emits well-tuned Metal kernels for those, and Inductor's wrapper
just adds overhead.

**When V2 would win.** On a CUDA GPU with `max-autotune` mode and a
larger model (≥ ~12 layers, or seq ≥ 1024), Inductor's
`max_autotune_gemm` flag can find faster kernels than cuBLAS and
the fusion of LayerNorm + residual + GELU pays off. On MPS at this
size, skip it.

---

## V3 — Manual fusion (QKV + GELU-tanh)

**Solution name:** `v3_fused`  (file: `solutions/v3_fused.py`)

**Description:** Two hand-rolled fusions, no graph compiler.

1. **Fused QKV:** keep the same `q_proj / k_proj / v_proj`
   parameters (so weight copy still works), but at forward time
   concatenate their weight matrices into one `(3*d_model, d_model)`
   tensor and do **one** `F.linear` instead of three. The fused
   weight is built lazily on the first forward and cached as a
   buffer.
2. **GELU-tanh:** switch `F.gelu(x, approximate='none')` to
   `F.gelu(x, approximate='tanh')` in the FFN. The tanh
   approximation is one `tanh` + a cubic instead of an `erf` call.

**Steps:**

* Define `FusedQKVSelfAttention` (same param names as baseline).
* Define `FusedTransformerBlock` that uses it and uses
  `approximate='tanh'`.
* `UserOptimizedTransformer.__init__` builds its own
  `nn.ModuleList` of `FusedTransformerBlock`s (skipping the parent
  layer construction) and re-uses the standard forward loop.

**Result:** `optimized = 14.02 ms` vs `baseline = 14.18 ms` →
**1.01×**. Accuracy passes (max_abs ≈ 4.9e-4, max_rel can be huge
for near-zero values but every element satisfies `abs_error ≤ 0.002
OR rel_error ≤ 0.02`).

**Why it barely moves the needle.** The QKV fusion saves
kernel-launch overhead: three `(1024, 512) @ (512, 512)` matmuls
collapse to one `(1024, 512) @ (1536, 512)`. At fp32 on the M4 Pro
each of those GEMMs is small enough to be memory-bandwidth bound
rather than launch-overhead bound, so the saved launches are
drowned out by the actual compute time. The GELU-tanh change is
similar — `erf` and `tanh` are both tiny elementwise kernels on
Metal and the saved flops don't show up in wall time.

**Lesson.** At this problem size the baseline is GEMM dominated.
Rearranging GEMM borders (QKV fusion) or replacing tiny kernels
(GELU) doesn't change the dominant cost. To get a big win you have
to attack the **attention** itself (V1) or write a kernel that
fuses attention with its surrounding ops (FlashAttention-style /
Triton).

---

## V4 — SDPA + `torch.compile` (combined)

**Solution name:** `v4_sdpa_compile`  (file: `solutions/v4_sdpa_compile.py`)

**Description:** Use `SDPASelfAttention` (the V1 attention) **and**
wrap the whole forward in `torch.compile`. On a CUDA stack this is
the strongest single config that doesn't require a custom kernel.

**Steps:** Same as V1 for the attention; same as V2 for the
`torch.compile` wrapper.

**Result:** `optimized = 39.63 ms` vs `baseline = 14.32 ms` →
**0.36×**. Slower than baseline by ~3×. Accuracy passes
(max_abs ≈ 3.1e-6).

**Why it is slower even with SDPA underneath.** The forward graph
contains the fast `scaled_dot_product_attention` call, but every
invocation of it now goes through Inductor's generated wrapper
(per-call dispatch, shape/stride/contiguity guards, autotune-miss
fallback). The cost of those wrapper ops eats most of V1's 8.5×
attention win. The rest of the wrapper (LayerNorm + GELU +
residual fusion) does help, but not enough to recover the wrapper
overhead on this backend.

**When V4 would win.** On a CUDA GPU V4 typically beats V1 by
another 1.2–1.4× because Inductor can fuse the LayerNorm + GELU
+ residual into a single Triton kernel that matches SDPA's matmul
time. On MPS, where the Inductor → Metal codegen is not yet
competitive, the best current single change is **V1 (SDPA eager)**.

---

## Cross-cutting analysis

### 1. The single biggest lever is the attention itself

Going from the manual `QKᵀ → mask → softmax → ·V` to
`scaled_dot_product_attention` is a 1-line change in the user code
and gives **8.5×** at the default config, **20×** at seq 512. The
reason is structural: the baseline pays O(S²) memory traffic per
layer, SDPA pays O(S·D), and the gap widens with sequence length.

### 2. `torch.compile` is not free on MPS

V2 and V4 are both **3× slower** than baseline. The same code on
CUDA is usually 1.2–1.5× faster than eager; on Apple Silicon the
Inductor codegen for Metal is in an earlier state, the autotune
path doesn't apply, and the per-op wrapper overhead exceeds the
fusion savings for this size of model. Worth revisiting when the
PyTorch MPS backend matures.

### 3. Micro-fusions don't help when you're already GEMM bound

V3's QKV + GELU-tanh gives 1.01×. Useful as a "negative result"
that confirms the bottleneck is not in launch overhead or in tiny
pointwise kernels — it's in the six large `Linear(512, 2048)`
matmuls and the attention matmul. To speed those up you need a
real kernel (Triton, MPSGraph hand-rolled, fused attention), not
PyTorch op rearrangement.

### 4. Combined optimizations are not additive

V4 = V1 + V2 should logically be ≥ V1, but on this backend it
loses to baseline. The lesson: when stacking optimizations, the
second one must at least break even on its own overhead, otherwise
it cancels the first. Here Inductor's wrapper overhead is so large
that even a free attention change can't beat it.

### 5. Recommendation

For inference of fp32 transformer blocks at this size on Apple
Silicon, use `F.scaled_dot_product_attention` (V1). For
production-style deployments, also consider:

* mixed-precision (bf16) — V1 in bf16 with relaxed tolerance still
  gives 7.5× (extra F);
* longer sequences, where V1's win grows super-linearly (extras A
  and G);
* a hand-rolled fused-attention kernel in MPSGraph or Metal
  compute shaders if you need to beat SDPA further, but on the
  M4 Pro SDPA already runs at > 1 M tokens / s which is hard to
  improve on with hand-written code.

---

## Reproducing these results

```bash
# 1. Create the venv (only needed once).
/Users/ary/.workbuddy/binaries/python/versions/3.13.12/bin/python3 -m venv \
  /Users/ary/.workbuddy/binaries/python/envs/torchjam
/Users/ary/.workbuddy/binaries/python/envs/torchjam/bin/pip install torch torchvision

# 2. Run every variant at the default config.
bash run_all.sh

# 3. Run a single variant with custom config.
/Users/ary/.workbuddy/binaries/python/envs/torchjam/bin/python3 \
  solutions/v1_sdpa.py --device mps --dtype float32 \
  --batch-size 8 --seq-len 256 --d-model 512 --heads 8 \
  --ffn-dim 2048 --layers 6 --causal
```

The per-variant logs are written to `results/<variant>.log`; the
per-variant write-ups are in `results/<variant>_results.md`; the
cross-experiment table is in `results/summary.md`.
