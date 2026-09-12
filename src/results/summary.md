# Results Summary

Default config (every row below uses these flags):

```
--device mps --dtype float32
--batch-size 8 --seq-len 128 --d-model 512
--heads 8 --ffn-dim 2048 --layers 6
--warmup 20 --repeats 100 --benchmark-rounds 3 --accuracy-trials 5
--rtol 0.02 --atol 0.002
```

All numbers come from the per-variant logs in this directory.
"Optimized" is the variant being benchmarked; "Baseline" is the
reference `BaselineTransformer` the harness constructs alongside it.

## Main result table

| Variant | Description | Accuracy | Optimized median | Baseline median | **Speedup** |
| ------- | ----------- | :------: | ---------------: | --------------: | ----------: |
| V0      | No change (sanity)              | PASS (max_abs = 0)        | 14.04 ms | 14.10 ms | **1.00×** |
| V1      | `F.scaled_dot_product_attention` | PASS (max_abs ≈ 2.1e-6)    | **1.58 ms** | 13.40 ms | **8.47×** |
| V2      | `torch.compile` (default mode)   | PASS (max_abs ≈ 2.9e-6)    | 41.64 ms | 14.46 ms | **0.35×** ❌ |
| V3      | Fused QKV + GELU-tanh            | PASS (max_abs ≈ 4.9e-4)    | 14.02 ms | 14.18 ms | **1.01×** |
| V4      | SDPA + `torch.compile`           | PASS (max_abs ≈ 3.1e-6)    | 39.63 ms | 14.32 ms | **0.36×** ❌ |

## Throughput (tokens / second)

| Variant | Optimized tok/s | Baseline tok/s | Δ |
| ------- | --------------: | -------------: | ---: |
| V0      |   72,915        |   72,619        |  +0.4 % |
| V1      |  **647,026**    |   76,413        | **+747 %** |
| V2      |   24,594        |   70,814        |  -65 % |
| V3      |   73,023        |   72,206        |  +1.1 % |
| V4      |   25,837        |   71,522        |  -64 % |

## Extras (varied config, not the default)

| Tag | Setup                              | Speedup | Note |
| --- | ---------------------------------- | ------: | ---- |
| A   | V1, `seq-len 256`                  | **15.0×** | SDPA scales with attention-matrix size |
| B   | V1, default + `--causal`           |  8.62× | Causal path uses `is_causal=True` |
| C   | V2 with `mode='reduce-overhead'`   |   0.35× | Same as V2; the mode is a no-op for our shape |
| D   | V1, `dtype bfloat16`, default atol |  (FAIL)  | bf16 abs error is ~6e-2 > atol 2e-3 — expected |
| E   | V3, default config                 |   1.01× | Confirms main run; included for completeness |
| F   | V1, `dtype bfloat16`, `--atol 0.1` |   7.51× | SDPA wins in bf16 once tolerance matches precision |
| G₀  | V0, `seq-len 512`                  |   1.00× | Baseline reference at long seq |
| G₁  | V1, `seq-len 512`                  | **20.2×** | Baseline attention blows up; SDPA barely slows down |

## V5 — hybrid router (new)

V5 builds on V1 and switches between all 4 solutions based on input
parameters (plus a 5th chunked-streaming strategy for huge sequences).
Sheet-sweep details: [V5 hybrid results](v5_hybrid_results.md).

| Router rule | Condition | Strategy | Measured effect |
| ----------- | --------- | -------- | --------------- |
| Extreme seq | `seq_len >= 4096` | chunked (O(S·d) streaming) | only runnable path (row 14) |
| Tiny batch | `batch_size <= 2` | SDPA + `torch.compile` | 2.30× vs 1.25× (row 2) |
| Tiny width | `d_model <= 64` | fused QKV | best of the four (row 7) |
| Default | — | SDPA | best on 10/13 rows |

Across the 13 runnable sheet rows, V5's auto-routing matched the best of
V1–V4 on 12 rows (the one miss is −2%). Row 14 (seq=100000): all of
V0–V4 OOM; V5's chunked path is numerically exact (1e-7 vs SDPA) and works
at moderate scale, but the full batch=32/d=1024 config exceeds the 48 GB
unified-memory budget — full-scale run abandoned.

## Per-variant deep dives

* [V0 baseline](v0_baseline_results.md)
* [V1 SDPA](v1_sdpa_results.md)
* [V2 torch.compile](v2_compile_results.md)
* [V3 manual fusion](v3_fused_results.md)
* [V4 SDPA + compile](v4_sdpa_compile_results.md)
* [V5 hybrid router](v5_hybrid_results.md)

The cross-cutting analysis and recommendations live in
[`../REPORT.md`](../REPORT.md).
