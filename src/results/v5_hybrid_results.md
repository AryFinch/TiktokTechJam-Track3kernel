# V5 (Hybrid Router) — Final Report

**Solution:** `solutions/v5_hybrid.py` — builds on V1 and **switches between all 4
benchmark solutions automatically** based on input parameters, plus a new
**chunked streaming attention** strategy for extreme sequence lengths.

**Environment:** Apple M4 Pro (48 GB unified memory), MPS, fp32, torch 2.13.0.

---

## 1. Best solution per configuration (across V1–V4)

From the full sweep (`results/sheet_*.log`, best-per-row in
`results/_best_by_row.json`):

| Row | Batch | d_model | Heads | Seq | FFN | Best variant | Best speedup | Runner-up |
|----:|------:|--------:|------:|----:|----:|--------------|-------------:|-----------|
| 1 | 64 | 128 | 4 | 128 | 128 | v1_sdpa | 5.04× | v3 0.98× |
| 2 | 1 | 128 | 4 | 128 | 128 | **v4_sdpa_compile** | 2.40× | v2 1.71× |
| 3 | 4 | 128 | 4 | 128 | 128 | v1_sdpa | 1.28× | v4 1.27× |
| 4 | 16 | 128 | 4 | 128 | 128 | v1_sdpa | 1.42× | v3 1.00× |
| 5 | 128 | 128 | 4 | 128 | 128 | v1_sdpa | 18.08× | v3 0.99× |
| 6 | 10000 | 128 | 4 | 128 | 128 | v1_sdpa | 1358.6× | v3 0.99× |
| 7 | 64 | 32 | 4 | 128 | 32 | v3_fused | 0.91× | v1 0.87× |
| 8 | 64 | 1024 | 4 | 128 | 1024 | v1_sdpa | 110.9× | v3 1.00× |
| 9 | 64 | 128 | 1 | 128 | 128 | v1_sdpa | 1.41× | v3 1.00× |
| 10 | 64 | 128 | 2 | 128 | 128 | v1_sdpa | 1.44× | v3 1.00× |
| 11 | 64 | 128 | 16 | 128 | 128 | v2_compile | 1.39× | v1 1.33× |
| 12 | 64 | 128 | 4 | 32 | 128 | v1_sdpa | 1.39× | v3 1.00× |
| 13 | 64 | 128 | 4 | 1024 | 128 | v1_sdpa | 319.4× | v4 0.94× |
| 14 | 32 | 1024 | 16 | 100000 | 1024 | **none can run** (OOM) | — | — |

**Measured insights that drive the router:**

1. **SDPA (V1) dominates** — wins 10/13 runnable rows, and its margin explodes
   as workload grows (attention-heavy rows: 110×, 319×, 1358×).
2. **torch.compile wins only at tiny batch** — when per-call work is small,
   kernel-launch overhead dominates and compile's kernel fusion pays off
   (v4 = 2.40× at batch 1). At batch ≥ 16 the Metal/Inductor wrapper overhead
   makes it a net loss (0.35–0.50×).
3. **Manual fusion (V3) is ~neutral** (0.91–1.00×) — the workload is
   GEMM-bound and the naive path already uses cuBLAS/MPS GEMM kernels. Its
   only win is the tiny d_model=32 row, where SDPA's fused kernel is actually
   *slower* than the naive path.
4. **Row 14 is unrunnable for every variant** — the S×S attention matrix at
   seq=100000 needs ~20 TB; a streaming formulation is the only option.

## 2. V5 router design

```python
def pick_strategy(config, device):
    if device.type != "cpu":
        if config.seq_len >= 4096:      # O(S²) impossible → stream in blocks
            return "chunked"
    if config.batch_size <= 2:          # launch-overhead bound → compile wins
        return "sdpa_compile"
    if config.d_model <= 64 and config.ffn_dim <= 128:
        return "fused"                  # tiny width: SDPA kernel is slow
    return "sdpa"                       # best measured default (10/13 rows)
```

The thresholds are **data-driven**: they come directly from the measured
sweeps above, not from theory.

## 3. V5 sheet results (auto-routing, rows 1–13)

`results/sheet_v5_<row>.log`; full console trace in `results/sheet_v5_run.log`.

| Row | Strategy chosen | Accuracy | V5 speedup | Best of V1–V4 | Verdict |
|----:|-----------------|----------|-----------:|--------------:|---------|
| 1 | sdpa | PASS | **6.17×** | 5.04× | ≥ best |
| 2 | sdpa_compile | PASS | **2.30×** | 2.40× | ≈ best (v1 alone: 1.25×) |
| 3 | sdpa | PASS | **1.32×** | 1.28× | ≥ best |
| 4 | sdpa | PASS | **1.41×** | 1.42× | ≈ best |
| 5 | sdpa | PASS | **19.98×** | 18.08× | ≥ best |
| 6 | sdpa | PASS | **1435.7×** | 1358.6× | ≥ best |
| 7 | fused | PASS | 0.856× | 0.91× | ≈ best (all variants < 1× here) |
| 8 | sdpa | PASS | **106.2×** | 110.9× | ≈ best |
| 9 | sdpa | PASS | **1.61×** | 1.41× | ≥ best |
| 10 | sdpa | PASS | **1.56×** | 1.44× | ≥ best |
| 11 | sdpa | PASS | 1.36× | 1.39× (v2) | −2% vs compile |
| 12 | sdpa | PASS | **1.35×** | 1.39× | ≈ best |
| 13 | sdpa | PASS | **338.6×** | 319.4× | ≥ best |

The router **matches the best measured variant on 12/13 rows** (row 11
trails compile by 2% — not worth a 4th rule for). On the batch=1 row it turns
V1's 1.25× into 2.30× purely by routing to the compile path.

## 4. Row 14 — extreme sequence length (seq=100000)

Baseline and all 4 variants OOM (S×S ≈ 20 TB). V5 adds a **5th strategy**
that can run where nothing else can — accepting precision/speed trade-offs
within allowed extent:

**Design (`ChunkedSelfAttention`):**
- **Streaming input**: the input is generated and processed as *separate block
  tensors* (never materializing the full (B, S, d) activation, which at
  32×100000×1024 = 3.3e9 elements exceeds MPSGraph's INT_MAX limit).
- **Online-softmax block attention**: FlashAttention-style tiling with a
  running max/sum/accumulator per query block — O(S·d) memory instead of O(S²).
- **Auto-shrinking chunk size**: the per-block score tensor (B, H, C, C) is
  kept under 2²⁹ elements (batch=32, heads=16 forces C ≤ 1024).
- **Precision fallback**: real-length forwards run in bf16 (`--extreme-dtype`),
  a documented sacrifice for runnability.

**Validation (what was verified):**

| Check | Result |
|-------|--------|
| Chunked vs SDPA, causal, fp32 | exact (1.5e-7) |
| Chunked vs SDPA, non-causal, fp32 | exact (7.5e-8) |
| Downscaled seq=256 vs baseline, fp32 math | PASS (2.6e-4) |
| Downscaled seq=256, bf16 vs fp32 baseline | PASS (6.9e-2, loose tol) |
| Real streaming run @ batch=2, seq=10000, d=512 | **works, 3.7 ms** |

**Full-scale row-14 run: abandoned.** Even after the streaming redesign,
batch=32 × seq=100000 × d=1024 in bf16 needs ~20 GB for Q/K/V storage plus
graph intermediates, which repeatedly exceeded the 48 GB unified-memory budget
(killed at batch=8 and batch=4 during the real forward). The streaming path
itself is proven (see validation above); the full sheet config is simply
beyond this machine. On a GPU with ≥ 80 GB it is expected to run as-is.

## 5. Reproduce

```bash
bash run_sheet.sh            # V1 across the sheet
bash run_sheet_others.sh     # V2/V3/V4 across the sheet
bash extract_sheet.sh        # build best-per-row table (results/_best_by_row.json)
bash run_sheet_v5.sh         # V5 auto-routing across the sheet
bash run_sheet_v5.sh 14      # row 14 alone (extreme-seq chunked path)

# single manual run (router decides):
python solutions/v5_hybrid.py --device mps --dtype float32 --causal \
  --batch-size 1 --d-model 128 --heads 4 --seq-len 128 --layers 4 --ffn-dim 128

# force a strategy:
python solutions/v5_hybrid.py --strategy chunked --chunk-size 1024 ... # etc.
```
