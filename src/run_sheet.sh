#!/usr/bin/env bash
# Run solutions/v1_sdpa.py (SDPA) across the 14 configurations from the test
# sheet. Each row:  Batch | QKV Dim (=d_model) | Heads | Seq Len | Layers |
# Causal | FFN Dim.  Logs go to results/sheet_<row>.log
#
# Timing flags are scaled per row so huge rows (batch 10000, seq 100000)
# don't take forever, and every row uses the same flags where possible so
# speedups stay comparable.

set -eo pipefail
cd "$(dirname "$0")"

PYBIN="/Users/ary/.workbuddy/binaries/python/envs/torchjam/bin/python3"
mkdir -p results

# Standard timing for small rows: warmup 10, repeats 50, 2 rounds, 2 trials.
STD="--warmup 10 --repeats 50 --benchmark-rounds 2 --accuracy-trials 2"
# Reduced timing for giant rows.
BIG="--warmup 3 --repeats 10 --benchmark-rounds 2 --accuracy-trials 2"

run_row () {
  local row="$1"; local bs="$2"; local dm="$3"; local h="$4"; local sl="$5"
  local ly="$6"; local ff="$7"; local tf="$8"
  echo "=================================================="
  echo "  Row ${row}: batch=${bs} d_model=${dm} heads=${h} seq=${sl} layers=${ly} causal ffn=${ff}"
  echo "=================================================="
  "$PYBIN" solutions/v1_sdpa.py \
    --device mps --dtype float32 --causal \
    --batch-size "$bs" --d-model "$dm" --heads "$h" --seq-len "$sl" \
    --layers "$ly" --ffn-dim "$ff" --rtol 0.02 --atol 0.002 $tf \
    2>&1 | tee "results/sheet_${row}.log"
  echo
}

#        row  batch  d_model heads seq   layers ffn    flags
run_row   1    64     128    4     128   4      128    "$STD"
run_row   2    1      128    4     128   4      128    "$STD"
run_row   3    4      128    4     128   4      128    "$STD"
run_row   4    16     128    4     128   4      128    "$STD"
run_row   5    128    128    4     128   4      128    "$STD"
run_row   6    10000  128    4     128   4      128    "$BIG"
run_row   7    64     32     4     128   4      32     "$STD"
run_row   8    64     1024   4     128   4      1024   "$STD"
run_row   9    64     128    1     128   4      128    "$STD"
run_row   10   64     128    2     128   4      128    "$STD"
run_row   11   64     128    16    128   4      128    "$STD"
run_row   12   64     128    4     32    4      128    "$STD"
run_row   13   64     128    4     1024  4      128    "$BIG"
run_row   14   32     1024   16    100000 2     1024   "$BIG"

echo "All sheet rows complete. Logs in results/sheet_*.log"
