#!/usr/bin/env bash
# Sweep v2 (compile), v3 (fused), v4 (sdpa+compile) over the same 14-row
# sheet used for v1, so we can extract the best variant per configuration.
# Row failures (e.g. row 14 OOM for every variant) are tolerated per-row:
# the script keeps going and the per-row log records the crash.
#
# Logs: results/sheet_<variant>_<row>.log

set -uo pipefail
cd "$(dirname "$0")"

PYBIN="/Users/ary/.workbuddy/binaries/python/envs/torchjam/bin/python3"
mkdir -p results

STD="--warmup 10 --repeats 50 --benchmark-rounds 2 --accuracy-trials 2"
BIG="--warmup 3 --repeats 10 --benchmark-rounds 2 --accuracy-trials 2"

run_row () {
  local variant="$1"; local row="$2"; local bs="$3"; local dm="$4"; local h="$5"
  local sl="$6"; local ly="$7"; local ff="$8"; local tf="$9"
  local log="results/sheet_${variant}_${row}.log"
  echo "=================================================="
  echo "  ${variant} | Row ${row}: batch=${bs} d_model=${dm} heads=${h} seq=${sl} layers=${ly} causal ffn=${ff}"
  echo "=================================================="
  # do not abort the sweep if this row crashes
  "$PYBIN" "solutions/${variant}.py" \
    --device mps --dtype float32 --causal \
    --batch-size "$bs" --d-model "$dm" --heads "$h" --seq-len "$sl" \
    --layers "$ly" --ffn-dim "$ff" --rtol 0.02 --atol 0.002 $tf \
    > "$log" 2>&1 || { echo "  !! row ${row} FAILED (see ${log})"; }
  tail -n 12 "$log"
  echo
}

for variant in v2_compile v3_fused v4_sdpa_compile; do
  run_row "$variant"  1  64   128   4   128  4   128   "$STD"
  run_row "$variant"  2  1    128   4   128  4   128   "$STD"
  run_row "$variant"  3  4    128   4   128  4   128   "$STD"
  run_row "$variant"  4  16   128   4   128  4   128   "$STD"
  run_row "$variant"  5  128  128   4   128  4   128   "$STD"
  run_row "$variant"  6  10000 128  4   128  4   128   "$BIG"
  run_row "$variant"  7  64   32    4   128  4   32    "$STD"
  run_row "$variant"  8  64   1024  4   128  4   1024  "$STD"
  run_row "$variant"  9  64   128   1   128  4   128   "$STD"
  run_row "$variant" 10  64   128   2   128  4   128   "$STD"
  run_row "$variant" 11  64   128  16   128  4   128   "$STD"
  run_row "$variant" 12  64   128   4    32  4   128   "$STD"
  run_row "$variant" 13  64   128   4   1024  4   128   "$BIG"
  run_row "$variant" 14  32   1024 16  100000 2  1024   "$BIG"
done

echo "All variant sweeps complete. Logs in results/sheet_<variant>_<row>.log"
