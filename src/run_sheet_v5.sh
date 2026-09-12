#!/usr/bin/env bash
# Run V5 (hybrid router) across the 14-row sheet in "auto" mode, i.e. let
# V5 choose the strategy per configuration. Row 14 (seq=100000) uses the
# chunked streaming path automatically. Logs: results/sheet_v5_<row>.log
#
# Usage:
#   bash run_sheet_v5.sh           # full sheet, auto routing
#   bash run_sheet_v5.sh 1 5 13    # only selected rows

set -uo pipefail
cd "$(dirname "$0")"

PYBIN="/Users/ary/.workbuddy/binaries/python/envs/torchjam/bin/python3"
mkdir -p results

STD="--warmup 10 --repeats 50 --benchmark-rounds 2 --accuracy-trials 2"
BIG="--warmup 3 --repeats 10 --benchmark-rounds 2 --accuracy-trials 2"
# Row 14 is extremely slow (chunked streaming over 100k tokens), so use a
# tiny timing budget. Override with V5_REPEATS if you want more samples.
ROW14_REPEATS="${V5_REPEATS:-3}"

run_row () {
  local row="$1"; local bs="$2"; local dm="$3"; local h="$4"; local sl="$5"
  local ly="$6"; local ff="$7"; local tf="$8"
  local log="results/sheet_v5_${row}.log"
  echo "=================================================="
  echo "  v5_hybrid (auto) | Row ${row}: batch=${bs} d_model=${dm} heads=${h} seq=${sl} layers=${ly} causal ffn=${ff}"
  echo "=================================================="
  "$PYBIN" solutions/v5_hybrid.py \
    --device mps --dtype float32 --causal \
    --batch-size "$bs" --d-model "$dm" --heads "$h" --seq-len "$sl" \
    --layers "$ly" --ffn-dim "$ff" --rtol 0.02 --atol 0.002 $tf \
    > "$log" 2>&1 || { echo "  !! row ${row} FAILED (see ${log})"; }
  grep -E "V5 strategy|summary|speedup|optimized:|downscaled" "$log" | head -8
  echo
}

if [ "$#" -gt 0 ]; then
  for row in "$@"; do
    case "$row" in
      1)  run_row  1  64   128   4   128   4  128  "$STD";;
      2)  run_row  2  1    128   4   128   4  128  "$STD";;
      3)  run_row  3  4    128   4   128   4  128  "$STD";;
      4)  run_row  4  16   128   4   128   4  128  "$STD";;
      5)  run_row  5  128  128   4   128   4  128  "$STD";;
      6)  run_row  6  10000 128  4   128   4  128  "$BIG";;
      7)  run_row  7  64   32    4   128   4  32   "$STD";;
      8)  run_row  8  64   1024  4   128   4  1024 "$STD";;
      9)  run_row  9  64   128   1   128   4  128  "$STD";;
      10) run_row 10  64   128   2   128   4  128  "$STD";;
      11) run_row 11  64   128  16   128   4  128  "$STD";;
      12) run_row 12  64   128   4    32   4  128  "$STD";;
      13) run_row 13  64   128   4   1024   4  128  "$BIG";;
      14) run_row 14  32   1024 16  100000  2  1024 \
            "--warmup 1 --repeats ${ROW14_REPEATS} --benchmark-rounds 1 --accuracy-trials 1";;
      *) echo "unknown row $row"; exit 1;;
    esac
  done
else
  for row in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    case "$row" in
      1)  run_row  1  64   128   4   128   4  128  "$STD";;
      2)  run_row  2  1    128   4   128   4  128  "$STD";;
      3)  run_row  3  4    128   4   128   4  128  "$STD";;
      4)  run_row  4  16   128   4   128   4  128  "$STD";;
      5)  run_row  5  128  128   4   128   4  128  "$STD";;
      6)  run_row  6  10000 128  4   128   4  128  "$BIG";;
      7)  run_row  7  64   32    4   128   4  32   "$STD";;
      8)  run_row  8  64   1024  4   128   4  1024 "$STD";;
      9)  run_row  9  64   128   1   128   4  128  "$STD";;
      10) run_row 10  64   128   2   128   4  128  "$STD";;
      11) run_row 11  64   128  16   128   4  128  "$STD";;
      12) run_row 12  64   128   4    32   4  128  "$STD";;
      13) run_row 13  64   128   4   1024   4  128  "$BIG";;
      14) run_row 14  32   1024 16  100000  2  1024 \
            "--warmup 1 --repeats ${ROW14_REPEATS} --benchmark-rounds 1 --accuracy-trials 1";;
    esac
  done
fi

echo "V5 sheet complete. Logs in results/sheet_v5_*.log"
