#!/usr/bin/env bash
# Run the default-config benchmark for every solution and dump the full
# log into results/. The log is what each per-variant results.md quotes
# from; this script is what the README's "How to run all" refers to.
#
# Usage:
#   bash run_all.sh                  # default config, all 5 variants
#   bash run_all.sh --causal         # enable causal masking
#   bash run_all.sh --padding 0.25   # 25% padding
#   bash run_all.sh --bf16           # bfloat16 dtype
#
# Each variant is run with the SAME arguments so the numbers are
# directly comparable.

set -eo pipefail

cd "$(dirname "$0")"

PYBIN="/Users/ary/.workbuddy/binaries/python/envs/torchjam/bin/python3"
COMMON_FLAGS=(
  --device mps
  --dtype float32
  --batch-size 8
  --seq-len 128
  --d-model 512
  --heads 8
  --ffn-dim 2048
  --layers 6
  --warmup 20
  --repeats 100
  --benchmark-rounds 3
  --accuracy-trials 5
  --rtol 0.02
  --atol 0.002
)

# Allow extra flags to be passed in to compare causal/padding/dtype.
if [ "$#" -gt 0 ]; then
  EXTRA_FLAGS=("$@")
else
  EXTRA_FLAGS=()
fi

mkdir -p results

for variant in v0_baseline v1_sdpa v2_compile v3_fused v4_sdpa_compile; do
  echo "=========================================="
  echo "  Running ${variant}"
  echo "=========================================="
  log_file="results/${variant}.log"
  "$PYBIN" "solutions/${variant}.py" "${COMMON_FLAGS[@]}" "${EXTRA_FLAGS[@]}" 2>&1 \
    | tee "${log_file}"
  echo
done

echo
echo "All variants complete. Logs in results/*.log"
