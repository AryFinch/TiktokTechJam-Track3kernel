#!/usr/bin/env bash
# Extra experiments to characterize when each optimization pays off.
# Output goes to results/extras_<tag>.log

set -eo pipefail
cd "$(dirname "$0")"

PYBIN="/Users/ary/.workbuddy/binaries/python/envs/torchjam/bin/python3"
mkdir -p results

run() {
  local tag="$1"
  shift
  echo "=========================================="
  echo "  ${tag}"
  echo "=========================================="
  local log="results/extras_${tag}.log"
  "$PYBIN" "$@" 2>&1 | tee "${log}"
  echo
}

# A: larger sequence length (attention dominates more)
run "A_sdpa_seq256" solutions/v1_sdpa.py \
  --device mps --dtype float32 --batch-size 8 --seq-len 256 --d-model 512 \
  --heads 8 --ffn-dim 2048 --layers 6 --warmup 10 --repeats 50 \
  --benchmark-rounds 2 --accuracy-trials 3

# B: causal masking on the default config (typical autoregressive setup)
run "B_sdpa_causal" solutions/v1_sdpa.py \
  --device mps --dtype float32 --batch-size 8 --seq-len 128 --d-model 512 \
  --heads 8 --ffn-dim 2048 --layers 6 --warmup 20 --repeats 100 \
  --benchmark-rounds 3 --accuracy-trials 3 --causal

# C: compile in reduce-overhead mode (CUDA-only flag, but try it on MPS)
run "C_compile_reduce" solutions/v2_compile.py \
  --device mps --dtype float32 --batch-size 8 --seq-len 128 --d-model 512 \
  --heads 8 --ffn-dim 2048 --layers 6 --warmup 20 --repeats 100 \
  --benchmark-rounds 3 --accuracy-trials 3

# D: bfloat16 — compile may behave differently under lower precision
run "D_sdpa_bf16" solutions/v1_sdpa.py \
  --device mps --dtype bfloat16 --batch-size 8 --seq-len 128 --d-model 512 \
  --heads 8 --ffn-dim 2048 --layers 6 --warmup 20 --repeats 100 \
  --benchmark-rounds 3 --accuracy-trials 3

# E: V3 fused with default config (so the report quotes the same data)
run "E_v3_default" solutions/v3_fused.py \
  --device mps --dtype float32 --batch-size 8 --seq-len 128 --d-model 512 \
  --heads 8 --ffn-dim 2048 --layers 6 --warmup 20 --repeats 100 \
  --benchmark-rounds 3 --accuracy-trials 3
