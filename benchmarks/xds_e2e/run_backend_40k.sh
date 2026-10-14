#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/yangxc/workspace
LMCACHE_ROOT=$ROOT/LMCache-v043-xds
VLLM_ROOT=$ROOT/vllm-v0180
PYTHON=/home/yangxc/miniconda3/envs/xds043/bin/python
MODEL=/mnt/adminserver-nfsrdma/Model/Qwen3-30B-A3B
BACKEND=${BACKEND:-}
BENCHMARK_LOG=${BENCHMARK_LOG:-}
BENCHMARK_DRY_RUN=${BENCHMARK_DRY_RUN:-0}
NUM_PROMPTS=${NUM_PROMPTS:-5}
REPEAT_COUNT=${REPEAT_COUNT:-20}
BENCHMARK_CPUSET=${BENCHMARK_CPUSET:-}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
PREFIX_LEN=${PREFIX_LEN:-0}
INPUT_LENGTH_RANGE=${INPUT_LENGTH_RANGE:-40000:40000}
OUTPUT_LEN=${OUTPUT_LEN:-1}

case "$BACKEND" in
  disk)
    config=$LMCACHE_ROOT/benchmarks/xds_e2e/disk_40k.yaml
    ;;
  disk_odirect)
    config=$LMCACHE_ROOT/benchmarks/xds_e2e/disk_odirect_40k.yaml
    ;;
  cascadekv)
    config=$LMCACHE_ROOT/benchmarks/xds_e2e/cascadekv_40k.yaml
    ;;
  xds)
    config=$LMCACHE_ROOT/benchmarks/xds_e2e/xds_30k40k_shared5k_n100x2.yaml
    ;;
  *)
    printf 'unsupported BACKEND=%s; expected disk, disk_odirect, cascadekv, or xds\n' "$BACKEND" >&2
    exit 2
    ;;
esac
config=${LMCACHE_CONFIG_FILE:-$config}

if [[ -z "$BENCHMARK_LOG" ]]; then
  printf 'BENCHMARK_LOG must be set\n' >&2
  exit 2
fi

launcher=()
if [[ -n "$BENCHMARK_CPUSET" ]]; then
  launcher=(taskset -c "$BENCHMARK_CPUSET")
fi

prefix_caching_arg=--no-enable-prefix-caching
if [[ "$ENABLE_PREFIX_CACHING" == 1 ]]; then
  prefix_caching_arg=--enable-prefix-caching
elif [[ "$ENABLE_PREFIX_CACHING" != 0 ]]; then
  printf 'ENABLE_PREFIX_CACHING must be 0 or 1: %s\n' \
    "$ENABLE_PREFIX_CACHING" >&2
  exit 2
fi

cmd=(
  "${launcher[@]}"
  env
  CUDA_VISIBLE_DEVICES=4,5,6,7
  PYTHONHASHSEED=0
  "PYTHONPATH=$ROOT/xDS/sdk/src:$LMCACHE_ROOT:$VLLM_ROOT"
  LMCACHE_USE_EXPERIMENTAL=True
  LMCACHE_LOG_LEVEL=INFO
  "LMCACHE_CONFIG_FILE=$config"
  "$PYTHON"
  "$LMCACHE_ROOT/benchmarks/xds_e2e/benchmark_prefix_caching_metrics.py"
  --model "$MODEL"
  --enforce-eager
  --max-model-len 40960
  --trust-remote-code
  --cpu-offload-gb 0
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
  --prefix-len "$PREFIX_LEN"
  --tensor-parallel-size 4
  --input-length-range "$INPUT_LENGTH_RANGE"
  --num-prompts "$NUM_PROMPTS"
  --repeat-count "$REPEAT_COUNT"
  --output-len "$OUTPUT_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  "$prefix_caching_arg"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --seed 0
)

if [[ "$BENCHMARK_DRY_RUN" == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$(dirname "$BENCHMARK_LOG")"
"${cmd[@]}" >"$BENCHMARK_LOG" 2>&1
