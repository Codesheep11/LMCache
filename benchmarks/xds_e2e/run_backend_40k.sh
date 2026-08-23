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

case "$BACKEND" in
  disk)
    config=$LMCACHE_ROOT/benchmarks/xds_e2e/disk_40k.yaml
    ;;
  cascadekv)
    config=$LMCACHE_ROOT/benchmarks/xds_e2e/cascadekv_40k.yaml
    ;;
  *)
    printf 'unsupported BACKEND=%s; expected disk or cascadekv\n' "$BACKEND" >&2
    exit 2
    ;;
esac

if [[ -z "$BENCHMARK_LOG" ]]; then
  printf 'BENCHMARK_LOG must be set\n' >&2
  exit 2
fi

cmd=(
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
  --prefix-len 0
  --tensor-parallel-size 4
  --input-length-range 40000:40000
  --num-prompts "$NUM_PROMPTS"
  --repeat-count "$REPEAT_COUNT"
  --output-len 1
  --max-num-seqs 1
  --no-enable-prefix-caching
  --gpu-memory-utilization 0.85
  --seed 0
)

if [[ "$BENCHMARK_DRY_RUN" == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$(dirname "$BENCHMARK_LOG")"
"${cmd[@]}" >"$BENCHMARK_LOG" 2>&1
