#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yangxc/workspace
LMCACHE_ROOT=$ROOT/LMCache-v043-xds
VLLM_ROOT=$ROOT/vllm-v0180
PYTHON=/home/yangxc/miniconda3/envs/xds043/bin/python
MODEL=/mnt/adminserver-nfsrdma/Model/Qwen3-30B-A3B
CONFIG=${LMCACHE_CONFIG_FILE:-$LMCACHE_ROOT/benchmarks/xds_e2e/cascadekv_tp2_40k.yaml}
BENCHMARK_LOG=${BENCHMARK_LOG:-}
NUM_PROMPTS=${NUM_PROMPTS:-1}
REPEAT_COUNT=${REPEAT_COUNT:-2}

if [[ -z "$BENCHMARK_LOG" ]]; then
  printf 'BENCHMARK_LOG must be set\n' >&2
  exit 2
fi

mkdir -p "$(dirname "$BENCHMARK_LOG")"
env \
  CUDA_VISIBLE_DEVICES=4,5 \
  PYTHONHASHSEED=0 \
  PYTHONPATH="$ROOT/xDS/sdk/src:$LMCACHE_ROOT:$VLLM_ROOT" \
  LMCACHE_USE_EXPERIMENTAL=True \
  LMCACHE_LOG_LEVEL=INFO \
  LMCACHE_CONFIG_FILE="$CONFIG" \
  "$PYTHON" \
  "$LMCACHE_ROOT/benchmarks/xds_e2e/benchmark_prefix_caching_metrics.py" \
  --model "$MODEL" \
  --enforce-eager \
  --max-model-len 40960 \
  --trust-remote-code \
  --cpu-offload-gb 0 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --prefix-len 0 \
  --tensor-parallel-size 2 \
  --input-length-range 40000:40000 \
  --num-prompts "$NUM_PROMPTS" \
  --repeat-count "$REPEAT_COUNT" \
  --output-len 1 \
  --max-num-seqs 1 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.85 \
  --seed 0 >"$BENCHMARK_LOG" 2>&1
