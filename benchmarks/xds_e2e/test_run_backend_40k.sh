#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/yangxc/workspace/LMCache-v043-xds
RUNNER=$ROOT/benchmarks/xds_e2e/run_backend_40k.sh
CASCADE_CONFIG=$ROOT/benchmarks/xds_e2e/cascadekv_40k.yaml
SERVICE_CONFIG=/home/yangxc/workspace/xDS/configs/setup_qwen3_30b_tp4_numa1_layerwise.json

disk=$(BACKEND=disk BENCHMARK_LOG=/tmp/disk.log BENCHMARK_DRY_RUN=1 bash "$RUNNER")
cascade=$(BACKEND=cascadekv BENCHMARK_LOG=/tmp/cascade.log BENCHMARK_DRY_RUN=1 bash "$RUNNER")
smoke=$(BACKEND=cascadekv BENCHMARK_LOG=/tmp/cascade-smoke.log \
  BENCHMARK_DRY_RUN=1 NUM_PROMPTS=1 REPEAT_COUNT=2 bash "$RUNNER")
override=$(BACKEND=cascadekv BENCHMARK_LOG=/tmp/cascade-override.log \
  BENCHMARK_DRY_RUN=1 LMCACHE_CONFIG_FILE=/tmp/custom-cascadekv.yaml \
  bash "$RUNNER")
isolated=$(BACKEND=cascadekv BENCHMARK_LOG=/tmp/cascade-isolated.log \
  BENCHMARK_CPUSET=40-63,96-127 BENCHMARK_DRY_RUN=1 bash "$RUNNER")
prefix_swap=$(BACKEND=cascadekv BENCHMARK_LOG=/tmp/cascade-prefix.log \
  BENCHMARK_DRY_RUN=1 ENABLE_PREFIX_CACHING=1 \
  GPU_MEMORY_UTILIZATION=0.27 MAX_NUM_SEQS=1 bash "$RUNNER")
long_workload=$(BACKEND=cascadekv BENCHMARK_LOG=/tmp/cascade-workload.log \
  BENCHMARK_DRY_RUN=1 PREFIX_LEN=5000 INPUT_LENGTH_RANGE=30000:40000 \
  OUTPUT_LEN=1 NUM_PROMPTS=100 REPEAT_COUNT=2 bash "$RUNNER")

[[ "$disk" == *disk_40k.yaml* ]]
[[ "$cascade" == *cascadekv_40k.yaml* ]]
[[ "$cascade" == *benchmark_prefix_caching_metrics.py* ]]
for output in "$disk" "$cascade"; do
  [[ "$output" == *"/home/yangxc/miniconda3/envs/xds043/bin/python"* ]]
  [[ "$output" == *"benchmark_prefix_caching_metrics.py"* ]]
  [[ "$output" == *"--model /mnt/adminserver-nfsrdma/Model/Qwen3-30B-A3B"* ]]
  [[ "$output" == *"--max-model-len 40960"* ]]
  [[ "$output" == *"--input-length-range 40000:40000"* ]]
  [[ "$output" == *"--num-prompts 5"* ]]
  [[ "$output" == *"--repeat-count 20"* ]]
  [[ "$output" == *"--output-len 1"* ]]
  [[ "$output" == *"--tensor-parallel-size 4"* ]]
  [[ "$output" == *"--no-enable-prefix-caching"* ]]
  [[ "$output" != *"--enable-prefix-caching"* ]]
done
[[ "$smoke" == *"--num-prompts 1"* ]]
[[ "$smoke" == *"--repeat-count 2"* ]]
[[ "$override" == *"LMCACHE_CONFIG_FILE=/tmp/custom-cascadekv.yaml"* ]]
[[ "$isolated" == taskset\ -c\ 40-63\\,96-127* ]]
[[ "$prefix_swap" == *"--enable-prefix-caching"* ]]
[[ "$prefix_swap" != *"--no-enable-prefix-caching"* ]]
[[ "$prefix_swap" == *"--gpu-memory-utilization 0.27"* ]]
[[ "$long_workload" == *"--prefix-len 5000"* ]]
[[ "$long_workload" == *"--input-length-range 30000:40000"* ]]
[[ "$long_workload" == *"--num-prompts 100"* ]]
[[ "$long_workload" == *"--repeat-count 2"* ]]
[[ "$long_workload" == *"--output-len 1"* ]]
[[ "$long_workload" == *"--no-enable-prefix-caching"* ]]

rg -q '^max_local_cpu_size: 2$' "$CASCADE_CONFIG"
rg -q '^xds_max_size: 6$' "$CASCADE_CONFIG"
rg -q '^xds_enable_chunked_overlap: true$' "$CASCADE_CONFIG"
rg -q '^use_layerwise: true$' "$CASCADE_CONFIG"
rg -q '^xds_chunked_size_kb: 4096$' "$CASCADE_CONFIG"
rg -q '^xds_write_window_size: 64$' "$CASCADE_CONFIG"
rg -q '"xds_cluster_size_kb": 128' "$SERVICE_CONFIG"
rg -q '"xds_blob_size_bytes": 131072' "$SERVICE_CONFIG"
rg -q '"xds_blob_count": 196608' "$SERVICE_CONFIG"
if rg -q '^cascadekv_(slot|read_slot|write_slot|io_queue)' "$CASCADE_CONFIG"; then
  echo "deprecated CascadeKV slot settings are still present" >&2
  exit 1
fi

if BACKEND=unknown BENCHMARK_LOG=/tmp/unknown.log BENCHMARK_DRY_RUN=1 \
  bash "$RUNNER" >/dev/null 2>&1; then
  echo "unknown backend unexpectedly succeeded" >&2
  exit 1
else
  status=$?
  [[ "$status" == 2 ]]
fi

printf 'LMCACHE_V043_BACKEND_RUNNER_TESTS_OK\n'
