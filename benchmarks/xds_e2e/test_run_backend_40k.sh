#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/yangxc/workspace/LMCache-v043-xds
RUNNER=$ROOT/benchmarks/xds_e2e/run_backend_40k.sh
CASCADE_CONFIG=$ROOT/benchmarks/xds_e2e/cascadekv_40k.yaml

disk=$(BACKEND=disk BENCHMARK_LOG=/tmp/disk.log BENCHMARK_DRY_RUN=1 bash "$RUNNER")
cascade=$(BACKEND=cascadekv BENCHMARK_LOG=/tmp/cascade.log BENCHMARK_DRY_RUN=1 bash "$RUNNER")
smoke=$(BACKEND=cascadekv BENCHMARK_LOG=/tmp/cascade-smoke.log \
  BENCHMARK_DRY_RUN=1 NUM_PROMPTS=1 REPEAT_COUNT=2 bash "$RUNNER")

[[ "$disk" == *disk_40k.yaml* ]]
[[ "$cascade" == *cascadekv_40k.yaml* ]]
for output in "$disk" "$cascade"; do
  [[ "$output" == *"/home/yangxc/miniconda3/envs/xds043/bin/python"* ]]
  [[ "$output" == *"/home/yangxc/workspace/vllm-v0180/benchmarks/benchmark_prefix_caching.py"* ]]
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

rg -q '^max_local_cpu_size: 2$' "$CASCADE_CONFIG"
rg -q '^xds_max_size: 500$' "$CASCADE_CONFIG"
rg -q '^xds_enable_chunked_overlap: true$' "$CASCADE_CONFIG"
rg -q '^xds_chunked_size_kb: 4096$' "$CASCADE_CONFIG"
rg -q '^xds_write_window_size: 64$' "$CASCADE_CONFIG"
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
