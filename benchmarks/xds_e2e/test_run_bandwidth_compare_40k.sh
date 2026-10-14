#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/yangxc/workspace/LMCache-v043-xds
output=$(BENCHMARK_DRY_RUN=1 RESULT_DIR=/tmp/lmcache-bandwidth-dry-run \
  bash "$ROOT/benchmarks/xds_e2e/run_bandwidth_compare_40k.sh")

for marker in \
  '0000:aa:00.0' \
  'raw SSD baseline: buffered + O_DIRECT' \
  'BACKEND=disk NUM_PROMPTS=5 REPEAT_COUNT=10' \
  'BACKEND=disk_odirect NUM_PROMPTS=5 REPEAT_COUNT=10' \
  'BACKEND=cascadekv NUM_PROMPTS=5 REPEAT_COUNT=10' \
  'setup_qwen3_30b_tp4_numa1_layerwise.json' \
  'switch_lmcache_test_ssds.sh xds --yes --confirm ERASE-aa-d3-ec-ed'; do
  [[ "$output" == *"$marker"* ]]
done

printf 'LMCACHE_V043_BANDWIDTH_ORCHESTRATOR_TESTS_OK\n'
