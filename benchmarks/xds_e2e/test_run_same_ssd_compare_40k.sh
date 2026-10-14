#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/yangxc/workspace/LMCache-v043-xds
RUNNER=$ROOT/benchmarks/xds_e2e/run_same_ssd_compare_40k.sh

output=$(BENCHMARK_DRY_RUN=1 RESULT_DIR=/tmp/lmcache-v043-same-ssd-dry-run \
  bash "$RUNNER")

filesystem_marker='switch_lmcache_test_ssds.sh filesystem --yes --confirm ERASE-aa-d3-ec-ed'
disk_marker='BACKEND=disk'
xds_marker='switch_lmcache_test_ssds.sh xds --yes --confirm ERASE-aa-d3-ec-ed'
service_marker='xds_service --config /home/yangxc/workspace/xDS/configs/setup_qwen3_30b_tp4_numa1_layerwise.json --service-name xds0'
cascade_marker='BACKEND=cascadekv'

for marker in "$filesystem_marker" "$disk_marker" "$xds_marker" \
  "$service_marker" "$cascade_marker"; do
  [[ "$output" == *"$marker"* ]]
done

filesystem_tail=${output#*"$filesystem_marker"}
disk_tail=${filesystem_tail#*"$disk_marker"}
xds_tail=${disk_tail#*"$xds_marker"}
service_tail=${xds_tail#*"$service_marker"}
[[ "$filesystem_tail" != "$output" ]]
[[ "$disk_tail" != "$filesystem_tail" ]]
[[ "$xds_tail" != "$disk_tail" ]]
[[ "$service_tail" != "$xds_tail" ]]
[[ "$service_tail" == *"$cascade_marker"* ]]

[[ "$output" == *'0000:aa:00.0'* ]]
[[ "$output" == *'0000:d3:00.0'* ]]
[[ "$output" == *'0000:ec:00.0'* ]]
[[ "$output" == *'0000:ed:00.0'* ]]

cascade_only=$(RUN_MODE=cascadekv BENCHMARK_DRY_RUN=1 \
  RESULT_DIR=/tmp/lmcache-v043-same-ssd-dry-run bash "$RUNNER")
[[ "$cascade_only" != *"$filesystem_marker"* ]]
[[ "$cascade_only" != *"$disk_marker"* ]]
[[ "$cascade_only" == *"$service_marker"* ]]
[[ "$cascade_only" == *"$cascade_marker"* ]]

if rg -q 'lmcache\.__version__' "$RUNNER"; then
  echo 'manifest must not require lmcache.__version__' >&2
  exit 1
fi
rg -q 'metadata\.version\("lmcache"\)' "$RUNNER"
if rg -q '\| rg ' "$RUNNER"; then
  echo 'orchestrator must not require rg in the root login PATH' >&2
  exit 1
fi

printf 'LMCACHE_V043_SAME_SSD_ORCHESTRATOR_TESTS_OK\n'
