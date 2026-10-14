#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yangxc/workspace
LMCACHE_ROOT=$ROOT/LMCache-v043-xds
SERVICE=$ROOT/xDS/src/service/xds_service
SERVICE_CONFIG=$ROOT/xDS/configs/setup_qwen3_30b_tp2_numa1_layerwise.json
RUNNER=$LMCACHE_ROOT/benchmarks/xds_e2e/run_cascadekv_tp2_40k.sh
RESULT_DIR=${RESULT_DIR:-$ROOT/test/results/lmcache-v043-cascadekv-tp2-$(date -u +%Y%m%dT%H%M%SZ)}
NUM_PROMPTS=${NUM_PROMPTS:-1}
REPEAT_COUNT=${REPEAT_COUNT:-2}
SERVICE_PID=

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVICE_PID" ]] && kill -0 "$SERVICE_PID" 2>/dev/null; then
    kill -INT "$SERVICE_PID" || true
    wait "$SERVICE_PID" || true
  fi
  rm -f /tmp/xds_xds0.sock /tmp/xds_xds0.shm
  mount -o remount,ro /dev/hugepages 2>/dev/null || true
  if id yangxc >/dev/null 2>&1; then
    chown -R yangxc:"$(id -gn yangxc)" "$RESULT_DIR" || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

[[ "$EUID" == 0 ]] || {
  printf 'run as root\n' >&2
  exit 1
}

mkdir -p "$RESULT_DIR"
mount -o remount,rw /dev/hugepages
rm -f /tmp/xds_xds0.sock /tmp/xds_xds0.shm

env XDS_SERVICE_ALLOWED_UID=0 "$SERVICE" \
  --config "$SERVICE_CONFIG" --service-name xds0 \
  >"$RESULT_DIR/xds-service.log" 2>&1 &
SERVICE_PID=$!

for _attempt in $(seq 1 120); do
  if [[ -S /tmp/xds_xds0.sock ]]; then
    break
  fi
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    wait "$SERVICE_PID" || true
    printf 'xDS service exited before becoming ready\n' >&2
    exit 1
  fi
  sleep 1
done
[[ -S /tmp/xds_xds0.sock ]] || {
  printf 'timed out waiting for xDS service\n' >&2
  exit 1
}

BENCHMARK_LOG="$RESULT_DIR/cascadekv.log" \
NUM_PROMPTS="$NUM_PROMPTS" REPEAT_COUNT="$REPEAT_COUNT" \
bash "$RUNNER"

printf 'TP2_CASCADEKV_PROFILE_OK result_dir=%s\n' "$RESULT_DIR"
