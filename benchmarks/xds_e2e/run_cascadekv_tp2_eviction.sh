#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/yangxc/workspace
LMCACHE_ROOT=$ROOT/LMCache-v043-xds
SERVICE=$ROOT/xDS/src/service/xds_service
SERVICE_CONFIG=${SERVICE_CONFIG:-$ROOT/xDS/configs/setup_qwen3_30b_tp2_numa1_block_blob_evict.json}
CASCADE_CONFIG=${CASCADE_CONFIG:-$LMCACHE_ROOT/benchmarks/xds_e2e/cascadekv_tp2_40k_block_blob_evict.yaml}
RUNNER=$LMCACHE_ROOT/benchmarks/xds_e2e/run_cascadekv_tp2_40k.sh
SWITCH=$ROOT/test/scripts/switch_lmcache_test_ssds.sh
RESULT_DIR=${RESULT_DIR:-$ROOT/test/results/lmcache-v043-cascadekv-tp2-eviction-$(date -u +%Y%m%dT%H%M%SZ)}
NODE1_HUGEPAGES=${NODE1_HUGEPAGES:-20480}
HIT_INDEX_SEQUENCE=${HIT_INDEX_SEQUENCE:-2,1,1,0,0,2,2}
SERVICE_PID=
HUGEPAGES_REMOUNTED=0
ORIGINAL_NODE1_HUGEPAGES=

stop_service() {
  if [[ -n "$SERVICE_PID" ]] && kill -0 "$SERVICE_PID" 2>/dev/null; then
    kill -INT "$SERVICE_PID" || true
    for _attempt in $(seq 1 50); do
      kill -0 "$SERVICE_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$SERVICE_PID" 2>/dev/null; then
      kill -TERM "$SERVICE_PID" || true
    fi
    wait "$SERVICE_PID" || true
  fi
  SERVICE_PID=
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_service
  rm -f /tmp/xds_xds0.sock /tmp/xds_xds0.shm
  if [[ -n "$ORIGINAL_NODE1_HUGEPAGES" ]]; then
    printf '%s\n' "$ORIGINAL_NODE1_HUGEPAGES" \
      >/sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages || true
  fi
  if [[ "$HUGEPAGES_REMOUNTED" == 1 ]]; then
    mount -o remount,ro /dev/hugepages || true
  fi
  if [[ -d "$RESULT_DIR" ]] && id yangxc >/dev/null 2>&1; then
    chown -R yangxc:"$(id -gn yangxc)" "$RESULT_DIR" || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

[[ "$EUID" == 0 ]] || {
  printf 'run as root\n' >&2
  exit 1
}

state=$(bash "$SWITCH" status)
[[ "$state" == *'"state":"xds"'* ]] || {
  printf 'test SSDs are not in xDS state: %s\n' "$state" >&2
  exit 1
}
if pgrep -f '[x]ds_service' >/dev/null || \
  [[ -e /tmp/xds_xds0.sock || -e /tmp/xds_xds0.shm ]]; then
  printf 'xDS service or IPC namespace is already active\n' >&2
  exit 1
fi
busy_gpus=$(nvidia-smi --query-gpu=index,memory.used \
  --format=csv,noheader,nounits | \
  awk -F, '$1 + 0 >= 4 && $1 + 0 <= 5 && $2 + 0 > 16 {print $0}')
[[ -z "$busy_gpus" ]] || {
  printf 'GPU 4-5 are busy:\n%s\n' "$busy_gpus" >&2
  exit 1
}

mkdir -p "$RESULT_DIR"
chmod 0755 "$RESULT_DIR"
cp "$CASCADE_CONFIG" "$RESULT_DIR/cascadekv-config.yaml"
cp "$SERVICE_CONFIG" "$RESULT_DIR/xds-service-config.json"
printf 'hit_index_sequence=%s\n' "$HIT_INDEX_SEQUENCE" >"$RESULT_DIR/manifest.txt"

ORIGINAL_NODE1_HUGEPAGES=$(cat \
  /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages)
printf '%s\n' "$NODE1_HUGEPAGES" \
  >/sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
hugepage_options=$(findmnt -no OPTIONS /dev/hugepages)
case ",$hugepage_options," in
  *,rw,*) ;;
  *)
    mount -o remount,rw /dev/hugepages
    HUGEPAGES_REMOUNTED=1
    ;;
esac

env XDS_SERVICE_ALLOWED_UID=0 "$SERVICE" --config "$SERVICE_CONFIG" \
  --service-name xds0 >"$RESULT_DIR/xds-service.log" 2>&1 &
SERVICE_PID=$!
for _attempt in $(seq 1 300); do
  [[ -S /tmp/xds_xds0.sock ]] && break
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

env \
  BENCHMARK_LOG="$RESULT_DIR/cascadekv.log" \
  NUM_PROMPTS=3 \
  REPEAT_COUNT=2 \
  LMCACHE_CONFIG_FILE="$CASCADE_CONFIG" \
  LMCACHE_BENCH_HIT_INDEX_SEQUENCE="$HIT_INDEX_SEQUENCE" \
  bash "$RUNNER"

stop_service
printf 'TP2_CASCADEKV_EVICTION_OK result_dir=%s\n' "$RESULT_DIR"
