#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/yangxc/workspace
LMCACHE_ROOT=$ROOT/LMCache-v043-xds
RUNNER=$LMCACHE_ROOT/benchmarks/xds_e2e/run_backend_40k.sh
SWITCH=$ROOT/test/scripts/switch_lmcache_test_ssds.sh
SERVICE=$ROOT/xDS/src/service/xds_service
SERVICE_CONFIG=${SERVICE_CONFIG:-$ROOT/xDS/configs/setup_qwen3_30b_tp4_numa1_layerwise.json}
PYTHON=/home/yangxc/miniconda3/envs/xds043/bin/python
MOUNT_POINT=/mnt/lmcache-raid0
CACHE_DIR=$MOUNT_POINT/disk-cache
CONFIRM_TOKEN=ERASE-aa-d3-ec-ed
BDFS=(0000:aa:00.0 0000:d3:00.0 0000:ec:00.0 0000:ed:00.0)
RESULT_DIR=${RESULT_DIR:-$ROOT/test/results/lmcache-v043-cascadekv-disk-40k-$(date -u +%Y%m%dT%H%M%SZ)}
BENCHMARK_DRY_RUN=${BENCHMARK_DRY_RUN:-0}
RUN_MODE=${RUN_MODE:-all}
SERVICE_PID=
HUGEPAGES_REMOUNTED=0

dry_run() {
  printf 'approved SSDs: %s\n' "${BDFS[*]}"
  if [[ "$RUN_MODE" == all ]]; then
    printf '%s filesystem --yes --confirm %s\n' "$SWITCH" "$CONFIRM_TOKEN"
    printf 'BACKEND=disk BENCHMARK_LOG=%s/disk.log %s\n' "$RESULT_DIR" "$RUNNER"
    printf '%s xds --yes --confirm %s\n' "$SWITCH" "$CONFIRM_TOKEN"
  fi
  printf '%s --config %s --service-name xds0\n' "$SERVICE" "$SERVICE_CONFIG"
  printf 'BACKEND=cascadekv BENCHMARK_LOG=%s/cascadekv.log %s\n' \
    "$RESULT_DIR" "$RUNNER"
}

case "$RUN_MODE" in
  all|cascadekv) ;;
  *)
    printf 'unsupported RUN_MODE=%s; expected all or cascadekv\n' "$RUN_MODE" >&2
    exit 2
    ;;
esac

if [[ "$BENCHMARK_DRY_RUN" == 1 ]]; then
  dry_run
  exit 0
fi

stop_service() {
  if [[ -n "$SERVICE_PID" ]] && kill -0 "$SERVICE_PID" 2>/dev/null; then
    kill -INT "$SERVICE_PID" || true
    wait "$SERVICE_PID" || true
  fi
  SERVICE_PID=
}

restore_xds_state() {
  local state
  state=$(bash "$SWITCH" status 2>/dev/null || true)
  if [[ "$state" != *'"state":"xds"'* ]]; then
    bash "$SWITCH" xds --yes --confirm "$CONFIRM_TOKEN" || true
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_service
  rm -f /tmp/xds_xds0.sock /tmp/xds_xds0.shm
  restore_xds_state
  if [[ "$HUGEPAGES_REMOUNTED" == 1 ]]; then
    mount -o remount,ro /dev/hugepages || true
  fi
  if [[ -d "$RESULT_DIR" ]] && id yangxc >/dev/null 2>&1; then
    chown -R yangxc:$(id -gn yangxc) "$RESULT_DIR" || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_for_service() {
  local attempt
  for attempt in $(seq 1 120); do
    if [[ -S /tmp/xds_xds0.sock ]]; then
      return 0
    fi
    if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
      wait "$SERVICE_PID" || true
      return 1
    fi
    sleep 1
  done
  return 1
}

preflight() {
  if [[ "$EUID" != 0 ]]; then
    printf 'run as root; SSD switching and xDS service require root privileges\n' >&2
    return 1
  fi
  for path in "$RUNNER" "$SWITCH" "$SERVICE" "$SERVICE_CONFIG" \
    "$PYTHON" /mnt/adminserver-nfsrdma/Model/Qwen3-30B-A3B; do
    if [[ ! -e "$path" ]]; then
      printf 'required path is missing: %s\n' "$path" >&2
      return 1
    fi
  done
  if pgrep -f '[x]ds_service' >/dev/null || \
    [[ -e /tmp/xds_xds0.sock || -e /tmp/xds_xds0.shm ]]; then
    printf 'xDS service or IPC namespace is already active\n' >&2
    return 1
  fi
  local state
  state=$(bash "$SWITCH" status)
  if [[ "$state" != *'"state":"xds"'* ]]; then
    printf 'test SSDs are not in the expected xDS state: %s\n' "$state" >&2
    return 1
  fi
  local busy_gpus
  busy_gpus=$(nvidia-smi --query-gpu=index,memory.used \
    --format=csv,noheader,nounits | \
    awk -F, '$1 + 0 >= 4 && $1 + 0 <= 7 && $2 + 0 > 16 {print $0}')
  if [[ -n "$busy_gpus" ]]; then
    printf 'GPU 4-7 are not idle:\n%s\n' "$busy_gpus" >&2
    return 1
  fi
}

capture_manifest() {
  {
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'lmcache_commit=%s\n' "$(git -C "$LMCACHE_ROOT" rev-parse HEAD)"
    printf 'vllm_commit=%s\n' "$(git -C "$ROOT/vllm-v0180" rev-parse HEAD)"
    printf 'ssd_bdfs=%s\n' "${BDFS[*]}"
    printf 'gpu_visible=4,5,6,7\n'
    "$PYTHON" -c \
      'from importlib import metadata; import torch; print("lmcache=" + metadata.version("lmcache")); print("vllm=" + metadata.version("vllm")); print(f"torch={torch.__version__}")'
    bash "$SWITCH" status
    nvidia-smi --query-gpu=index,name,memory.total,memory.used \
      --format=csv,noheader
    local bdf
    for bdf in "${BDFS[@]}"; do
      lspci -nnk -s "$bdf"
    done
  } >"$RESULT_DIR/manifest.txt"
  cp "$LMCACHE_ROOT/benchmarks/xds_e2e/disk_40k.yaml" \
    "$RESULT_DIR/disk_40k.yaml"
  cp "$LMCACHE_ROOT/benchmarks/xds_e2e/cascadekv_40k.yaml" \
    "$RESULT_DIR/cascadekv_40k.yaml"
  cp "$SERVICE_CONFIG" "$RESULT_DIR/xds-service-config.json"
}

main() {
  preflight
  mkdir -p "$RESULT_DIR"
  chmod 0755 "$RESULT_DIR"
  capture_manifest

  if [[ "$RUN_MODE" == all ]]; then
    printf '[1/5] switching the four approved SSDs to mdadm/ext4\n'
    bash "$SWITCH" filesystem --yes --confirm "$CONFIRM_TOKEN"
    mkdir -p "$CACHE_DIR"
    chmod 0755 "$CACHE_DIR"

    printf '[2/5] running buffered Local Disk with one 40K group\n'
    BACKEND=disk BENCHMARK_LOG="$RESULT_DIR/disk.log" bash "$RUNNER"
    sync

    printf '[3/5] destroying mdadm/ext4 metadata and returning SSDs to xDS\n'
    bash "$SWITCH" xds --yes --confirm "$CONFIRM_TOKEN"
  else
    printf '[resume] preserving the completed Local Disk result\n'
  fi

  local hugepage_options
  hugepage_options=$(findmnt -no OPTIONS /dev/hugepages)
  case ",$hugepage_options," in
    *,rw,*) ;;
    *)
      mount -o remount,rw /dev/hugepages
      HUGEPAGES_REMOUNTED=1
      ;;
  esac

  printf '[4/5] starting xDS service\n'
  env XDS_SERVICE_ALLOWED_UID=0 "$SERVICE" \
    --config "$SERVICE_CONFIG" --service-name xds0 \
    >"$RESULT_DIR/xds-service.log" 2>&1 &
  SERVICE_PID=$!
  if ! wait_for_service; then
    printf 'xDS service failed to become ready\n' >&2
    return 1
  fi

  printf '[5/5] running CascadeKV with one 40K group\n'
  BACKEND=cascadekv BENCHMARK_LOG="$RESULT_DIR/cascadekv.log" bash "$RUNNER"
  stop_service

  local final_state
  final_state=$(bash "$SWITCH" status)
  printf '%s\n' "$final_state" >"$RESULT_DIR/final-ssd-state.json"
  [[ "$final_state" == *'"state":"xds"'* ]]
  printf 'LMCACHE_V043_SAME_SSD_COMPARISON_OK result_dir=%s\n' "$RESULT_DIR"
}

main "$@"
