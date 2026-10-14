#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/yangxc/workspace
LMCACHE_ROOT=$ROOT/LMCache-v043-xds
RUNNER=$LMCACHE_ROOT/benchmarks/xds_e2e/run_backend_40k.sh
ANALYZER=$LMCACHE_ROOT/benchmarks/xds_e2e/analyze_prefix_swap.py
SWITCH=$ROOT/test/scripts/switch_lmcache_test_ssds.sh
SERVICE=$ROOT/xDS/src/service/xds_service
SERVICE_CONFIG=$ROOT/xDS/configs/setup_qwen3_30b_tp4_numa1_layerwise_swap8.json
CASCADE_CONFIG=$LMCACHE_ROOT/benchmarks/xds_e2e/cascadekv_40k_prefix_swap8.yaml
PYTHON=/home/yangxc/miniconda3/envs/xds043/bin/python
MOUNT_POINT=/mnt/lmcache-raid0
CACHE_DIR=$MOUNT_POINT/disk-cache
CONFIRM_TOKEN=ERASE-aa-d3-ec-ed
BDFS=(0000:aa:00.0 0000:d3:00.0 0000:ec:00.0 0000:ed:00.0)
RESULT_DIR=${RESULT_DIR:-$ROOT/test/results/lmcache-v043-prefix-swap8x20-$(date -u +%Y%m%dT%H%M%SZ)}
NUM_PROMPTS=8
REPEAT_COUNT=20
GPU_MEMORY_UTILIZATION=0.27
SERVICE_PID=
HUGEPAGES_REMOUNTED=0

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
  for attempt in $(seq 1 300); do
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
  [[ "$EUID" == 0 ]] || {
    printf 'run as root\n' >&2
    return 1
  }
  for path in "$RUNNER" "$ANALYZER" "$SWITCH" "$SERVICE" \
    "$SERVICE_CONFIG" "$CASCADE_CONFIG" "$PYTHON"; do
    [[ -e "$path" ]] || {
      printf 'required path is missing: %s\n' "$path" >&2
      return 1
    }
  done
  local state
  state=$(bash "$SWITCH" status)
  [[ "$state" == *'"state":"xds"'* ]] || {
    printf 'test SSDs are not in xDS state: %s\n' "$state" >&2
    return 1
  }
  if pgrep -f '[x]ds_service' >/dev/null || \
    [[ -e /tmp/xds_xds0.sock || -e /tmp/xds_xds0.shm ]]; then
    printf 'xDS service or IPC namespace is already active\n' >&2
    return 1
  fi
  local busy_gpus
  busy_gpus=$(nvidia-smi --query-gpu=index,memory.used \
    --format=csv,noheader,nounits | \
    awk -F, '$1 + 0 >= 4 && $1 + 0 <= 7 && $2 + 0 > 16 {print $0}')
  [[ -z "$busy_gpus" ]] || {
    printf 'GPU 4-7 are busy:\n%s\n' "$busy_gpus" >&2
    return 1
  }
}

run_backend() {
  local backend=$1
  local log_path=$2
  shift 2
  env \
    ENABLE_PREFIX_CACHING=1 \
    GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    MAX_NUM_SEQS=1 \
    NUM_PROMPTS="$NUM_PROMPTS" \
    REPEAT_COUNT="$REPEAT_COUNT" \
    LMCACHE_BENCH_COLD_SETTLE_SECONDS=10 \
    BACKEND="$backend" \
    BENCHMARK_LOG="$log_path" \
    "$@" \
    bash "$RUNNER"
}

main() {
  preflight
  mkdir -p "$RESULT_DIR"
  chmod 0755 "$RESULT_DIR"
  {
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'num_prompts=%s\nrepeat_count=%s\n' "$NUM_PROMPTS" "$REPEAT_COUNT"
    printf 'prefix_caching=true\ngpu_memory_utilization=%s\n' "$GPU_MEMORY_UTILIZATION"
    printf 'ssd_bdfs=%s\n' "${BDFS[*]}"
    git -C "$LMCACHE_ROOT" rev-parse HEAD
    git -C "$ROOT/vllm-v0180" rev-parse HEAD
    bash "$SWITCH" status
  } >"$RESULT_DIR/manifest.txt"
  cp "$CASCADE_CONFIG" "$RESULT_DIR/cascadekv-config.yaml"
  cp "$SERVICE_CONFIG" "$RESULT_DIR/xds-service-config.json"

  printf '[1/6] switching four approved SSDs to mdadm/ext4 RAID0\n'
  bash "$SWITCH" filesystem --yes --confirm "$CONFIRM_TOKEN"
  mkdir -p "$CACHE_DIR"
  find "$CACHE_DIR" -mindepth 1 -maxdepth 1 -type f -delete

  printf '[2/6] Local Disk O_DIRECT: 8 unique 40K prompts x 20 cycles\n'
  run_backend disk_odirect "$RESULT_DIR/disk-odirect.log" \
    LMCACHE_BENCH_DROP_CACHE_AFTER_COLD=1
  sync

  printf '[3/6] returning the same SSDs to xDS RAID0\n'
  bash "$SWITCH" xds --yes --confirm "$CONFIRM_TOKEN"

  local hugepage_options
  hugepage_options=$(findmnt -no OPTIONS /dev/hugepages)
  case ",$hugepage_options," in
    *,rw,*) ;;
    *)
      mount -o remount,rw /dev/hugepages
      HUGEPAGES_REMOUNTED=1
      ;;
  esac

  printf '[4/6] starting the 32 GiB layerwise xDS pool\n'
  env XDS_SERVICE_ALLOWED_UID=0 "$SERVICE" --config "$SERVICE_CONFIG" \
    --service-name xds0 >"$RESULT_DIR/xds-service.log" 2>&1 &
  SERVICE_PID=$!
  wait_for_service

  printf '[5/6] CascadeKV: 8 unique 40K prompts x 20 cycles\n'
  run_backend cascadekv "$RESULT_DIR/cascadekv.log" \
    LMCACHE_CONFIG_FILE="$CASCADE_CONFIG"
  stop_service

  printf '[6/6] summarizing results\n'
  "$PYTHON" "$ANALYZER" "$RESULT_DIR" >"$RESULT_DIR/summary.stdout.json"
  bash "$SWITCH" status >"$RESULT_DIR/final-ssd-state.json"
  printf 'LMCACHE_V043_PREFIX_SWAP_COMPARISON_OK result_dir=%s\n' "$RESULT_DIR"
}

main "$@"
