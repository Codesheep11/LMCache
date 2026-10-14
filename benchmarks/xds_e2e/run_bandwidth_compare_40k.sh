#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/yangxc/workspace
LMCACHE_ROOT=$ROOT/LMCache-v043-xds
RUNNER=$LMCACHE_ROOT/benchmarks/xds_e2e/run_backend_40k.sh
ANALYZER=$LMCACHE_ROOT/benchmarks/xds_e2e/analyze_bandwidth_compare.py
SWITCH=$ROOT/test/scripts/switch_lmcache_test_ssds.sh
SERVICE=$ROOT/xDS/src/service/xds_service
SERVICE_CONFIG=$ROOT/xDS/configs/setup_qwen3_30b_tp4_numa1_layerwise.json
PYTHON=/home/yangxc/miniconda3/envs/xds043/bin/python
MOUNT_POINT=/mnt/lmcache-raid0
CACHE_DIR=$MOUNT_POINT/disk-cache
BASELINE_FILE=$MOUNT_POINT/lmcache-ssd-read-baseline.bin
BASELINE_BLOCK_MIB=6
BASELINE_COUNT=21845
CONFIRM_TOKEN=ERASE-aa-d3-ec-ed
BDFS=(0000:aa:00.0 0000:d3:00.0 0000:ec:00.0 0000:ed:00.0)
RESULT_DIR=${RESULT_DIR:-$ROOT/test/results/lmcache-v043-bandwidth-5x10-$(date -u +%Y%m%dT%H%M%SZ)}
NUM_PROMPTS=${NUM_PROMPTS:-5}
REPEAT_COUNT=${REPEAT_COUNT:-10}
BENCHMARK_DRY_RUN=${BENCHMARK_DRY_RUN:-0}
SERVICE_PID=
HUGEPAGES_REMOUNTED=0

dry_run() {
  printf 'approved SSDs: %s\n' "${BDFS[*]}"
  printf '%s filesystem --yes --confirm %s\n' "$SWITCH" "$CONFIRM_TOKEN"
  printf 'raw SSD baseline: buffered + O_DIRECT\n'
  printf 'BACKEND=disk NUM_PROMPTS=%s REPEAT_COUNT=%s\n' "$NUM_PROMPTS" "$REPEAT_COUNT"
  printf 'BACKEND=disk_odirect NUM_PROMPTS=%s REPEAT_COUNT=%s\n' "$NUM_PROMPTS" "$REPEAT_COUNT"
  printf '%s xds --yes --confirm %s\n' "$SWITCH" "$CONFIRM_TOKEN"
  printf '%s --config %s --service-name xds0\n' "$SERVICE" "$SERVICE_CONFIG"
  printf 'BACKEND=cascadekv NUM_PROMPTS=%s REPEAT_COUNT=%s\n' "$NUM_PROMPTS" "$REPEAT_COUNT"
}

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
  [[ "$EUID" == 0 ]] || {
    printf 'run as root\n' >&2
    return 1
  }
  local state
  state=$(bash "$SWITCH" status)
  [[ "$state" == *'"state":"xds"'* ]] || {
    printf 'test SSDs are not in xDS state: %s\n' "$state" >&2
    return 1
  }
  if pgrep -f '[x]ds_service' >/dev/null; then
    printf 'xDS service is already active\n' >&2
    return 1
  fi
  local busy_gpus
  busy_gpus=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F, '$1 + 0 >= 4 && $1 + 0 <= 7 && $2 + 0 > 16 {print $0}')
  [[ -z "$busy_gpus" ]] || {
    printf 'GPU 4-7 are busy:\n%s\n' "$busy_gpus" >&2
    return 1
  }
}

record_read() {
  local mode=$1
  local start_ns end_ns byte_count seconds
  sync
  printf '3\n' >/proc/sys/vm/drop_caches
  start_ns=$(date +%s%N)
  if [[ "$mode" == odirect ]]; then
    dd if="$BASELINE_FILE" of=/dev/null bs="${BASELINE_BLOCK_MIB}M" \
      count="$BASELINE_COUNT" iflag=direct,fullblock status=none
  else
    dd if="$BASELINE_FILE" of=/dev/null bs="${BASELINE_BLOCK_MIB}M" \
      count="$BASELINE_COUNT" iflag=fullblock status=none
  fi
  end_ns=$(date +%s%N)
  byte_count=$((BASELINE_BLOCK_MIB * 1024 * 1024 * BASELINE_COUNT))
  seconds=$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN {printf "%.9f", (end-start)/1000000000}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$byte_count" "$start_ns" "$end_ns" "$seconds" >>"$RESULT_DIR/ssd-baseline.tsv"
}

clear_disk_cache() {
  mkdir -p "$CACHE_DIR"
  find "$CACHE_DIR" -mindepth 1 -maxdepth 1 -type f -delete
}

main() {
  preflight
  mkdir -p "$RESULT_DIR"
  chmod 0755 "$RESULT_DIR"
  printf 'mode\tbytes\tstart_ns\tend_ns\tseconds\n' >"$RESULT_DIR/ssd-baseline.tsv"
  git -C "$LMCACHE_ROOT" rev-parse HEAD >"$RESULT_DIR/lmcache-commit.txt"
  git -C "$ROOT/vllm-v0180" rev-parse HEAD >"$RESULT_DIR/vllm-commit.txt"

  printf '[1/8] switching approved SSDs to mdadm/ext4\n'
  bash "$SWITCH" filesystem --yes --confirm "$CONFIRM_TOKEN"

  printf '[2/8] preparing 128 GiB direct-I/O baseline file\n'
  dd if=/dev/zero of="$BASELINE_FILE" bs="${BASELINE_BLOCK_MIB}M" \
    count="$BASELINE_COUNT" oflag=direct status=none
  record_read buffered
  record_read odirect
  rm -f "$BASELINE_FILE"

  printf '[3/8] running buffered LocalDisk, %s unique x %s\n' "$NUM_PROMPTS" "$REPEAT_COUNT"
  clear_disk_cache
  LMCACHE_BENCH_DROP_CACHE_AFTER_COLD=1 BACKEND=disk NUM_PROMPTS="$NUM_PROMPTS" \
    REPEAT_COUNT="$REPEAT_COUNT" BENCHMARK_LOG="$RESULT_DIR/disk-buffered.log" bash "$RUNNER"

  printf '[4/8] running O_DIRECT LocalDisk, %s unique x %s\n' "$NUM_PROMPTS" "$REPEAT_COUNT"
  clear_disk_cache
  LMCACHE_BENCH_DROP_CACHE_AFTER_COLD=1 BACKEND=disk_odirect NUM_PROMPTS="$NUM_PROMPTS" REPEAT_COUNT="$REPEAT_COUNT" \
    BENCHMARK_LOG="$RESULT_DIR/disk-odirect.log" bash "$RUNNER"

  printf '[5/8] returning SSDs to xDS\n'
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

  printf '[6/8] starting xDS service\n'
  env XDS_SERVICE_ALLOWED_UID=0 "$SERVICE" --config "$SERVICE_CONFIG" \
    --service-name xds0 >"$RESULT_DIR/xds-service.log" 2>&1 &
  SERVICE_PID=$!
  wait_for_service

  printf '[7/8] running CascadeKV/xDS, %s unique x %s\n' "$NUM_PROMPTS" "$REPEAT_COUNT"
  BACKEND=cascadekv NUM_PROMPTS="$NUM_PROMPTS" REPEAT_COUNT="$REPEAT_COUNT" \
    BENCHMARK_LOG="$RESULT_DIR/cascadekv.log" bash "$RUNNER"
  stop_service

  printf '[8/8] analyzing measurements\n'
  "$PYTHON" "$ANALYZER" "$RESULT_DIR"
  bash "$SWITCH" status >"$RESULT_DIR/final-ssd-state.json"
  printf 'LMCACHE_V043_BANDWIDTH_COMPARISON_OK result_dir=%s\n' "$RESULT_DIR"
}

main "$@"
