#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/yangxc/workspace
LMCACHE_ROOT=$ROOT/LMCache-v043-xds
RUNNER=$LMCACHE_ROOT/benchmarks/xds_e2e/run_backend_40k.sh
SERVICE=${SERVICE:-$ROOT/xDS/src/service/xds_service}
SERVICE_CONFIG=${SERVICE_CONFIG:-$ROOT/xDS/configs/setup_qwen3_30b_tp4_numa1_layerwise_swap8.json}
CASCADE_CONFIG=${CASCADE_CONFIG:-$LMCACHE_ROOT/benchmarks/xds_e2e/cascadekv_40k_prefix_swap8.yaml}
WORKLOAD_BACKEND=${WORKLOAD_BACKEND:-cascadekv}
WORKLOAD_CONFIG=${WORKLOAD_CONFIG:-$CASCADE_CONFIG}
SWITCH=$ROOT/test/scripts/switch_lmcache_test_ssds.sh
RESULT_DIR=${RESULT_DIR:-$ROOT/test/results/lmcache-v043-prefix-pressure-ab-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ONLY=${RUN_ONLY:-all}
NODE1_HUGEPAGES=${NODE1_HUGEPAGES:-}
BENCHMARK_CPUSET=${BENCHMARK_CPUSET:-40-63,104-127}
SERVICE_PID=
HUGEPAGES_REMOUNTED=0
ORIGINAL_NODE1_HUGEPAGES=

stop_service() {
  if [[ -n "$SERVICE_PID" ]] && kill -0 "$SERVICE_PID" 2>/dev/null; then
    kill -INT "$SERVICE_PID" || true
    local attempt
    for attempt in $(seq 1 50); do
      kill -0 "$SERVICE_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$SERVICE_PID" 2>/dev/null; then
      kill -TERM "$SERVICE_PID" || true
      for attempt in $(seq 1 20); do
        kill -0 "$SERVICE_PID" 2>/dev/null || break
        sleep 0.1
      done
    fi
    if kill -0 "$SERVICE_PID" 2>/dev/null; then
      printf 'xDS service did not stop after INT/TERM; sending KILL\n' >&2
      kill -KILL "$SERVICE_PID" || true
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

run_case() {
  local name=$1
  local num_prompts=$2
  local repeat_count=$3
  local prefix_caching=$4
  local gpu_util=$5
  printf 'running %s: n=%s repeats=%s prefix=%s gpu_util=%s\n' \
    "$name" "$num_prompts" "$repeat_count" "$prefix_caching" "$gpu_util"
  env \
    BACKEND="$WORKLOAD_BACKEND" \
    BENCHMARK_LOG="$RESULT_DIR/$name.log" \
    NUM_PROMPTS="$num_prompts" \
    REPEAT_COUNT="$repeat_count" \
    ENABLE_PREFIX_CACHING="$prefix_caching" \
    GPU_MEMORY_UTILIZATION="$gpu_util" \
    MAX_NUM_SEQS=1 \
    BENCHMARK_CPUSET="$BENCHMARK_CPUSET" \
    LMCACHE_BENCH_COLD_SETTLE_SECONDS=10 \
    LMCACHE_CONFIG_FILE="$WORKLOAD_CONFIG" \
    bash "$RUNNER"
}

main() {
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
  if ps -eo comm= | grep -Eq '^xds_service' || \
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

  mkdir -p "$RESULT_DIR"
  chmod 0755 "$RESULT_DIR"
  cp "$WORKLOAD_CONFIG" "$RESULT_DIR/lmcache-config.yaml"
  cp "$SERVICE_CONFIG" "$RESULT_DIR/xds-service-config.json"

  if [[ -n "$NODE1_HUGEPAGES" ]]; then
    ORIGINAL_NODE1_HUGEPAGES=$(cat \
      /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages)
    printf '%s\n' "$NODE1_HUGEPAGES" \
      >/sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
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

  env XDS_SERVICE_ALLOWED_UID=0 "$SERVICE" --config "$SERVICE_CONFIG" \
    --service-name xds0 >"$RESULT_DIR/xds-service.log" 2>&1 &
  SERVICE_PID=$!
  wait_for_service

  case "$RUN_ONLY" in
    all)
      run_case a_prefix_off_gpu085_n1x20 1 20 0 0.85
      run_case b_prefix_off_gpu027_n8x20 8 20 0 0.27
      ;;
    d)
      run_case d_prefix_off_gpu085_n8x20 8 20 0 0.85
      ;;
    e)
      run_case e_prefix_on_gpu027_n8x20_staging8 \
        "${E_NUM_PROMPTS:-8}" "${E_REPEAT_COUNT:-20}" 1 0.27
      ;;
    workload)
      run_case workload_prefix_off_shared5k_n100x2 \
        "${WORKLOAD_NUM_PROMPTS:-100}" "${WORKLOAD_REPEAT_COUNT:-2}" 0 \
        "${WORKLOAD_GPU_UTILIZATION:-0.85}"
      ;;
    *)
      printf 'unsupported RUN_ONLY=%s; expected all, d, e, or workload\n' \
        "$RUN_ONLY" >&2
      return 2
      ;;
  esac
  stop_service
  printf 'LMCACHE_PREFIX_PRESSURE_AB_OK result_dir=%s\n' "$RESULT_DIR"
}

main "$@"
