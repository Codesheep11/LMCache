# SPDX-License-Identifier: Apache-2.0
"""Summarize the TP4 40K prefix-cache swap experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from analyze_bandwidth_compare import (
    GIB,
    active_bandwidth,
    marker_records,
    merged_interval_seconds,
    percentile,
    request_bandwidth,
)


def describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def summarize_native_xds_io(path: Path) -> dict[str, object]:
    """Report device I/O separately from staging allocation and rank skew."""
    records = marker_records(path, "[CASCADEKV_NATIVE_WINDOW_METRICS]")
    if not records:
        return {"count": 0}

    by_worker: dict[int, list[dict]] = {}
    for record in records:
        by_worker.setdefault(int(record["worker_id"]), []).append(record)

    worker_gib_s: dict[str, float] = {}
    for worker_id, worker_records in sorted(by_worker.items()):
        worker_bytes = sum(int(record["bytes"]) for record in worker_records)
        worker_seconds = sum(float(record["io_seconds"]) for record in worker_records)
        worker_gib_s[str(worker_id)] = worker_bytes / GIB / worker_seconds

    timed_records = [
        record
        for record in records
        if int(record.get("io_end_ns", 0)) > int(record.get("io_start_ns", 0)) > 0
    ]
    active_seconds = merged_interval_seconds(timed_records, "io_start_ns", "io_end_ns")
    active_gib_s = (
        sum(int(record["bytes"]) for record in timed_records) / GIB / active_seconds
        if active_seconds
        else None
    )
    request_count = min((len(items) for items in by_worker.values()), default=0)
    per_request_gib_s = []
    if len(timed_records) == len(records):
        for index in range(request_count):
            request_records = [items[index] for items in by_worker.values()]
            started_ns = min(int(record["io_start_ns"]) for record in request_records)
            ended_ns = max(int(record["io_end_ns"]) for record in request_records)
            if ended_ns > started_ns:
                request_bytes = sum(int(record["bytes"]) for record in request_records)
                per_request_gib_s.append(
                    request_bytes / GIB / ((ended_ns - started_ns) / 1e9)
                )

    return {
        "count": len(records),
        # The rank-local xDS queues run concurrently, so aggregate device bandwidth
        # is the sum of their independently measured byte/time rates.
        "aggregate_gib_s": sum(worker_gib_s.values()),
        "active_gib_s": active_gib_s,
        "active_seconds": active_seconds,
        "per_request_gib_s": describe(per_request_gib_s),
        "per_request_gib_s_values": per_request_gib_s,
        "per_worker_gib_s": worker_gib_s,
        "io_ms": describe([1000.0 * float(record["io_seconds"]) for record in records]),
        "allocation_ms": describe(
            [1000.0 * float(record["allocation_seconds"]) for record in records]
        ),
        "progress_callback_ms": describe(
            [
                1000.0 * float(record.get("progress_callback_seconds", 0.0))
                for record in records
            ]
        ),
    }


def summarize(path: Path, backend: str, num_prompts: int) -> dict:
    prefix_records = marker_records(path, "[PREFIX_BENCH_METRICS]")
    if not prefix_records:
        raise RuntimeError(f"missing prefix metrics in {path}")
    prefix = prefix_records[-1]
    hits = [request for request in prefix["requests"] if request["phase"] == "hit"]
    latencies = [float(request["inference_ms"]) for request in hits]
    cached_tokens = [int(request["num_cached_tokens"]) for request in hits]
    cycle_latencies = [
        describe(latencies[start : start + num_prompts])
        for start in range(0, len(latencies), num_prompts)
    ]
    e2e = request_bandwidth(marker_records(path, "[LMCACHE_SSD_GPU_METRICS]"))
    if backend == "disk_odirect":
        storage_records = marker_records(path, "[LOCAL_DISK_READ_METRICS]")
        storage_bw, storage_bytes, storage_seconds = active_bandwidth(
            storage_records, "start_ns", "end_ns"
        )
    else:
        storage_records = marker_records(path, "[CASCADEKV_READ_METRICS]")
        storage_bw, storage_bytes, storage_seconds = active_bandwidth(
            storage_records, "storage_start_ns", "storage_end_ns"
        )
    native_xds_io = summarize_native_xds_io(path) if backend != "disk_odirect" else None
    native_request_bandwidths = (
        native_xds_io.get("per_request_gib_s_values", [])
        if native_xds_io is not None
        else []
    )
    e2e_to_native = [
        e2e_value / native_value * 100
        for e2e_value, native_value in zip(e2e, native_request_bandwidths, strict=False)
        if native_value > 0
    ]
    return {
        "backend": backend,
        "cold_wall_seconds": float(prefix["cold_wall_seconds"]),
        "cost_time_seconds": float(prefix["hit_wall_seconds"]),
        "hot_requests": len(hits),
        # output_len=1 leaves the generated token outside the restored prefix,
        # so vLLM reports 39,999 cached tokens for a complete 40K lookup.
        "full_40k_prefix_hits": sum(tokens >= 39999 for tokens in cached_tokens),
        "cached_tokens": describe([float(value) for value in cached_tokens]),
        "prefill_latency_ms": describe(latencies),
        "ssd_to_gpu_gib_s": describe(e2e),
        "ssd_to_gpu_vs_native_pct": describe(e2e_to_native),
        # This application-visible interval includes rank arrival skew, CPU staging
        # allocation, descriptor construction, and xDS I/O. Keep it distinct from
        # the exact native xDS device-I/O measurement below.
        "storage_path_active_gib_s": storage_bw,
        "storage_active_gib_s": storage_bw,
        "storage_read_gib": storage_bytes / GIB,
        "storage_active_seconds": storage_seconds,
        "native_xds_io": native_xds_io,
        "cycle_prefill_latency_ms": cycle_latencies,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir
    num_prompts = 8
    results = {
        "num_unique_prompts": num_prompts,
        "cycles": 20,
        "prefix_caching": True,
        "gpu_memory_utilization": 0.27,
        "backends": [
            summarize(result_dir / "disk-odirect.log", "disk_odirect", num_prompts),
            summarize(result_dir / "cascadekv.log", "cascadekv", num_prompts),
        ],
    }
    (result_dir / "summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
