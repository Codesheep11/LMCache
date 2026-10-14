# SPDX-License-Identifier: Apache-2.0
"""Summarize the TP4 variable-length shared-prefix CascadeKV workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from analyze_bandwidth_compare import (
    GIB,
    marker_records,
    merged_interval_seconds,
    percentile,
)
from analyze_prefix_swap import describe


BYTES_PER_TOKEN_PER_TP4_RANK = 24 * 1024


def aggregate_request_bandwidth(
    records: list[dict], min_tokens: int, seconds_key: str = "seconds"
) -> list[float]:
    by_worker: dict[int, list[dict]] = {}
    for record in records:
        if int(record.get("retrieved_tokens", 0)) >= min_tokens:
            by_worker.setdefault(int(record["worker_id"]), []).append(record)
    request_count = min((len(rows) for rows in by_worker.values()), default=0)
    values = []
    for index in range(request_count):
        rows = [worker_rows[index] for worker_rows in by_worker.values()]
        if not all(seconds_key in row for row in rows):
            continue
        seconds = max(float(row[seconds_key]) for row in rows)
        if seconds > 0:
            values.append(sum(int(row["bytes"]) for row in rows) / GIB / seconds)
    return values


def native_request_bandwidth(records: list[dict], minimum_bytes: int) -> list[float]:
    by_worker: dict[int, list[dict]] = {}
    for record in records:
        if int(record.get("bytes", 0)) >= minimum_bytes:
            by_worker.setdefault(int(record["worker_id"]), []).append(record)
    request_count = min((len(rows) for rows in by_worker.values()), default=0)
    values = []
    for index in range(request_count):
        rows = [worker_rows[index] for worker_rows in by_worker.values()]
        seconds = (
            max(int(row["io_end_ns"]) for row in rows)
            - min(int(row["io_start_ns"]) for row in rows)
        ) / 1e9
        if seconds > 0:
            values.append(sum(int(row["bytes"]) for row in rows) / GIB / seconds)
    return values


def describe_bandwidth(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p05": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def filtered_active_bandwidth(
    records: list[dict], start_key: str, end_key: str, minimum_bytes: int
) -> float:
    selected = [row for row in records if int(row.get("bytes", 0)) >= minimum_bytes]
    seconds = merged_interval_seconds(selected, start_key, end_key)
    return sum(int(row["bytes"]) for row in selected) / GIB / seconds if seconds else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--num-prompts", type=int, default=100)
    args = parser.parse_args()

    prefix = marker_records(args.log, "[PREFIX_BENCH_METRICS]")[-1]
    first_cycle = prefix["requests"][: args.num_prompts]
    second_cycle = prefix["requests"][args.num_prompts :]
    exact_hits = [
        int(row["num_cached_tokens"]) >= int(row["prompt_tokens"]) - 1
        for row in second_cycle
    ]
    recomputed_tail_tokens = [
        max(
            0,
            int(row["prompt_tokens"]) - 1 - int(row["num_cached_tokens"]),
        )
        for row in second_cycle
    ]
    chunk_complete_hits = [tail < 256 for tail in recomputed_tail_tokens]
    partial_hits = [int(row["num_cached_tokens"]) for row in first_cycle]

    minimum_bytes = 30000 * BYTES_PER_TOKEN_PER_TP4_RANK
    native = marker_records(args.log, "[CASCADEKV_NATIVE_WINDOW_METRICS]")
    xds_direct = marker_records(args.log, "[XDS_DIRECT_READ_METRICS]")
    xds_layerwise = marker_records(args.log, "[XDS_LAYERWISE_READ_METRICS]")
    if not native:
        native = xds_direct if xds_direct else xds_layerwise
    native_full = [row for row in native if int(row["bytes"]) >= minimum_bytes]
    native_by_worker: dict[int, list[dict]] = {}
    for row in native_full:
        native_by_worker.setdefault(int(row["worker_id"]), []).append(row)
    native_aggregate = sum(
        sum(int(row["bytes"]) for row in rows)
        / GIB
        / sum(float(row["io_seconds"]) for row in rows)
        for rows in native_by_worker.values()
    )

    app = marker_records(args.log, "[CASCADEKV_READ_METRICS]")
    xds_app = xds_direct if xds_direct else xds_layerwise
    e2e = marker_records(args.log, "[LMCACHE_SSD_GPU_METRICS]")
    native_per_request = native_request_bandwidth(native_full, minimum_bytes)
    e2e_per_request = aggregate_request_bandwidth(e2e, min_tokens=30000)
    gpu_ready_per_request = aggregate_request_bandwidth(
        e2e, min_tokens=30000, seconds_key="gpu_ready_seconds"
    )
    # Native xDS returns from its direct hook only after each staging blob has
    # been scattered into paged KV and the transfer event has synchronized.
    # It therefore has no separate gpu_ready timestamp: the ordinary direct
    # endpoint already is paged-KV ready.
    if not gpu_ready_per_request and xds_direct:
        gpu_ready_per_request = e2e_per_request
    e2e_to_native = [
        e2e_value / native_value * 100
        for e2e_value, native_value in zip(
            e2e_per_request, native_per_request, strict=True
        )
    ]
    result = {
        "workload": {
            "num_unique_prompts": args.num_prompts,
            "repeat_count": int(prefix["repeat_count"]),
            "prefix_caching": False,
            "shared_prefix_tokens_requested": 5000,
            "prompt_tokens": describe(
                [float(row["prompt_tokens"]) for row in first_cycle]
            ),
        },
        "first_cycle": {
            "wall_seconds": float(prefix["cold_wall_seconds"]),
            "latency_ms": describe(
                [float(row["inference_ms"]) for row in first_cycle]
            ),
            "cached_tokens": describe([float(value) for value in partial_hits]),
            "shared_prefix_hits_after_first": sum(
                value >= 4864 for value in partial_hits[1:]
            ),
        },
        "second_cycle": {
            "cost_time_seconds": float(prefix["hit_wall_seconds"]),
            "latency_ms": describe(
                [float(row["inference_ms"]) for row in second_cycle]
            ),
            "exact_full_hits": sum(exact_hits),
            "complete_chunk_hits": sum(chunk_complete_hits),
            "requests": len(second_cycle),
            "recomputed_tail_tokens": describe(
                [float(value) for value in recomputed_tail_tokens]
            ),
            "cached_tokens": describe(
                [float(row["num_cached_tokens"]) for row in second_cycle]
            ),
        },
        "bandwidth": {
            "native_aggregate_gib_s": native_aggregate,
            "native_active_gib_s": filtered_active_bandwidth(
                native_full, "io_start_ns", "io_end_ns", minimum_bytes
            ),
            "application_visible_ssd_gib_s": filtered_active_bandwidth(
                app if app else xds_app,
                "storage_start_ns" if app else "transfer_start_ns",
                "storage_end_ns" if app else "transfer_end_ns",
                minimum_bytes,
            ),
            "native_per_request_gib_s": describe_bandwidth(native_per_request),
            "ssd_to_hbm_gib_s": describe_bandwidth(e2e_per_request),
            # Retain the legacy field name for existing result consumers. New
            # records declare endpoint=h2d_copy_complete explicitly.
            "ssd_to_gpu_gib_s": describe_bandwidth(e2e_per_request),
            "ssd_to_gpu_vs_native_pct": describe_bandwidth(e2e_to_native),
            "ssd_to_paged_kv_gib_s": describe_bandwidth(gpu_ready_per_request),
            "native_full_read_records": len(native_full),
        },
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    (args.log.parent / "shared-prefix-summary.json").write_text(output + "\n")


if __name__ == "__main__":
    main()
