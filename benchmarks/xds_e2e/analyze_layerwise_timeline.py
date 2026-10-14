#!/usr/bin/env python3
"""Summarize CascadeKV per-layer SSD-to-GPU timeline log records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TIMELINE_MARKER = "[LMCACHE_LAYERWISE_TIMELINE] "
METRICS_MARKER = "[PREFIX_BENCH_METRICS] "
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
JSON_DECODER = json.JSONDecoder()


def _duration_ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[index]


def load_log(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    timelines: list[dict[str, Any]] = []
    benchmark_metrics = None
    for raw_line in path.read_text(errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        if TIMELINE_MARKER in line:
            payload = line.split(TIMELINE_MARKER, 1)[1]
            timelines.append(JSON_DECODER.raw_decode(payload)[0])
        elif METRICS_MARKER in line:
            payload = line.split(METRICS_MARKER, 1)[1]
            benchmark_metrics = JSON_DECODER.raw_decode(payload)[0]
    if not timelines:
        raise ValueError(f"no {TIMELINE_MARKER.strip()} records in {path}")
    return sorted(
        timelines, key=lambda trace: (trace["request_start_ns"], trace["worker_id"])
    ), benchmark_metrics


def select_request_traces(
    traces: list[dict[str, Any]],
    request_index: int | None,
    min_bytes: int,
) -> list[dict[str, Any]]:
    """Select the same chronological request from every TP worker."""
    by_worker: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        trace_bytes = trace.get(
            "bytes", sum(layer.get("bytes", 0) for layer in trace["layers"])
        )
        if trace_bytes < min_bytes:
            continue
        by_worker.setdefault(str(trace["worker_id"]), []).append(trace)
    if not by_worker:
        raise ValueError(f"no timeline records have at least {min_bytes} bytes")
    if request_index is None:
        counts = {worker: len(records) for worker, records in by_worker.items()}
        if any(count != 1 for count in counts.values()):
            raise ValueError(
                "log contains multiple requests per worker; pass --request-index "
                f"after applying --min-bytes (counts={counts})"
            )
        return [records[0] for records in by_worker.values()]
    selected = []
    for worker, records in by_worker.items():
        try:
            selected.append(records[request_index])
        except IndexError as error:
            raise ValueError(
                f"worker {worker} has only {len(records)} matching requests; "
                f"cannot select index {request_index}"
            ) from error
    return selected


def summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    layers = trace["layers"]
    start_ns = trace["request_start_ns"]
    total_bytes = sum(layer["bytes"] for layer in layers)
    xds_layers = [
        layer
        for layer in layers
        if "xds_io_start_ns" in layer and "xds_io_end_ns" in layer
    ]
    xds_io_layer_ms = [
        _duration_ms(layer["xds_io_start_ns"], layer["xds_io_end_ns"])
        for layer in xds_layers
    ]
    xds_staging_acquire_layer_ms = [
        _duration_ms(
            layer["xds_staging_acquire_start_ns"],
            layer["xds_staging_acquire_end_ns"],
        )
        for layer in xds_layers
        if "xds_staging_acquire_start_ns" in layer
        and "xds_staging_acquire_end_ns" in layer
    ]
    xds_descriptor_prepare_layer_ms = [
        _duration_ms(
            layer["xds_staging_acquire_end_ns"], layer["xds_io_start_ns"]
        )
        for layer in xds_layers
        if "xds_staging_acquire_end_ns" in layer
    ]
    xds_inter_layer_gap_ms = sum(
        max(
            0.0,
            _duration_ms(previous["xds_io_end_ns"], current["xds_io_start_ns"]),
        )
        for previous, current in zip(xds_layers, xds_layers[1:], strict=False)
    )
    future_wait_ms = sum(
        _duration_ms(layer["future_wait_start_ns"], layer["future_wait_end_ns"])
        for layer in layers
    )
    h2d_submit_cpu_ms = sum(
        _duration_ms(layer["h2d_submit_start_ns"], layer["h2d_submit_end_ns"])
        for layer in layers
    )
    inter_layer_submit_gap_ms = sum(
        _duration_ms(
            previous["h2d_submit_end_ns"], current["h2d_submit_start_ns"]
        )
        for previous, current in zip(layers, layers[1:], strict=False)
    )
    h2d_ready_ns = trace.get(
        "pipeline_h2d_complete_ns", layers[-1]["h2d_complete_ns"]
    )
    paged_ready_ns = trace.get(
        "pipeline_gpu_ready_complete_ns", layers[-1]["h2d_complete_ns"]
    )
    io_ms = (
        _duration_ms(trace["io_start_ns"], trace["io_end_ns"])
        if "io_start_ns" in trace and "io_end_ns" in trace
        else None
    )
    h2d_ready_ms = _duration_ms(start_ns, h2d_ready_ns)
    paged_ready_ms = _duration_ms(start_ns, paged_ready_ns)
    callback_sync_wait_ms = sum(
        _duration_ms(layer["h2d_sync_start_ns"], layer["h2d_complete_ns"])
        for layer in layers
    )
    between_layer_compute_ms = sum(
        _duration_ms(
            previous["h2d_complete_ns"], current["h2d_sync_start_ns"]
        )
        for previous, current in zip(layers, layers[1:], strict=False)
    )
    return {
        "worker_id": trace["worker_id"],
        "bytes": total_bytes,
        "gib": total_bytes / 2**30,
        "request_ms": _duration_ms(start_ns, trace["request_end_ns"]),
        "key_scan_ms": (
            _duration_ms(trace["key_scan_start_ns"], trace["key_scan_end_ns"])
            if "key_scan_start_ns" in trace and "key_scan_end_ns" in trace
            else None
        ),
        "layer_key_build_ms": (
            _duration_ms(
                trace["layer_key_build_start_ns"],
                trace["layer_key_build_end_ns"],
            )
            if "layer_key_build_start_ns" in trace
            and "layer_key_build_end_ns" in trace
            else None
        ),
        "producer_start_ms": _duration_ms(start_ns, trace["producer_start_ns"]),
        "producer_ms": _duration_ms(
            trace["producer_start_ns"], trace["producer_end_ns"]
        ),
        "gpu_connector_setup_ms": (
            _duration_ms(
                trace["gpu_connector_setup_start_ns"],
                trace["gpu_connector_setup_end_ns"],
            )
            if "gpu_connector_setup_start_ns" in trace
            and "gpu_connector_setup_end_ns" in trace
            else None
        ),
        "slot_mapping_build_ms": (
            _duration_ms(
                trace["slot_mapping_build_start_ns"],
                trace["slot_mapping_build_end_ns"],
            )
            if "slot_mapping_build_start_ns" in trace
            and "slot_mapping_build_end_ns" in trace
            else None
        ),
        "gpu_staging_allocate_ms": (
            _duration_ms(
                trace["gpu_staging_allocate_start_ns"],
                trace["gpu_staging_allocate_end_ns"],
            )
            if "gpu_staging_allocate_start_ns" in trace
            and "gpu_staging_allocate_end_ns" in trace
            else None
        ),
        "xds_layer_count": len(xds_layers),
        "xds_io_span_ms": (
            _duration_ms(
                xds_layers[0]["xds_io_start_ns"],
                xds_layers[-1]["xds_io_end_ns"],
            )
            if xds_layers
            else None
        ),
        "xds_io_sum_ms": sum(xds_io_layer_ms) if xds_io_layer_ms else None,
        "xds_io_mean_ms": (
            sum(xds_io_layer_ms) / len(xds_io_layer_ms)
            if xds_io_layer_ms
            else None
        ),
        "xds_io_p50_ms": _percentile(xds_io_layer_ms, 0.50),
        "xds_io_p95_ms": _percentile(xds_io_layer_ms, 0.95),
        "xds_io_max_ms": max(xds_io_layer_ms) if xds_io_layer_ms else None,
        "xds_staging_acquire_sum_ms": (
            sum(xds_staging_acquire_layer_ms)
            if xds_staging_acquire_layer_ms
            else None
        ),
        "xds_staging_acquire_mean_ms": (
            sum(xds_staging_acquire_layer_ms)
            / len(xds_staging_acquire_layer_ms)
            if xds_staging_acquire_layer_ms
            else None
        ),
        "xds_staging_acquire_p95_ms": _percentile(
            xds_staging_acquire_layer_ms, 0.95
        ),
        "xds_descriptor_prepare_sum_ms": (
            sum(xds_descriptor_prepare_layer_ms)
            if xds_descriptor_prepare_layer_ms
            else None
        ),
        "xds_descriptor_prepare_mean_ms": (
            sum(xds_descriptor_prepare_layer_ms)
            / len(xds_descriptor_prepare_layer_ms)
            if xds_descriptor_prepare_layer_ms
            else None
        ),
        "xds_inter_layer_idle_gap_ms": xds_inter_layer_gap_ms,
        "metadata_lookup_ms": (
            _duration_ms(
                trace["metadata_lookup_start_ns"],
                trace["metadata_lookup_end_ns"],
            )
            if "metadata_lookup_start_ns" in trace
            and "metadata_lookup_end_ns" in trace
            else None
        ),
        "storage_ms": (
            _duration_ms(trace["storage_start_ns"], trace["storage_end_ns"])
            if "storage_start_ns" in trace and "storage_end_ns" in trace
            else None
        ),
        "allocation_ms": (
            _duration_ms(trace["allocation_start_ns"], trace["allocation_end_ns"])
            if "allocation_start_ns" in trace and "allocation_end_ns" in trace
            else None
        ),
        "staging_acquire_ms": (
            _duration_ms(
                trace["staging_acquire_start_ns"], trace["staging_acquire_end_ns"]
            )
            if "staging_acquire_start_ns" in trace
            and "staging_acquire_end_ns" in trace
            else None
        ),
        "descriptor_build_ms": (
            _duration_ms(
                trace["descriptor_build_start_ns"], trace["descriptor_build_end_ns"]
            )
            if "descriptor_build_start_ns" in trace
            and "descriptor_build_end_ns" in trace
            else None
        ),
        "descriptor_template_cache_hit": trace.get(
            "descriptor_template_cache_hit"
        ),
        "io_ms": io_ms,
        "storage_pre_io_ms": (
            _duration_ms(trace["storage_start_ns"], trace["io_start_ns"])
            if "storage_start_ns" in trace and "io_start_ns" in trace
            else None
        ),
        "storage_post_io_ms": (
            _duration_ms(trace["io_end_ns"], trace["storage_end_ns"])
            if "io_end_ns" in trace and "storage_end_ns" in trace
            else None
        ),
        "setup_to_io_ms": (
            _duration_ms(start_ns, trace["io_start_ns"])
            if "io_start_ns" in trace
            else None
        ),
        "io_to_gpu_tail_ms": (
            _duration_ms(trace["io_end_ns"], h2d_ready_ns)
            if "io_end_ns" in trace
            else None
        ),
        "h2d_ready_ms": h2d_ready_ms,
        "paged_kv_ready_ms": paged_ready_ms,
        "h2d_to_paged_kv_tail_ms": _duration_ms(h2d_ready_ns, paged_ready_ns),
        "h2d_ready_to_io_pct": (
            io_ms / h2d_ready_ms * 100 if io_ms is not None else None
        ),
        "first_ssd_ready_ms": _duration_ms(start_ns, layers[0]["ssd_ready_ns"]),
        "first_gpu_ready_ms": _duration_ms(
            start_ns, layers[0]["h2d_complete_ns"]
        ),
        "last_ssd_ready_ms": _duration_ms(start_ns, layers[-1]["ssd_ready_ns"]),
        "last_h2d_complete_ms": _duration_ms(
            start_ns, layers[-1]["h2d_complete_ns"]
        ),
        "pipeline_h2d_complete_ms": h2d_ready_ms,
        "pipeline_gpu_ready_complete_ms": paged_ready_ms,
        "data_ready_to_request_end_ms": _duration_ms(
            h2d_ready_ns, trace["request_end_ns"]
        ),
        "callback_sync_wait_ms": callback_sync_wait_ms,
        "callback_sync_wait_max_ms": max(
            _duration_ms(
                layer["h2d_sync_start_ns"], layer["h2d_complete_ns"]
            )
            for layer in layers
        ),
        "callback_wait_layers_over_0_1ms": sum(
            _duration_ms(
                layer["h2d_sync_start_ns"], layer["h2d_complete_ns"]
            )
            > 0.1
            for layer in layers
        ),
        "between_layer_compute_ms": between_layer_compute_ms,
        "between_layer_compute_mean_ms": between_layer_compute_ms
        / max(len(layers) - 1, 1),
        "future_wait_ms": future_wait_ms,
        "layer0_future_wait_ms": _duration_ms(
            layers[0]["future_wait_start_ns"], layers[0]["future_wait_end_ns"]
        ),
        "later_future_wait_ms": future_wait_ms
        - _duration_ms(
            layers[0]["future_wait_start_ns"], layers[0]["future_wait_end_ns"]
        ),
        "h2d_submit_cpu_ms": h2d_submit_cpu_ms,
        "host_coalesce_ms": sum(
            _duration_ms(layer["host_coalesce_start_ns"], layer["host_coalesce_end_ns"])
            for layer in layers
            if "host_coalesce_start_ns" in layer
            and "host_coalesce_end_ns" in layer
        ),
        "host_coalesced_run_count": sum(
            int(layer.get("host_coalesced_run_count", 0)) for layer in layers
        ),
        "copy_enqueue_cpu_ms": sum(
            _duration_ms(layer["copy_enqueue_start_ns"], layer["copy_enqueue_end_ns"])
            for layer in layers
            if "copy_enqueue_start_ns" in layer and "copy_enqueue_end_ns" in layer
        ),
        "scatter_enqueue_cpu_ms": sum(
            _duration_ms(
                layer["scatter_enqueue_start_ns"], layer["scatter_enqueue_end_ns"]
            )
            for layer in layers
            if "scatter_enqueue_start_ns" in layer
            and "scatter_enqueue_end_ns" in layer
        ),
        "future_publish_ms": sum(
            _duration_ms(layer["ssd_ready_ns"], layer["future_publish_complete_ns"])
            for layer in layers
            if "future_publish_complete_ns" in layer
        ),
        "h2d_pacing_wait_ms": sum(
            _duration_ms(layer["pacing_wait_start_ns"], layer["pacing_wait_end_ns"])
            for layer in layers
            if "pacing_wait_start_ns" in layer and "pacing_wait_end_ns" in layer
        ),
        "h2d_pacing_blocked_layers_over_0_1ms": sum(
            _duration_ms(layer["pacing_wait_start_ns"], layer["pacing_wait_end_ns"])
            > 0.1
            for layer in layers
            if "pacing_wait_start_ns" in layer and "pacing_wait_end_ns" in layer
        ),
        "consumer_submit_wait_ms": sum(
            _duration_ms(
                layer["consumer_submit_wait_start_ns"],
                layer["consumer_submit_wait_end_ns"],
            )
            for layer in layers
            if "consumer_submit_wait_start_ns" in layer
            and "consumer_submit_wait_end_ns" in layer
        ),
        "h2d_cuda_sum_ms": sum(layer["h2d_cuda_ms"] for layer in layers),
        "h2d_cuda_mean_ms": sum(layer["h2d_cuda_ms"] for layer in layers)
        / len(layers),
        "gpu_copy_cuda_sum_ms": sum(
            layer.get("gpu_copy_cuda_ms", 0.0) for layer in layers
        ),
        "gpu_scatter_cuda_sum_ms": sum(
            layer.get("gpu_scatter_cuda_ms", 0.0) for layer in layers
        ),
        "inter_layer_submit_gap_ms": inter_layer_submit_gap_ms,
        "mean_ssd_ready_to_submit_ms": sum(
            _duration_ms(layer["ssd_ready_ns"], layer["h2d_submit_start_ns"])
            for layer in layers
        )
        / len(layers),
        "last_ssd_ready_to_submit_ms": _duration_ms(
            layers[-1]["ssd_ready_ns"], layers[-1]["h2d_submit_start_ns"]
        ),
        "max_ssd_ready_to_submit_ms": max(
            _duration_ms(layer["ssd_ready_ns"], layer["h2d_submit_start_ns"])
            for layer in layers
        ),
        "first_progress_callback_ms": (
            _duration_ms(start_ns, trace["progress_callbacks"][0]["start_ns"])
            if trace.get("progress_callbacks")
            else None
        ),
        "last_progress_callback_ms": (
            _duration_ms(start_ns, trace["progress_callbacks"][-1]["end_ns"])
            if trace.get("progress_callbacks")
            else None
        ),
        "progress_callback_ms": sum(
            _duration_ms(callback["start_ns"], callback["end_ns"])
            for callback in trace.get("progress_callbacks", [])
        ),
    }


def summarize(
    path: Path, request_index: int | None = None, min_bytes: int = 0
) -> dict[str, Any]:
    all_traces, benchmark_metrics = load_log(path)
    traces = select_request_traces(all_traces, request_index, min_bytes)
    workers = [summarize_trace(trace) for trace in traces]
    total_bytes = sum(worker["bytes"] for worker in workers)
    total_gib = total_bytes / 2**30

    def global_span_ms(start_key: str, end_key: str) -> float:
        return _duration_ms(
            min(trace[start_key] for trace in traces),
            max(trace[end_key] for trace in traces),
        )

    request_ms = global_span_ms("request_start_ns", "request_end_ns")
    complete_storage_traces = [
        trace
        for trace in traces
        if "storage_start_ns" in trace and "storage_end_ns" in trace
    ]
    complete_io_traces = [
        trace for trace in traces if "io_start_ns" in trace and "io_end_ns" in trace
    ]
    storage_ms = (
        _duration_ms(
            min(trace["storage_start_ns"] for trace in complete_storage_traces),
            max(trace["storage_end_ns"] for trace in complete_storage_traces),
        )
        if complete_storage_traces
        else 0.0
    )
    xds_traces = [
        trace
        for trace in traces
        if trace.get("layers")
        and "xds_io_start_ns" in trace["layers"][0]
        and "xds_io_end_ns" in trace["layers"][-1]
    ]
    xds_io_ms = (
        _duration_ms(
            min(trace["layers"][0]["xds_io_start_ns"] for trace in xds_traces),
            max(trace["layers"][-1]["xds_io_end_ns"] for trace in xds_traces),
        )
        if xds_traces
        else 0.0
    )
    io_ms = (
        _duration_ms(
            min(trace["io_start_ns"] for trace in complete_io_traces),
            max(trace["io_end_ns"] for trace in complete_io_traces),
        )
        if complete_io_traces
        else 0.0
    )
    h2d_ready_ms = _duration_ms(
        min(trace["request_start_ns"] for trace in traces),
        max(
            trace.get(
                "pipeline_h2d_complete_ns",
                trace["layers"][-1]["h2d_complete_ns"],
            )
            for trace in traces
        ),
    )
    paged_ready_ms = _duration_ms(
        min(trace["request_start_ns"] for trace in traces),
        max(
            trace.get(
                "pipeline_gpu_ready_complete_ns",
                trace["layers"][-1]["h2d_complete_ns"],
            )
            for trace in traces
        ),
    )
    aggregate = {
        "bytes": total_bytes,
        "gib": total_gib,
        "request_ms": request_ms,
        "request_gib_s": total_gib / (request_ms / 1000),
        "h2d_ready_ms": h2d_ready_ms,
        "h2d_ready_gib_s": total_gib / (h2d_ready_ms / 1000),
        "paged_kv_ready_ms": paged_ready_ms,
        "paged_kv_ready_gib_s": total_gib / (paged_ready_ms / 1000),
        "storage_ms": storage_ms,
        "storage_gib_s": total_gib / (storage_ms / 1000) if storage_ms else None,
        "io_ms": io_ms,
        "io_gib_s": total_gib / (io_ms / 1000) if io_ms else None,
        "xds_io_ms": xds_io_ms,
        "xds_io_gib_s": total_gib / (xds_io_ms / 1000) if xds_io_ms else None,
        "xds_io_to_h2d_tail_ms": (
            _duration_ms(
                max(
                    trace["layers"][-1]["xds_io_end_ns"]
                    for trace in xds_traces
                ),
                max(
                    trace.get(
                        "pipeline_h2d_complete_ns",
                        trace["layers"][-1]["h2d_complete_ns"],
                    )
                    for trace in traces
                ),
            )
            if xds_traces
            else None
        ),
        "rank_start_skew_ms": _duration_ms(
            min(trace["request_start_ns"] for trace in traces),
            max(trace["request_start_ns"] for trace in traces),
        ),
        "request_to_io_ms": _duration_ms(
            min(trace["request_start_ns"] for trace in traces),
            min(trace["io_start_ns"] for trace in complete_io_traces),
        )
        if complete_io_traces
        else None,
        "io_to_h2d_tail_ms": _duration_ms(
            max(trace["io_end_ns"] for trace in complete_io_traces),
            max(
                trace.get(
                    "pipeline_h2d_complete_ns",
                    trace["layers"][-1]["h2d_complete_ns"],
                )
                for trace in traces
            ),
        )
        if complete_io_traces
        else None,
    }
    aggregate["application_to_native_pct"] = (
        aggregate["storage_gib_s"] / aggregate["io_gib_s"] * 100
        if aggregate["io_gib_s"] is not None
        else None
    )
    aggregate["h2d_ready_to_io_pct"] = (
        aggregate["h2d_ready_gib_s"] / aggregate["io_gib_s"] * 100
        if aggregate["io_gib_s"] is not None
        else None
    )
    aggregate["h2d_ready_to_xds_io_pct"] = (
        aggregate["h2d_ready_gib_s"] / aggregate["xds_io_gib_s"] * 100
        if aggregate["xds_io_gib_s"] is not None
        else None
    )
    return {
        "log": str(path),
        "workers": workers,
        "aggregate": aggregate,
        "benchmark_metrics": benchmark_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--request-index",
        type=int,
        help="zero-based chronological matching request index per TP worker",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=0,
        help="ignore smaller (for example shared-prefix-only) timeline records",
    )
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(args.log, args.request_index, args.min_bytes),
            indent=None if args.compact else 2,
        )
    )


if __name__ == "__main__":
    main()
