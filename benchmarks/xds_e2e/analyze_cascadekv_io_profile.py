#!/usr/bin/env python3
"""Summarize CascadeKV per-I/O, xDS service, and layer H2D profiles."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


IO_MARKER = "[CASCADEKV_IO_PROFILE] "
TIMELINE_MARKER = "[LMCACHE_LAYERWISE_TIMELINE] "
DIRECT_PIPELINE_MARKER = "[CASCADEKV_DIRECT_PIPELINE_PROFILE] "
DIRECT_READ_MARKER = "[CASCADEKV_DIRECT_READ_METRICS] "


def marker_records(path: Path, marker: str) -> list[dict[str, Any]]:
    records = []
    decoder = json.JSONDecoder()
    for line in path.read_text(errors="replace").splitlines():
        position = line.find(marker)
        if position < 0:
            continue
        # LMCache appends an ANSI-formatted ``(file.py:line:function)`` suffix
        # after the message. Decode only the leading JSON object.
        record, _end = decoder.raw_decode(line[position + len(marker) :])
        records.append(record)
    return records


def percentile(values: Iterable[float], percent: int) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = math.ceil((len(ordered) - 1) * percent / 100)
    return ordered[index]


def aggregate_io_metric(
    records: list[dict[str, Any]], metric: str
) -> dict[str, float | int]:
    summaries = [record.get(metric, {}) for record in records]
    summaries = [item for item in summaries if int(item.get("count", 0)) > 0]
    count = sum(int(item["count"]) for item in summaries)
    if count == 0:
        return {"samples": 0}
    return {
        "samples": count,
        "weighted_mean_us": sum(
            float(item["mean"]) * int(item["count"]) for item in summaries
        )
        / count,
        "median_task_p50_us": statistics.median(
            float(item["p50"]) for item in summaries
        ),
        "median_task_p95_us": statistics.median(
            float(item["p95"]) for item in summaries
        ),
        "p95_task_p95_us": percentile((float(item["p95"]) for item in summaries), 95),
        "worst_us": max(float(item["max"]) for item in summaries),
    }


def parse_service_profile(path: Path | None) -> dict[str, float | int]:
    if path is None or not path.exists():
        return {}
    result: dict[str, float | int] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "service_sample_profile " not in line:
            continue
        payload = line[line.index("service_sample_profile ") :]
        for key, raw_value in re.findall(r"([a-zA-Z0-9_]+)=([^ ]+)", payload):
            try:
                result[key] = float(raw_value) if "." in raw_value else int(raw_value)
            except ValueError:
                continue
    return result


def layer_samples(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {
        "ssd_ready_to_h2d_submit_us": [],
        "future_wait_us": [],
        "h2d_host_submit_us": [],
        "h2d_cuda_us": [],
        "h2d_sync_us": [],
        "staging_release_us": [],
    }
    for record in records:
        for layer in record.get("layers", []):
            pairs = {
                "ssd_ready_to_h2d_submit_us": (
                    "ssd_ready_ns",
                    "h2d_submit_start_ns",
                ),
                "future_wait_us": ("future_wait_start_ns", "future_wait_end_ns"),
                "h2d_host_submit_us": (
                    "h2d_submit_start_ns",
                    "h2d_submit_end_ns",
                ),
                "h2d_sync_us": ("h2d_sync_start_ns", "h2d_complete_ns"),
                "staging_release_us": (
                    "staging_release_start_ns",
                    "staging_release_end_ns",
                ),
            }
            for name, (start, end) in pairs.items():
                if start in layer and end in layer:
                    samples[name].append((int(layer[end]) - int(layer[start])) / 1e3)
            if "h2d_cuda_ms" in layer:
                samples["h2d_cuda_us"].append(float(layer["h2d_cuda_ms"]) * 1e3)
    return samples


def request_pipeline_samples(
    records: list[dict[str, Any]],
) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {
        "request_total_us": [],
        "storage_producer_us": [],
        "ssd_ready_span_us": [],
        "first_ssd_to_first_h2d_us": [],
        "h2d_pipeline_us": [],
    }
    for record in records:
        layers = record.get("layers", [])
        pairs = {
            "request_total_us": ("request_start_ns", "request_end_ns", record),
            "storage_producer_us": (
                "producer_start_ns",
                "producer_end_ns",
                record,
            ),
        }
        for name, (start, end, source) in pairs.items():
            if start in source and end in source:
                samples[name].append((int(source[end]) - int(source[start])) / 1e3)
        if not layers:
            continue
        first, last = layers[0], layers[-1]
        if "ssd_ready_ns" in first and "ssd_ready_ns" in last:
            samples["ssd_ready_span_us"].append(
                (int(last["ssd_ready_ns"]) - int(first["ssd_ready_ns"])) / 1e3
            )
        if "ssd_ready_ns" in first and "h2d_submit_start_ns" in first:
            samples["first_ssd_to_first_h2d_us"].append(
                (int(first["h2d_submit_start_ns"]) - int(first["ssd_ready_ns"])) / 1e3
            )
        if "h2d_submit_start_ns" in first and "pipeline_h2d_complete_ns" in record:
            samples["h2d_pipeline_us"].append(
                (
                    int(record["pipeline_h2d_complete_ns"])
                    - int(first["h2d_submit_start_ns"])
                )
                / 1e3
            )
    return samples


def plain_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"samples": 0}
    return {
        "samples": len(values),
        "mean_us": statistics.fmean(values),
        "p50_us": percentile(values, 50),
        "p95_us": percentile(values, 95),
        "p99_us": percentile(values, 99),
        "max_us": max(values),
    }


def fmt(value: Any) -> str:
    if value is None or isinstance(value, float) and math.isnan(value):
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.2f}"


def build_report(
    io_records: list[dict[str, Any]],
    timelines: list[dict[str, Any]],
    service: dict[str, float | int],
    direct_pipeline_records: list[dict[str, Any]] | None = None,
    direct_read_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    direct_pipeline_records = direct_pipeline_records or []
    direct_read_records = direct_read_records or []
    metric_labels = {
        "engine_queue_us": "引擎排队：task accepted → native submit",
        "native_submit_us": "Python/native submit 调用",
        "submit_to_python_us": "native submit 完成 → Python 收到 completion",
        "submit_to_service_publish_us": "submit → 服务端发布 completion（采样）",
        "completion_ring_us": "completion ring 等待（采样）",
        "dequeue_to_python_us": "native dequeue → Python poll 返回（采样）",
        "layer_callback_us": "layer-ready callback",
    }
    operations: dict[str, Any] = {}
    lines = ["# CascadeKV per-I/O profiling", ""]
    for operation in ("read", "write"):
        selected = [item for item in io_records if item.get("operation") == operation]
        if not selected:
            continue
        metrics = {name: aggregate_io_metric(selected, name) for name in metric_labels}
        operations[operation] = {
            "tasks": len(selected),
            "io_requests": sum(int(item["io_count"]) for item in selected),
            "bytes": sum(int(item["bytes"]) for item in selected),
            "task_ms": plain_summary(
                [float(item["task_ms"]) * 1e3 for item in selected]
            ),
            "metrics": metrics,
        }
        lines.extend(
            [
                f"## {operation}",
                "",
                f"tasks={len(selected)}, I/O={operations[operation]['io_requests']}, "
                f"bytes={operations[operation]['bytes']}",
                "",
                "| 阶段 | 样本数 | 加权均值 (us) | task中位p50 | "
                "task中位p95 | task-p95的p95 | 最大值 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, label in metric_labels.items():
            item = metrics[name]
            lines.append(
                f"| {label} | {fmt(item.get('samples'))} | "
                f"{fmt(item.get('weighted_mean_us'))} | "
                f"{fmt(item.get('median_task_p50_us'))} | "
                f"{fmt(item.get('median_task_p95_us'))} | "
                f"{fmt(item.get('p95_task_p95_us'))} | "
                f"{fmt(item.get('worst_us'))} |"
            )
        lines.append("")

    layer = {
        name: plain_summary(values) for name, values in layer_samples(timelines).items()
    }
    request_pipeline = {
        name: plain_summary(values)
        for name, values in request_pipeline_samples(timelines).items()
    }
    if timelines:
        lines.extend(
            [
                "## Per-request pipeline",
                "",
                "| 阶段 | 样本数 | 均值 (ms) | p50 | p95 | p99 | 最大值 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, item in request_pipeline.items():
            lines.append(
                f"| {name} | {fmt(item.get('samples'))} | "
                f"{fmt(float(item.get('mean_us', math.nan)) / 1e3)} | "
                f"{fmt(float(item.get('p50_us', math.nan)) / 1e3)} | "
                f"{fmt(float(item.get('p95_us', math.nan)) / 1e3)} | "
                f"{fmt(float(item.get('p99_us', math.nan)) / 1e3)} | "
                f"{fmt(float(item.get('max_us', math.nan)) / 1e3)} |"
            )
        lines.append("")
    if timelines:
        lines.extend(
            [
                "## SSD → H2D layer pipeline",
                "",
                "| 阶段 | 样本数 | 均值 (us) | p50 | p95 | p99 | 最大值 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, item in layer.items():
            lines.append(
                f"| {name} | {fmt(item.get('samples'))} | {fmt(item.get('mean_us'))} | "
                f"{fmt(item.get('p50_us'))} | {fmt(item.get('p95_us'))} | "
                f"{fmt(item.get('p99_us'))} | {fmt(item.get('max_us'))} |"
            )
        lines.append("")

    direct_metric_labels = {
        "ready_get_us": "请求线程等待 ready queue",
        "ssd_completion_gap_us": "相邻 SSD completion 间隔",
        "ready_to_h2d_us": "SSD ready → H2D submit",
        "storage_release_us": "SSD reader / Block Buffer owner 释放",
        "h2d_host_submit_us": "to_gpu 主机提交",
        "h2d_host_gap_us": "相邻 H2D 主机提交间隔",
        "event_record_us": "CUDA completion event record",
        "event_query_us": "CUDA completion event query",
        "event_sync_us": "最终 CUDA event synchronize",
        "event_timing_read_us": "CUDA timing event 读取",
        "h2d_cuda_us": "单 block H2D CUDA（采样）",
        "memory_release_us": "H2D 后 MemoryObj 释放",
    }
    direct_pipeline = {
        name: aggregate_io_metric(direct_pipeline_records, name)
        for name in direct_metric_labels
    }
    direct_pipeline["transfer_ms"] = plain_summary(
        [float(item["transfer_ms"]) * 1e3 for item in direct_pipeline_records]
    )
    direct_pipeline["final_drain_us"] = plain_summary(
        [float(item["final_drain_us"]) for item in direct_pipeline_records]
    )
    if direct_pipeline_records:
        lines.extend(
            [
                "## Direct block pipeline",
                "",
                f"requests={len(direct_pipeline_records)}, "
                "blocks="
                f"{sum(int(item['blocks']) for item in direct_pipeline_records)}, "
                f"CUDA sample shift="
                f"{direct_pipeline_records[0].get('cuda_sample_shift')}",
                "",
                "| 阶段 | 样本数 | 加权均值 (us) | request中位p50 | "
                "request中位p95 | request-p95的p95 | 最大值 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, label in direct_metric_labels.items():
            item = direct_pipeline[name]
            lines.append(
                f"| {label} | {fmt(item.get('samples'))} | "
                f"{fmt(item.get('weighted_mean_us'))} | "
                f"{fmt(item.get('median_task_p50_us'))} | "
                f"{fmt(item.get('median_task_p95_us'))} | "
                f"{fmt(item.get('p95_task_p95_us'))} | "
                f"{fmt(item.get('worst_us'))} |"
            )
        lines.extend(
            [
                "",
                "| 请求级阶段 | 样本数 | 均值 (ms) | p50 | p95 | p99 | 最大值 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, label in (
            ("transfer_ms", "完整 direct transfer"),
            ("final_drain_us", "最终 H2D drain"),
        ):
            item = direct_pipeline[name]
            lines.append(
                f"| {label} | {fmt(item.get('samples'))} | "
                f"{fmt(float(item.get('mean_us', math.nan)) / 1e3)} | "
                f"{fmt(float(item.get('p50_us', math.nan)) / 1e3)} | "
                f"{fmt(float(item.get('p95_us', math.nan)) / 1e3)} | "
                f"{fmt(float(item.get('p99_us', math.nan)) / 1e3)} | "
                f"{fmt(float(item.get('max_us', math.nan)) / 1e3)} |"
            )
        lines.append("")

    direct_read_labels = {
        "storage_release_us": "SSD reader / Block Buffer 释放",
        "h2d_host_submit_us": "H2D engine to_gpu 主机提交",
        "event_record_us": "CUDA batch event record",
        "event_sync_us": "CUDA batch event synchronize",
        "memory_release_us": "H2D后单block MemoryObj释放",
    }
    direct_read = {
        name: aggregate_io_metric(direct_read_records, name)
        for name in direct_read_labels
    }
    if direct_read_records:
        direct_read_blocks = sum(
            int(item.get("block_count", 0)) for item in direct_read_records
        )
        lines.extend(
            [
                "## Direct-read pipeline stages",
                "",
                f"requests={len(direct_read_records)}, blocks={direct_read_blocks}",
                "",
                "| 阶段 | 样本数 | 加权均值 (us) | request中位p50 | "
                "request中位p95 | request-p95的p95 | 最大值 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, label in direct_read_labels.items():
            item = direct_read[name]
            lines.append(
                f"| {label} | {fmt(item.get('samples'))} | "
                f"{fmt(item.get('weighted_mean_us'))} | "
                f"{fmt(item.get('median_task_p50_us'))} | "
                f"{fmt(item.get('median_task_p95_us'))} | "
                f"{fmt(item.get('p95_task_p95_us'))} | "
                f"{fmt(item.get('worst_us'))} |"
            )
        lines.append("")

    latest_by_worker: dict[int, dict[str, Any]] = {}
    for record in direct_read_records:
        worker = int(record.get("worker_id", -1))
        if float(record.get("engine_profile_read_active_ms", 0)) >= float(
            latest_by_worker.get(worker, {}).get("engine_profile_read_active_ms", -1)
        ):
            latest_by_worker[worker] = record
    engine_records = list(latest_by_worker.values())
    read_active_ms = sum(
        float(item.get("engine_profile_read_active_ms", 0)) for item in engine_records
    )
    engine_profile: dict[str, Any] = {
        "workers": len(engine_records),
        "read_active_ms": read_active_ms,
    }
    for name in (
        "engine_read_avg_qd",
        "engine_read_qd0_pct",
        "engine_read_qd_lt4_pct",
        "engine_read_qd_lt8_pct",
        "engine_read_qd_lt16_pct",
        "engine_read_qd_lt32_pct",
    ):
        engine_profile[name] = (
            sum(
                float(item.get(name, 0))
                * float(item.get("engine_profile_read_active_ms", 0))
                for item in engine_records
            )
            / read_active_ms
            if read_active_ms
            else math.nan
        )
    engine_timing_names = (
        "read_engine_queue",
        "poll_call",
        "completion_batch_processing",
        "read_descriptor_callback",
        "read_native_submit",
        "read_completion_to_refill",
    )
    engine_profile["timings"] = {}
    for name in engine_timing_names:
        count = sum(int(item.get(f"engine_{name}_count", 0)) for item in engine_records)
        total_ms = sum(
            float(item.get(f"engine_{name}_total_ms", 0)) for item in engine_records
        )
        engine_profile["timings"][name] = {
            "count": count,
            "mean_us": total_ms * 1e3 / count if count else math.nan,
            "max_us": max(
                (
                    float(item.get(f"engine_{name}_max_us", 0))
                    for item in engine_records
                ),
                default=math.nan,
            ),
            "total_ms": total_ms,
        }
    engine_profile["ring_full"] = sum(
        int(item.get("engine_ring_full", 0)) for item in engine_records
    )
    engine_profile["poll_empty"] = sum(
        int(item.get("engine_poll_empty", 0)) for item in engine_records
    )
    engine_profile["poll_completions"] = sum(
        int(item.get("engine_poll_completions", 0)) for item in engine_records
    )
    if engine_records:
        average_qd = fmt(engine_profile["engine_read_avg_qd"])
        qd0_pct = fmt(engine_profile["engine_read_qd0_pct"])
        qd_lt4_pct = fmt(engine_profile["engine_read_qd_lt4_pct"])
        qd_lt8_pct = fmt(engine_profile["engine_read_qd_lt8_pct"])
        qd_lt16_pct = fmt(engine_profile["engine_read_qd_lt16_pct"])
        qd_lt32_pct = fmt(engine_profile["engine_read_qd_lt32_pct"])
        engine_header = (
            f"workers={len(engine_records)}, "
            f"read-active={read_active_ms / 1e3:.3f}s, "
            f"ring-full={engine_profile['ring_full']}"
        )
        lines.extend(
            [
                "## StorageIOEngine scheduling",
                "",
                engine_header,
                "",
                "| QD指标 | 值 |",
                "|---|---:|",
                f"| read active期平均QD | {average_qd} |",
                f"| QD=0时间占比 | {qd0_pct}% |",
                f"| QD<4时间占比 | {qd_lt4_pct}% |",
                f"| QD<8时间占比 | {qd_lt8_pct}% |",
                f"| QD<16时间占比 | {qd_lt16_pct}% |",
                f"| QD<32时间占比 | {qd_lt32_pct}% |",
                "",
                "| Engine阶段 | 次数 | 均值 (us) | 最大 (us) | 累计 (ms) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name in engine_timing_names:
            item = engine_profile["timings"][name]
            lines.append(
                f"| {name} | {item['count']} | {fmt(item['mean_us'])} | "
                f"{fmt(item['max_us'])} | {fmt(item['total_ms'])} |"
            )
        lines.append("")

    service_stages = [
        ("request_ring", "request_ring", "request ring"),
        ("request_pre_poll", "request_pre_poll", "service pre-poll"),
        ("request_in_batch", "request_in_batch", "service batch wait"),
        ("service_prepare", "prepare", "service prepare"),
        ("raid_submit", "raid_submit", "RAID submit"),
        ("device", "device", "SSD device"),
        ("completion_publish", "completion_publish", "completion publish"),
    ]
    if service:
        lines.extend(
            [
                "## xDS service sampled stages",
                "",
                "| 阶段 | count | 平均 (us) | p95 (us) | p99 (us) | 最大 (us) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for mean_prefix, tail_prefix, label in service_stages:
            lines.append(
                f"| {label} | {fmt(service.get(mean_prefix + '_count'))} | "
                f"{fmt(float(service.get(mean_prefix + '_avg_ns', math.nan)) / 1e3)} | "
                f"{fmt(float(service.get(tail_prefix + '_p95_ns', math.nan)) / 1e3)} | "
                f"{fmt(float(service.get(tail_prefix + '_p99_ns', math.nan)) / 1e3)} | "
                f"{fmt(float(service.get(tail_prefix + '_max_ns', math.nan)) / 1e3)} |"
            )
        lines.append("")

    result = {
        "io_profile_records": len(io_records),
        "timeline_records": len(timelines),
        "direct_pipeline_records": len(direct_pipeline_records),
        "operations": operations,
        "request_pipeline": request_pipeline,
        "layer_pipeline": layer,
        "direct_pipeline": direct_pipeline,
        "direct_read": direct_read,
        "engine_profile": engine_profile,
        "service_sample_profile": service,
    }
    return result, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--service-log", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result, report = build_report(
        marker_records(args.server_log, IO_MARKER),
        marker_records(args.server_log, TIMELINE_MARKER),
        parse_service_profile(args.service_log),
        marker_records(args.server_log, DIRECT_PIPELINE_MARKER),
        marker_records(args.server_log, DIRECT_READ_MARKER),
    )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "per-io-profile.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        (args.output_dir / "per-io-profile.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
