#!/usr/bin/env python3
"""Correlate LMCache retrieve stages with client TTFT and decode time."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Callable

REQUEST_PATTERN = re.compile(r"stress-s(?P<session>\d+)-t(?P<turn>\d+)")


def marker_records(path: Path, marker: str) -> list[dict[str, Any]]:
    """Read structured JSON log records for one marker."""
    prefix = f"[{marker}]"
    records = []
    decoder = json.JSONDecoder()
    for line in path.read_text(errors="replace").splitlines():
        position = line.find(prefix)
        if position < 0:
            continue
        try:
            record, _ = decoder.raw_decode(line[position + len(prefix) :].lstrip())
        except json.JSONDecodeError:
            continue
        records.append(record)
    return records


def percentile(values: list[float], percent: int) -> float:
    """Return a nearest-rank percentile."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = ((len(ordered) - 1) * percent + 99) // 100
    return ordered[index]


def summary(values: list[float]) -> dict[str, float | int]:
    """Summarize millisecond samples."""
    if not values:
        return {"samples": 0}
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": max(values),
    }


def request_key(req_id: str) -> tuple[int, int] | None:
    """Extract the stress session and turn from a vLLM request ID."""
    match = REQUEST_PATTERN.search(req_id)
    if match is None:
        return None
    return int(match["session"]), int(match["turn"])


def load_clients(run_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Load successful client records."""
    clients = {}
    for path in sorted((run_dir / "clients").glob("session-*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("success"):
                clients[(int(record["session_id"]), int(record["turn"]))] = record
    return clients


def duration_ms(record: dict[str, Any], start: str, end: str) -> float:
    """Return a monotonic timestamp interval in milliseconds."""
    if start not in record or end not in record:
        return 0.0
    return (int(record[end]) - int(record[start])) / 1e6


def correlate_requests(
    clients: dict[tuple[int, int], dict[str, Any]],
    records: list[dict[str, Any]],
    start_field: str,
) -> dict[str, dict[str, float | int]]:
    """Build the reuse-request critical path across all TP ranks."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = request_key(str(record.get("req_id", "")))
        if key is not None:
            grouped.setdefault(key, []).append(record)

    stages = {name: [] for name in ("pre_lmcache", "lmcache", "post_lmcache", "decode")}
    for key, client in clients.items():
        if key[1] == 0 or key not in grouped:
            continue
        ranks = grouped[key]
        cache_start = min(int(record[start_field]) for record in ranks)
        cache_end = max(int(record["request_end_ns"]) for record in ranks)
        stages["pre_lmcache"].append((cache_start - int(client["start_ns"])) / 1e6)
        stages["lmcache"].append((cache_end - cache_start) / 1e6)
        stages["post_lmcache"].append((int(client["first_token_ns"]) - cache_end) / 1e6)
        stages["decode"].append(
            (int(client["end_ns"]) - int(client["first_token_ns"])) / 1e6
        )
    return {name: summary(values) for name, values in stages.items()}


def wall_tail(
    clients: dict[tuple[int, int], dict[str, Any]],
    records: list[dict[str, Any]],
    start_field: str,
) -> dict[str, Any]:
    """Split wall time at the cold-turn barrier and profile the last response."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = request_key(str(record.get("req_id", "")))
        if key is not None:
            grouped.setdefault(key, []).append(record)
    first_start = min(int(client["start_ns"]) for client in clients.values())
    cold = max(
        (client for key, client in clients.items() if key[1] == 0),
        key=lambda client: int(client["end_ns"]),
    )
    final_key, final = max(
        clients.items(), key=lambda item: int(item[1]["end_ns"])
    )
    result: dict[str, Any] = {
        "cold_barrier_ms": (int(cold["end_ns"]) - first_start) / 1e6,
        "post_cold_tail_ms": (int(final["end_ns"]) - int(cold["end_ns"]))
        / 1e6,
        "total_ms": (int(final["end_ns"]) - first_start) / 1e6,
        "cold_last": {"session": int(cold["session_id"]), "turn": 0},
        "final": {"session": final_key[0], "turn": final_key[1]},
    }
    if final_key in grouped:
        ranks = grouped[final_key]
        cache_start = min(int(record[start_field]) for record in ranks)
        cache_end = max(int(record["request_end_ns"]) for record in ranks)
        result["final_stages_ms"] = {
            "pre_lmcache": (cache_start - int(final["start_ns"])) / 1e6,
            "lmcache": (cache_end - cache_start) / 1e6,
            "post_lmcache": (int(final["first_token_ns"]) - cache_end) / 1e6,
            "decode": (int(final["end_ns"]) - int(final["first_token_ns"]))
            / 1e6,
        }
    return result


def summarize_fields(
    records: list[dict[str, Any]],
    fields: dict[str, Callable[[dict[str, Any]], float]],
    *,
    require_reuse_id: bool = True,
) -> dict[str, dict[str, float | int]]:
    """Summarize rank-local stage extractors for reuse requests."""
    selected = records
    if require_reuse_id:
        selected = [
            record
            for record in records
            if (key := request_key(str(record.get("req_id", "")))) is not None
            and key[1] > 0
        ]
    return {
        name: summary([extract(record) for record in selected])
        for name, extract in fields.items()
    }


def render_section(title: str, stages: dict[str, dict[str, float | int]]) -> list[str]:
    """Render one stage table."""
    lines = [
        f"## {title}",
        "",
        "| Stage | Samples | Mean ms | p50 | p95 | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in stages.items():
        lines.append(
            f"| {name} | {item.get('samples', 0)} | "
            f"{item.get('mean_ms', 0):.3f} | {item.get('p50_ms', 0):.3f} | "
            f"{item.get('p95_ms', 0):.3f} | {item.get('max_ms', 0):.3f} |"
        )
    lines.append("")
    return lines


def main() -> None:
    """Parse logs and write JSON and Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    server_log = args.run_dir / "server.log"
    fine = marker_records(server_log, "LMCACHE_RETRIEVE_FINE_PROFILE")
    timelines = marker_records(server_log, "LMCACHE_LAYERWISE_TIMELINE")
    disk_reads = marker_records(server_log, "LOCAL_DISK_READ_METRICS")
    clients = load_clients(args.run_dir)

    result: dict[str, Any] = {}
    lines = ["# Retrieve fine profile", ""]
    if fine:
        critical = correlate_requests(clients, fine, "request_start_ns")
        local = summarize_fields(
            fine,
            {
                "request_total": lambda r: duration_ms(
                    r, "request_start_ns", "request_end_ns"
                ),
                "chunk_scan": lambda r: duration_ms(
                    r, "chunk_scan_start_ns", "chunk_scan_end_ns"
                ),
                "mapping": lambda r: duration_ms(
                    r, "mapping_start_ns", "mapping_end_ns"
                ),
                "storage": lambda r: sum(
                    (int(call["end_ns"]) - int(call["start_ns"])) / 1e6
                    for call in r.get("storage_calls", [])
                ),
                "h2d_host": lambda r: duration_ms(
                    r, "h2d_host_start_ns", "h2d_host_end_ns"
                ),
                "h2d_cuda": lambda r: float(r.get("h2d_cuda_ms", 0.0)),
                "release": lambda r: duration_ms(
                    r, "release_start_ns", "release_end_ns"
                ),
            },
        )
        tail = wall_tail(clients, fine, "request_start_ns")
        result.update(
            {"critical_path": critical, "rank_local": local, "wall_tail": tail}
        )
        lines.extend(render_section("Reuse request critical path", critical))
        lines.extend(render_section("Reuse rank-local LMCache stages", local))
    if timelines:
        critical = correlate_requests(clients, timelines, "cache_retrieve_start_ns")
        local = summarize_fields(
            timelines,
            {
                "request_total": lambda r: duration_ms(
                    r, "cache_retrieve_start_ns", "request_end_ns"
                ),
                "key_scan": lambda r: duration_ms(
                    r, "key_scan_start_ns", "key_scan_end_ns"
                ),
                "storage_producer": lambda r: duration_ms(
                    r, "producer_start_ns", "producer_end_ns"
                ),
                "h2d_pipeline": lambda r: (
                    duration_ms(
                        r, "request_start_ns", "pipeline_h2d_complete_ns"
                    )
                    if r.get("pipeline_h2d_complete_ns") is not None
                    else duration_ms(r, "request_start_ns", "request_end_ns")
                ),
            },
            # Layerwise retrieval batches do not currently preserve vLLM req_id.
            # Every emitted record in this stress test is nevertheless a cache hit.
            require_reuse_id=False,
        )
        tail = wall_tail(clients, timelines, "cache_retrieve_start_ns")
        result.update(
            {"critical_path": critical, "rank_local": local, "wall_tail": tail}
        )
        lines.extend(render_section("Reuse request critical path", critical))
        lines.extend(render_section("Reuse rank-local layerwise stages", local))

    if "wall_tail" in result:
        tail = result["wall_tail"]
        lines.extend(
            [
                "## Wall-time tail",
                "",
                "| Cold barrier ms | Post-cold tail ms | Total ms | Cold-last | Final |",
                "|---:|---:|---:|---|---|",
                f"| {tail['cold_barrier_ms']:.3f} | {tail['post_cold_tail_ms']:.3f} | "
                f"{tail['total_ms']:.3f} | s{tail['cold_last']['session']:03d}-t00 | "
                f"s{tail['final']['session']:03d}-t{tail['final']['turn']:02d} |",
                "",
            ]
        )
        if stages := tail.get("final_stages_ms"):
            lines.extend(
                [
                    "| Final response stage | ms |",
                    "|---|---:|",
                    *[f"| {name} | {value:.3f} |" for name, value in stages.items()],
                    "",
                ]
            )

    if disk_reads:
        total_bytes = sum(int(record.get("bytes", 0)) for record in disk_reads)
        total_files = sum(int(record.get("file_count", 0)) for record in disk_reads)
        total_seconds = sum(float(record.get("seconds", 0.0)) for record in disk_reads)
        disk = {
            "batches": len(disk_reads),
            "files": total_files,
            "bytes": total_bytes,
            "active_seconds": total_seconds,
            "active_gib_per_s": total_bytes / (1024**3) / total_seconds,
            "mean_file_us": total_seconds * 1e6 / total_files,
            "batch_latency_ms": summary(
                [float(record.get("seconds", 0.0)) * 1e3 for record in disk_reads]
            ),
        }
        result["local_disk"] = disk
        batch = disk["batch_latency_ms"]
        lines.extend(
            [
                "## Local Disk reads",
                "",
                "| Batches | Files | GiB | Active GiB/s | Mean us/file | Batch mean ms | p95 | Max |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| {disk['batches']} | {disk['files']} | {total_bytes / 1024**3:.3f} | "
                f"{disk['active_gib_per_s']:.3f} | {disk['mean_file_us']:.3f} | "
                f"{batch['mean_ms']:.3f} | {batch['p95_ms']:.3f} | "
                f"{batch['max_ms']:.3f} |",
                "",
            ]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    (args.output_dir / "retrieve-fine-profile.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output_dir / "retrieve-fine-profile.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
