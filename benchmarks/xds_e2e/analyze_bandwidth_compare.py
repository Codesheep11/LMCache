# SPDX-License-Identifier: Apache-2.0
"""Summarize raw SSD and vLLM SSD-to-GPU bandwidth measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


GIB = 1024**3


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def marker_records(path: Path, marker: str) -> list[dict]:
    records = []
    for line in path.read_text(errors="replace").splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        # LMCache's colored logger inserts an ANSI reset before the JSON and
        # appends the source location after it.  Parse only the JSON object so
        # benchmark logs work with both colored and plain logging.
        object_start = payload.find("{")
        object_end = payload.rfind("}")
        if object_start < 0 or object_end < object_start:
            continue
        try:
            records.append(json.loads(payload[object_start : object_end + 1]))
        except json.JSONDecodeError:
            continue
    return records


def merged_interval_seconds(records: list[dict], start_key: str, end_key: str) -> float:
    intervals = sorted(
        (int(record[start_key]), int(record[end_key]))
        for record in records
        if int(record.get(end_key, 0)) > int(record.get(start_key, 0)) > 0
    )
    if not intervals:
        return 0.0
    total_ns = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total_ns += current_end - current_start
            current_start, current_end = start, end
    total_ns += current_end - current_start
    return total_ns / 1e9


def active_bandwidth(
    records: list[dict], start_key: str, end_key: str
) -> tuple[float, int, float]:
    seconds = merged_interval_seconds(records, start_key, end_key)
    byte_count = sum(int(record.get("bytes", 0)) for record in records)
    bandwidth = byte_count / seconds / GIB if seconds else 0.0
    return bandwidth, byte_count, seconds


def request_bandwidth(records: list[dict]) -> list[float]:
    hits = [
        record
        for record in records
        if int(record.get("bytes", 0)) > 0
        and int(record.get("retrieved_tokens", 0)) >= 40000
    ]
    per_worker: dict[int, list[dict]] = {}
    for record in hits:
        per_worker.setdefault(int(record["worker_id"]), []).append(record)
    if not per_worker:
        return []
    request_count = min(len(items) for items in per_worker.values())
    bandwidths = []
    for index in range(request_count):
        rank_records = [items[index] for items in per_worker.values()]
        byte_count = sum(int(record["bytes"]) for record in rank_records)
        seconds = max(float(record["seconds"]) for record in rank_records)
        if seconds > 0:
            bandwidths.append(byte_count / seconds / GIB)
    return bandwidths


def inference_latencies(path: Path) -> list[float]:
    records = marker_records(path, "[PREFIX_BENCH_METRICS]")
    if not records:
        return []
    return [
        float(request["inference_ms"])
        for request in records[-1]["requests"]
        if request["phase"] == "hit"
    ]


def raw_baselines(path: Path) -> dict[str, tuple[int, float, float]]:
    result = {}
    for line in path.read_text().splitlines()[1:]:
        mode, byte_count, _start_ns, _end_ns, seconds = line.split("\t")
        byte_count_int = int(byte_count)
        seconds_float = float(seconds)
        result[mode] = (
            byte_count_int,
            seconds_float,
            byte_count_int / seconds_float / GIB,
        )
    return result


def describe(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"mean {statistics.mean(values):.3f}, "
        f"P50 {percentile(values, 0.50):.3f}, "
        f"P95 {percentile(values, 0.95):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir

    raw = raw_baselines(result_dir / "ssd-baseline.tsv")
    benchmark_shape = marker_records(
        result_dir / "disk-buffered.log", "[PREFIX_BENCH_METRICS]"
    )[-1]
    num_unique_prompts = int(benchmark_shape["num_unique_prompts"])
    repeat_count = int(benchmark_shape["repeat_count"])
    backend_rows = []
    for name, filename in (
        ("LocalDisk buffered", "disk-buffered.log"),
        ("LocalDisk O_DIRECT", "disk-odirect.log"),
        ("CascadeKV/xDS", "cascadekv.log"),
    ):
        path = result_dir / filename
        e2e = request_bandwidth(
            marker_records(path, "[LMCACHE_SSD_GPU_METRICS]")
        )
        inference = inference_latencies(path)
        if name.startswith("LocalDisk"):
            storage_records = marker_records(path, "[LOCAL_DISK_READ_METRICS]")
            storage_bw, storage_bytes, storage_seconds = active_bandwidth(
                storage_records, "start_ns", "end_ns"
            )
        else:
            storage_records = marker_records(path, "[CASCADEKV_READ_METRICS]")
            storage_bw, storage_bytes, storage_seconds = active_bandwidth(
                storage_records, "storage_start_ns", "storage_end_ns"
            )
        backend_rows.append(
            (
                name,
                storage_bw,
                storage_bytes,
                storage_seconds,
                e2e,
                inference,
                len(e2e),
            )
        )

    lines = [
        "# LMCache 0.4.3：同 SSD 带宽与 40K KV 端到端对比",
        "",
        "配置：Qwen3-30B-A3B，TP=4，chunk_size=256，"
        f"{num_unique_prompts} 个唯一 40K prompt × {repeat_count} 轮；"
        f"前 {num_unique_prompts} 条为冷写入，统计后续完整命中。",
        "",
        "## RAID0 顺序读取基线",
        "",
        "| 模式 | 数据量 (GiB) | 时间 (s) | 带宽 (GiB/s) |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("buffered", "odirect"):
        byte_count, seconds, bandwidth = raw[mode]
        lines.append(
            f"| {mode} | {byte_count / GIB:.3f} | {seconds:.3f} | {bandwidth:.3f} |"
        )
    lines.extend(
        [
            "",
            "## vLLM 端到端",
            "",
            "| 后端 | 观测 SSD/xDS 阶段带宽 (GiB/s) | "
            "SSD→GPU 有效带宽 (GiB/s) | 命中 Prefill latency (ms) | 完整命中样本 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, storage_bw, _bytes, _seconds, e2e, inference, count in backend_rows:
        lines.append(
            f"| {name} | {storage_bw:.3f} | {describe(e2e)} | "
            f"{describe(inference)} | {count} |"
        )
    lines.extend(
        [
            "",
            "说明：",
            "",
            "- SSD/xDS 阶段带宽按四个 TP 进程的实际读区间做时间并集，"
            "再用总读取字节数除以活跃时间，避免简单相加重复计算并发时间。",
            "- SSD→GPU 带宽以每条请求四个 TP rank 的 KV 总字节数，"
            "除以最慢 rank 从读取开始到 GPU 数据可用的时间；"
            "LocalDisk 基准在末层显式同步 CUDA stream。",
            "- buffered LocalDisk 在冷写入与命中读取之间执行 sync 和 "
            "drop_caches，避免把 Linux 页缓存误当成 SSD 带宽。",
            "- 命中 Prefill latency 来自 vLLM 的 scheduled_ts 到 last_token_ts；"
            "本测试 output_len=1，因此包含 KV 恢复、prefill 和首个输出 token。",
        ]
    )
    (result_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
