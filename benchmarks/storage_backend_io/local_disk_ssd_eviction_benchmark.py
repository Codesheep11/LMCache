# SPDX-License-Identifier: Apache-2.0
"""Measure LocalDisk read bandwidth while concurrent writes evict SSD blocks."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence
import argparse
import json
import random
import shutil
import statistics
import tempfile
import threading
import time

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

# Local
from local_disk_cpu_pressure_benchmark import (
    GIB,
    MIB,
    make_aligned_source,
    make_key,
    make_metadata,
    make_shape,
    start_loop,
)


class CountingLocalDiskBackend(LocalDiskBackend):
    """LocalDiskBackend that records eviction count and deletion time."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eviction_count = 0
        self.eviction_ns = 0
        self.eviction_stats_lock = threading.Lock()

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        """Remove disk objects and account for time in the removal path."""
        start_ns = time.monotonic_ns()
        removed = super().batched_remove(keys, force=force)
        elapsed_ns = time.monotonic_ns() - start_ns
        with self.eviction_stats_lock:
            self.eviction_count += removed
            self.eviction_ns += elapsed_ns
        return removed

    def reset_eviction_stats(self) -> None:
        """Reset eviction counters before a measured case."""
        with self.eviction_stats_lock:
            self.eviction_count = 0
            self.eviction_ns = 0

    def get_eviction_stats(self) -> tuple[int, int]:
        """Return eviction count and cumulative removal time."""
        with self.eviction_stats_lock:
            return self.eviction_count, self.eviction_ns


class CompletionTracker:
    """Track completion callbacks from LocalDiskBackend's write workers."""

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.completed = 0
        self.lock = threading.Lock()
        self.done = threading.Event()

    def callback(self, _key: CacheEngineKey) -> None:
        """Record one completed disk write."""
        with self.lock:
            self.completed += 1
            if self.completed == self.expected:
                self.done.set()

    def wait(self, timeout: float) -> None:
        """Wait for all writes or raise a timeout with the observed count."""
        if not self.done.wait(timeout):
            raise TimeoutError(
                f"disk writes timed out: completed={self.completed}, "
                f"expected={self.expected}"
            )


def submit_and_wait(
    backend: LocalDiskBackend,
    keys: Sequence[CacheEngineKey],
    source: Any,
    timeout: float = 300,
) -> None:
    """Submit a set of writes and wait until their metadata is published."""
    tracker = CompletionTracker(len(keys))
    backend.batched_submit_put_task(
        keys,
        [source] * len(keys),
        on_complete_callback=tracker.callback,
    )
    tracker.wait(timeout)


def read_block_stat(device: str) -> tuple[int, int, int, int, int] | None:
    """Read sectors, I/O time, and weighted time from one block device."""
    path = Path("/sys/class/block") / device / "stat"
    try:
        fields = [int(value) for value in path.read_text().split()]
    except (FileNotFoundError, PermissionError):
        return None
    if len(fields) < 11:
        return None
    return fields[2], fields[6], fields[3], fields[7], fields[10]


def block_stat_deltas(
    before: dict[str, tuple[int, int, int, int, int] | None],
    after: dict[str, tuple[int, int, int, int, int] | None],
    elapsed_sec: float,
) -> dict[str, dict[str, float]]:
    """Convert block-device snapshots into throughput and timing metrics."""
    output: dict[str, dict[str, float]] = {}
    elapsed_ms = elapsed_sec * 1000
    for device, start in before.items():
        end = after[device]
        if start is None or end is None:
            continue
        output[device] = {
            "read_gib": (end[0] - start[0]) * 512 / GIB,
            "write_gib": (end[1] - start[1]) * 512 / GIB,
            "read_ms": float(end[2] - start[2]),
            "write_ms": float(end[3] - start[3]),
            "average_queue_depth": (end[4] - start[4]) / elapsed_ms,
        }
    return output


def run_case(
    backend: CountingLocalDiskBackend,
    cpu_backend: LocalCPUBackend,
    read_keys: list[CacheEngineKey],
    write_keys: list[CacheEngineKey],
    source: Any,
    concurrency: int,
    eviction: bool,
    devices: list[str],
    seed: int,
    initial_bytes: int,
    write_bytes: int,
) -> dict[str, Any]:
    """Read pinned blocks while an equal-size write stream runs concurrently."""
    backend.max_cache_size = initial_bytes if eviction else initial_bytes + write_bytes
    backend.reset_eviction_stats()

    ordered_keys = list(read_keys)
    random.Random(seed).shuffle(ordered_keys)
    gate = threading.Event()
    tracker = CompletionTracker(len(write_keys))
    admission_times: dict[str, int] = {}

    def load_one(key: CacheEngineKey) -> tuple[int, int, int, int, bool]:
        gate.wait()
        profile = {"metadata_ns": 0, "allocate_ns": 0, "read_ns": 0}
        memory_obj = backend.get_blocking(key, profile)
        if memory_obj is None:
            return (
                0,
                profile["metadata_ns"],
                profile["allocate_ns"],
                profile["read_ns"],
                True,
            )
        size = memory_obj.get_size()
        memory_obj.ref_count_down()
        return (
            size,
            profile["metadata_ns"],
            profile["allocate_ns"],
            profile["read_ns"],
            False,
        )

    def write_stream() -> None:
        gate.wait()
        admission_times["start_ns"] = time.monotonic_ns()
        backend.batched_submit_put_task(
            write_keys,
            [source] * len(write_keys),
            on_complete_callback=tracker.callback,
        )
        admission_times["end_ns"] = time.monotonic_ns()

    reader_executor = ThreadPoolExecutor(max_workers=concurrency)
    read_futures = [reader_executor.submit(load_one, key) for key in ordered_keys]
    writer = threading.Thread(target=write_stream)
    writer.start()

    device_before = {device: read_block_stat(device) for device in devices}
    start_ns = time.monotonic_ns()
    gate.set()
    samples = [future.result() for future in read_futures]
    read_end_ns = time.monotonic_ns()
    writer.join()
    tracker.wait(timeout=300)
    all_end_ns = time.monotonic_ns()
    device_after = {device: read_block_stat(device) for device in devices}
    reader_executor.shutdown(wait=True)

    total_read_bytes = sum(sample[0] for sample in samples)
    read_elapsed_sec = (read_end_ns - start_ns) / 1e9
    makespan_sec = (all_end_ns - start_ns) / 1e9
    evictions, eviction_ns = backend.get_eviction_stats()
    read_failures = sum(sample[4] for sample in samples)
    expected_evictions = len(write_keys) if eviction else 0
    if read_failures:
        raise RuntimeError(f"unexpected disk-read failures: {read_failures}")
    if evictions != expected_evictions:
        raise RuntimeError(
            f"unexpected SSD eviction count: observed={evictions}, "
            f"expected={expected_evictions}"
        )
    result = {
        "mode": "ssd_full_eviction" if eviction else "ssd_headroom",
        "concurrency": concurrency,
        "read_bytes": total_read_bytes,
        "write_bytes": write_bytes,
        "read_elapsed_sec": read_elapsed_sec,
        "makespan_sec": makespan_sec,
        "read_gib_per_sec": total_read_bytes / GIB / read_elapsed_sec,
        "aggregate_gib_per_sec": (total_read_bytes + write_bytes) / GIB / makespan_sec,
        "read_failures": read_failures,
        "ssd_evictions": evictions,
        "ssd_eviction_ms_sum": eviction_ns / 1e6,
        "write_admission_ms": (admission_times["end_ns"] - admission_times["start_ns"])
        / 1e6,
        "metadata_ms_sum": sum(sample[1] for sample in samples) / 1e6,
        "allocate_ms_sum": sum(sample[2] for sample in samples) / 1e6,
        "read_ms_sum": sum(sample[3] for sample in samples) / 1e6,
        "devices": block_stat_deltas(device_before, device_after, makespan_sec),
    }
    print(
        f"{result['mode']}: readers={concurrency} "
        f"read={result['read_gib_per_sec']:.2f} GiB/s "
        f"aggregate={result['aggregate_gib_per_sec']:.2f} GiB/s "
        f"evictions={evictions} failures={result['read_failures']}",
        flush=True,
    )
    return result


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize median read throughput and eviction-induced change."""

    def percentile(values: Sequence[float], percent: int) -> float:
        """Return a linearly interpolated percentile."""
        ordered = sorted(values)
        position = (len(ordered) - 1) * percent / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    summary = []
    for concurrency in sorted({result["concurrency"] for result in results}):
        headroom = [
            result
            for result in results
            if result["concurrency"] == concurrency and result["mode"] == "ssd_headroom"
        ]
        eviction = [
            result
            for result in results
            if result["concurrency"] == concurrency
            and result["mode"] == "ssd_full_eviction"
        ]
        headroom_bw = statistics.median(
            result["read_gib_per_sec"] for result in headroom
        )
        eviction_bw = statistics.median(
            result["read_gib_per_sec"] for result in eviction
        )
        headroom_by_trial = {result["trial"]: result for result in headroom}
        eviction_by_trial = {result["trial"]: result for result in eviction}
        if headroom_by_trial.keys() != eviction_by_trial.keys():
            raise RuntimeError(
                f"unpaired headroom/full trials for concurrency {concurrency}"
            )
        trial_losses = [
            (
                headroom_by_trial[trial]["read_gib_per_sec"]
                - eviction_by_trial[trial]["read_gib_per_sec"]
            )
            / headroom_by_trial[trial]["read_gib_per_sec"]
            * 100
            for trial in sorted(headroom_by_trial)
        ]
        summary.append(
            {
                "concurrency": concurrency,
                "ssd_headroom_median_read_gib_per_sec": headroom_bw,
                "ssd_full_eviction_median_read_gib_per_sec": eviction_bw,
                "bandwidth_loss_percent": (1 - eviction_bw / headroom_bw) * 100,
                "trial_loss_count": len(trial_losses),
                "trial_loss_mean_percent": statistics.mean(trial_losses),
                "trial_loss_p50_percent": percentile(trial_losses, 50),
                "trial_loss_p90_percent": percentile(trial_losses, 90),
                "trial_loss_p99_percent": percentile(trial_losses, 99),
                "ssd_full_median_eviction_ms_sum": statistics.median(
                    result["ssd_eviction_ms_sum"] for result in eviction
                ),
                "ssd_full_median_write_admission_ms": statistics.median(
                    result["write_admission_ms"] for result in eviction
                ),
            }
        )
    return summary


def parse_csv_ints(value: str) -> list[int]:
    """Parse comma-separated positive integers."""
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def main() -> None:
    """Run the SSD-headroom versus SSD-eviction comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disk-dir", required=True)
    parser.add_argument("--dataset-gb", type=float, default=8.0)
    parser.add_argument("--cpu-size-gb", type=float, default=1.0)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--concurrency", type=parse_csv_ints, default=[1, 2, 4, 8])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--odirect", action="store_true")
    parser.add_argument("--devices", default="md127")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--keep-files", action="store_true")
    args = parser.parse_args()

    if args.dataset_gb <= 0 or args.cpu_size_gb <= 0 or args.chunk_mib <= 0:
        parser.error("dataset, CPU size, and chunk size must be positive")
    if args.trials <= 0:
        parser.error("trials must be positive")

    disk_parent = Path(args.disk_dir).resolve()
    disk_parent.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="lmcache-ssd-eviction-", dir=disk_parent))
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_bytes = args.chunk_mib * MIB
    block_count = max(1, int(args.dataset_gb * GIB) // chunk_bytes)
    dataset_bytes = block_count * chunk_bytes
    initial_bytes = dataset_bytes * 2
    shape = make_shape(chunk_bytes)
    metadata = make_metadata(shape)
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=shape[2],
        local_cpu=True,
        max_local_cpu_size=args.cpu_size_gb,
        lmcache_instance_id="local_disk_ssd_eviction",
    )
    config.local_disk = str(run_root)
    config.max_local_disk_size = initial_bytes * 2 / GIB
    config.extra_config = {
        "use_odirect": args.odirect,
        "local_cpu.pinned_align_bytes": args.alignment,
    }

    loop, _loop_thread = start_loop()
    cpu_backend = LocalCPUBackend(
        config=config,
        metadata=metadata,
        dst_device="cpu",
    )
    disk_backend = CountingLocalDiskBackend(
        config=config,
        loop=loop,
        local_cpu_backend=cpu_backend,
        dst_device="cpu",
        metadata=metadata,
    )
    source, source_keepalive = make_aligned_source(shape, args.alignment)

    read_keys = [make_key(10_000_000 + index) for index in range(block_count)]
    filler_keys = [make_key(20_000_000 + index) for index in range(block_count)]
    all_created_keys = list(read_keys) + list(filler_keys)
    results: list[dict[str, Any]] = []

    try:
        print(
            f"prepare: {len(read_keys)} read blocks + {len(filler_keys)} fillers, "
            f"{initial_bytes / GIB:.2f} GiB",
            flush=True,
        )
        submit_and_wait(disk_backend, all_created_keys, source)
        for key in read_keys:
            if not disk_backend.pin(key):
                raise RuntimeError(f"failed to pin read key {key}")

        case_index = 0
        for concurrency in args.concurrency:
            for trial in range(args.trials):
                mode_order = (False, True) if trial % 2 == 0 else (True, False)
                for eviction in mode_order:
                    write_keys = [
                        make_key(100_000_000 + case_index * block_count + index)
                        for index in range(block_count)
                    ]
                    case_index += 1
                    result = run_case(
                        disk_backend,
                        cpu_backend,
                        read_keys,
                        write_keys,
                        source,
                        concurrency,
                        eviction,
                        [device for device in args.devices.split(",") if device],
                        seed=trial,
                        initial_bytes=initial_bytes,
                        write_bytes=dataset_bytes,
                    )
                    result["trial"] = trial
                    results.append(result)

                    if not eviction:
                        disk_backend.batched_remove(write_keys)
                        disk_backend.current_cache_size = initial_bytes

        report = {
            "parameters": {
                "disk_dir": str(run_root),
                "dataset_bytes_per_direction": dataset_bytes,
                "chunk_bytes": chunk_bytes,
                "block_count": block_count,
                "initial_cache_bytes": initial_bytes,
                "cpu_size_bytes": int(args.cpu_size_gb * GIB),
                "concurrency": args.concurrency,
                "trials": args.trials,
                "odirect": args.odirect,
            },
            "results": results,
            "summary": summarize(results),
        }
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2))
        print(f"wrote {output_path}")
    finally:
        del source_keepalive
        cpu_backend.clear()
        cpu_backend.close()
        if not args.keep_files:
            shutil.rmtree(run_root)


if __name__ == "__main__":
    main()
