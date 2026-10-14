# SPDX-License-Identifier: Apache-2.0
"""Measure CascadeKV reads with and without concurrent SSD eviction."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence
import argparse
import asyncio
import json
import random
import statistics
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    XDSServiceMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cascadekv_backend import (
    BlockTable,
    CascadeKVBackend,
)
from lmcache.v1.xds_utils import connect_xds_client


GIB = 1024**3
MIB = 1024**2
DTYPE = torch.bfloat16
FORMAT = MemoryFormat.KV_2LTD
MODEL_NAME = "cascadekv_ssd_eviction"


class CountingBlockTable(BlockTable):
    """Record durable lookup and eviction lock-path time."""

    def __init__(self) -> None:
        super().__init__()
        self._stats_lock = threading.Lock()
        self._lookup_count = 0
        self._lookup_ns = 0
        self._eviction_count = 0
        self._eviction_ns = 0

    def acquire(self, key: CacheEngineKey):
        """Acquire one reader and account for metadata lookup time."""
        start_ns = time.monotonic_ns()
        block = super().acquire(key)
        elapsed_ns = time.monotonic_ns() - start_ns
        with self._stats_lock:
            self._lookup_count += 1
            self._lookup_ns += elapsed_ns
        return block

    def evict_oldest(self) -> int | None:
        """Evict one block and account for victim-selection time."""
        start_ns = time.monotonic_ns()
        handle = super().evict_oldest()
        elapsed_ns = time.monotonic_ns() - start_ns
        with self._stats_lock:
            self._eviction_count += int(handle is not None)
            self._eviction_ns += elapsed_ns
        return handle

    def reset_stats(self) -> None:
        """Reset counters between measured cases."""
        with self._stats_lock:
            self._lookup_count = 0
            self._lookup_ns = 0
            self._eviction_count = 0
            self._eviction_ns = 0

    def get_stats(self) -> dict[str, int]:
        """Return a consistent counter snapshot."""
        with self._stats_lock:
            return {
                "lookup_count": self._lookup_count,
                "lookup_ns_sum": self._lookup_ns,
                "eviction_count": self._eviction_count,
                "eviction_ns_sum": self._eviction_ns,
            }


class CompletionTracker:
    """Wait for one callback from every accepted CascadeKV write."""

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.completed = 0
        self._lock = threading.Lock()
        self._done = threading.Event()

    def callback(self, _key: CacheEngineKey) -> None:
        """Record one durable write."""
        with self._lock:
            self.completed += 1
            if self.completed == self.expected:
                self._done.set()

    def wait(self, backend: CascadeKVBackend, timeout: float) -> None:
        """Wait for all writes while surfacing scheduler failures."""
        deadline = time.monotonic() + timeout
        while not self._done.wait(0.1):
            backend.io_engine.check_health()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "CascadeKV writes timed out: "
                    f"completed={self.completed}, expected={self.expected}"
                )


def parse_concurrency(value: str) -> list[int]:
    """Parse and validate a comma-separated thread-count list."""
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("thread counts must be positive")
    return values


def make_shape(chunk_bytes: int) -> torch.Size:
    """Build a one-layer KV_2LTD shape with exactly chunk_bytes."""
    bytes_per_token = 2 * DTYPE.itemsize
    if chunk_bytes % bytes_per_token:
        raise ValueError("chunk size must be divisible by four bytes")
    return torch.Size([2, 1, chunk_bytes // bytes_per_token, 1])


def make_metadata(shape: torch.Size) -> LMCacheMetadata:
    """Build metadata consistent with the benchmark block shape."""
    return LMCacheMetadata(
        model_name=MODEL_NAME,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=DTYPE,
        kv_shape=(1, 2, shape[2], 1, 1),
        chunk_size=shape[2],
    )


def make_key(index: int) -> CacheEngineKey:
    """Create a deterministic one-layer cache key."""
    return CacheEngineKey(MODEL_NAME, 1, 0, index, DTYPE)


def wait_until(predicate: Any, timeout: float, description: str) -> None:
    """Wait for an asynchronous state transition."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {description}")
        time.sleep(0.01)


def submit_and_wait(
    backend: CascadeKVBackend,
    keys: Sequence[CacheEngineKey],
    source: MemoryObj,
    timeout: float,
    batch_size: int | None = None,
) -> None:
    """Submit identical immutable blocks and wait until all are durable."""
    width = len(keys) if batch_size is None else batch_size
    for start in range(0, len(keys), width):
        batch = keys[start : start + width]
        tracker = CompletionTracker(len(batch))
        backend.batched_submit_put_task(
            batch,
            [source] * len(batch),
            on_complete_callback=tracker.callback,
        )
        tracker.wait(backend, timeout)
        if batch_size is not None:
            print(f"prepare durable: {start + len(batch)}/{len(keys)}", flush=True)


def cleanup_headroom_writes(
    backend: CascadeKVBackend,
    keys: Sequence[CacheEngineKey],
    expected_blocks: int,
    timeout: float,
) -> None:
    """Remove headroom-only writes and wait for handle releases."""
    removed = sum(backend.remove(key) for key in keys)
    if removed != len(keys):
        raise RuntimeError(f"headroom cleanup removed {removed} of {len(keys)} blocks")
    wait_until(
        lambda: (
            len(backend.block_table) == expected_blocks
            and backend.io_engine.metrics()["leased_handles"] == expected_blocks
        ),
        timeout,
        "headroom handle cleanup",
    )


def run_case(
    backend: CascadeKVBackend,
    block_table: CountingBlockTable,
    read_keys: list[CacheEngineKey],
    write_keys: list[CacheEngineKey],
    source: MemoryObj,
    concurrency: int,
    eviction: bool,
    initial_blocks: int,
    write_batch_size: int,
    timeout: float,
    seed: int,
) -> dict[str, Any]:
    """Read pinned blocks while an equal-size write stream runs concurrently."""
    target_blocks = initial_blocks if eviction else initial_blocks + len(write_keys)
    leased_handles = int(backend.io_engine.metrics()["leased_handles"])
    if leased_handles != initial_blocks:
        raise RuntimeError(
            f"case starts with {leased_handles} handles, expected {initial_blocks}"
        )
    backend.io_engine.max_blocks = target_blocks
    backend.io_engine.peak_inflight = 0
    backend.io_engine.peak_active_tasks = 0
    block_table.reset_stats()

    ordered_keys = list(read_keys)
    random.Random(seed).shuffle(ordered_keys)
    gate = threading.Event()
    write_times: dict[str, int] = {}
    write_errors: list[BaseException] = []

    def load_one(key: CacheEngineKey) -> tuple[int, bool]:
        gate.wait()
        memory_obj = backend.get_blocking(key)
        if memory_obj is None:
            return 0, True
        size = memory_obj.get_size()
        memory_obj.ref_count_down()
        return size, False

    def write_stream() -> None:
        gate.wait()
        write_times["start_ns"] = time.monotonic_ns()
        try:
            for start in range(0, len(write_keys), write_batch_size):
                batch = write_keys[start : start + write_batch_size]
                tracker = CompletionTracker(len(batch))
                backend.batched_submit_put_task(
                    batch,
                    [source] * len(batch),
                    on_complete_callback=tracker.callback,
                )
                tracker.wait(backend, timeout)
        except BaseException as error:
            write_errors.append(error)
        finally:
            write_times["end_ns"] = time.monotonic_ns()

    reader_executor = ThreadPoolExecutor(max_workers=concurrency)
    read_futures = [reader_executor.submit(load_one, key) for key in ordered_keys]
    writer = threading.Thread(target=write_stream)
    writer.start()

    start_ns = time.monotonic_ns()
    gate.set()
    samples = [future.result() for future in read_futures]
    read_end_ns = time.monotonic_ns()
    writer.join()
    if write_errors:
        raise RuntimeError("CascadeKV write stream failed") from write_errors[0]
    all_end_ns = time.monotonic_ns()
    reader_executor.shutdown(wait=True)

    stats = block_table.get_stats()
    read_failures = sum(sample[1] for sample in samples)
    read_bytes = sum(sample[0] for sample in samples)
    expected_evictions = len(write_keys) if eviction else 0
    if read_failures:
        raise RuntimeError(f"unexpected CascadeKV read failures: {read_failures}")
    if stats["lookup_count"] != len(read_keys):
        raise RuntimeError(
            f"observed {stats['lookup_count']} lookups, expected {len(read_keys)}"
        )
    if stats["eviction_count"] != expected_evictions:
        raise RuntimeError(
            "unexpected CascadeKV eviction count: "
            f"observed={stats['eviction_count']}, expected={expected_evictions}"
        )

    read_elapsed_sec = (read_end_ns - start_ns) / 1e9
    makespan_sec = (all_end_ns - start_ns) / 1e9
    result = {
        "mode": "ssd_full_eviction" if eviction else "ssd_headroom",
        "concurrency": concurrency,
        "read_bytes": read_bytes,
        "write_bytes": len(write_keys) * backend.blob_size,
        "read_elapsed_sec": read_elapsed_sec,
        "makespan_sec": makespan_sec,
        "read_gib_per_sec": read_bytes / GIB / read_elapsed_sec,
        "aggregate_gib_per_sec": (read_bytes + len(write_keys) * backend.blob_size)
        / GIB
        / makespan_sec,
        "read_failures": read_failures,
        "ssd_evictions": stats["eviction_count"],
        "ssd_eviction_ms_sum": stats["eviction_ns_sum"] / 1e6,
        "metadata_lookup_ms_sum": stats["lookup_ns_sum"] / 1e6,
        "write_stream_ms": (write_times["end_ns"] - write_times["start_ns"]) / 1e6,
        "engine_peak_qd": backend.io_engine.peak_inflight,
        "engine_peak_active_tasks": backend.io_engine.peak_active_tasks,
        "leased_handles": backend.io_engine.metrics()["leased_handles"],
    }
    print(
        f"{result['mode']}: readers={concurrency} "
        f"read={result['read_gib_per_sec']:.2f} GiB/s "
        f"aggregate={result['aggregate_gib_per_sec']:.2f} GiB/s "
        f"evictions={result['ssd_evictions']} failures={read_failures}",
        flush=True,
    )
    return result


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize read bandwidth and eviction overhead per thread count."""

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
        full = [
            result
            for result in results
            if result["concurrency"] == concurrency
            and result["mode"] == "ssd_full_eviction"
        ]
        headroom_bw = statistics.median(
            result["read_gib_per_sec"] for result in headroom
        )
        full_bw = statistics.median(result["read_gib_per_sec"] for result in full)
        headroom_by_trial = {result["trial"]: result for result in headroom}
        full_by_trial = {result["trial"]: result for result in full}
        if headroom_by_trial.keys() != full_by_trial.keys():
            raise RuntimeError(
                f"unpaired headroom/full trials for concurrency {concurrency}"
            )
        trial_losses = [
            (
                headroom_by_trial[trial]["read_gib_per_sec"]
                - full_by_trial[trial]["read_gib_per_sec"]
            )
            / headroom_by_trial[trial]["read_gib_per_sec"]
            * 100
            for trial in sorted(headroom_by_trial)
        ]
        summary.append(
            {
                "concurrency": concurrency,
                "ssd_headroom_median_read_gib_per_sec": headroom_bw,
                "ssd_full_eviction_median_read_gib_per_sec": full_bw,
                "bandwidth_loss_percent": (headroom_bw - full_bw) / headroom_bw * 100,
                "trial_loss_count": len(trial_losses),
                "trial_loss_mean_percent": statistics.mean(trial_losses),
                "trial_loss_p50_percent": percentile(trial_losses, 50),
                "trial_loss_p90_percent": percentile(trial_losses, 90),
                "trial_loss_p99_percent": percentile(trial_losses, 99),
                "ssd_full_median_eviction_ms_sum": statistics.median(
                    result["ssd_eviction_ms_sum"] for result in full
                ),
                "ssd_full_median_metadata_lookup_ms_sum": statistics.median(
                    result["metadata_lookup_ms_sum"] for result in full
                ),
            }
        )
    return summary


def main() -> int:
    """Run the CascadeKV SSD-eviction bandwidth matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-name", required=True)
    parser.add_argument("--dataset-gb", type=float, default=8)
    parser.add_argument("--cpu-size-gb", type=float, default=0.25)
    parser.add_argument("--chunk-mib", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        type=parse_concurrency,
        default=parse_concurrency(",".join(str(value) for value in range(1, 17))),
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--write-batch-size", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if args.dataset_gb <= 0 or args.cpu_size_gb <= 0 or args.chunk_mib <= 0:
        parser.error("dataset, CPU size, and chunk size must be positive")
    if args.trials <= 0 or args.write_batch_size <= 0 or args.timeout <= 0:
        parser.error("trials, write batch size, and timeout must be positive")

    chunk_bytes = args.chunk_mib * MIB
    block_count = max(1, int(args.dataset_gb * GIB) // chunk_bytes)
    dataset_bytes = block_count * chunk_bytes
    initial_blocks = block_count * 2
    headroom_blocks = initial_blocks + block_count
    shape = make_shape(chunk_bytes)
    metadata = make_metadata(shape)
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=shape[2],
        local_cpu=False,
        max_local_cpu_size=args.cpu_size_gb,
        lmcache_instance_id=MODEL_NAME,
    )
    config.xds_service_name = args.service_name
    config.cascadekv_backend = True
    config.xds_backend = False
    config.xds_require_direct = False
    config.xds_max_size = headroom_blocks * chunk_bytes / GIB
    config.use_layerwise = False
    config.extra_config = {
        "benchmark_transfer_metrics": False,
        "cascadekv_read_block_window": max(args.concurrency),
    }

    client = connect_xds_client(config, metadata)
    service_metadata = client.metadata
    if int(service_metadata.blob_size_bytes) != chunk_bytes:
        client.close()
        raise RuntimeError(
            f"service blob size is {service_metadata.blob_size_bytes}, "
            f"expected {chunk_bytes}"
        )
    service_blocks = int(service_metadata.capacity_bytes) // chunk_bytes
    if service_blocks < headroom_blocks:
        client.close()
        raise RuntimeError(
            f"service has {service_blocks} blocks, need {headroom_blocks}"
        )

    allocator = XDSServiceMemoryAllocator(
        client,
        int(args.cpu_size_gb * GIB),
        int(service_metadata.io_unit_size),
        close_client=True,
    )
    loop = asyncio.new_event_loop()
    backend = CascadeKVBackend(config, metadata, loop, allocator, client, "cpu")
    block_table = CountingBlockTable()
    backend.block_table = block_table
    backend.io_engine.block_table = block_table
    backend.io_engine.max_blocks = headroom_blocks

    source = backend.allocate(shape, DTYPE, FORMAT)
    if source is None:
        backend.close()
        loop.close()
        raise RuntimeError("failed to allocate the CascadeKV source block")
    assert source.tensor is not None
    source.tensor.fill_(7)

    read_keys = [make_key(10_000_000 + index) for index in range(block_count)]
    filler_keys = [make_key(20_000_000 + index) for index in range(block_count)]
    initial_keys = read_keys + filler_keys
    results: list[dict[str, Any]] = []
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(
            f"prepare: {len(read_keys)} read blocks + {len(filler_keys)} fillers, "
            f"{initial_blocks * chunk_bytes / GIB:.2f} GiB",
            flush=True,
        )
        submit_and_wait(
            backend,
            initial_keys,
            source,
            args.timeout,
            batch_size=512,
        )
        for key in read_keys:
            if not backend.pin(key):
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
                        backend,
                        block_table,
                        read_keys,
                        write_keys,
                        source,
                        concurrency,
                        eviction,
                        initial_blocks,
                        args.write_batch_size,
                        args.timeout,
                        seed=trial,
                    )
                    result["trial"] = trial
                    results.append(result)
                    if not eviction:
                        cleanup_headroom_writes(
                            backend,
                            write_keys,
                            initial_blocks,
                            args.timeout,
                        )

        report = {
            "parameters": {
                "service_name": args.service_name,
                "dataset_bytes_per_direction": dataset_bytes,
                "chunk_bytes": chunk_bytes,
                "block_count": block_count,
                "initial_cache_bytes": initial_blocks * chunk_bytes,
                "headroom_cache_bytes": headroom_blocks * chunk_bytes,
                "cpu_size_bytes": int(args.cpu_size_gb * GIB),
                "concurrency": args.concurrency,
                "trials": args.trials,
                "write_batch_size": args.write_batch_size,
                "xds_io_unit_size": int(service_metadata.io_unit_size),
                "xds_service_capacity_bytes": int(service_metadata.capacity_bytes),
            },
            "results": results,
            "summary": summarize(results),
        }
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2))
        print(f"wrote {output_path}")
    finally:
        if source.is_valid():
            source.ref_count_down()
        backend.close()
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
