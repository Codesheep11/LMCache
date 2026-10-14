# SPDX-License-Identifier: Apache-2.0
"""Measure synchronous LocalDisk reads with and without CPU-cache eviction."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import argparse
import asyncio
import json
import random
import shutil
import statistics
import tempfile
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


GIB = 1024**3
MIB = 1024**2
DTYPE = torch.bfloat16
FORMAT = MemoryFormat.KV_2LTD


@dataclass
class DeviceStat:
    """Relevant counters from Linux's per-block-device stat file."""

    sectors_read: int
    read_ms: int
    io_ms: int
    weighted_io_ms: int


class CountingLocalCPUBackend(LocalCPUBackend):
    """LocalCPUBackend that records evictions made by allocate()."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eviction_count = 0
        self.eviction_ns = 0
        self.eviction_lock = threading.Lock()

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        """Remove objects and account for the time spent doing so."""
        start_ns = time.monotonic_ns()
        removed = super().batched_remove(keys, force=force)
        elapsed_ns = time.monotonic_ns() - start_ns
        with self.eviction_lock:
            self.eviction_count += removed
            self.eviction_ns += elapsed_ns
        return removed

    def reset_eviction_stats(self) -> None:
        """Reset eviction counters before a measured trial."""
        with self.eviction_lock:
            self.eviction_count = 0
            self.eviction_ns = 0

    def get_eviction_stats(self) -> tuple[int, int]:
        """Return the number of evictions and cumulative eviction time."""
        with self.eviction_lock:
            return self.eviction_count, self.eviction_ns


def start_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start the event loop required by LocalDiskBackend."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def make_shape(chunk_bytes: int) -> torch.Size:
    """Build a valid KV_2LTD shape with exactly ``chunk_bytes`` bytes."""
    bytes_per_token = 2 * DTYPE.itemsize
    if chunk_bytes % bytes_per_token:
        raise ValueError("chunk size must be divisible by four bytes")
    return torch.Size([2, 1, chunk_bytes // bytes_per_token, 1])


def make_metadata(shape: torch.Size) -> LMCacheMetadata:
    """Build model metadata consistent with the benchmark chunk shape."""
    return LMCacheMetadata(
        model_name="local_disk_cpu_pressure",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=DTYPE,
        kv_shape=(1, 2, shape[2], 1, 1),
        chunk_size=shape[2],
    )


def make_aligned_source(
    shape: torch.Size,
    alignment: int,
) -> tuple[TensorMemoryObj, torch.Tensor]:
    """Create one page-aligned source object used to prepare disk files."""
    num_bytes = shape.numel() * DTYPE.itemsize
    base = torch.empty(num_bytes + alignment, dtype=torch.uint8)
    offset = (-base.data_ptr()) % alignment
    raw_data = base[offset : offset + num_bytes]
    memory_obj = TensorMemoryObj(
        raw_data=raw_data,
        metadata=MemoryObjMetadata(
            shape=shape,
            dtype=DTYPE,
            address=0,
            phy_size=num_bytes,
            ref_count=1,
            pin_count=0,
            fmt=FORMAT,
            shapes=[shape],
            dtypes=[DTYPE],
        ),
        parent_allocator=None,
    )
    assert memory_obj.tensor is not None
    memory_obj.tensor.fill_(7)
    return memory_obj, base


def make_key(index: int) -> CacheEngineKey:
    """Create a deterministic benchmark cache key."""
    return CacheEngineKey("local_disk_cpu_pressure", 1, 0, index, DTYPE)


def disk_path(root: Path, key: CacheEngineKey) -> Path:
    """Return the LocalDiskBackend-compatible path for a cache key."""
    return root / f"{key.to_string().replace('/', '-')}.pt"


def prepare_dataset(
    backend: LocalDiskBackend,
    root: Path,
    keys: Sequence[CacheEngineKey],
    shape: torch.Size,
    alignment: int,
) -> int:
    """Create the raw KV files and publish their LocalDisk metadata."""
    source, keepalive = make_aligned_source(shape, alignment)
    del keepalive
    size = source.get_size()
    start = time.perf_counter()
    for index, key in enumerate(keys, start=1):
        path = disk_path(root, key)
        backend.write_file(source.byte_array, str(path))
        backend.insert_key(key, size, shape, DTYPE, FORMAT)
        if index % 128 == 0 or index == len(keys):
            elapsed = time.perf_counter() - start
            gib = index * size / GIB
            print(
                f"prepare: {index}/{len(keys)} files, {gib:.2f} GiB, "
                f"{gib / elapsed:.2f} GiB/s",
                flush=True,
            )
    return size


def fill_cpu_cache(
    cpu_backend: CountingLocalCPUBackend,
    shape: torch.Size,
) -> int:
    """Fill the CPU allocator completely with evictable hot-cache objects."""
    count = 0
    while True:
        memory_obj = cpu_backend.allocate(
            shape,
            DTYPE,
            FORMAT,
            eviction=False,
        )
        if memory_obj is None:
            break
        key = make_key(1_000_000_000 + count)
        cpu_backend.submit_put_task(key, memory_obj)
        memory_obj.ref_count_down()
        count += 1
    if count == 0:
        raise RuntimeError("CPU pool cannot hold one benchmark chunk")
    return count


def read_device_stat(device: str) -> DeviceStat | None:
    """Read Linux block statistics for one device, if it is visible."""
    stat_path = Path("/sys/class/block") / device / "stat"
    try:
        fields = [int(value) for value in stat_path.read_text().split()]
    except (FileNotFoundError, PermissionError):
        return None
    if len(fields) < 11:
        return None
    return DeviceStat(
        sectors_read=fields[2],
        read_ms=fields[3],
        io_ms=fields[9],
        weighted_io_ms=fields[10],
    )


def device_deltas(
    before: dict[str, DeviceStat | None],
    after: dict[str, DeviceStat | None],
    elapsed_sec: float,
) -> dict[str, dict[str, float]]:
    """Convert block-counter snapshots into per-trial device metrics."""
    metrics: dict[str, dict[str, float]] = {}
    elapsed_ms = elapsed_sec * 1000
    for device, start in before.items():
        end = after[device]
        if start is None or end is None:
            continue
        read_bytes = (end.sectors_read - start.sectors_read) * 512
        metrics[device] = {
            "read_gib": read_bytes / GIB,
            "read_gib_per_sec": read_bytes / GIB / elapsed_sec,
            "read_ms": float(end.read_ms - start.read_ms),
            "util_percent": (end.io_ms - start.io_ms) / elapsed_ms * 100,
            "average_queue_depth": (end.weighted_io_ms - start.weighted_io_ms)
            / elapsed_ms,
        }
    return metrics


def run_trial(
    backend: LocalDiskBackend,
    cpu_backend: CountingLocalCPUBackend,
    keys: list[CacheEngineKey],
    concurrency: int,
    pressure: bool,
    shape: torch.Size,
    devices: list[str],
    seed: int,
) -> dict[str, Any]:
    """Run one synchronous-read trial and return application/device metrics."""
    cpu_backend.clear()
    filler_count = fill_cpu_cache(cpu_backend, shape) if pressure else 0
    cpu_backend.reset_eviction_stats()

    ordered_keys = list(keys)
    random.Random(seed).shuffle(ordered_keys)

    def load_one(key: CacheEngineKey) -> tuple[int, int, int, int, bool]:
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
        # StorageManager writes successful disk loads back to the CPU hot cache.
        # Do this in both modes so eviction is the only behavioral difference.
        cpu_backend.submit_put_task(key, memory_obj)
        memory_obj.ref_count_down()
        return (
            size,
            profile["metadata_ns"],
            profile["allocate_ns"],
            profile["read_ns"],
            False,
        )

    device_before = {device: read_device_stat(device) for device in devices}
    start_ns = time.monotonic_ns()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        samples = list(executor.map(load_one, ordered_keys))
    end_ns = time.monotonic_ns()
    elapsed_sec = (end_ns - start_ns) / 1e9
    device_after = {device: read_device_stat(device) for device in devices}

    total_bytes = sum(sample[0] for sample in samples)
    misses = sum(sample[4] for sample in samples)
    evictions, eviction_ns = cpu_backend.get_eviction_stats()
    result = {
        "mode": "cpu_full_eviction" if pressure else "cpu_available",
        "concurrency": concurrency,
        "elapsed_sec": elapsed_sec,
        "bytes": total_bytes,
        "gib_per_sec": total_bytes / GIB / elapsed_sec,
        "files": len(keys),
        "allocation_failures": misses,
        "cpu_filler_count": filler_count,
        "cpu_hot_cache_count_after": len(cpu_backend.get_keys()),
        "cpu_evictions": evictions,
        "cpu_eviction_ms_sum": eviction_ns / 1e6,
        "metadata_ms_sum": sum(sample[1] for sample in samples) / 1e6,
        "allocate_ms_sum": sum(sample[2] for sample in samples) / 1e6,
        "read_ms_sum": sum(sample[3] for sample in samples) / 1e6,
        "devices": device_deltas(device_before, device_after, elapsed_sec),
    }
    print(
        f"{result['mode']}: qd={concurrency} "
        f"bandwidth={result['gib_per_sec']:.2f} GiB/s "
        f"evictions={evictions} failures={misses}",
        flush=True,
    )
    return result


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return median bandwidth and pressure slowdown for each concurrency."""
    summary = []
    concurrencies = sorted({result["concurrency"] for result in results})
    for concurrency in concurrencies:
        by_mode = {
            mode: [
                result["gib_per_sec"]
                for result in results
                if result["concurrency"] == concurrency and result["mode"] == mode
            ]
            for mode in ("cpu_available", "cpu_full_eviction")
        }
        available = statistics.median(by_mode["cpu_available"])
        pressure = statistics.median(by_mode["cpu_full_eviction"])
        summary.append(
            {
                "concurrency": concurrency,
                "cpu_available_median_gib_per_sec": available,
                "cpu_full_eviction_median_gib_per_sec": pressure,
                "pressure_over_available": pressure / available,
                "bandwidth_loss_percent": (1 - pressure / available) * 100,
            }
        )
    return summary


def parse_csv_ints(value: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def main() -> None:
    """Prepare a dataset and run the CPU-pressure comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disk-dir", required=True)
    parser.add_argument("--dataset-gb", type=float, default=8.0)
    parser.add_argument("--cpu-size-gb", type=float, default=8.0)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--concurrency", type=parse_csv_ints, default=[1, 2, 4, 8])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--odirect", action="store_true")
    parser.add_argument(
        "--devices",
        default="md127,nvme0n1,nvme1n1,nvme2n1,nvme3n1",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--keep-files", action="store_true")
    args = parser.parse_args()

    if args.dataset_gb <= 0 or args.cpu_size_gb <= 0 or args.chunk_mib <= 0:
        parser.error("dataset, CPU size, and chunk size must be positive")
    if args.trials <= 0:
        parser.error("trials must be positive")

    disk_parent = Path(args.disk_dir).resolve()
    disk_parent.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="lmcache-cpu-pressure-", dir=disk_parent))
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_bytes = args.chunk_mib * MIB
    num_files = max(1, int(args.dataset_gb * GIB) // chunk_bytes)
    dataset_bytes = num_files * chunk_bytes
    cpu_size_bytes = int(args.cpu_size_gb * GIB)
    if cpu_size_bytes < dataset_bytes:
        parser.error(
            "--cpu-size-gb must cover the dataset so the CPU-available "
            "case does not begin evicting partway through the trial"
        )
    shape = make_shape(chunk_bytes)
    metadata = make_metadata(shape)
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=shape[2],
        local_cpu=True,
        max_local_cpu_size=args.cpu_size_gb,
        lmcache_instance_id="local_disk_cpu_pressure",
    )
    config.local_disk = str(run_root)
    config.max_local_disk_size = max(args.dataset_gb * 2, 1.0)
    config.extra_config = {
        "use_odirect": args.odirect,
        "local_cpu.pinned_align_bytes": args.alignment,
    }

    loop, _loop_thread = start_loop()
    cpu_backend = CountingLocalCPUBackend(
        config=config,
        metadata=metadata,
        dst_device="cpu",
    )
    disk_backend = LocalDiskBackend(
        config=config,
        loop=loop,
        local_cpu_backend=cpu_backend,
        dst_device="cpu",
        metadata=metadata,
    )

    keys = [make_key(index) for index in range(num_files)]
    results: list[dict[str, Any]] = []
    try:
        actual_chunk_bytes = prepare_dataset(
            disk_backend,
            run_root,
            keys,
            shape,
            args.alignment,
        )
        for concurrency in args.concurrency:
            for trial in range(args.trials):
                pressure_order = (False, True) if trial % 2 == 0 else (True, False)
                for pressure in pressure_order:
                    result = run_trial(
                        disk_backend,
                        cpu_backend,
                        keys,
                        concurrency,
                        pressure,
                        shape,
                        [item for item in args.devices.split(",") if item],
                        seed=trial,
                    )
                    result["trial"] = trial
                    results.append(result)

        report = {
            "parameters": {
                "disk_dir": str(run_root),
                "dataset_bytes": dataset_bytes,
                "chunk_bytes": actual_chunk_bytes,
                "num_files": num_files,
                "cpu_size_bytes": cpu_size_bytes,
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
        cpu_backend.clear()
        cpu_backend.close()
        if not args.keep_files:
            shutil.rmtree(run_root)


if __name__ == "__main__":
    main()
