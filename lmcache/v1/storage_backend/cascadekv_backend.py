# SPDX-License-Identifier: Apache-2.0
"""CascadeKV block buffer, metadata tables, and rank-local xDS scheduler."""

# Standard
from collections import OrderedDict, deque
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Any, Callable, Optional, Sequence
import asyncio
import errno
import json
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.cache_controller.locks import RWLockWithTimeout
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
    XDSBufferAllocatorInterface,
    XDSIORegion,
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.transfer_result import BackendTransferResult

logger = init_logger(__name__)

_BLOCK_BUFFER_WAIT_SECS = 300.0
_BLOB_WAIT_SECS = 30.0
_IO_POLL_BURST = 64
_WRITE_ACTIVATE_BURST = 64
_IO_WAIT_SECS = 0.00005
_READ_BLOCK_WINDOW = 4
_H2D_SUBMIT_BATCH = 16
_IO_TIMEOUT_SECS = 30.0


def _is_xds_ring_full(error: BaseException) -> bool:
    """Recognize the SDK's current and errno-preserving ENOSPC forms."""
    return isinstance(error, OSError) and (
        error.errno == errno.ENOSPC or f"error code: {-errno.ENOSPC}" in str(error)
    )


def _latency_summary_us(samples_ns: Sequence[int]) -> dict[str, float | int]:
    """Summarize request-local profiling samples in microseconds."""
    if not samples_ns:
        return {"count": 0}
    ordered = sorted(samples_ns)

    def percentile(percent: int) -> float:
        index = (len(ordered) - 1) * percent // 100
        return ordered[index] / 1e3

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered) / 1e3,
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": ordered[-1] / 1e3,
    }


class _IOEngineProfile:
    """Lock-free counters written only by the StorageIOEngine owner thread."""

    def __init__(self) -> None:
        self.last_state_ns = time.monotonic_ns()
        self.qd = 0
        self.read_qd = 0
        self.active_tasks = 0
        self.active_reads = 0
        self.active_ns = 0
        self.read_active_ns = 0
        self.qd_time_ns = [0] * 256
        self.read_qd_time_ns = [0] * 256
        self.timings = {
            name: [0, 0, 0]
            for name in (
                "read_engine_queue",
                "poll_call",
                "completion_batch_processing",
                "read_descriptor_callback",
                "read_native_submit",
                "write_native_submit",
                "read_completion_to_refill",
            )
        }
        self.counters = {
            "poll_empty": 0,
            "poll_completions": 0,
            "poll_batches": 0,
            "ring_full": 0,
        }

    @staticmethod
    def _add_depth_time(values: list[int], depth: int, elapsed_ns: int) -> None:
        if depth >= len(values):
            values.extend([0] * (depth + 1 - len(values)))
        values[depth] += elapsed_ns

    def state(
        self, qd: int, read_qd: int, active_tasks: int, active_reads: int
    ) -> None:
        """Account time in the previous scheduler state, then publish the new one."""
        now_ns = time.monotonic_ns()
        elapsed_ns = now_ns - self.last_state_ns
        if self.active_tasks:
            self.active_ns += elapsed_ns
            self._add_depth_time(self.qd_time_ns, self.qd, elapsed_ns)
        if self.active_reads:
            self.read_active_ns += elapsed_ns
            self._add_depth_time(self.read_qd_time_ns, self.read_qd, elapsed_ns)
        self.last_state_ns = now_ns
        self.qd = qd
        self.read_qd = read_qd
        self.active_tasks = active_tasks
        self.active_reads = active_reads

    def observe(self, name: str, elapsed_ns: int) -> None:
        """Add one duration without retaining per-I/O Python objects."""
        metric = self.timings[name]
        metric[0] += 1
        metric[1] += elapsed_ns
        metric[2] = max(metric[2], elapsed_ns)

    def increment(self, name: str, count: int = 1) -> None:
        """Increment one event counter."""
        self.counters[name] += count

    def snapshot(self) -> dict[str, float | int]:
        """Return cumulative scalar metrics suitable for request logs."""
        result: dict[str, float | int] = {
            "engine_profile_active_ms": self.active_ns / 1e6,
            "engine_profile_read_active_ms": self.read_active_ns / 1e6,
        }
        if self.active_ns:
            result["engine_avg_qd"] = (
                sum(depth * elapsed for depth, elapsed in enumerate(self.qd_time_ns))
                / self.active_ns
            )
            result["engine_qd0_pct"] = self.qd_time_ns[0] * 100 / self.active_ns
        if self.read_active_ns:
            result["engine_read_avg_qd"] = (
                sum(
                    depth * elapsed
                    for depth, elapsed in enumerate(self.read_qd_time_ns)
                )
                / self.read_active_ns
            )
            result["engine_read_qd0_pct"] = (
                self.read_qd_time_ns[0] * 100 / self.read_active_ns
            )
            for threshold in (4, 8, 16, 32):
                result[f"engine_read_qd_lt{threshold}_pct"] = (
                    sum(self.read_qd_time_ns[:threshold]) * 100 / self.read_active_ns
                )
        for name, (count, total_ns, max_ns) in self.timings.items():
            result[f"engine_{name}_count"] = count
            result[f"engine_{name}_mean_us"] = total_ns / count / 1e3 if count else 0.0
            result[f"engine_{name}_max_us"] = max_ns / 1e3
            result[f"engine_{name}_total_ms"] = total_ns / 1e6
        result.update({f"engine_{key}": value for key, value in self.counters.items()})
        return result


class BlockMemory:
    """Own one complete CPU block and its zero-copy MemoryObj view."""

    def __init__(
        self,
        owner: TensorMemoryObj,
        block_size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        on_free: Callable[[], None],
    ) -> None:
        self.owner = owner
        self.block_size = block_size
        self.shape = shape
        self.dtype = dtype
        self.fmt = fmt
        self.logical_size = shape.numel() * dtype.itemsize
        self._on_free = on_free
        self._refs = 1
        self._pins = 0

    def acquire(self) -> None:
        """Add one logical owner reference."""
        with self.owner.lock:
            if self._refs <= 0:
                raise RuntimeError("acquiring released CascadeKV BlockMemory")
            self._refs += 1

    def release(self) -> None:
        """Drop one reference and free the physical allocation at zero."""
        free = False
        with self.owner.lock:
            self._refs -= 1
            if self._refs < 0:
                raise RuntimeError("CascadeKV BlockMemory reference underflow")
            free = self._refs == 0
        if free:
            self.owner.ref_count_down()
            self._on_free()

    def view(self) -> "BlockMemoryObj":
        """Return a logical view while retaining the physical block."""
        self.acquire()
        return BlockMemoryObj(self)

    def io_region(self, allocator: XDSBufferAllocatorInterface) -> XDSIORegion:
        """Return the complete physical block used by one xDS I/O."""
        owner_region = allocator.get_io_region(self.owner)
        return XDSIORegion(
            owner_region.buffer_ref,
            owner_region.offset,
            self.block_size,
        )

    def pin(self) -> bool:
        """Add a Buffer Table pin."""
        with self.owner.lock:
            self._pins += 1
        return True

    def unpin(self) -> bool:
        """Drop a Buffer Table pin if present."""
        with self.owner.lock:
            if self._pins == 0:
                return False
            self._pins -= 1
        return True

    def pin_count(self) -> int:
        """Return the current Buffer Table pin count."""
        with self.owner.lock:
            return self._pins

    def free(self, memory_obj: MemoryObj) -> None:
        """Release the block reference owned by a standard TensorMemoryObj view."""
        memory_obj.invalidate()
        self.release()


class BlockMemoryObj(TensorMemoryObj):
    """A standard LMCache MemoryObj view retaining one complete BlockMemory."""

    def __init__(self, block: BlockMemory) -> None:
        owner = block.owner
        super().__init__(
            owner.raw_data[: block.logical_size],
            MemoryObjMetadata(
                shape=block.shape,
                dtype=block.dtype,
                address=owner.metadata.address,
                phy_size=block.logical_size,
                ref_count=1,
                pin_count=0,
                fmt=block.fmt,
                shapes=[block.shape],
                dtypes=[block.dtype],
            ),
            parent_allocator=block,  # type: ignore[arg-type]
        )
        self.block = block

    def pin(self) -> bool:
        return False

    def unpin(self) -> bool:
        return False


class BlockBuffer:
    """Allocate complete blocks from the rank-local HugePage allocator."""

    def __init__(
        self,
        allocator: XDSBufferAllocatorInterface,
        block_size: int,
        timeout_secs: float = _BLOCK_BUFFER_WAIT_SECS,
    ) -> None:
        self.allocator = allocator
        self.block_size = block_size
        self.timeout_secs = timeout_secs
        self.capacity = int(allocator.size) // self.block_size
        if self.capacity <= 0:
            raise ValueError("CascadeKV Block Buffer cannot fit one block")
        self._condition = threading.Condition()
        self._in_use = 0
        self._reserved = 0
        self._waiting_batches = 0

    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        wait: bool = True,
    ) -> Optional[BlockMemory]:
        """Allocate one full block, optionally waiting for bounded pressure."""
        specs = [(shape, dtype, fmt)]
        if not self._reserve(specs, wait=wait, batch=False):
            return None
        return self._materialize(specs)[0]

    def allocate_many(
        self,
        specs: Sequence[tuple[torch.Size, torch.dtype, MemoryFormat]],
    ) -> list[BlockMemory]:
        """Atomically reserve a complete read batch before allocating it."""
        if not specs:
            return []
        self._reserve(specs, wait=True, batch=True)
        return self._materialize(specs)

    def _reserve(
        self,
        specs: Sequence[tuple[torch.Size, torch.dtype, MemoryFormat]],
        *,
        wait: bool,
        batch: bool,
    ) -> bool:
        count = len(specs)
        if count > self.capacity:
            raise ValueError(
                f"CascadeKV read needs {count} blocks, capacity is {self.capacity}"
            )
        if any(
            shape.numel() * dtype.itemsize > self.block_size
            for shape, dtype, _fmt in specs
        ):
            raise ValueError("KV payload exceeds CascadeKV blob size")
        deadline = time.monotonic() + self.timeout_secs
        with self._condition:
            self._waiting_batches += int(batch)
            try:
                while self._in_use + self._reserved + count > self.capacity or (
                    not batch and self._waiting_batches > 0
                ):
                    if not wait:
                        return False
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "CascadeKV Block Buffer allocation timed out "
                            f"(requested={count}, in_use={self._in_use}, "
                            f"reserved={self._reserved}, capacity={self.capacity}, "
                            f"waiters={self._waiting_batches})"
                        )
                    self._condition.wait(remaining)
                self._reserved += count
            finally:
                self._waiting_batches -= int(batch)
                self._condition.notify_all()
        return True

    def _materialize(
        self, specs: Sequence[tuple[torch.Size, torch.dtype, MemoryFormat]]
    ) -> list[BlockMemory]:
        blocks: list[BlockMemory] = []
        try:
            for shape, dtype, fmt in specs:
                blocks.append(self._allocate_reserved(shape, dtype, fmt))
        except BaseException:
            with self._condition:
                self._reserved -= len(specs) - len(blocks)
                self._condition.notify_all()
            for block in blocks:
                block.release()
            raise
        return blocks

    def _allocate_reserved(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
    ) -> BlockMemory:
        owner = self.allocator.allocate(
            [torch.Size([self.block_size])],
            [torch.uint8],
            fmt,
        )
        if owner is not None and not isinstance(owner, TensorMemoryObj):
            owner.ref_count_down()
            raise TypeError("CascadeKV allocator must return TensorMemoryObj")
        with self._condition:
            self._reserved -= 1
            if owner is not None:
                self._in_use += 1
            self._condition.notify_all()
        if owner is None:
            raise RuntimeError(
                "CascadeKV allocator rejected an internally reserved block"
            )
        return BlockMemory(
            owner,
            self.block_size,
            shape,
            dtype,
            fmt,
            self._release_slot,
        )

    def _release_slot(self) -> None:
        """Return one physical block and wake batched allocation waiters."""
        with self._condition:
            self._in_use -= 1
            if self._in_use < 0:
                raise RuntimeError("CascadeKV Block Buffer accounting underflow")
            self._condition.notify_all()

    def metrics(self) -> dict[str, int]:
        """Return allocation counters for pressure diagnostics."""
        with self._condition:
            return {
                "block_buffer_capacity": self.capacity,
                "block_buffer_in_use": self._in_use,
                "block_buffer_reserved": self._reserved,
                "block_buffer_waiters": self._waiting_batches,
            }


class BufferTable:
    """Map visible, D2H-complete keys to immutable BlockMemory objects."""

    def __init__(self) -> None:
        self._lock = RWLockWithTimeout()
        self._blocks: dict[CacheEngineKey, BlockMemory] = {}

    def __contains__(self, key: CacheEngineKey) -> bool:
        with self._lock.read_lock():
            return key in self._blocks

    def __len__(self) -> int:
        with self._lock.read_lock():
            return len(self._blocks)

    def lookup(self, key: CacheEngineKey) -> Optional[BlockMemory]:
        """Return an acquired block on hit."""
        with self._lock.read_lock():
            block = self._blocks.get(key)
            if block is not None:
                block.acquire()
            return block

    def publish(
        self, key: CacheEngineKey, block: BlockMemory, blocks: "BlockTable"
    ) -> bool:
        """Publish the first complete writer if neither table has the key."""
        with self._lock.write_lock():
            if key in self._blocks or blocks.contains(key):
                return False
            block.acquire()
            self._blocks[key] = block
            return True

    def commit(
        self,
        key: CacheEngineKey,
        block: BlockMemory,
        durable: "SSDBlock",
        blocks: "BlockTable",
    ) -> bool:
        """Publish SSD metadata before removing the identical Buffer entry."""
        with self._lock.read_lock():
            if self._blocks.get(key) is not block:
                return False
            durable.set_pins(block.pin_count())
            if not blocks.publish(key, durable):
                return False
        removed = self.remove_if_same(key, block)
        if removed is not None:
            removed.release()
        return True

    def remove_if_same(
        self, key: CacheEngineKey, block: BlockMemory
    ) -> Optional[BlockMemory]:
        """Remove key only when it still names the expected block."""
        with self._lock.write_lock():
            if self._blocks.get(key) is not block:
                return None
            return self._blocks.pop(key)

    def remove(self, key: CacheEngineKey) -> Optional[BlockMemory]:
        """Remove one pending block."""
        with self._lock.write_lock():
            return self._blocks.pop(key, None)

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin one pending block while serializing the transition."""
        with self._lock.write_lock():
            block = self._blocks.get(key)
            return block.pin() if block is not None else False

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin one pending block while serializing the transition."""
        with self._lock.write_lock():
            block = self._blocks.get(key)
            return block.unpin() if block is not None else False

    def is_pinned(self, key: CacheEngineKey) -> bool:
        """Return whether a pending block has any pins."""
        with self._lock.read_lock():
            block = self._blocks.get(key)
            return block is not None and block.pin_count() > 0

    def clear(self) -> list[BlockMemory]:
        """Detach and return every pending block."""
        with self._lock.write_lock():
            blocks = list(self._blocks.values())
            self._blocks.clear()
            return blocks


class SSDBlock:
    """Own one durable blob handle and its reader, pin, and LRU state."""

    def __init__(
        self,
        handle: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
    ) -> None:
        self._handle = handle
        self.shape = shape
        self.dtype = dtype
        self.fmt = fmt
        self._readers = 0
        self._pins = 0
        self._in_table = True
        self._last_access = time.monotonic_ns()
        self._touch_enqueued = False
        self._lock = threading.Lock()

    @property
    def blob_handle(self) -> int:
        """Return the live blob handle."""
        with self._lock:
            if self._handle <= 0:
                raise RuntimeError("CascadeKV SSD block no longer owns a blob")
            return self._handle

    def acquire_reader(self) -> tuple[bool, bool]:
        """Acquire a reader and report whether its deferred LRU touch is new."""
        with self._lock:
            if not self._in_table or self._handle <= 0:
                return False, False
            self._readers += 1
            return True, self._touch_locked()

    def release_reader(self) -> Optional[int]:
        """Drop a reader and return a detached handle ready for release."""
        with self._lock:
            if self._readers <= 0:
                raise RuntimeError("CascadeKV SSD reader reference underflow")
            self._readers -= 1
            if not self._in_table and self._readers == 0:
                return self._take_handle()
            return None

    def pin(self) -> bool:
        """Add one durable pin."""
        with self._lock:
            if not self._in_table:
                return False
            self._pins += 1
            return True

    def unpin(self) -> tuple[bool, bool]:
        """Drop one durable pin and make the block recent."""
        with self._lock:
            if self._pins == 0:
                return False, False
            self._pins -= 1
            return True, self._touch_locked()

    def is_pinned(self) -> bool:
        """Return whether this durable block has any pins."""
        with self._lock:
            return self._pins > 0

    def set_pins(self, pins: int) -> None:
        """Transfer pending pins during Buffer-to-Block publication."""
        with self._lock:
            self._pins = pins

    def consume_touch(self) -> Optional[int]:
        """Consume one coalesced LRU notification and return its latest time."""
        with self._lock:
            if not self._touch_enqueued:
                return None
            self._touch_enqueued = False
            if not self._in_table:
                return None
            return self._last_access

    def detach(self) -> Optional[int]:
        """Detach the table entry and return an immediately free handle."""
        with self._lock:
            self._in_table = False
            if self._readers == 0:
                return self._take_handle()
            return None

    def try_take_for_reuse(self) -> Optional[int]:
        """Detach an evictable block and transfer its handle to a writer."""
        with self._lock:
            if self._readers or self._pins or not self._in_table or self._handle <= 0:
                return None
            self._in_table = False
            return self._take_handle()

    def _touch_locked(self) -> bool:
        self._last_access = time.monotonic_ns()
        if self._touch_enqueued:
            return False
        self._touch_enqueued = True
        return True

    def _take_handle(self) -> Optional[int]:
        if self._handle <= 0:
            return None
        handle = self._handle
        self._handle = 0
        return handle


class BlockTable:
    """Map durable keys to independently synchronized SSDBlock objects."""

    def __init__(self) -> None:
        self._lock = RWLockWithTimeout()
        self._blocks: OrderedDict[CacheEngineKey, SSDBlock] = OrderedDict()
        self._touches: SimpleQueue[tuple[CacheEngineKey, SSDBlock]] = SimpleQueue()

    def __contains__(self, key: CacheEngineKey) -> bool:
        return self.contains(key)

    def __len__(self) -> int:
        with self._lock.read_lock():
            return len(self._blocks)

    def __getitem__(self, key: CacheEngineKey) -> SSDBlock:
        with self._lock.read_lock():
            return self._blocks[key]

    def contains(self, key: CacheEngineKey) -> bool:
        """Return whether durable metadata contains key."""
        with self._lock.read_lock():
            return key in self._blocks

    def acquire(self, key: CacheEngineKey) -> Optional[SSDBlock]:
        """Acquire one reader while the mapping is stable."""
        with self._lock.read_lock():
            block = self._blocks.get(key)
            if block is None:
                return None
            acquired, notify = block.acquire_reader()
            if not acquired:
                return None
            if notify:
                self._touches.put((key, block))
            return block

    def publish(self, key: CacheEngineKey, block: SSDBlock) -> bool:
        """Publish durable metadata if key is absent."""
        with self._lock.write_lock():
            if key in self._blocks:
                return False
            self._blocks[key] = block
            return True

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin a durable block without taking the table write lock."""
        with self._lock.read_lock():
            block = self._blocks.get(key)
            return block.pin() if block is not None else False

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin a durable block without taking the table write lock."""
        with self._lock.read_lock():
            block = self._blocks.get(key)
            if block is None:
                return False
            unpinned, notify = block.unpin()
            if notify:
                self._touches.put((key, block))
            return unpinned

    def is_pinned(self, key: CacheEngineKey) -> bool:
        """Return whether a durable block has any pins."""
        with self._lock.read_lock():
            block = self._blocks.get(key)
            return block is not None and block.is_pinned()

    def remove(self, key: CacheEngineKey) -> tuple[bool, Optional[int]]:
        """Detach key and return a handle if no reader remains."""
        with self._lock.write_lock():
            block = self._blocks.pop(key, None)
        return (False, None) if block is None else (True, block.detach())

    def evict_oldest(self) -> Optional[int]:
        """Remove the oldest evictable block from the maintained LRU order."""
        with self._lock.write_lock():
            self._drain_touches_locked()
            for _ in range(len(self._blocks)):
                key, block = next(iter(self._blocks.items()))
                handle = block.try_take_for_reuse()
                if handle is not None:
                    self._blocks.popitem(last=False)
                    return handle
                # A reader or pin makes this block newer than an idle victim.
                self._blocks.move_to_end(key)
            return None

    def clear(self) -> list[int]:
        """Detach all metadata and return immediately releasable handles."""
        with self._lock.write_lock():
            blocks = list(self._blocks.values())
            self._blocks.clear()
            while True:
                try:
                    self._touches.get_nowait()
                except Empty:
                    break
        return [handle for block in blocks if (handle := block.detach()) is not None]

    def _drain_touches_locked(self) -> None:
        """Apply coalesced read-side touches while already holding the write lock."""
        touched: list[tuple[int, CacheEngineKey, SSDBlock]] = []
        while True:
            try:
                key, block = self._touches.get_nowait()
            except Empty:
                break
            timestamp = block.consume_touch()
            if timestamp is not None:
                touched.append((timestamp, key, block))
        touched.sort(key=lambda item: item[0])
        for _timestamp, key, block in touched:
            if self._blocks.get(key) is block:
                self._blocks.move_to_end(key)


@dataclass(frozen=True)
class IODescriptor:
    """One xDS operation against a service-owned CPU region."""

    blob_handle: int
    region: XDSIORegion
    blob_offset: int
    size: int
    tag: int = -1


class IOTaskHandle:
    """Waitable handle for a task accepted by StorageIOEngine."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._error: Optional[BaseException] = None

    @property
    def done(self) -> bool:
        """Return whether the task has drained."""
        return self._event.is_set()

    @property
    def error(self) -> Optional[BaseException]:
        """Return the task error, if any."""
        return self._error

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for completion and raise the task error."""
        if not self._event.wait(timeout):
            raise TimeoutError("timed out waiting for CascadeKV I/O task")
        if self._error is not None:
            raise self._error

    def finish(self, error: Optional[BaseException]) -> None:
        """Complete the handle. Only StorageIOEngine should call this."""
        self._error = error
        self._event.set()


@dataclass(eq=False)
class _IOTask:
    operation: str
    descriptors: deque[IODescriptor]
    callback: Callable[[Optional[BaseException]], None]
    descriptor_callback: Optional[Callable[[IODescriptor], None]] = None
    window: int = 1
    inflight: int = 0
    error: Optional[BaseException] = None
    last_progress: Optional[float] = None
    queued_ns: int = 0


@dataclass(frozen=True)
class _InflightIO:
    task: _IOTask
    descriptor: IODescriptor


@dataclass
class _WriteCommand:
    region: XDSIORegion
    callback: Callable[[int, Optional[BaseException]], bool]
    deadline: float


@dataclass(frozen=True)
class _ReleaseCommand:
    handle: int


@dataclass(frozen=True)
class _PipelineEvent:
    """One SSD or H2D transition delivered to the owning request thread."""

    stage: str
    position: int = -1
    error: Optional[BaseException] = None
    timestamp_ns: int = 0
    submit_start_ns: int = 0
    host_submit_ns: int = 0
    event_record_ns: int = 0
    event_sync_ns: int = 0


@dataclass(frozen=True)
class _H2DCommand:
    """One host-to-device copy owned by the rank-local H2D engine."""

    position: int
    connector: Any
    memory_obj: MemoryObj
    start: int
    end: int
    kwargs: dict[str, Any]
    callback: Callable[[_PipelineEvent], None]


@dataclass(frozen=True)
class _H2DBatch:
    """CUDA event and commands that can retire together."""

    event: Any
    commands: list[_H2DCommand]
    submit_starts_ns: list[int]
    host_submit_ns: list[int]
    event_record_ns: int


class _H2DTransferEngine:
    """Sole rank-local producer for the shared CUDA load stream."""

    def __init__(self) -> None:
        self._commands: SimpleQueue[_H2DCommand] = SimpleQueue()
        self._retire: SimpleQueue[Optional[_H2DBatch]] = SimpleQueue()
        self._wake = threading.Event()
        self._accepting = True
        self._submitter = threading.Thread(
            target=self._submit_loop,
            name="CascadeKV-H2DSubmitter",
            daemon=True,
        )
        self._reclaimer = threading.Thread(
            target=self._retire_loop,
            name="CascadeKV-H2DReclaimer",
            daemon=True,
        )
        self._submitter.start()
        self._reclaimer.start()

    def submit(
        self,
        position: int,
        connector: Any,
        memory_obj: MemoryObj,
        start: int,
        end: int,
        kwargs: dict[str, Any],
        callback: Callable[[_PipelineEvent], None],
    ) -> None:
        """Queue one H2D copy and return without waiting for CUDA."""
        if not self._accepting:
            raise RuntimeError("CascadeKV H2D engine is closed")
        self._commands.put(
            _H2DCommand(
                position,
                connector,
                memory_obj,
                start,
                end,
                kwargs,
                callback,
            )
        )
        self._wake.set()

    def close(self) -> None:
        """Drain submitted H2D work and stop both owner threads."""
        self._accepting = False
        self._wake.set()
        self._submitter.join(_IO_TIMEOUT_SECS)
        if self._submitter.is_alive():
            raise TimeoutError("CascadeKV H2D submitter did not drain")
        self._reclaimer.join(_IO_TIMEOUT_SECS)
        if self._reclaimer.is_alive():
            raise TimeoutError("CascadeKV H2D reclaimer did not drain")

    def _submit_loop(self) -> None:
        while self._accepting or not self._commands.empty():
            try:
                first = self._commands.get_nowait()
            except Empty:
                self._wake.wait(0.001)
                self._wake.clear()
                continue
            batch = [first]
            while len(batch) < _H2D_SUBMIT_BATCH:
                try:
                    batch.append(self._commands.get_nowait())
                except Empty:
                    break
            connector = first.connector
            if any(command.connector is not connector for command in batch):
                error = RuntimeError("CascadeKV received multiple GPU connectors")
                for command in batch:
                    command.callback(
                        _PipelineEvent("h2d", command.position, error=error)
                    )
                continue

            submit_starts_ns: list[int] = []
            host_submit_ns: list[int] = []
            try:
                stream = getattr(connector, "load_stream", None)
                stream_context = (
                    torch.cuda.stream(stream) if stream is not None else nullcontext()
                )
                with stream_context:
                    for command in batch:
                        started_ns = time.monotonic_ns()
                        connector.to_gpu(
                            command.memory_obj,
                            command.start,
                            command.end,
                            **command.kwargs,
                        )
                        submit_starts_ns.append(started_ns)
                        host_submit_ns.append(time.monotonic_ns() - started_ns)
                    record_started_ns = time.monotonic_ns()
                    event = connector.record_transfer_event("load")
                    event_record_ns = time.monotonic_ns() - record_started_ns
                self._retire.put(
                    _H2DBatch(
                        event,
                        batch,
                        submit_starts_ns,
                        host_submit_ns,
                        event_record_ns,
                    )
                )
            except BaseException as error:
                for command in batch:
                    command.callback(
                        _PipelineEvent("h2d", command.position, error=error)
                    )
        self._retire.put(None)

    def _retire_loop(self) -> None:
        while (batch := self._retire.get()) is not None:
            sync_started_ns = time.monotonic_ns()
            try:
                batch.event.synchronize()
                error: Optional[BaseException] = None
            except BaseException as caught:
                error = caught
            completed_ns = time.monotonic_ns()
            sync_ns = completed_ns - sync_started_ns
            for index, command in enumerate(batch.commands):
                command.callback(
                    _PipelineEvent(
                        "h2d",
                        command.position,
                        error=error,
                        timestamp_ns=completed_ns,
                        submit_start_ns=batch.submit_starts_ns[index],
                        host_submit_ns=batch.host_submit_ns[index],
                        event_record_ns=batch.event_record_ns if index == 0 else 0,
                        event_sync_ns=sync_ns if index == 0 else 0,
                    )
                )


class StorageIOEngine:
    """Sole rank-local producer for xDS shared-memory submission rings."""

    def __init__(
        self,
        client: Any,
        block_table: BlockTable,
        io_timeout_secs: float = _IO_TIMEOUT_SECS,
        profile: bool = False,
        read_block_window: int = _READ_BLOCK_WINDOW,
    ) -> None:
        self.client = client
        self.block_table = block_table
        self.io_timeout_secs = io_timeout_secs
        metadata = client.metadata
        capacity = int(metadata.capacity_bytes)
        blob_size = int(metadata.blob_size_bytes)
        self.max_blocks = capacity // blob_size
        if self.max_blocks <= 0:
            raise ValueError("invalid CascadeKV xDS capacity")
        if read_block_window <= 0:
            raise ValueError("CascadeKV read block window must be positive")
        self.read_block_window = read_block_window
        self._leased_handles = 0
        self._commands: SimpleQueue[Any] = SimpleQueue()
        self._writes: deque[_WriteCommand] = deque()
        self._ready: deque[_IOTask] = deque()
        self._active: set[_IOTask] = set()
        self._requests: dict[int, _InflightIO] = {}
        self._read_inflight = 0
        self._active_reads = 0
        self._refill_ready_ns: deque[int] = deque()
        self._accepting = True
        self._fatal_error: Optional[BaseException] = None
        self._wake = threading.Event()
        self.peak_inflight = 0
        self.peak_active_tasks = 0
        self.completion_batches = 0
        self.completed_requests = 0
        self._profile = _IOEngineProfile() if profile else None
        self._thread = threading.Thread(
            target=self._run, name="CascadeKV-StorageIOEngine", daemon=True
        )
        self._thread.start()

    def submit_write(
        self,
        region: XDSIORegion,
        callback: Callable[[int, Optional[BaseException]], bool],
    ) -> None:
        """Queue one block write and return before handle allocation or I/O."""
        self._submit(
            _WriteCommand(
                region,
                callback,
                time.monotonic() + _BLOB_WAIT_SECS,
            )
        )

    def submit_read(
        self,
        descriptors: Sequence[IODescriptor],
        callback: Callable[[Optional[BaseException]], None],
        descriptor_callback: Optional[Callable[[IODescriptor], None]] = None,
    ) -> IOTaskHandle:
        """Queue a read whose request-private CPU block is already allocated."""
        handle = IOTaskHandle()

        def complete(error: Optional[BaseException]) -> None:
            try:
                callback(error)
            finally:
                handle.finish(error)

        self._submit(
            _IOTask(
                "read",
                deque(descriptors),
                complete,
                descriptor_callback,
                self.read_block_window,
                queued_ns=time.monotonic_ns(),
            )
        )
        return handle

    def release_blob(self, handle: int) -> None:
        """Return a detached handle through the sole xDS owner thread."""
        self._submit(_ReleaseCommand(handle), allow_closing=True)

    def check_health(self) -> None:
        """Raise an asynchronous scheduler failure on the calling thread."""
        if self._fatal_error is not None:
            raise RuntimeError(
                "CascadeKV StorageIOEngine failed"
            ) from self._fatal_error

    def metrics(self) -> dict[str, float | int]:
        """Return low-cost concurrency and handle counters."""
        result: dict[str, float | int] = {
            "engine_peak_qd": self.peak_inflight,
            "engine_peak_active_tasks": self.peak_active_tasks,
            "completion_batches": self.completion_batches,
            "completed_requests": self.completed_requests,
            "leased_handles": self._leased_handles,
        }
        if self._profile is not None:
            result.update(self._profile.snapshot())
        return result

    def close(self, timeout: float = _IO_TIMEOUT_SECS + _BLOB_WAIT_SECS) -> None:
        """Drain accepted work and stop the owner thread."""
        self._accepting = False
        self._wake.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("CascadeKV StorageIOEngine did not drain")
        self.check_health()

    def _submit(self, command: Any, allow_closing: bool = False) -> None:
        if not self._accepting and not allow_closing:
            raise RuntimeError("CascadeKV StorageIOEngine is closed")
        self.check_health()
        self._commands.put(command)
        self._wake.set()

    def _run(self) -> None:
        try:
            while (
                self._accepting
                or not self._commands.empty()
                or self._writes
                or self._active
            ):
                self._drain_commands()
                self._poll_io()
                self._expire_tasks()
                self._start_writes()
                self._submit_ready()
                if not self._requests:
                    self._wake.wait(0.0005)
                    self._wake.clear()
        except BaseException as error:
            logger.exception("CascadeKV StorageIOEngine stopped")
            self._fatal_error = error
            self._fail_all(error)

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except Empty:
                return
            if isinstance(command, _ReleaseCommand):
                self.client.release_blob(command.handle)
                self._leased_handles -= 1
            elif isinstance(command, _WriteCommand):
                self._writes.append(command)
            else:
                self._activate(command)

    def _start_writes(self) -> None:
        for _ in range(min(len(self._writes), _WRITE_ACTIVATE_BURST)):
            command = self._writes.popleft()
            handle = self._acquire_handle()
            if handle is None:
                if time.monotonic() >= command.deadline:
                    command.callback(
                        0, TimeoutError("CascadeKV SSD capacity wait timed out")
                    )
                else:
                    self._writes.append(command)
                    # One failed search proves that no block is currently reusable.
                    break
                continue

            def complete(
                error: Optional[BaseException],
                *,
                blob_handle: int = handle,
                callback: Callable[[int, Optional[BaseException]], bool] = (
                    command.callback
                ),
            ) -> None:
                accepted = callback(blob_handle, error)
                if error is not None or not accepted:
                    self.client.release_blob(blob_handle)
                    self._leased_handles -= 1

            self._activate(
                _IOTask(
                    "write",
                    deque(
                        [IODescriptor(handle, command.region, 0, command.region.size)]
                    ),
                    complete,
                )
            )

    def _acquire_handle(self) -> Optional[int]:
        if self._leased_handles < self.max_blocks:
            handle = int(self.client.get_blob(timeout=5.0))
            if handle <= 0:
                raise RuntimeError(f"xDS returned invalid blob handle {handle}")
            self._leased_handles += 1
            return handle
        return self.block_table.evict_oldest()

    def _activate(self, task: _IOTask) -> None:
        self._active.add(task)
        self._active_reads += int(task.operation == "read")
        self._profile_state()
        if self._profile is not None and task.queued_ns:
            self._profile.observe(
                f"{task.operation}_engine_queue",
                time.monotonic_ns() - task.queued_ns,
            )
        self.peak_active_tasks = max(self.peak_active_tasks, len(self._active))
        if task.descriptors:
            self._ready.append(task)
        else:
            self._finish_task(task)

    def _poll_io(self) -> None:
        if not self._requests:
            return
        poll_start_ns = time.monotonic_ns() if self._profile is not None else 0
        completions = self.client.wait_burst(_IO_POLL_BURST, _IO_WAIT_SECS)
        poll_end_ns = time.monotonic_ns() if self._profile is not None else 0
        if self._profile is not None:
            self._profile.observe("poll_call", poll_end_ns - poll_start_ns)
            self._profile.increment("poll_empty", int(not completions))
        if completions:
            self.completion_batches += 1
            self.completed_requests += len(completions)
            if self._profile is not None:
                self._profile.increment("poll_completions", len(completions))
                self._profile.increment("poll_batches")
        batch_start_ns = time.monotonic_ns() if completions and self._profile else 0
        refill_candidates: dict[_IOTask, list[int]] = {}
        for completion in completions:
            owner = self._requests.pop(int(completion.request_id), None)
            if owner is None:
                continue
            task, descriptor = owner.task, owner.descriptor
            self._read_inflight -= int(task.operation == "read")
            self._profile_state()
            task.inflight -= 1
            task.last_progress = time.monotonic()
            try:
                self._validate(completion, descriptor.size)
                if task.descriptor_callback:
                    callback_start_ns = (
                        time.monotonic_ns() if self._profile is not None else 0
                    )
                    task.descriptor_callback(descriptor)
                    if self._profile is not None:
                        self._profile.observe(
                            f"{task.operation}_descriptor_callback",
                            time.monotonic_ns() - callback_start_ns,
                        )
            except BaseException as error:
                task.error = task.error or error
                task.descriptors.clear()
            if task.error is None and task.descriptors:
                if self._profile is not None:
                    refill_candidates.setdefault(task, []).append(poll_end_ns)
                if task not in self._ready:
                    self._ready.append(task)
            elif task.inflight == 0:
                self._finish_task(task)
        if self._profile is not None:
            for task, timestamps in refill_candidates.items():
                self._refill_ready_ns.extend(timestamps[: len(task.descriptors)])
            if completions:
                self._profile.observe(
                    "completion_batch_processing",
                    time.monotonic_ns() - batch_start_ns,
                )

    def _submit_ready(self) -> None:
        """Fill each ready task's block window without a global task/QD cap."""
        while self._ready:
            task = self._ready.popleft()
            if task not in self._active or task.error is not None:
                continue
            while task.descriptors and task.inflight < task.window:
                descriptor = task.descriptors[0]
                try:
                    submit_start_ns = time.monotonic_ns() if self._profile else 0
                    request_id = self._submit_descriptor(task.operation, descriptor)
                except BaseException as error:
                    if _is_xds_ring_full(error):
                        if self._profile is not None:
                            self._profile.increment("ring_full")
                        self._ready.append(task)
                        return
                    task.error = task.error or error
                    task.descriptors.clear()
                    if task.inflight == 0:
                        self._finish_task(task)
                    break
                submit_end_ns = time.monotonic_ns() if self._profile else 0
                if self._profile is not None:
                    self._profile.observe(
                        f"{task.operation}_native_submit",
                        submit_end_ns - submit_start_ns,
                    )
                    if self._refill_ready_ns:
                        self._profile.observe(
                            "read_completion_to_refill",
                            submit_end_ns - self._refill_ready_ns.popleft(),
                        )
                task.descriptors.popleft()
                self._requests[request_id] = _InflightIO(task, descriptor)
                self._read_inflight += int(task.operation == "read")
                self._profile_state()
                task.inflight += 1
                task.last_progress = time.monotonic()
                self.peak_inflight = max(self.peak_inflight, len(self._requests))
            if not task.descriptors and task.inflight == 0:
                self._finish_task(task)

    def _submit_descriptor(self, operation: str, descriptor: IODescriptor) -> int:
        args = (
            descriptor.blob_handle,
            descriptor.region.buffer_ref,
            descriptor.region.offset,
            descriptor.blob_offset,
            descriptor.size,
        )
        submit = (
            self.client.submit_write
            if operation == "write"
            else self.client.submit_read
        )
        return int(submit(*args))

    def _expire_tasks(self) -> None:
        now = time.monotonic()
        for task in tuple(self._active):
            if (
                task.error is None
                and task.last_progress is not None
                and now - task.last_progress > self.io_timeout_secs
            ):
                task.error = TimeoutError("CascadeKV xDS task made no progress")
                task.descriptors.clear()
                if task.inflight == 0:
                    self._finish_task(task)

    def _finish_task(self, task: _IOTask) -> None:
        if task not in self._active:
            return
        self._active.remove(task)
        self._active_reads -= int(task.operation == "read")
        self._profile_state()
        task.callback(task.error)

    def _fail_all(self, error: BaseException) -> None:
        self._requests.clear()
        self._read_inflight = 0
        self._profile_state()
        for task in tuple(self._active):
            task.error = task.error or error
            task.descriptors.clear()
            task.inflight = 0
            self._finish_task(task)
        while self._writes:
            self._writes.popleft().callback(0, error)

    def _profile_state(self) -> None:
        if self._profile is not None:
            self._profile.state(
                len(self._requests),
                self._read_inflight,
                len(self._active),
                self._active_reads,
            )

    @staticmethod
    def _validate(completion: Any, expected_size: int) -> None:
        if int(completion.status) != 0:
            raise RuntimeError(
                f"xDS request failed: status={completion.status}, "
                f"error={completion.error_code}"
            )
        if int(completion.completed_bytes) != expected_size:
            raise RuntimeError(
                f"short xDS I/O: {completion.completed_bytes} != {expected_size}"
            )


@dataclass
class _DirectRead:
    """State owned by one direct-transfer block."""

    position: int
    start: int
    end: int
    memory_obj: Optional[MemoryObj]
    ssd_block: Optional[SSDBlock] = None
    block: Optional[BlockMemory] = None
    ready_ns: int = 0
    size: int = 0

    def __post_init__(self) -> None:
        if self.memory_obj is not None:
            self.size = self.memory_obj.get_size()

    def release_storage(self, io_engine: StorageIOEngine) -> None:
        if self.ssd_block is not None:
            handle = self.ssd_block.release_reader()
            if handle is not None:
                io_engine.release_blob(handle)
            self.ssd_block = None
        if self.block is not None:
            self.block.release()
            self.block = None


class CascadeKVBackend(AllocatorBackendInterface):
    """Store block-owned KV data through one Buffer Table and one Block Table."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: Any,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: XDSBufferAllocatorInterface,
        xds_client: Any,
        dst_device: str = "cuda",
        lmcache_worker: Optional[Any] = None,
        lookup_server: Optional[Any] = None,
    ) -> None:
        super().__init__(dst_device)
        del loop, lmcache_worker, lookup_server
        if not isinstance(memory_allocator, XDSBufferAllocatorInterface):
            raise TypeError("CascadeKV requires an xDS buffer allocator")
        if config.use_layerwise:
            raise ValueError("CascadeKV requires use_layerwise: false")
        self.config = config
        self.metadata = metadata
        self.memory_allocator = memory_allocator
        self.block_blob_layout = True
        self.blob_size = int(xds_client.metadata.blob_size_bytes)
        self.block_buffer = BlockBuffer(memory_allocator, self.blob_size)
        self.buffer_table = BufferTable()
        self.block_table = BlockTable()
        self._profiling = bool(
            config.get_extra_config_value("benchmark_transfer_metrics", False)
        )
        self.read_block_window = int(
            config.get_extra_config_value(
                "cascadekv_read_block_window", _READ_BLOCK_WINDOW
            )
        )
        if self.read_block_window <= 0:
            raise ValueError("cascadekv_read_block_window must be positive")
        self.io_engine = StorageIOEngine(
            xds_client,
            self.block_table,
            profile=self._profiling,
            read_block_window=self.read_block_window,
        )
        self.h2d_engine = _H2DTransferEngine()
        self._async_error: Optional[BaseException] = None
        self._closed = False

    def __str__(self) -> str:
        return "CascadeKVBackend"

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: Any
    ) -> MemoryAllocatorInterface:
        """Return the allocator created by the backend factory."""
        del config, metadata
        return self.memory_allocator  # type: ignore[return-value]

    def get_allocator_backend(self) -> "CascadeKVBackend":
        """Return this backend as the owner of Block Buffer allocation."""
        return self

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        """Return the rank-local HugePage allocator."""
        return self.memory_allocator  # type: ignore[return-value]

    def allocate(
        self,
        shapes: Any,
        dtypes: Any,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """Allocate one complete block."""
        del eviction
        shape, dtype = self._one_spec(shapes, dtypes)
        block = self.block_buffer.allocate(shape, dtype, fmt, wait=busy_loop)
        if block is None:
            return None
        view = block.view()
        block.release()
        return view

    def batched_allocate(
        self,
        shapes: Any,
        dtypes: Any,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        """Allocate one complete Block Buffer object per batch item."""
        del eviction, busy_loop
        shape, dtype = self._one_spec(shapes, dtypes)
        blocks = self.block_buffer.allocate_many([(shape, dtype, fmt)] * batch_size)
        views: list[MemoryObj] = []
        for block in blocks:
            views.append(block.view())
            block.release()
        return views

    def calculate_chunk_budget(self) -> int:
        """Return the number of complete blocks that fit in Block Buffer."""
        return max(1, int(self.memory_allocator.size) // self.blob_size)

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Return whether complete pending or durable data is visible."""
        self._ensure_open()
        block_key = self._block_key(key)
        found = block_key in self.buffer_table or self.block_table.contains(block_key)
        return self.pin(block_key) if found and pin else found

    def batched_contains(self, keys: list[CacheEngineKey], pin: bool = False) -> int:
        """Return the visible prefix length."""
        hits = 0
        for key in keys:
            if not self.contains(key, pin):
                break
            hits += 1
        return hits

    async def batched_async_contains(
        self, lookup_id: str, keys: list[CacheEngineKey], pin: bool = False
    ) -> int:
        """Provide the asynchronous metadata lookup interface."""
        del lookup_id
        return self.batched_contains(keys, pin)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Return whether a complete block is visible in Buffer Table."""
        return self._block_key(key) in self.buffer_table

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin a durable block, or a pending block during publication."""
        block_key = self._block_key(key)
        return self.block_table.pin(block_key) or self.buffer_table.pin(block_key)

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin a durable block, or a pending block during publication."""
        block_key = self._block_key(key)
        return self.block_table.unpin(block_key) or self.buffer_table.unpin(block_key)

    def _submit_block_write(
        self,
        key: CacheEngineKey,
        block: BlockMemory,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        def write_complete(handle: int, write_error: Optional[BaseException]) -> bool:
            committed = False
            if write_error is None and not self._closed:
                committed = self.buffer_table.commit(
                    key,
                    block,
                    SSDBlock(handle, block.shape, block.dtype, block.fmt),
                    self.block_table,
                )
            elif write_error is not None:
                self._async_error = write_error
            if not committed:
                removed = self.buffer_table.remove_if_same(key, block)
                if removed is not None:
                    removed.release()
            block.release()
            if committed and on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except BaseException:
                    logger.exception("CascadeKV write completion callback failed")
            return committed

        self.io_engine.submit_write(
            block.io_region(self.memory_allocator), write_complete
        )

    def batched_get_blocking(
        self, keys: Sequence[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        """Read Buffer first, then issue one xDS request per durable block."""
        self._ensure_open()
        if not keys:
            return []
        results: list[Optional[MemoryObj]] = [None] * len(keys)
        durable_entries: list[tuple[int, SSDBlock]] = []
        durable: list[tuple[int, SSDBlock, BlockMemory]] = []
        try:
            for position, original_key in enumerate(keys):
                key = self._block_key(original_key)
                pending = self.buffer_table.lookup(key)
                if pending is not None:
                    results[position] = pending.view()
                    pending.release()
                    continue
                ssd_block = self.block_table.acquire(key)
                if ssd_block is None:
                    self._release_objects(results)
                    self._release_ssd_blocks(durable_entries)
                    return []
                durable_entries.append((position, ssd_block))

            read_blocks = self.block_buffer.allocate_many(
                [
                    (ssd_block.shape, ssd_block.dtype, ssd_block.fmt)
                    for _position, ssd_block in durable_entries
                ]
            )
            durable = [
                (position, ssd_block, block)
                for (position, ssd_block), block in zip(
                    durable_entries, read_blocks, strict=True
                )
            ]
            durable_entries.clear()
            for position, _ssd_block, block in durable:
                results[position] = block.view()

            if durable:
                descriptors = [
                    IODescriptor(
                        ssd_block.blob_handle,
                        block.io_region(self.memory_allocator),
                        0,
                        self.blob_size,
                    )
                    for _position, ssd_block, block in durable
                ]
                io_start_ns = time.monotonic_ns()
                handle = self.io_engine.submit_read(
                    descriptors,
                    lambda _error: None,
                )
                handle.wait(_IO_TIMEOUT_SECS * 2)
                io_end_ns = time.monotonic_ns()
                if self.config.get_extra_config_value(
                    "benchmark_transfer_metrics", False
                ):
                    logger.info(
                        "[CASCADEKV_READ_METRICS] %s",
                        json.dumps(
                            {
                                "worker_id": self.metadata.worker_id,
                                "bytes": len(descriptors) * self.blob_size,
                                "block_count": len(durable),
                                "io_request_count": len(descriptors),
                                "start_ns": io_start_ns,
                                "end_ns": io_end_ns,
                                "seconds": (io_end_ns - io_start_ns) / 1e9,
                                "submission_policy": "ring_capacity",
                            },
                            separators=(",", ":"),
                        ),
                    )
                self._release_durable(durable)
                durable.clear()
            return results
        except BaseException:
            self._release_objects(results)
            self._release_durable(durable)
            self._release_ssd_blocks(durable_entries)
            raise

    def batched_get_to_gpu_blocking(
        self,
        keys: list[CacheEngineKey],
        starts: list[int],
        ends: list[int],
        gpu_connector: Any,
        **kwargs: Any,
    ) -> Optional[BackendTransferResult]:
        """Pipeline bounded SSD windows through the rank-local H2D engine."""
        self._ensure_open()
        if not (len(keys) == len(starts) == len(ends)):
            raise ValueError("CascadeKV direct-transfer inputs must have equal lengths")
        if not keys:
            return BackendTransferResult(
                completed=(),
                requested_bytes=0,
                completed_bytes=0,
                device=self.dst_device,
                window_size=1,
                window_count=0,
                peak_staging_bytes=0,
                storage_seconds=0.0,
                gpu_seconds=0.0,
                overlap_seconds=0.0,
                bank_wait_seconds=0.0,
                short_io_chunks=0,
                timeout_chunks=0,
            )

        transfer_start_ns = time.monotonic_ns()
        pipeline: SimpleQueue[_PipelineEvent] = SimpleQueue()
        entries: dict[int, _DirectRead] = {}
        pending_reads: deque[tuple[int, SSDBlock]] = deque()
        durable_positions: set[int] = set()
        read_handle: Optional[IOTaskHandle] = None
        completed = [False] * len(keys)
        ready_to_h2d_ns: list[int] = []
        storage_release_ns: list[int] = []
        h2d_host_submit_ns: list[int] = []
        event_record_ns: list[int] = []
        event_sync_ns: list[int] = []
        memory_release_ns: list[int] = []
        io_start_ns = 0
        io_end_ns = 0
        gpu_start_ns = 0
        gpu_end_ns = 0
        active_slots = 0
        io_request_count = 0

        def submit_h2d(entry: _DirectRead) -> None:
            release_started_ns = time.monotonic_ns()
            entry.release_storage(self.io_engine)
            storage_release_ns.append(time.monotonic_ns() - release_started_ns)
            memory_obj = entry.memory_obj
            assert memory_obj is not None
            self.h2d_engine.submit(
                entry.position,
                gpu_connector,
                memory_obj,
                entry.start,
                entry.end,
                kwargs,
                pipeline.put,
            )

        def schedule_reads() -> None:
            nonlocal active_slots, io_request_count, read_handle
            if active_slots or not pending_reads:
                return
            descriptors: list[IODescriptor] = []
            while pending_reads and len(descriptors) < self.read_block_window:
                position, ssd_block = pending_reads.popleft()
                block = self.block_buffer.allocate(
                    ssd_block.shape,
                    ssd_block.dtype,
                    ssd_block.fmt,
                )
                assert block is not None
                memory_obj = block.view()
                entry = _DirectRead(
                    position,
                    starts[position],
                    ends[position],
                    memory_obj,
                    ssd_block,
                    block,
                )
                entries[position] = entry
                descriptors.append(
                    IODescriptor(
                        ssd_block.blob_handle,
                        block.io_region(self.memory_allocator),
                        0,
                        self.blob_size,
                        tag=position,
                    )
                )
                durable_positions.add(position)
                active_slots += 1
                io_request_count += 1

            def block_ready(completed_descriptor: IODescriptor) -> None:
                pipeline.put(
                    _PipelineEvent(
                        "ssd",
                        completed_descriptor.tag,
                        timestamp_ns=time.monotonic_ns(),
                    )
                )

            def read_complete(error: Optional[BaseException]) -> None:
                if error is not None:
                    pipeline.put(_PipelineEvent("io", error=error))

            read_handle = self.io_engine.submit_read(
                descriptors,
                read_complete,
                descriptor_callback=block_ready,
            )

        try:
            for position, (original_key, start, end) in enumerate(
                zip(keys, starts, ends, strict=True)
            ):
                key = self._block_key(original_key)
                pending = self.buffer_table.lookup(key)
                if pending is not None:
                    memory_obj = pending.view()
                    pending.release()
                    entries[position] = _DirectRead(
                        position, start, end, memory_obj, ready_ns=time.monotonic_ns()
                    )
                    continue
                ssd_block = self.block_table.acquire(key)
                if ssd_block is not None:
                    pending_reads.append((position, ssd_block))

            target_count = len(entries) + len(pending_reads)
            for entry in tuple(entries.values()):
                submit_h2d(entry)
            io_start_ns = time.monotonic_ns()
            schedule_reads()

            completed_count = 0
            while completed_count < target_count:
                try:
                    event = pipeline.get(timeout=_IO_TIMEOUT_SECS * 2)
                except Empty as error:
                    raise TimeoutError(
                        "timed out waiting for CascadeKV transfer completion"
                    ) from error
                if event.error is not None:
                    raise event.error
                entry = entries[event.position]
                if event.stage == "ssd":
                    entry.ready_ns = event.timestamp_ns
                    io_end_ns = max(io_end_ns, event.timestamp_ns)
                    submit_h2d(entry)
                    continue
                if event.stage != "h2d":
                    raise RuntimeError(f"unknown CascadeKV stage {event.stage}")

                gpu_start_ns = (
                    event.submit_start_ns
                    if gpu_start_ns == 0
                    else min(gpu_start_ns, event.submit_start_ns)
                )
                gpu_end_ns = max(gpu_end_ns, event.timestamp_ns)
                ready_to_h2d_ns.append(event.submit_start_ns - entry.ready_ns)
                h2d_host_submit_ns.append(event.host_submit_ns)
                if event.event_record_ns:
                    event_record_ns.append(event.event_record_ns)
                if event.event_sync_ns:
                    event_sync_ns.append(event.event_sync_ns)
                release_started_ns = time.monotonic_ns()
                memory_obj = entry.memory_obj
                assert memory_obj is not None
                memory_obj.ref_count_down()
                entry.memory_obj = None
                memory_release_ns.append(time.monotonic_ns() - release_started_ns)
                completed[event.position] = True
                completed_count += 1
                if event.position in durable_positions:
                    active_slots -= 1
                    schedule_reads()

            if io_end_ns == 0:
                io_end_ns = io_start_ns

            requested_bytes = sum(entry.size for entry in entries.values())
            completed_bytes = sum(
                entries[position].size
                for position, outcome in enumerate(completed)
                if outcome and position in entries
            )
            transfer_end_ns = time.monotonic_ns()
            storage_seconds = max(0, io_end_ns - io_start_ns) / 1e9
            gpu_seconds = (
                max(0, gpu_end_ns - gpu_start_ns) / 1e9 if gpu_start_ns else 0.0
            )
            pipeline_start_ns = min(
                value for value in (io_start_ns, gpu_start_ns) if value > 0
            )
            pipeline_end_ns = max(io_end_ns, gpu_end_ns)
            pipeline_seconds = max(0, pipeline_end_ns - pipeline_start_ns) / 1e9
            overlap_seconds = max(0.0, storage_seconds + gpu_seconds - pipeline_seconds)
            if self.config.get_extra_config_value("benchmark_transfer_metrics", False):
                logger.info(
                    "[CASCADEKV_DIRECT_READ_METRICS] %s",
                    json.dumps(
                        {
                            "worker_id": self.metadata.worker_id,
                            "req_id": str(kwargs.get("req_id", "unspecified")),
                            "bytes": completed_bytes,
                            "block_count": sum(completed),
                            "io_request_count": io_request_count,
                            "transfer_start_ns": transfer_start_ns,
                            "transfer_end_ns": transfer_end_ns,
                            "io_start_ns": io_start_ns,
                            "io_end_ns": io_end_ns,
                            "gpu_start_ns": gpu_start_ns,
                            "gpu_end_ns": gpu_end_ns,
                            "storage_seconds": storage_seconds,
                            "gpu_seconds": gpu_seconds,
                            "overlap_seconds": overlap_seconds,
                            "ready_to_h2d_mean_us": (
                                sum(ready_to_h2d_ns) / len(ready_to_h2d_ns) / 1e3
                                if ready_to_h2d_ns
                                else 0.0
                            ),
                            "ready_to_h2d_max_us": (
                                max(ready_to_h2d_ns) / 1e3 if ready_to_h2d_ns else 0.0
                            ),
                            "storage_release_us": _latency_summary_us(
                                storage_release_ns
                            ),
                            "h2d_host_submit_us": _latency_summary_us(
                                h2d_host_submit_ns
                            ),
                            "event_record_us": _latency_summary_us(event_record_ns),
                            "event_sync_us": _latency_summary_us(event_sync_ns),
                            "memory_release_us": _latency_summary_us(memory_release_ns),
                            "request_io_window": self.read_block_window,
                            **self.io_engine.metrics(),
                            "submission_policy": "ring_capacity",
                        },
                        separators=(",", ":"),
                    ),
                )
            return BackendTransferResult(
                completed=tuple(completed),
                requested_bytes=requested_bytes,
                completed_bytes=completed_bytes,
                device=self.dst_device,
                window_size=max(1, len(entries)),
                window_count=1 if entries else 0,
                peak_staging_bytes=min(io_request_count, self.read_block_window)
                * self.blob_size,
                storage_seconds=storage_seconds,
                gpu_seconds=gpu_seconds,
                overlap_seconds=overlap_seconds,
                bank_wait_seconds=0.0,
                short_io_chunks=0,
                timeout_chunks=0,
                logical_chunk_count=len(keys),
                read_slot_count=self.block_buffer.capacity,
                io_queue_depth=0,
                peak_inflight_io=min(self.io_engine.peak_inflight, io_request_count),
            )
        finally:
            if read_handle is not None and not read_handle.done:
                try:
                    read_handle.wait()
                except BaseException:
                    pass
            for entry in entries.values():
                entry.release_storage(self.io_engine)
                memory_obj = entry.memory_obj
                if memory_obj is not None and memory_obj.is_valid():
                    memory_obj.ref_count_down()
            self._release_ssd_blocks(tuple(pending_reads))

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: list[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Publish whole blocks and submit one asynchronous SSD write each."""
        del transfer_spec
        self._ensure_open()
        if len(keys) != len(objs):
            raise ValueError("CascadeKV keys and objects must have equal length")
        for original_key, obj in zip(keys, objs, strict=True):
            if not isinstance(obj, BlockMemoryObj):
                raise TypeError("CascadeKV writes require Block Buffer objects")
            key = self._block_key(original_key)
            block = obj.block
            block.acquire()
            if not self.buffer_table.publish(key, block, self.block_table):
                block.release()
                continue
            self._submit_block_write(key, block, on_complete_callback)

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: list[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Submit a one-layer compatibility write without extra scheduling."""
        self.batched_submit_put_task(keys, objs, transfer_spec, on_complete_callback)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Return one object for a one-layer backend."""
        result = self.batched_get_blocking([key])
        return result[0] if result else None

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Provide the compatibility coroutine around blocking reads."""
        del lookup_id, transfer_spec
        return [obj for obj in self.batched_get_blocking(keys) if obj is not None]

    def get_non_blocking(
        self, key: CacheEngineKey, location: Optional[str] = None
    ) -> Optional[Future]:
        """Return None because CascadeKV does not implement legacy prefetch."""
        del key, location
        return None

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Submit one layer for a one-layer backend."""
        self.batched_submit_put_task([key], [memory_obj], None, on_complete_callback)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Detach both metadata entries and release resources asynchronously."""
        block_key = self._block_key(key)
        if not force and (
            self.block_table.is_pinned(block_key)
            or self.buffer_table.is_pinned(block_key)
        ):
            return False
        pending = self.buffer_table.remove(block_key)
        found, handle = self.block_table.remove(block_key)
        if pending is not None:
            pending.release()
        if handle is not None:
            self.io_engine.release_blob(handle)
        return pending is not None or found

    def touch_cache(self) -> None:
        """Do nothing because reader acquisition updates durable LRU time."""

    def metrics(self) -> dict[str, int]:
        """Return rank-local scheduler and table counters."""
        return {
            **self.io_engine.metrics(),
            **self.block_buffer.metrics(),
            "buffer_blocks": len(self.buffer_table),
            "durable_blocks": len(self.block_table),
        }

    def close(self) -> None:
        """Reject new work, drain accepted work, and release rank resources."""
        if self._closed:
            return
        self._closed = True
        for block in self.buffer_table.clear():
            block.release()
        for handle in self.block_table.clear():
            self.io_engine.release_blob(handle)
        self.io_engine.close()
        self.h2d_engine.close()
        self.memory_allocator.close()

    def _release_durable(
        self, durable: Sequence[tuple[int, SSDBlock, BlockMemory]]
    ) -> None:
        for _position, ssd_block, block in durable:
            handle = ssd_block.release_reader()
            if handle is not None:
                self.io_engine.release_blob(handle)
            block.release()

    def _release_ssd_blocks(self, durable: Sequence[tuple[int, SSDBlock]]) -> None:
        for _position, ssd_block in durable:
            handle = ssd_block.release_reader()
            if handle is not None:
                self.io_engine.release_blob(handle)

    @staticmethod
    def _release_objects(objects: Sequence[Optional[MemoryObj]]) -> None:
        for memory_obj in objects:
            if memory_obj is not None and memory_obj.is_valid():
                memory_obj.ref_count_down()

    @staticmethod
    def _one_spec(shapes: Any, dtypes: Any) -> tuple[torch.Size, torch.dtype]:
        if isinstance(shapes, torch.Size):
            shape = shapes
        elif isinstance(shapes, list) and len(shapes) == 1:
            shape = shapes[0]
        else:
            raise TypeError("CascadeKV requires exactly one logical tensor shape")
        if isinstance(dtypes, torch.dtype):
            dtype = dtypes
        elif isinstance(dtypes, list) and len(dtypes) == 1:
            dtype = dtypes[0]
        else:
            raise TypeError("CascadeKV requires exactly one logical tensor dtype")
        if not isinstance(shape, torch.Size) or not isinstance(dtype, torch.dtype):
            raise TypeError("invalid CascadeKV tensor shape or dtype")
        return shape, dtype

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("CascadeKV backend is closed")
        if self._async_error is not None:
            raise RuntimeError("CascadeKV asynchronous operation failed") from (
                self._async_error
            )
        self.io_engine.check_health()

    @staticmethod
    def _block_key(key: CacheEngineKey) -> CacheEngineKey:
        if not isinstance(key, LayerCacheEngineKey):
            return key
        return CacheEngineKey(
            key.model_name,
            key.world_size,
            key.worker_id,
            key.chunk_hash,
            key.dtype,
            key.request_configs,
        )
