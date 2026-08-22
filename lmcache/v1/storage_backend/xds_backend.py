# SPDX-License-Identifier: Apache-2.0
"""Direct SSD-to-GPU LMCache backend using an external xDS service."""

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Callable, Optional, Sequence
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj, TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.cascadekv_backend import CascadeKVBackend
from lmcache.v1.storage_backend.transfer_result import (
    BackendTransferResult,
    BackendWriteSubmissionResult,
)
from lmcache.v1.storage_backend.xds_read_ring import XDSReadRingPump
from lmcache.v1.storage_backend.xds_write_scheduler import (
    LeaseState,
    SchedulerSnapshot,
    StagingLease,
    XDSAsyncWriteScheduler,
)
from lmcache.v1.xds_utils import XDSBlobMetadata, align_to_io_unit


logger = init_logger(__name__)


class XDSBackend(CascadeKVBackend):
    """Allocator/storage backend whose registered pool resides on the GPU."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        xds_client: Any,
        dst_device: str = "cuda",
    ) -> None:
        AllocatorBackendInterface.__init__(self, dst_device=dst_device)
        self.config = config
        self.metadata = metadata
        self.client = xds_client
        self._closed = False
        self._entries: OrderedDict[CacheEngineKey, XDSBlobMetadata] = OrderedDict()
        self._entries_lock = threading.RLock()
        self._put_tasks: set[CacheEngineKey] = set()
        self._pending_objects: dict[CacheEngineKey, MemoryObj] = {}
        self._put_tasks_lock = threading.RLock()
        self._async_admission_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._keys_in_request: list[CacheEngineKey] = []

        service_metadata = xds_client.metadata
        self.io_unit_size = int(service_metadata.io_unit_size)
        self.blob_size = align_to_io_unit(
            int(service_metadata.blob_size_bytes),
            self.io_unit_size,
        )
        self.registered_size = int(service_metadata.gpu_buffer_size_bytes)
        if self.registered_size < self.blob_size:
            raise ValueError(
                "xDS registered GPU pool must hold at least one blob: "
                f"pool={self.registered_size}, blob={self.blob_size}"
            )
        if config.xds_windowed_io:
            required_size = (
                config.xds_read_slot_count + config.xds_write_slot_count
            ) * self.blob_size
            if self.registered_size < required_size:
                raise ValueError(
                    f"xDS windowed I/O requires {required_size} bytes: "
                    f"pool={self.registered_size}, blob={self.blob_size}, "
                    f"read_slots={config.xds_read_slot_count}, "
                    f"write_slots={config.xds_write_slot_count}"
                )
        if self.registered_size % self.io_unit_size != 0:
            raise ValueError(
                "xDS registered GPU pool must be I/O-unit aligned: "
                f"pool={self.registered_size}, io_unit={self.io_unit_size}"
            )

        self.gpu_tensor = torch.empty(
            self.registered_size,
            dtype=torch.uint8,
            device=dst_device,
        )
        self.registered_base_ptr = self.gpu_tensor.data_ptr()
        self.memory_allocator = TensorMemoryAllocator(
            self.gpu_tensor,
            align_bytes=self.io_unit_size,
        )
        self.buffer_ref = int(
            xds_client.register_gpu_buffer(
                self.registered_base_ptr,
                self.registered_size,
            )
        )
        if self.buffer_ref <= 0:
            raise RuntimeError(
                f"xDS returned an invalid GPU buffer reference {self.buffer_ref}"
            )

        # The inherited service I/O state machine uses these generic names.
        self.staging_tensor = self.gpu_tensor
        self.staging_base_ptr = self.registered_base_ptr
        self.staging_size = self.registered_size
        self.staging_buffer_ref = self.buffer_ref

        self._write_scheduler: Optional[XDSAsyncWriteScheduler] = None
        if config.xds_async_write:
            self._write_scheduler = XDSAsyncWriteScheduler(
                write_budget_bytes=config.xds_write_slot_count * self.blob_size,
                submit_write=self._submit_async_lease,
                commit=self._commit_async_lease,
                rollback=self._rollback_async_lease,
            )

    def calculate_chunk_budget(self) -> int:
        return self.registered_size // self.blob_size

    def _make_transfer_result(
        self,
        completed: Sequence[bool],
        *,
        requested_bytes: int,
        completed_bytes: int,
        window_size: int,
        window_count: int,
        peak_staging_bytes: int,
        storage_seconds: float,
        gpu_seconds: float,
        slot_wait_seconds: float,
        short_io_chunks: int,
        peak_inflight_io: int,
    ) -> BackendTransferResult:
        return BackendTransferResult(
            completed=tuple(completed),
            requested_bytes=requested_bytes,
            completed_bytes=completed_bytes,
            device=self.dst_device,
            window_size=window_size,
            window_count=window_count,
            peak_staging_bytes=peak_staging_bytes,
            storage_seconds=storage_seconds,
            gpu_seconds=gpu_seconds,
            overlap_seconds=0.0,
            bank_wait_seconds=slot_wait_seconds,
            short_io_chunks=short_io_chunks,
            timeout_chunks=0,
            logical_chunk_count=len(completed),
            read_slot_count=self.config.xds_read_slot_count,
            write_slot_count=self.config.xds_write_slot_count,
            io_queue_depth=self.config.xds_io_queue_depth,
            peak_inflight_io=peak_inflight_io,
            slot_wait_seconds=slot_wait_seconds,
        )

    def _write_window(
        self,
        task_infos: Sequence[dict[str, Any]],
        objects: Sequence[MemoryObj],
    ) -> tuple[list[bool], int, int]:
        """Persist one slot-sized window with bounded xDS queue depth."""
        if len(task_infos) != len(objects):
            raise ValueError("xDS write window inputs must have equal lengths")
        if not task_infos:
            return [], 0, 0

        items: list[dict[str, Any]] = []
        released_handles: set[int] = set()
        committed_handles: set[int] = set()
        try:
            for index, (info, memory_obj) in enumerate(
                zip(task_infos, objects, strict=True)
            ):
                blob_handle = int(self.client.get_blob(timeout=5.0))
                if blob_handle <= 0:
                    raise RuntimeError(
                        f"xDS returned an invalid blob handle {blob_handle}"
                    )
                try:
                    offset, size = self._buffer_descriptor(memory_obj)
                except BaseException:
                    self.client.release_blob(blob_handle)
                    released_handles.add(blob_handle)
                    raise
                items.append(
                    {
                        "index": index,
                        "info": info,
                        "memory_obj": memory_obj,
                        "blob_handle": blob_handle,
                        "offset": offset,
                        "size": size,
                    }
                )

            pump = XDSReadRingPump(
                queue_depth=self.config.xds_io_queue_depth,
                submit=lambda item: int(
                    self.client.submit_write(
                        item["blob_handle"],
                        self.buffer_ref,
                        item["offset"],
                        0,
                        item["size"],
                    )
                ),
                wait_any=self.client.wait_any_result,
                timeout=30.0,
            )
            for item in items:
                pump.put(item)
            pump.close_input()
            with self._io_lock:
                pump.run()

            outcomes = [False] * len(items)
            completed_bytes = 0
            short_io_chunks = 0
            first_error: Optional[BaseException] = None
            for _ in items:
                ring_completion = pump.get_completion()
                if ring_completion.error is not None:
                    if first_error is None:
                        first_error = ring_completion.error
                    continue
                item = ring_completion.item
                completion = ring_completion.completion
                status = int(completion.status)
                completed_size = int(completion.completed_bytes)
                if status == 0 and completed_size == item["size"]:
                    metadata = XDSBlobMetadata(
                        blob_handle=item["blob_handle"],
                        size=item["size"],
                        shapes=item["memory_obj"].get_shapes(),
                        dtypes=item["memory_obj"].get_dtypes(),
                        fmt=item["memory_obj"].get_memory_format(),
                    )
                    with self._entries_lock:
                        self._entries[item["info"]["key"]] = metadata
                    committed_handles.add(item["blob_handle"])
                    outcomes[item["index"]] = True
                    completed_bytes += item["size"]
                else:
                    self.client.release_blob(item["blob_handle"])
                    released_handles.add(item["blob_handle"])
                    if status == 0:
                        short_io_chunks += 1

            if first_error is None and pump.error is not None:
                first_error = pump.error
            if first_error is not None:
                raise first_error
            return outcomes, completed_bytes, short_io_chunks
        except BaseException:
            for item in items:
                blob_handle = item["blob_handle"]
                if (
                    blob_handle not in committed_handles
                    and blob_handle not in released_handles
                ):
                    self.client.release_blob(blob_handle)
                    released_handles.add(blob_handle)
            raise

    def batched_get_to_gpu_blocking(
        self,
        keys: list[CacheEngineKey],
        starts: list[int],
        ends: list[int],
        gpu_connector: Any,
        **kwargs: Any,
    ) -> Optional[BackendTransferResult]:
        """Read a logical batch through a bounded registered-GPU slot ring."""
        if not self.config.xds_windowed_io:
            return None
        if not (len(keys) == len(starts) == len(ends)):
            raise ValueError("xDS streaming read inputs must have equal lengths")

        completed = [False] * len(keys)
        pending_infos: list[dict[str, Any]] = []
        pending_indexes: set[int] = set()
        with self._put_tasks_lock:
            for index, (key, start, end) in enumerate(
                zip(keys, starts, ends, strict=True)
            ):
                memory_obj = self._pending_objects.get(key)
                if memory_obj is None:
                    continue
                memory_obj.ref_count_up()
                pending_indexes.add(index)
                pending_infos.append(
                    {
                        "index": index,
                        "memory_obj": memory_obj,
                        "start": start,
                        "end": end,
                        "size": memory_obj.get_physical_size(),
                    }
                )
        task_infos: list[dict[str, Any]] = []
        for index, (key, start, end) in enumerate(
            zip(keys, starts, ends, strict=True)
        ):
            if index in pending_indexes:
                continue
            with self._entries_lock:
                blob_metadata = self._entries.get(key)
                if blob_metadata is None:
                    continue
                blob_metadata.pin()
                self._entries.move_to_end(key)
            task_infos.append(
                {
                    "index": index,
                    "metadata": blob_metadata,
                    "start": start,
                    "end": end,
                }
            )

        requested_bytes = sum(info["size"] for info in pending_infos) + sum(
            info["metadata"].size for info in task_infos
        )
        pending_completed_bytes = 0
        pending_gpu_seconds = 0.0
        pending_wait_seconds = 0.0
        try:
            for info in pending_infos:
                gpu_start = time.perf_counter()
                gpu_connector.to_gpu(
                    info["memory_obj"],
                    info["start"],
                    info["end"],
                    **kwargs,
                )
                event = gpu_connector.record_transfer_event("load")
                wait_start = time.perf_counter()
                event.synchronize()
                pending_wait_seconds += time.perf_counter() - wait_start
                pending_gpu_seconds += time.perf_counter() - gpu_start
                completed[info["index"]] = True
                pending_completed_bytes += info["size"]
        except BaseException:
            for info in task_infos:
                with self._entries_lock:
                    info["metadata"].unpin()
            raise
        finally:
            for info in pending_infos:
                info["memory_obj"].ref_count_down()

        if not task_infos:
            return self._make_transfer_result(
                completed,
                requested_bytes=requested_bytes,
                completed_bytes=pending_completed_bytes,
                window_size=self.config.xds_read_window_size,
                window_count=0,
                peak_staging_bytes=0,
                storage_seconds=0.0,
                gpu_seconds=pending_gpu_seconds,
                slot_wait_seconds=pending_wait_seconds,
                short_io_chunks=0,
                peak_inflight_io=0,
            )

        pump = XDSReadRingPump(
            queue_depth=self.config.xds_io_queue_depth,
            submit=lambda entry: int(
                self.client.submit_read(
                    entry["metadata"].blob_handle,
                    self.buffer_ref,
                    entry["offset"],
                    0,
                    entry["size"],
                )
            ),
            wait_any=self.client.wait_any_result,
            timeout=30.0,
        )

        def run_pump() -> None:
            with self._io_lock:
                pump.run()

        pump_thread = threading.Thread(target=run_pump, name="xds-read-ring")
        live_entries: dict[int, dict[str, Any]] = {}
        next_task = 0
        input_closed = False
        peak_staging_bytes = 0
        submitted_count = 0
        completed_bytes = pending_completed_bytes
        short_io_chunks = 0
        storage_seconds = 0.0
        gpu_seconds = pending_gpu_seconds
        slot_wait_seconds = pending_wait_seconds

        def release_entry(entry: dict[str, Any]) -> None:
            entry["memory_obj"].ref_count_down()
            with self._entries_lock:
                entry["metadata"].unpin()

        def fill_free_slots() -> None:
            nonlocal next_task, input_closed, peak_staging_bytes, submitted_count
            while (
                len(live_entries) < self.config.xds_read_slot_count
                and next_task < len(task_infos)
            ):
                info = task_infos[next_task]
                next_task += 1
                memory_obj = self.allocate(
                    info["metadata"].shapes,
                    info["metadata"].dtypes,
                    info["metadata"].fmt,
                    busy_loop=False,
                )
                if memory_obj is None:
                    with self._entries_lock:
                        info["metadata"].unpin()
                    continue
                try:
                    offset, size = self._buffer_descriptor(memory_obj)
                    if size != info["metadata"].size:
                        raise RuntimeError(
                            "xDS allocation size does not match stored blob: "
                            f"allocated={size}, stored={info['metadata'].size}"
                        )
                    entry = {
                        **info,
                        "memory_obj": memory_obj,
                        "offset": offset,
                        "size": size,
                    }
                    pump.put(entry)
                except BaseException:
                    memory_obj.ref_count_down()
                    with self._entries_lock:
                        info["metadata"].unpin()
                    raise
                live_entries[id(entry)] = entry
                submitted_count += 1
                peak_staging_bytes = max(
                    peak_staging_bytes,
                    sum(item["size"] for item in live_entries.values()),
                )
            if next_task == len(task_infos) and not input_closed:
                pump.close_input()
                input_closed = True

        pump_thread.start()
        try:
            fill_free_slots()
            while live_entries:
                storage_start = time.perf_counter()
                ring_completion = pump.get_completion()
                storage_seconds += time.perf_counter() - storage_start
                if ring_completion.error is not None:
                    raise ring_completion.error
                entry = ring_completion.item
                live_entries.pop(id(entry))
                try:
                    completion = ring_completion.completion
                    if (
                        int(completion.status) == 0
                        and int(completion.completed_bytes) == entry["size"]
                    ):
                        gpu_start = time.perf_counter()
                        gpu_connector.to_gpu(
                            entry["memory_obj"],
                            entry["start"],
                            entry["end"],
                            **kwargs,
                        )
                        event = gpu_connector.record_transfer_event("load")
                        wait_start = time.perf_counter()
                        event.synchronize()
                        slot_wait_seconds += time.perf_counter() - wait_start
                        gpu_seconds += time.perf_counter() - gpu_start
                        completed[entry["index"]] = True
                        completed_bytes += entry["size"]
                    elif int(completion.status) == 0:
                        short_io_chunks += 1
                finally:
                    release_entry(entry)
                fill_free_slots()
        finally:
            if not input_closed:
                pump.close_input()
            pump_thread.join(timeout=30.0)
            for entry in live_entries.values():
                release_entry(entry)
            for info in task_infos[next_task:]:
                with self._entries_lock:
                    info["metadata"].unpin()
        if pump_thread.is_alive():
            raise TimeoutError("timed out stopping xDS read ring")
        if pump.error is not None:
            raise pump.error

        window_count = (
            len(task_infos) + self.config.xds_read_window_size - 1
        ) // self.config.xds_read_window_size
        return self._make_transfer_result(
            completed,
            requested_bytes=requested_bytes,
            completed_bytes=completed_bytes,
            window_size=self.config.xds_read_window_size,
            window_count=window_count,
            peak_staging_bytes=peak_staging_bytes,
            storage_seconds=storage_seconds,
            gpu_seconds=gpu_seconds,
            slot_wait_seconds=slot_wait_seconds,
            short_io_chunks=short_io_chunks,
            peak_inflight_io=min(
                self.config.xds_io_queue_depth, submitted_count
            ),
        )

    def batched_put_from_gpu_blocking(
        self,
        keys: Sequence[CacheEngineKey],
        starts: list[int],
        ends: list[int],
        gpu_connector: Any,
        **kwargs: Any,
    ) -> Optional[BackendTransferResult | BackendWriteSubmissionResult]:
        """Gather and persist a logical batch through bounded GPU slots."""
        if not self.config.xds_windowed_io:
            return None
        if not (len(keys) == len(starts) == len(ends)):
            raise ValueError("xDS streaming write inputs must have equal lengths")
        if self._write_scheduler is not None:
            return self._batched_put_from_gpu_async(
                keys, starts, ends, gpu_connector, **kwargs
            )

        completed = [False] * len(keys)
        task_infos = []
        with self._put_tasks_lock, self._entries_lock:
            for index, (key, start, end) in enumerate(
                zip(keys, starts, ends, strict=True)
            ):
                if key in self._put_tasks or key in self._entries:
                    continue
                self._put_tasks.add(key)
                task_infos.append(
                    {
                        "index": index,
                        "key": key,
                        "start": start,
                        "end": end,
                    }
                )

        requested_bytes = 0
        completed_bytes = 0
        peak_staging_bytes = 0
        short_io_chunks = 0
        gpu_seconds = 0.0
        storage_seconds = 0.0
        slot_wait_seconds = 0.0
        wall_start = time.perf_counter()
        try:
            for window_start in range(
                0, len(task_infos), self.config.xds_write_slot_count
            ):
                window_infos = task_infos[
                    window_start : window_start + self.config.xds_write_slot_count
                ]
                objects: list[MemoryObj] = []
                try:
                    for info in window_infos:
                        memory_obj = self.allocate(
                            self.metadata.get_shapes(info["end"] - info["start"]),
                            self.metadata.get_dtypes(),
                            busy_loop=False,
                        )
                        if memory_obj is None:
                            raise MemoryError("xDS write slot allocation failed")
                        objects.append(memory_obj)
                    sizes = [obj.get_physical_size() for obj in objects]
                    requested_bytes += sum(sizes)
                    peak_staging_bytes = max(peak_staging_bytes, sum(sizes))

                    gpu_start = time.perf_counter()
                    gpu_connector.batched_from_gpu(
                        objects,
                        [info["start"] for info in window_infos],
                        [info["end"] for info in window_infos],
                        **kwargs,
                    )
                    event = gpu_connector.record_transfer_event("store")
                    wait_start = time.perf_counter()
                    event.synchronize()
                    slot_wait_seconds += time.perf_counter() - wait_start
                    gpu_seconds += time.perf_counter() - gpu_start

                    storage_start = time.perf_counter()
                    outcomes, window_bytes, window_short = self._write_window(
                        window_infos, objects
                    )
                    storage_seconds += time.perf_counter() - storage_start
                    completed_bytes += window_bytes
                    short_io_chunks += window_short
                    for info, outcome in zip(window_infos, outcomes, strict=True):
                        completed[info["index"]] = outcome
                finally:
                    for memory_obj in objects:
                        memory_obj.ref_count_down()
                    with self._put_tasks_lock:
                        for info in window_infos:
                            self._put_tasks.discard(info["key"])
        finally:
            with self._put_tasks_lock:
                for info in task_infos:
                    self._put_tasks.discard(info["key"])

        window_count = (
            len(task_infos) + self.config.xds_write_window_size - 1
        ) // self.config.xds_write_window_size
        result = self._make_transfer_result(
            completed,
            requested_bytes=requested_bytes,
            completed_bytes=completed_bytes,
            window_size=self.config.xds_write_window_size,
            window_count=window_count,
            peak_staging_bytes=peak_staging_bytes,
            storage_seconds=storage_seconds,
            gpu_seconds=gpu_seconds,
            slot_wait_seconds=slot_wait_seconds,
            short_io_chunks=short_io_chunks,
            peak_inflight_io=min(
                self.config.xds_io_queue_depth, len(task_infos)
            ),
        )
        logger.info(
            "[xDS Window] direction=write chunks=%d slots=%d bytes=%d wall_ms=%.3f",
            len(task_infos),
            self.config.xds_write_slot_count,
            completed_bytes,
            (time.perf_counter() - wall_start) * 1000,
        )
        return result

    def _batched_put_from_gpu_async(
        self,
        keys: Sequence[CacheEngineKey],
        starts: list[int],
        ends: list[int],
        gpu_connector: Any,
        **kwargs: Any,
    ) -> BackendWriteSubmissionResult:
        scheduler = self._write_scheduler
        assert scheduler is not None
        scheduler.check_healthy()
        accepted = [False] * len(keys)
        accepted_bytes = 0
        staging_seconds = 0.0
        backpressure_seconds = 0.0

        with self._async_admission_lock:
            task_infos: list[dict[str, Any]] = []
            seen: set[CacheEngineKey] = set()
            with self._put_tasks_lock, self._entries_lock:
                for index, (key, start, end) in enumerate(
                    zip(keys, starts, ends, strict=True)
                ):
                    if key in seen or key in self._put_tasks or key in self._entries:
                        continue
                    seen.add(key)
                    task_infos.append(
                        {
                            "index": index,
                            "key": key,
                            "start": start,
                            "end": end,
                        }
                    )

            for lease_start in range(
                0, len(task_infos), self.config.xds_write_slot_count
            ):
                lease_infos = task_infos[
                    lease_start : lease_start + self.config.xds_write_slot_count
                ]
                reserved_bytes = len(lease_infos) * self.blob_size
                backpressure_seconds += scheduler.acquire_credits(
                    reserved_bytes,
                    timeout=float(self.config.blocking_timeout_secs),
                )
                objects: list[MemoryObj] = []
                pending_installed = False
                enqueued = False
                staging_start = time.perf_counter()
                try:
                    for info in lease_infos:
                        memory_obj = self.allocate(
                            self.metadata.get_shapes(
                                info["end"] - info["start"]
                            ),
                            self.metadata.get_dtypes(),
                            busy_loop=False,
                        )
                        if memory_obj is None:
                            raise MemoryError("xDS async write slot allocation failed")
                        objects.append(memory_obj)
                    sizes = [obj.get_physical_size() for obj in objects]
                    gpu_connector.batched_from_gpu(
                        objects,
                        [info["start"] for info in lease_infos],
                        [info["end"] for info in lease_infos],
                        **kwargs,
                    )
                    event = gpu_connector.record_transfer_event("store")
                    event.synchronize()
                    lease = StagingLease(
                        keys=tuple(info["key"] for info in lease_infos),
                        indexes=tuple(info["index"] for info in lease_infos),
                        objects=tuple(objects),
                        sizes=tuple(sizes),
                        credited_bytes=sum(sizes),
                        reserved_bytes=reserved_bytes,
                    )
                    lease.direct_from_gpu = True
                    lease.resources_released = False
                    lease.io_outcome = None
                    lease.transition(LeaseState.PENDING)
                    with self._put_tasks_lock:
                        self._put_tasks.update(lease.keys)
                        for key, memory_obj in zip(
                            lease.keys, lease.objects, strict=True
                        ):
                            self._pending_objects[key] = memory_obj
                    pending_installed = True
                    scheduler.enqueue(lease)
                    enqueued = True
                    for index in lease.indexes:
                        accepted[index] = True
                    accepted_bytes += sum(sizes)
                    staging_seconds += time.perf_counter() - staging_start
                except BaseException:
                    if pending_installed:
                        with self._put_tasks_lock:
                            for info in lease_infos:
                                self._put_tasks.discard(info["key"])
                                self._pending_objects.pop(info["key"], None)
                    if not enqueued:
                        for memory_obj in objects:
                            memory_obj.ref_count_down()
                        scheduler.release_credits(reserved_bytes)
                    raise

        snapshot = scheduler.snapshot()
        logger.info(
            "[xDS Async Submit] accepted=%d bytes=%d staging_ms=%.3f "
            "backpressure_ms=%.3f queue=%d outstanding_mib=%.1f",
            sum(accepted),
            accepted_bytes,
            staging_seconds * 1000,
            backpressure_seconds * 1000,
            snapshot.queued_leases,
            snapshot.outstanding_bytes / (1024**2),
        )
        return BackendWriteSubmissionResult(
            accepted=tuple(accepted),
            accepted_bytes=accepted_bytes,
            durable=False,
            staging_seconds=staging_seconds,
            backpressure_seconds=backpressure_seconds,
            queue_depth=snapshot.queued_leases,
            outstanding_bytes=snapshot.outstanding_bytes,
        )

    def _validate_registered_range(self, offset: int, size: int) -> None:
        if offset < 0 or size <= 0 or offset + size > self.registered_size:
            raise ValueError(
                "xDS descriptor is outside the registered range: "
                f"offset={offset}, size={size}, registered={self.registered_size}"
            )
        if offset % self.io_unit_size != 0 or size % self.io_unit_size != 0:
            raise ValueError(
                "xDS descriptor must be I/O-unit aligned: "
                f"offset={offset}, size={size}, io_unit={self.io_unit_size}"
            )

    def _buffer_descriptor(self, memory_obj: MemoryObj) -> tuple[int, int]:
        offset = memory_obj.raw_data.data_ptr() - self.registered_base_ptr
        size = memory_obj.get_physical_size()
        self._validate_registered_range(offset, size)
        if size > self.blob_size:
            raise ValueError(
                f"xDS memory object size {size} exceeds blob size {self.blob_size}"
            )
        return offset, size

    def _write_direct_async_lease(
        self,
        lease: StagingLease,
    ) -> tuple[int, ...]:
        items: list[dict[str, Any]] = []
        blob_handles: list[int] = []
        try:
            for memory_obj, expected_size in zip(
                lease.objects, lease.sizes, strict=True
            ):
                blob_handle = int(self.client.get_blob(timeout=5.0))
                if blob_handle <= 0:
                    raise RuntimeError(
                        f"xDS returned an invalid blob handle {blob_handle}"
                    )
                blob_handles.append(blob_handle)
                offset, size = self._buffer_descriptor(memory_obj)
                if size != expected_size:
                    raise RuntimeError(
                        "xDS async staging size changed after admission"
                    )
                items.append(
                    {
                        "blob_handle": blob_handle,
                        "offset": offset,
                        "size": size,
                    }
                )

            pump = XDSReadRingPump(
                queue_depth=self.config.xds_io_queue_depth,
                submit=lambda item: int(
                    self.client.submit_write(
                        item["blob_handle"],
                        self.buffer_ref,
                        item["offset"],
                        0,
                        item["size"],
                    )
                ),
                wait_any=self.client.wait_any_result,
                timeout=30.0,
            )
            for item in items:
                pump.put(item)
            pump.close_input()
            with self._io_lock:
                pump.run()

            failures: list[str] = []
            first_error: Optional[BaseException] = None
            for _ in items:
                ring_completion = pump.get_completion()
                if ring_completion.error is not None:
                    if first_error is None:
                        first_error = ring_completion.error
                    continue
                completion = ring_completion.completion
                item = ring_completion.item
                if (
                    int(completion.status) != 0
                    or int(completion.completed_bytes) != item["size"]
                ):
                    failures.append(
                        f"request {completion.request_id} "
                        f"status={completion.status} "
                        f"completed={completion.completed_bytes} "
                        f"expected={item['size']}"
                    )
            if first_error is None and pump.error is not None:
                first_error = pump.error
            if first_error is not None:
                raise first_error
            if failures:
                raise RuntimeError("xDS async write failed: " + "; ".join(failures))
            outcome = tuple(blob_handles)
            lease.io_outcome = outcome
            return outcome
        except BaseException:
            for blob_handle in blob_handles:
                self.client.release_blob(blob_handle)
            raise

    @staticmethod
    def _release_direct_lease_objects(lease: StagingLease) -> None:
        if getattr(lease, "resources_released", False):
            return
        for memory_obj in lease.objects:
            memory_obj.ref_count_down()
        lease.resources_released = True

    def _submit_async_lease(self, lease: StagingLease) -> Any:
        if getattr(lease, "direct_from_gpu", False):
            return self._write_direct_async_lease(lease)
        key = lease.keys[0]
        memory_obj = lease.objects[0]
        assert isinstance(key, CacheEngineKey)
        assert isinstance(memory_obj, MemoryObj)
        return self._write_one(key, memory_obj)

    def _finish_async_lease(
        self,
        lease: StagingLease,
        outcome: Optional[bool] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        key = lease.keys[0]
        memory_obj = lease.objects[0]
        future = lease.future
        callback = lease.on_complete_callback
        try:
            if error is not None:
                future.set_exception(error)
            else:
                if outcome and callback is not None:
                    try:
                        callback(key)
                    except Exception:
                        logger.exception("xDS put completion callback failed")
                future.set_result(bool(outcome))
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
        finally:
            memory_obj.ref_count_down()
            with self._put_tasks_lock:
                self._put_tasks.discard(key)

    def _commit_async_lease(self, lease: StagingLease, outcome: Any) -> None:
        if getattr(lease, "direct_from_gpu", False):
            blob_handles = tuple(outcome)
            if len(blob_handles) != len(lease.keys):
                raise RuntimeError("xDS async outcome has the wrong handle count")
            with self._entries_lock:
                for key, memory_obj, size, blob_handle in zip(
                    lease.keys,
                    lease.objects,
                    lease.sizes,
                    blob_handles,
                    strict=True,
                ):
                    self._entries[key] = XDSBlobMetadata(
                        blob_handle=blob_handle,
                        size=size,
                        shapes=memory_obj.get_shapes(),
                        dtypes=memory_obj.get_dtypes(),
                        fmt=memory_obj.get_memory_format(),
                    )
            with self._put_tasks_lock:
                for key in lease.keys:
                    self._put_tasks.discard(key)
                    self._pending_objects.pop(key, None)
            self._release_direct_lease_objects(lease)
            return
        self._finish_async_lease(lease, outcome=bool(outcome))

    def _rollback_async_lease(
        self,
        lease: StagingLease,
        error: BaseException,
    ) -> None:
        if getattr(lease, "direct_from_gpu", False):
            blob_handles = tuple(getattr(lease, "io_outcome", ()) or ())
            with self._entries_lock:
                for key, blob_handle in zip(
                    lease.keys, blob_handles, strict=False
                ):
                    metadata = self._entries.get(key)
                    if (
                        metadata is not None
                        and metadata.blob_handle == blob_handle
                    ):
                        self._entries.pop(key)
            for blob_handle in blob_handles:
                self.client.release_blob(blob_handle)
            with self._put_tasks_lock:
                for key in lease.keys:
                    self._put_tasks.discard(key)
                    self._pending_objects.pop(key, None)
            self._release_direct_lease_objects(lease)
            logger.error(
                "[xDS Async Rollback] chunks=%d bytes=%d error=%s",
                len(lease.keys),
                lease.credited_bytes,
                error,
            )
            return
        self._finish_async_lease(lease, error=error)

    def flush_async_writes(
        self,
        timeout: Optional[float] = None,
    ) -> SchedulerSnapshot:
        """Wait until all admitted asynchronous xDS writes are durable."""
        if self._write_scheduler is None:
            raise RuntimeError("xDS asynchronous writes are not enabled")
        return self._write_scheduler.drain(timeout)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: list[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[list[Future]]:
        if self._write_scheduler is None:
            return super().batched_submit_put_task(
                keys,
                objs,
                transfer_spec,
                on_complete_callback,
            )
        if len(keys) != len(objs):
            raise ValueError("keys and objs must have the same length")

        futures: list[Future] = []
        for index, (key, memory_obj) in enumerate(zip(keys, objs, strict=True)):
            future: Future = Future()
            futures.append(future)
            with self._put_tasks_lock, self._entries_lock:
                if key in self._put_tasks or key in self._entries:
                    future.set_result(False)
                    continue
                self._put_tasks.add(key)
            size = memory_obj.get_physical_size()
            credits_acquired = False
            reference_added = False
            try:
                self._write_scheduler.acquire_credits(size, timeout=30.0)
                credits_acquired = True
                memory_obj.ref_count_up()
                reference_added = True
                lease = StagingLease(
                    keys=(key,),
                    indexes=(index,),
                    objects=(memory_obj,),
                    sizes=(size,),
                    credited_bytes=size,
                )
                lease.future = future
                lease.on_complete_callback = on_complete_callback
                lease.transition(LeaseState.PENDING)
                self._write_scheduler.enqueue(lease)
            except Exception as exc:
                if reference_added:
                    memory_obj.ref_count_down()
                if credits_acquired:
                    self._write_scheduler.release_credits(size)
                with self._put_tasks_lock:
                    self._put_tasks.discard(key)
                future.set_exception(exc)
        return futures

    def batched_get_blocking(
        self,
        keys: list[CacheEngineKey],
    ) -> list[Optional[MemoryObj]]:
        if len(keys) <= 1 or self.config.xds_io_queue_depth <= 1:
            return super().batched_get_blocking(keys)

        results: list[Optional[MemoryObj]] = [None] * len(keys)
        read_items = []
        for index, key in enumerate(keys):
            with self._entries_lock:
                blob_metadata = self._entries.get(key)
                if blob_metadata is None:
                    continue
                blob_metadata.pin()
                self._entries.move_to_end(key)
            memory_obj = self.allocate(
                blob_metadata.shapes,
                blob_metadata.dtypes,
                blob_metadata.fmt,
            )
            if memory_obj is None:
                with self._entries_lock:
                    blob_metadata.unpin()
                continue
            try:
                offset, size = self._buffer_descriptor(memory_obj)
                if size != blob_metadata.size:
                    raise RuntimeError(
                        "xDS allocation size does not match stored blob: "
                        f"allocated={size}, stored={blob_metadata.size}"
                    )
            except Exception:
                memory_obj.ref_count_down()
                with self._entries_lock:
                    blob_metadata.unpin()
                raise
            read_items.append((index, blob_metadata, memory_obj, offset, size))

        if not read_items:
            return results

        def submit(item) -> int:
            _, blob_metadata, _, offset, size = item
            return int(
                self.client.submit_read(
                    blob_metadata.blob_handle,
                    self.buffer_ref,
                    offset,
                    0,
                    size,
                )
            )

        pump = XDSReadRingPump(
            queue_depth=self.config.xds_io_queue_depth,
            submit=submit,
            wait_any=self.client.wait_any_result,
            timeout=30.0,
        )
        for item in read_items:
            pump.put(item)
        pump.close_input()
        with self._io_lock:
            pump.run()

        first_error: Optional[BaseException] = None
        completed_objects: set[int] = set()
        for _ in read_items:
            ring_completion = pump.get_completion()
            if ring_completion.error is not None:
                if first_error is None:
                    first_error = ring_completion.error
                continue
            item = ring_completion.item
            index, _, memory_obj, _, size = item
            try:
                self._validate_io_result(ring_completion.completion, size, "read")
                results[index] = memory_obj
                completed_objects.add(id(memory_obj))
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        if first_error is None and pump.error is not None:
            first_error = pump.error

        for _, blob_metadata, memory_obj, _, _ in read_items:
            with self._entries_lock:
                blob_metadata.unpin()
            if id(memory_obj) not in completed_objects or first_error is not None:
                memory_obj.ref_count_down()

        if first_error is not None:
            raise first_error
        return results

    def close(self) -> None:
        if self._closed:
            return
        first_error: Optional[BaseException] = None
        if self._write_scheduler is not None:
            try:
                self._write_scheduler.close(timeout=30.0)
            except BaseException as error:
                first_error = error
        active_allocations = self.memory_allocator.num_active_allocations
        if active_allocations:
            active_error = RuntimeError(
                "cannot close XDSBackend with active registered allocations: "
                f"{active_allocations}"
            )
            if first_error is not None:
                raise first_error from active_error
            raise active_error
        with self._entries_lock:
            blob_handles = [metadata.blob_handle for metadata in self._entries.values()]
            self._entries.clear()
        for blob_handle in blob_handles:
            try:
                self.client.release_blob(blob_handle)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        try:
            self.memory_allocator.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
        try:
            self.client.unregister_buffer(self.buffer_ref)
        except BaseException as error:
            if first_error is None:
                first_error = error
        client_close = getattr(self.client, "close", None)
        if callable(client_close):
            try:
                client_close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._closed = True
        if first_error is not None:
            raise first_error
