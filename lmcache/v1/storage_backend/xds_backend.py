# SPDX-License-Identifier: Apache-2.0
"""Direct SSD-to-GPU LMCache backend using an external xDS service."""

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Sequence
import asyncio
import json
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    TensorMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
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


class _XDSReaderWriterGate:
    """Allow concurrent native reads while excluding other client I/O."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        """Enter a shared-reader section."""
        with self._condition:
            while self._writer or self._waiting_writers:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    def __enter__(self) -> "_XDSReaderWriterGate":
        """Enter an exclusive writer section."""
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
        return self

    def __exit__(self, *_args: object) -> None:
        """Leave an exclusive writer section."""
        with self._condition:
            self._writer = False
            self._condition.notify_all()


@dataclass
class XDSLayerwiseBlockMetadata:
    """One xDS blob containing every layer for one token block."""

    blob_handle: int
    size: int
    layer_io_size: int
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]
    fmt: MemoryFormat
    pin_count: int = 0

    def pin(self) -> bool:
        self.pin_count += 1
        return True

    def unpin(self) -> bool:
        if self.pin_count == 0:
            return False
        self.pin_count -= 1
        return True

    @property
    def is_pinned(self) -> bool:
        return self.pin_count > 0


@dataclass
class XDSLayerReadSlotLease:
    """Track one reusable GPU staging object in a native layer read stream."""

    stream: Any
    memory_obj: MemoryObj
    slot_id: int
    slot_count: int
    num_layers: int
    current_layer: int
    published: bool = False
    terminal: bool = False


class XDSBackend(AllocatorBackendInterface):
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
        self._readv_targets: dict[tuple[int, int], tuple[torch.Tensor, int]] = {}
        self._readv_poisoned = False
        self._readv_failed_staging: list[MemoryObj] = []
        self._entries: OrderedDict[
            CacheEngineKey, XDSBlobMetadata | XDSLayerwiseBlockMetadata
        ] = OrderedDict()
        self._entries_lock = threading.RLock()
        self._put_tasks: set[CacheEngineKey] = set()
        self._pending_objects: dict[CacheEngineKey, MemoryObj] = {}
        self._put_tasks_lock = threading.RLock()
        self._async_admission_lock = threading.Lock()
        self._io_lock = _XDSReaderWriterGate()
        self._layer_read_slots_lock = threading.Lock()
        self._layer_read_slots: dict[int, XDSLayerReadSlotLease] = {}
        max_concurrent_layer_reads = int(
            config.get_extra_config_value("xds_layerwise_max_concurrent_requests", 4)
        )
        if max_concurrent_layer_reads <= 0:
            raise ValueError("xds_layerwise_max_concurrent_requests must be positive")
        self._layer_read_admission = threading.BoundedSemaphore(
            max_concurrent_layer_reads
        )
        self._keys_in_request: list[CacheEngineKey] = []

        service_metadata = xds_client.metadata
        self.io_unit_size = int(service_metadata.io_unit_size)
        self.blob_size = align_to_io_unit(
            int(service_metadata.blob_size_bytes),
            self.io_unit_size,
        )
        self.registered_size = int(service_metadata.gpu_buffer_size_bytes)
        self.block_blob_layout = bool(
            config.get_extra_config_value("xds_layerwise_block_blob", False)
        )
        self.num_layers = int(metadata.kv_shape[0])
        self.block_blob_layer_stride = 0
        if self.block_blob_layout:
            if not config.use_layerwise:
                raise ValueError("xds_layerwise_block_blob requires use_layerwise=true")
            if self.num_layers <= 0 or self.blob_size % self.num_layers:
                raise ValueError(
                    "xDS block blob must divide evenly across model layers: "
                    f"blob_size={self.blob_size}, layers={self.num_layers}"
                )
            self.block_blob_layer_stride = self.blob_size // self.num_layers
            if self.block_blob_layer_stride % self.io_unit_size:
                raise ValueError(
                    "xDS block-blob layer stride must be I/O-unit aligned: "
                    f"stride={self.block_blob_layer_stride}, "
                    f"io_unit={self.io_unit_size}"
                )
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

    def __str__(self) -> str:
        return self.__class__.__name__

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> TensorMemoryAllocator:
        del config, metadata
        return self.memory_allocator

    def get_allocator_backend(self) -> "XDSBackend":
        return self

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        return self.memory_allocator

    def allocate(
        self,
        shapes: Any,
        dtypes: Any,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        del eviction, busy_loop
        return self.memory_allocator.allocate(shapes, dtypes, fmt)

    def batched_allocate(
        self,
        shapes: Any,
        dtypes: Any,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        del eviction, busy_loop
        return self.memory_allocator.batched_allocate(
            shapes,
            dtypes,
            batch_size,
            fmt,
        )

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self._put_tasks_lock:
            if key in self._put_tasks:
                return True
        with self._entries_lock:
            metadata = self._entries.get(key)
            if metadata is None:
                return False
            if pin:
                metadata.pin()
                self._keys_in_request.append(key)
            return True

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._put_tasks_lock:
            return key in self._put_tasks

    @staticmethod
    def _validate_io_result(
        completion: Any,
        expected_bytes: int,
        operation: str,
    ) -> None:
        if int(completion.status) != 0:
            raise RuntimeError(
                f"xDS {operation} failed with status={completion.status}, "
                f"error_code={completion.error_code}"
            )
        if int(completion.completed_bytes) != expected_bytes:
            raise RuntimeError(
                f"short xDS {operation}: expected {expected_bytes} bytes, "
                f"completed {completion.completed_bytes}"
            )

    def _wait_for_completion(
        self,
        request_id: int,
        expected_bytes: int,
        operation: str,
    ) -> None:
        completion = self.client.wait_any_result(timeout=30.0)
        if int(completion.request_id) != request_id:
            raise RuntimeError(
                "unexpected xDS completion request id: "
                f"expected={request_id}, got={completion.request_id}"
            )
        self._validate_io_result(completion, expected_bytes, operation)

    def _write_one(self, key: CacheEngineKey, memory_obj: MemoryObj) -> bool:
        with self._io_lock:
            return self._write_one_exclusive(key, memory_obj)

    def _write_one_exclusive(self, key: CacheEngineKey, memory_obj: MemoryObj) -> bool:
        """Write one object while owning exclusive xDS client completion access."""
        with self._entries_lock:
            if key in self._entries:
                return False

        blob_handle = int(self.client.get_blob(timeout=5.0))
        if blob_handle <= 0:
            raise RuntimeError(f"xDS returned an invalid blob handle {blob_handle}")
        try:
            buffer_offset, size = self._buffer_descriptor(memory_obj)
            request_id = int(
                self.client.submit_write(
                    blob_handle,
                    self.buffer_ref,
                    buffer_offset,
                    0,
                    size,
                )
            )
            self._wait_for_completion(request_id, size, "write")
            with self._entries_lock:
                self._entries[key] = XDSBlobMetadata(
                    blob_handle=blob_handle,
                    size=size,
                    shapes=memory_obj.get_shapes(),
                    dtypes=memory_obj.get_dtypes(),
                    fmt=memory_obj.get_memory_format(),
                )
            return True
        except Exception:
            self.client.release_blob(blob_handle)
            raise

    def _batched_submit_put_task_serial(
        self,
        keys: Sequence[CacheEngineKey],
        objs: list[MemoryObj],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> list[Future]:
        if len(keys) != len(objs):
            raise ValueError("keys and objs must have the same length")
        futures: list[Future] = []
        for key, memory_obj in zip(keys, objs, strict=True):
            future: Future = Future()
            futures.append(future)
            with self._put_tasks_lock, self._entries_lock:
                if key in self._put_tasks or key in self._entries:
                    future.set_result(False)
                    continue
                self._put_tasks.add(key)
            try:
                stored = self._write_one(key, memory_obj)
                if stored and on_complete_callback is not None:
                    try:
                        on_complete_callback(key)
                    except Exception:
                        logger.exception("xDS put completion callback failed")
                future.set_result(stored)
            except Exception as error:
                future.set_exception(error)
            finally:
                with self._put_tasks_lock:
                    self._put_tasks.discard(key)
        return futures

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: list[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        futures = await asyncio.to_thread(
            self.batched_submit_put_task,
            keys,
            objs,
            transfer_spec,
            on_complete_callback,
        )
        await asyncio.gather(*(asyncio.wrap_future(item) for item in futures or []))

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self._put_tasks_lock:
            pending_obj = self._pending_objects.get(key)
            if pending_obj is not None:
                pending_obj.ref_count_up()
                return pending_obj

        with self._entries_lock:
            blob_metadata = self._entries.get(key)
            if blob_metadata is None:
                return None
            blob_metadata.pin()
            self._entries.move_to_end(key)

        memory_obj: Optional[MemoryObj] = None
        try:
            memory_obj = self.allocate(
                blob_metadata.shapes,
                blob_metadata.dtypes,
                blob_metadata.fmt,
            )
            if memory_obj is None:
                return None
            buffer_offset, size = self._buffer_descriptor(memory_obj)
            if size != blob_metadata.size:
                raise RuntimeError(
                    "xDS allocation size does not match stored blob: "
                    f"allocated={size}, stored={blob_metadata.size}"
                )
            with self._io_lock:
                request_id = int(
                    self.client.submit_read(
                        blob_metadata.blob_handle,
                        self.buffer_ref,
                        buffer_offset,
                        0,
                        size,
                    )
                )
                self._wait_for_completion(request_id, size, "read")
            return memory_obj
        except Exception:
            if memory_obj is not None:
                memory_obj.ref_count_down()
            raise
        finally:
            with self._entries_lock:
                blob_metadata.unpin()

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        del location
        future: Future = Future()
        try:
            future.set_result(self.get_blocking(key))
        except Exception as error:
            future.set_exception(error)
        return future

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        del lookup_id
        return self.batched_contains(keys, pin)

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        del lookup_id, transfer_spec
        results: list[MemoryObj] = []
        for key in keys:
            memory_obj = await asyncio.to_thread(self.get_blocking, key)
            if memory_obj is None:
                break
            results.append(memory_obj)
        return results

    def pin(self, key: CacheEngineKey) -> bool:
        with self._entries_lock:
            metadata = self._entries.get(key)
            return metadata.pin() if metadata is not None else False

    def unpin(self, key: CacheEngineKey) -> bool:
        with self._entries_lock:
            metadata = self._entries.get(key)
            return metadata.unpin() if metadata is not None else False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        with self._entries_lock:
            metadata = self._entries.get(key)
            if metadata is None:
                return False
            if metadata.is_pinned and not force:
                return False
            del self._entries[key]
        self.client.release_blob(metadata.blob_handle)
        return True

    def touch_cache(self) -> None:
        with self._entries_lock:
            for key in reversed(self._keys_in_request):
                if key in self._entries:
                    self._entries.move_to_end(key)
            self._keys_in_request.clear()

    def calculate_chunk_budget(self) -> int:
        if self.block_blob_layout:
            return self.registered_size // self.block_blob_layer_stride
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
        with self._io_lock:
            return self._write_window_exclusive(task_infos, objects)

    def _write_window_exclusive(
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
        if self.config.get_extra_config_value("xds_readv_direct", False):
            direct = self._get_readv_to_gpu(keys, starts, ends, gpu_connector, **kwargs)
            if direct is not None:
                return direct
        transfer_start_ns = time.monotonic_ns()
        benchmark_metrics = bool(
            self.config.get_extra_config_value("benchmark_transfer_metrics", False)
        )
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
        for index, (key, start, end) in enumerate(zip(keys, starts, ends, strict=True)):
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
        io_start_ns = time.monotonic_ns()
        io_end_ns = io_start_ns

        def release_entry(entry: dict[str, Any]) -> None:
            entry["memory_obj"].ref_count_down()
            with self._entries_lock:
                entry["metadata"].unpin()

        def fill_free_slots() -> None:
            nonlocal next_task, input_closed, peak_staging_bytes, submitted_count
            while len(
                live_entries
            ) < self.config.xds_read_slot_count and next_task < len(task_infos):
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
                io_end_ns = time.monotonic_ns()
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
        result = self._make_transfer_result(
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
            peak_inflight_io=min(self.config.xds_io_queue_depth, submitted_count),
        )
        if benchmark_metrics:
            transfer_end_ns = time.monotonic_ns()
            logger.info(
                "[XDS_DIRECT_READ_METRICS] %s",
                json.dumps(
                    {
                        "worker_id": int(self.metadata.worker_id),
                        "bytes": completed_bytes,
                        "retrieved_tokens": int(kwargs.get("retrieved_tokens", 0)),
                        "transfer_start_ns": transfer_start_ns,
                        "transfer_end_ns": transfer_end_ns,
                        "transfer_seconds": (transfer_end_ns - transfer_start_ns) / 1e9,
                        "io_start_ns": io_start_ns,
                        "io_end_ns": io_end_ns,
                        "io_seconds": (io_end_ns - io_start_ns) / 1e9,
                        "storage_wait_seconds": storage_seconds,
                        "gpu_seconds": gpu_seconds,
                        "slot_wait_seconds": slot_wait_seconds,
                        "read_slots": self.config.xds_read_slot_count,
                        "io_queue_depth": self.config.xds_io_queue_depth,
                        "peak_inflight_io": result.peak_inflight_io,
                    },
                    sort_keys=True,
                ),
            )
        return result

    def blockwise_batched_submit_put_task(
        self,
        block_keys: Sequence[CacheEngineKey],
        memory_objs_by_layer: Sequence[Sequence[MemoryObj]],
        completion_event: Any = None,
    ) -> None:
        """Persist layer objects in one blob per block after GPU production.

        ``completion_event``, when supplied, must expose ``synchronize()``.
        Its errors propagate before any blob is allocated or write is submitted.
        """
        if completion_event is not None:
            completion_event.synchronize()
        with self._io_lock:
            self._blockwise_batched_submit_put_task_exclusive(
                block_keys, memory_objs_by_layer
            )

    def _blockwise_batched_submit_put_task_exclusive(
        self,
        block_keys: Sequence[CacheEngineKey],
        memory_objs_by_layer: Sequence[Sequence[MemoryObj]],
    ) -> None:
        """Persist a blockwise request while holding exclusive client access."""
        if not self.block_blob_layout:
            raise RuntimeError("xDS blockwise writes require block-blob layout")
        if len(memory_objs_by_layer) != self.num_layers:
            raise ValueError(
                "xDS blockwise write layer count mismatch: "
                f"expected={self.num_layers}, actual={len(memory_objs_by_layer)}"
            )
        if any(len(layer) != len(block_keys) for layer in memory_objs_by_layer):
            raise ValueError("xDS blockwise write requires one object per block/layer")

        selected_indexes: list[int] = []
        selected_keys: list[CacheEngineKey] = []
        with self._put_tasks_lock, self._entries_lock:
            for block_index, key in enumerate(block_keys):
                if key in self._put_tasks or key in self._entries:
                    continue
                self._put_tasks.add(key)
                selected_indexes.append(block_index)
                selected_keys.append(key)
        if not selected_indexes:
            return

        blob_handles: list[int] = []
        published = False
        try:
            for _ in selected_indexes:
                blob_handle = int(self.client.get_blob(timeout=5.0))
                if blob_handle <= 0:
                    raise RuntimeError(
                        f"xDS returned an invalid blob handle {blob_handle}"
                    )
                blob_handles.append(blob_handle)

            items: list[dict[str, Any]] = []
            for layer_id, layer_objects in enumerate(memory_objs_by_layer):
                blob_offset = layer_id * self.block_blob_layer_stride
                for selected_offset, block_index in enumerate(selected_indexes):
                    memory_obj = layer_objects[block_index]
                    buffer_offset, size = self._buffer_descriptor(memory_obj)
                    if size > self.block_blob_layer_stride:
                        raise ValueError(
                            "xDS layer object exceeds its block-blob stride: "
                            f"size={size}, stride={self.block_blob_layer_stride}"
                        )
                    items.append(
                        {
                            "blob_handle": blob_handles[selected_offset],
                            "buffer_offset": buffer_offset,
                            "blob_offset": blob_offset,
                            "size": size,
                        }
                    )

            pump = XDSReadRingPump(
                queue_depth=self.config.xds_io_queue_depth,
                submit=lambda item: int(
                    self.client.submit_write(
                        item["blob_handle"],
                        self.buffer_ref,
                        item["buffer_offset"],
                        item["blob_offset"],
                        item["size"],
                    )
                ),
                wait_any=self.client.wait_any_result,
                timeout=30.0,
            )
            for item in items:
                pump.put(item)
            pump.close_input()
            pump.run()
            first_error: Optional[BaseException] = None
            for _ in items:
                ring_completion = pump.get_completion()
                if ring_completion.error is not None:
                    if first_error is None:
                        first_error = ring_completion.error
                    continue
                try:
                    self._validate_io_result(
                        ring_completion.completion,
                        int(ring_completion.item["size"]),
                        "blockwise write",
                    )
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if first_error is None and pump.error is not None:
                first_error = pump.error
            if first_error is not None:
                raise first_error

            with self._entries_lock:
                for selected_offset, block_index in enumerate(selected_indexes):
                    first_obj = memory_objs_by_layer[0][block_index]
                    _, layer_io_size = self._buffer_descriptor(first_obj)
                    for layer_objects in memory_objs_by_layer[1:]:
                        other_obj = layer_objects[block_index]
                        _, other_size = self._buffer_descriptor(other_obj)
                        if (
                            other_obj.get_shapes() != first_obj.get_shapes()
                            or other_obj.get_dtypes() != first_obj.get_dtypes()
                            or other_obj.get_memory_format()
                            != first_obj.get_memory_format()
                            or other_size != layer_io_size
                        ):
                            raise ValueError(
                                "xDS block blob requires identical per-layer layouts"
                            )
                    self._entries[selected_keys[selected_offset]] = (
                        XDSLayerwiseBlockMetadata(
                            blob_handle=blob_handles[selected_offset],
                            size=self.blob_size,
                            layer_io_size=layer_io_size,
                            shapes=first_obj.get_shapes(),
                            dtypes=first_obj.get_dtypes(),
                            fmt=first_obj.get_memory_format(),
                        )
                    )
            published = True
        finally:
            if not published:
                for blob_handle in blob_handles:
                    self.client.release_blob(blob_handle)
            with self._put_tasks_lock:
                for key in selected_keys:
                    self._put_tasks.discard(key)

    @staticmethod
    def _combined_layer_layout(
        metadata_entries: Sequence[XDSLayerwiseBlockMetadata],
    ) -> tuple[list[torch.Size], list[torch.dtype], MemoryFormat, list[int]]:
        """Build one contiguous layer tensor plus per-block byte offsets."""
        first = metadata_entries[0]
        # VLLM's paged layerwise connector exposes KV_T2D staging as
        # [tokens, 2, hidden], even though the legacy enum helper predates that
        # token-major representation.
        token_dim = 0
        combined_shapes = [torch.Size(shape) for shape in first.shapes]
        combined_dtypes = list(first.dtypes)
        token_totals = [0] * len(combined_shapes)
        destination_offsets: list[int] = []
        logical_cursor = 0
        for entry_index, entry in enumerate(metadata_entries):
            if entry.fmt != first.fmt or entry.dtypes != first.dtypes:
                raise ValueError("xDS blockwise read metadata layout mismatch")
            if len(entry.shapes) != len(combined_shapes):
                raise ValueError("xDS blockwise read shape-group mismatch")
            entry_logical_size = 0
            for group_index, (shape, dtype, first_shape) in enumerate(
                zip(entry.shapes, entry.dtypes, first.shapes, strict=True)
            ):
                if any(
                    shape[dim] != first_shape[dim]
                    for dim in range(len(shape))
                    if dim != token_dim
                ):
                    raise ValueError("xDS blockwise read non-token shape mismatch")
                token_totals[group_index] += int(shape[token_dim])
                entry_logical_size += shape.numel() * dtype.itemsize
            if entry_index + 1 < len(metadata_entries) and (
                entry.layer_io_size != entry_logical_size
            ):
                raise ValueError(
                    "only the final xDS block may contain aligned tail padding"
                )
            destination_offsets.append(logical_cursor)
            logical_cursor += entry_logical_size
        for group_index, shape in enumerate(combined_shapes):
            shape_list = list(shape)
            shape_list[token_dim] = token_totals[group_index]
            combined_shapes[group_index] = torch.Size(shape_list)
        return combined_shapes, combined_dtypes, first.fmt, destination_offsets

    def _allocate_blockwise_layer_staging(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat,
    ) -> MemoryObj:
        deadline = time.monotonic() + float(self.config.blocking_timeout_secs)
        while True:
            memory_obj = self.allocate(shapes, dtypes, fmt, busy_loop=False)
            if memory_obj is not None:
                return memory_obj
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for xDS layer staging window")
            time.sleep(0.0001)

    def prepare_layerwise_readv(
        self,
        block_keys: list[CacheEngineKey],
        starts: list[int],
        ends: list[int],
        gpu_connector: Any,
        **kwargs: Any,
    ) -> Optional[Iterator[BackendTransferResult]]:
        """Preflight a block-blob restore and return a GPU-ready layer iterator.

        Returns None before DMA for unsupported metadata/layouts. Each next()
        completes one layer and applies the RDMA visibility fence before yielding.
        Close the iterator on cancellation. The caller owns all target blocks
        through completion; abandoned, unstarted iterators hold no metadata pins.
        I/O failures raise and never fall back after partial DMA.
        """
        from lmcache.v1.gpu_connector.xds_readv import (
            XDSReadvPlan,
            cuda_readv_flush_function,
            finish_readv_dma,
        )

        planner = getattr(gpu_connector, "plan_layerwise_xds_readv", None)
        if (
            not self.config.get_extra_config_value("xds_layerwise_readv", False)
            or not self.block_blob_layout
            or not callable(planner)
            or not callable(getattr(self.client, "submit_readv", None))
        ):
            return None
        if not (len(block_keys) == len(starts) == len(ends)):
            raise ValueError("xDS layerwise readv batch lengths differ")
        if not block_keys:
            return None
        with self._entries_lock:
            entries = [self._entries.get(key) for key in block_keys]
            if any(not isinstance(e, XDSLayerwiseBlockMetadata) for e in entries):
                return None
            chunks = [
                (e.shapes, e.dtypes, e.fmt, a, b)
                for e, a, b in zip(entries, starts, ends, strict=True)
            ]
            plans = planner(chunks, self.io_unit_size, **kwargs)
            if plans is None or len(plans) != self.num_layers:
                return None
            if any(len(layer) != len(entries) for layer in plans):
                return None
            if any(
                p.size != e.layer_io_size or p.size > self.block_blob_layer_stride
                for layer in plans
                for p, e in zip(layer, entries, strict=True)
            ):
                return None
        cuda_readv_flush_function()
        device = plans[0][0].tensors[0].device

        def restore() -> Iterator[BackendTransferResult]:
            pinned = False
            unsafe: dict[int, int] = {}
            worker: Optional[threading.Thread] = None
            processed = 0
            total = sum(len(layer) for layer in plans)
            completed = [0] * self.num_layers
            error: Optional[BaseException] = None

            def consume() -> None:
                nonlocal processed, error
                outcome = pump.get_completion()
                processed += 1
                layer_id, entry, plan = outcome.item
                if not outcome.safe_to_reuse:
                    unsafe[id(entry)] = unsafe.get(id(entry), 0) + 1
                    self._readv_poisoned = True
                if outcome.error is not None:
                    error = error or outcome.error
                elif (
                    outcome.completion.status
                    or outcome.completion.completed_bytes != plan.size
                ):
                    error = error or RuntimeError(
                        "xDS layerwise readv failed or was short"
                    )
                else:
                    completed[layer_id] += 1

            try:
                with self._io_lock:
                    if self._readv_poisoned:
                        raise RuntimeError(
                            "xDS direct KV target has an unresolved DMA failure"
                        )
                    with self._entries_lock:
                        if any(
                            self._entries.get(k) is not e
                            for k, e in zip(block_keys, entries, strict=True)
                        ):
                            raise RuntimeError(
                                "xDS metadata changed after readv preflight"
                            )
                        for entry in entries:
                            entry.pin()
                        pinned = True
                    torch.cuda.synchronize(device)
                    refs = []
                    for layer in plans:
                        tensor = layer[0].tensors[0]
                        target = (
                            tensor.data_ptr(),
                            tensor.numel() * tensor.element_size(),
                        )
                        if target not in self._readv_targets:
                            ref = int(self.client.register_gpu_buffer(*target))
                            if ref <= 0:
                                raise RuntimeError("invalid final KV buffer reference")
                            self._readv_targets[target] = (tensor, ref)
                        refs.append(self._readv_targets[target][1])

                def submit(item: Any) -> int:
                    if self._readv_poisoned:
                        raise RuntimeError(
                            "xDS direct KV target has an unresolved DMA failure"
                        )
                    layer_id, entry, plan = item
                    return plan.submit(
                        self.client,
                        entry.blob_handle,
                        layer_id * self.block_blob_layer_stride,
                        [refs[layer_id]],
                    )

                def submit_batch(items: list[Any]) -> list[int | BaseException]:
                    return XDSReadvPlan.submit_batch(
                        self.client,
                        [
                            (
                                plan,
                                entry.blob_handle,
                                layer_id * self.block_blob_layer_stride,
                                [refs[layer_id]],
                            )
                            for layer_id, entry, plan in items
                        ],
                    )

                batch_capable = bool(
                    getattr(self.client, "supports_readv_batch", False)
                )
                pump = XDSReadRingPump(
                    self.config.xds_io_queue_depth,
                    submit,
                    self.client.wait_any_result,
                    30.0,
                    submit_batch=submit_batch if batch_capable else None,
                    wait_burst=(
                        getattr(self.client, "wait_burst", None)
                        if batch_capable
                        else None
                    ),
                )
                for layer_id, layer in enumerate(plans):
                    for entry, plan in zip(entries, layer, strict=True):
                        pump.put((layer_id, entry, plan))
                pump.close_input()

                def produce() -> None:
                    # The producer owns completion access; the yielding consumer
                    # holds no I/O lock, so other operations cannot deadlock it.
                    with self._io_lock:
                        io_start_ns = time.monotonic_ns()
                        pump.run()
                        io_end_ns = time.monotonic_ns()
                    if self.config.get_extra_config_value(
                        "benchmark_transfer_metrics", False
                    ):
                        logger.info(
                            "[XDS_LAYERWISE_READ_METRICS] %s",
                            json.dumps(
                                {
                                    "worker_id": int(self.metadata.worker_id),
                                    "mode": "readv",
                                    "bytes": sum(
                                        p.size for layer in plans for p in layer
                                    ),
                                    "layers": self.num_layers,
                                    "blocks": len(block_keys),
                                    "io_start_ns": io_start_ns,
                                    "io_end_ns": io_end_ns,
                                    "io_seconds": (io_end_ns - io_start_ns) / 1e9,
                                    "pump_error": (
                                        str(pump.error)
                                        if pump.error is not None
                                        else None
                                    ),
                                },
                                sort_keys=True,
                            ),
                        )

                worker = threading.Thread(target=produce, name="xds-layer-readv")
                worker.start()
                for layer_id, layer in enumerate(plans):
                    begin = time.perf_counter()
                    while completed[layer_id] < len(layer):
                        consume()
                        if error is not None:
                            raise error
                    finish_readv_dma(device)
                    size = sum(plan.size for plan in layer)
                    yield self._make_transfer_result(
                        [True] * len(entries),
                        requested_bytes=size,
                        completed_bytes=size,
                        window_size=len(entries),
                        window_count=1,
                        peak_staging_bytes=0,
                        storage_seconds=time.perf_counter() - begin,
                        gpu_seconds=0,
                        slot_wait_seconds=0,
                        short_io_chunks=0,
                        peak_inflight_io=min(total, self.config.xds_io_queue_depth),
                    )
            finally:
                # Closing a partially consumed iterator drains every submitted
                # read (including prefetched layers) before releasing ownership.
                if worker is not None and worker.ident is not None:
                    worker.join()
                    while processed < total:
                        consume()
                    finish_readv_dma(device)
                if pinned:
                    with self._entries_lock:
                        for entry in entries:
                            retained = unsafe.get(id(entry), 0)
                            if retained:
                                unsafe[id(entry)] = retained - 1
                            else:
                                entry.unpin()
                if error is not None:
                    raise error

        return restore()

    def _blockwise_get_kv_major(
        self,
        entries: Sequence[XDSLayerwiseBlockMetadata],
        progress_callback: Callable[[int, list[MemoryObj]], None],
    ) -> list[MemoryObj]:
        """Fallback preserves per-chunk KV-major byte order in staging."""
        with self._io_lock:
            retained: set[int] = set()
            try:
                if self._readv_poisoned:
                    raise RuntimeError(
                        "xDS direct KV target has an unresolved DMA failure"
                    )
                for layer in range(self.num_layers):
                    objects = []
                    published = False
                    unsafe_ids: set[int] = set()
                    try:
                        for entry in entries:
                            objects.append(
                                self._allocate_blockwise_layer_staging(
                                    entry.shapes,
                                    entry.dtypes,
                                    entry.fmt,
                                )
                            )

                        def submit(item: Any, layer: int = layer) -> int:
                            entry, obj = item
                            offset, _ = self._buffer_descriptor(obj)
                            return int(
                                self.client.submit_read(
                                    entry.blob_handle,
                                    self.buffer_ref,
                                    offset,
                                    layer * self.block_blob_layer_stride,
                                    entry.layer_io_size,
                                )
                            )

                        pump = XDSReadRingPump(
                            self.config.xds_io_queue_depth,
                            submit,
                            self.client.wait_any_result,
                            30.0,
                        )
                        for entry, obj in zip(entries, objects, strict=True):
                            pump.put((entry, obj))
                        pump.close_input()
                        pump.run()
                        error = pump.error
                        for _ in entries:
                            out = pump.get_completion()
                            entry, obj = out.item
                            if not out.safe_to_reuse:
                                unsafe_ids.add(id(obj))
                                retained.add(id(entry))
                                self._readv_failed_staging.append(obj)
                                self._readv_poisoned = True
                            if out.error is not None:
                                error = error or out.error
                            elif (
                                out.completion.status
                                or out.completion.completed_bytes != entry.layer_io_size
                            ):
                                error = error or RuntimeError(
                                    "xDS layerwise staging read failed"
                                )
                        if error is not None:
                            raise error
                        # GPU consumer owns staging references after publication.
                        published = True
                        progress_callback((layer + 1) * len(entries), objects)
                    finally:
                        if not published:
                            for obj in objects:
                                if id(obj) not in unsafe_ids:
                                    obj.ref_count_down()
                return []
            finally:
                with self._entries_lock:
                    for entry in entries:
                        if id(entry) not in retained:
                            entry.unpin()

    def blockwise_batched_get_blocking(
        self,
        block_keys: Sequence[CacheEngineKey],
        progress_callback: Callable[[int, list[MemoryObj]], None],
        trace_context: Optional[dict[str, Any]] = None,
    ) -> list[MemoryObj]:
        """Read block-owned blobs at 128 KiB layer offsets into GPU staging."""
        if trace_context is not None:
            trace_context["xds_backend_enter_ns"] = time.monotonic_ns()
        if not self.block_blob_layout:
            raise RuntimeError("xDS blockwise reads require block-blob layout")
        if not block_keys:
            return []

        metadata_entries: list[XDSLayerwiseBlockMetadata] = []
        if trace_context is not None:
            trace_context["xds_metadata_lookup_start_ns"] = time.monotonic_ns()
        with self._entries_lock:
            for key in block_keys:
                entry = self._entries.get(key)
                if not isinstance(entry, XDSLayerwiseBlockMetadata):
                    return []
                entry.pin()
                self._entries.move_to_end(key)
                metadata_entries.append(entry)
        if trace_context is not None:
            trace_context["xds_metadata_lookup_end_ns"] = time.monotonic_ns()

        if trace_context is not None:
            trace_context["xds_layout_build_start_ns"] = time.monotonic_ns()
        if any(entry.fmt == MemoryFormat.KV_2TD for entry in metadata_entries):
            return self._blockwise_get_kv_major(metadata_entries, progress_callback)
        shapes, dtypes, fmt, destination_offsets = self._combined_layer_layout(
            metadata_entries
        )
        if trace_context is not None:
            trace_context["xds_layout_build_end_ns"] = time.monotonic_ns()
        create_layer_stream = getattr(self.client, "create_layer_read_stream", None)
        use_persistent_stream = bool(
            callable(create_layer_stream)
            and self.config.get_extra_config_value(
                "xds_layerwise_persistent_stream", True
            )
        )
        if use_persistent_stream:
            try:
                return self._blockwise_batched_get_persistent_stream(
                    metadata_entries,
                    block_keys,
                    shapes,
                    dtypes,
                    fmt,
                    destination_offsets,
                    progress_callback,
                    trace_context,
                )
            finally:
                with self._entries_lock:
                    for entry in metadata_entries:
                        entry.unpin()

        total_bytes = 0
        io_start_ns = 0
        io_end_ns = 0
        transfer_start_ns = time.monotonic_ns()
        published_layers = 0
        native_window_calls = 0
        requested_window_layers = max(
            1,
            int(
                self.config.get_extra_config_value("xds_layerwise_io_window_layers", 8)
            ),
        )
        try:
            logical_layer_bytes = sum(
                shape.numel() * dtype.itemsize
                for shape, dtype in zip(shapes, dtypes, strict=True)
            )
            aligned_layer_bytes = align_to_io_unit(
                logical_layer_bytes, self.io_unit_size
            )
            max_pool_layers = max(1, self.registered_size // aligned_layer_bytes)
            io_window_layers = min(requested_window_layers, max_pool_layers)
            if trace_context is not None:
                trace_context["xds_io_window_layers"] = io_window_layers
                trace_context["xds_layer_staging_bytes"] = aligned_layer_bytes

            block_count = len(metadata_entries)
            layer_bytes = sum(entry.layer_io_size for entry in metadata_entries)
            total_bytes = self.num_layers * layer_bytes
            io_start_ns = time.monotonic_ns()

            window_start_layer = 0
            while window_start_layer < self.num_layers:
                current_window_layers = min(
                    io_window_layers,
                    self.num_layers - window_start_layer,
                )
                allocation_deadline = time.monotonic() + float(
                    self.config.blocking_timeout_secs
                )
                window_memory_objs = None
                while window_memory_objs is None:
                    window_memory_objs = self.batched_allocate(
                        shapes,
                        dtypes,
                        batch_size=current_window_layers,
                        fmt=fmt,
                        busy_loop=False,
                    )
                    if window_memory_objs is not None:
                        break
                    if time.monotonic() >= allocation_deadline:
                        raise TimeoutError(
                            "timed out waiting for the xDS layerwise GPU window"
                        )
                    time.sleep(0.0001)

                published_in_window = 0
                try:
                    requests = []
                    window_io_start_ns = time.monotonic_ns()
                    for local_layer_id, memory_obj in enumerate(window_memory_objs):
                        layer_id = window_start_layer + local_layer_id
                        layer_trace = (
                            trace_context["layers"][layer_id]
                            if trace_context is not None
                            else None
                        )
                        if layer_trace is not None:
                            layer_trace["xds_staging_acquire_start_ns"] = (
                                window_io_start_ns
                            )
                            layer_trace["xds_staging_acquire_end_ns"] = (
                                time.monotonic_ns()
                            )
                        buffer_offset, allocation_size = (
                            self._registered_buffer_descriptor(memory_obj)
                        )
                        memory_obj.chunk_count = len(block_keys)  # type: ignore[attr-defined]
                        for entry, destination_offset in zip(
                            metadata_entries, destination_offsets, strict=True
                        ):
                            if (
                                destination_offset + entry.layer_io_size
                                > allocation_size
                            ):
                                raise ValueError(
                                    "xDS layer read exceeds aggregate staging "
                                    "allocation"
                                )
                            requests.append(
                                (
                                    entry.blob_handle,
                                    self.buffer_ref,
                                    buffer_offset + destination_offset,
                                    layer_id * self.block_blob_layer_stride,
                                    entry.layer_io_size,
                                )
                            )

                    def publish_window_progress(
                        completed_count: int,
                        window_layer_count: int = current_window_layers,
                        base_layer: int = window_start_layer,
                        window_start_ns: int = window_io_start_ns,
                        memory_objs: list[MemoryObj] = window_memory_objs,
                    ) -> None:
                        nonlocal published_in_window
                        nonlocal published_layers
                        completed_window_layers = min(
                            window_layer_count,
                            completed_count // block_count,
                        )
                        callback_ns = time.monotonic_ns()
                        for local_layer_id in range(
                            published_in_window, completed_window_layers
                        ):
                            layer_id = base_layer + local_layer_id
                            layer_trace = (
                                trace_context["layers"][layer_id]
                                if trace_context is not None
                                else None
                            )
                            if layer_trace is not None:
                                layer_trace["xds_io_start_ns"] = window_start_ns
                                layer_trace["xds_io_end_ns"] = callback_ns
                                layer_trace["bytes"] = layer_bytes
                            progress_callback(
                                (layer_id + 1) * block_count,
                                [memory_objs[local_layer_id]],
                            )
                            published_in_window = local_layer_id + 1
                            published_layers = layer_id + 1

                    with self._io_lock:
                        self.client.read_window(
                            requests,
                            queue_depth=self.config.xds_io_queue_depth,
                            timeout=30.0,
                            progress_interval=block_count,
                            progress_callback=publish_window_progress,
                            validate_only=True,
                        )
                    native_window_calls += 1
                    if published_in_window < current_window_layers:
                        publish_window_progress(current_window_layers * block_count)
                finally:
                    for memory_obj in window_memory_objs[published_in_window:]:
                        memory_obj.ref_count_down()
                window_start_layer += current_window_layers
            io_end_ns = time.monotonic_ns()
        finally:
            with self._entries_lock:
                for entry in metadata_entries:
                    entry.unpin()

        if self.config.get_extra_config_value("benchmark_transfer_metrics", False):
            transfer_end_ns = time.monotonic_ns()
            logger.info(
                "[XDS_LAYERWISE_READ_METRICS] %s",
                json.dumps(
                    {
                        "worker_id": int(self.metadata.worker_id),
                        "bytes": total_bytes,
                        "layers": self.num_layers,
                        "blocks": len(block_keys),
                        "io_start_ns": io_start_ns,
                        "io_end_ns": io_end_ns,
                        "io_seconds": (io_end_ns - io_start_ns) / 1e9,
                        "transfer_start_ns": transfer_start_ns,
                        "transfer_end_ns": transfer_end_ns,
                        "transfer_seconds": (transfer_end_ns - transfer_start_ns) / 1e9,
                        "io_queue_depth": self.config.xds_io_queue_depth,
                        "io_window_layers": io_window_layers,
                        "native_window_calls": native_window_calls,
                    },
                    sort_keys=True,
                ),
            )
        return []

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
            peak_inflight_io=min(self.config.xds_io_queue_depth, len(task_infos)),
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
                            self.metadata.get_shapes(info["end"] - info["start"]),
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

    def _registered_buffer_descriptor(self, memory_obj: MemoryObj) -> tuple[int, int]:
        offset = memory_obj.raw_data.data_ptr() - self.registered_base_ptr
        size = memory_obj.get_physical_size()
        self._validate_registered_range(offset, size)
        return offset, size

    def _buffer_descriptor(self, memory_obj: MemoryObj) -> tuple[int, int]:
        offset, size = self._registered_buffer_descriptor(memory_obj)
        if size > self.blob_size:
            raise ValueError(
                f"xDS memory object size {size} exceeds blob size {self.blob_size}"
            )
        return offset, size

    def _write_direct_async_lease(
        self,
        lease: StagingLease,
    ) -> tuple[int, ...]:
        with self._io_lock:
            return self._write_direct_async_lease_exclusive(lease)

    def _write_direct_async_lease_exclusive(
        self,
        lease: StagingLease,
    ) -> tuple[int, ...]:
        """Persist a direct lease while owning exclusive completion access."""
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
                    raise RuntimeError("xDS async staging size changed after admission")
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
                for key, blob_handle in zip(lease.keys, blob_handles, strict=False):
                    metadata = self._entries.get(key)
                    if metadata is not None and metadata.blob_handle == blob_handle:
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
            del transfer_spec
            return self._batched_submit_put_task_serial(
                keys,
                objs,
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
            return [self.get_blocking(key) for key in keys]

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

    def release_layerwise_staging(self, memory_objs: Sequence[MemoryObj]) -> set[int]:
        """Release completed layer views or retain their slots for native refill.

        A persistent xDS stream owns each staging object until its final layer.
        Intermediate scatter completions release the native slot without freeing
        the underlying registered GPU allocation.
        """
        retained_ids: set[int] = set()
        with self._layer_read_slots_lock:
            for memory_obj in memory_objs:
                object_id = id(memory_obj)
                lease = self._layer_read_slots.get(object_id)
                if lease is None:
                    continue
                current_layer = lease.current_layer
                next_layer = current_layer + lease.slot_count
                if lease.terminal or next_layer >= lease.num_layers:
                    del self._layer_read_slots[object_id]
                    continue
                if not lease.published:
                    raise RuntimeError(
                        "xDS layer staging slot released before publication: "
                        f"slot={lease.slot_id}, layer={current_layer}"
                    )
                lease.stream.release_slot(current_layer)
                lease.current_layer = next_layer
                lease.published = False
                retained_ids.add(object_id)
        return retained_ids

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
        for target_key, (_, buffer_ref) in list(self._readv_targets.items()):
            try:
                self.client.unregister_buffer(buffer_ref)
                del self._readv_targets[target_key]
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

    def _blockwise_batched_get_persistent_stream(
        self,
        metadata_entries: Sequence[XDSLayerwiseBlockMetadata],
        block_keys: Sequence[CacheEngineKey],
        shapes: Sequence[torch.Size],
        dtypes: Sequence[torch.dtype],
        fmt: MemoryFormat,
        destination_offsets: Sequence[int],
        progress_callback: Callable[[int, list[MemoryObj]], None],
        trace_context: Optional[dict[str, Any]],
    ) -> list[MemoryObj]:
        """Admit a bounded concurrent read before it can reserve staging."""
        if trace_context is not None:
            trace_context["xds_persistent_enter_ns"] = time.monotonic_ns()
            trace_context["xds_admission_wait_start_ns"] = time.monotonic_ns()
        acquired = self._layer_read_admission.acquire(
            timeout=float(self.config.blocking_timeout_secs)
        )
        if not acquired:
            raise TimeoutError("timed out waiting for xDS layer read admission")
        if trace_context is not None:
            trace_context["xds_admission_wait_end_ns"] = time.monotonic_ns()
        try:
            return self._blockwise_batched_get_persistent_stream_admitted(
                metadata_entries,
                block_keys,
                shapes,
                dtypes,
                fmt,
                destination_offsets,
                progress_callback,
                trace_context,
            )
        finally:
            self._layer_read_admission.release()

    def _blockwise_batched_get_persistent_stream_admitted(
        self,
        metadata_entries: Sequence[XDSLayerwiseBlockMetadata],
        block_keys: Sequence[CacheEngineKey],
        shapes: Sequence[torch.Size],
        dtypes: Sequence[torch.dtype],
        fmt: MemoryFormat,
        destination_offsets: Sequence[int],
        progress_callback: Callable[[int, list[MemoryObj]], None],
        trace_context: Optional[dict[str, Any]],
    ) -> list[MemoryObj]:
        """Read all layers through one native stream with reusable GPU slots."""
        logical_layer_bytes = sum(
            shape.numel() * dtype.itemsize
            for shape, dtype in zip(shapes, dtypes, strict=True)
        )
        aligned_layer_bytes = align_to_io_unit(logical_layer_bytes, self.io_unit_size)
        requested_slots = min(
            6,
            max(
                1,
                int(
                    self.config.get_extra_config_value(
                        "xds_layerwise_io_window_layers", 6
                    )
                ),
            ),
        )
        max_pool_slots = max(1, self.registered_size // aligned_layer_bytes)
        slot_count = min(requested_slots, max_pool_slots, self.num_layers)
        allocation_deadline = time.monotonic() + float(
            self.config.blocking_timeout_secs
        )
        slot_memory_objs = None
        if trace_context is not None:
            trace_context["xds_staging_acquire_start_ns"] = time.monotonic_ns()
        while slot_memory_objs is None:
            allocated_slots = self.memory_allocator.allocate_at_most_contiguous(
                shapes,
                dtypes,
                max_batch_size=slot_count,
                fmt=fmt,
            )
            if allocated_slots:
                slot_memory_objs = allocated_slots
                slot_count = len(slot_memory_objs)
                break
            if time.monotonic() >= allocation_deadline:
                raise TimeoutError(
                    "timed out waiting for persistent xDS layer staging slots"
                )
            time.sleep(0.0001)
        if trace_context is not None:
            trace_context["xds_staging_acquire_end_ns"] = time.monotonic_ns()

        block_count = len(metadata_entries)
        layer_bytes = sum(entry.layer_io_size for entry in metadata_entries)
        total_bytes = self.num_layers * layer_bytes
        slot_descriptors: list[tuple[int, int, int]] = []
        stream = None
        native_stream_stats: dict[str, Any] = {}
        registered_leases = False
        published_layers = 0
        io_start_ns = 0
        try:
            if trace_context is not None:
                trace_context["xds_descriptor_build_start_ns"] = time.monotonic_ns()
            for memory_obj in slot_memory_objs:
                buffer_offset, allocation_size = self._registered_buffer_descriptor(
                    memory_obj
                )
                memory_obj.chunk_count = len(block_keys)  # type: ignore[attr-defined]
                for entry, destination_offset in zip(
                    metadata_entries, destination_offsets, strict=True
                ):
                    if destination_offset + entry.layer_io_size > allocation_size:
                        raise ValueError(
                            "xDS layer read exceeds persistent staging allocation"
                        )
                    slot_descriptors.append(
                        (
                            self.buffer_ref,
                            buffer_offset + destination_offset,
                            entry.layer_io_size,
                        )
                    )
            if trace_context is not None:
                trace_context["xds_descriptor_build_end_ns"] = time.monotonic_ns()

            def publish_native_layer(layer_id: int) -> None:
                nonlocal published_layers
                if layer_id != published_layers:
                    raise RuntimeError(
                        "xDS native stream published a noncontiguous layer: "
                        f"actual={layer_id}, expected={published_layers}"
                    )
                callback_ns = time.monotonic_ns()
                memory_obj = slot_memory_objs[layer_id % slot_count]
                with self._layer_read_slots_lock:
                    lease = self._layer_read_slots.get(id(memory_obj))
                    if lease is None or lease.current_layer != layer_id:
                        raise RuntimeError(
                            "xDS native stream/staging generation mismatch: "
                            f"layer={layer_id}"
                        )
                    lease.published = True
                if trace_context is not None:
                    layer_trace = trace_context["layers"][layer_id]
                    layer_trace["xds_staging_acquire_start_ns"] = io_start_ns
                    layer_trace["xds_staging_acquire_end_ns"] = io_start_ns
                    layer_trace["xds_io_start_ns"] = io_start_ns
                    layer_trace["xds_io_end_ns"] = callback_ns
                    layer_trace["bytes"] = layer_bytes
                progress_callback(
                    (layer_id + 1) * block_count,
                    [memory_obj],
                )
                published_layers += 1

            stream = self.client.create_layer_read_stream(
                [entry.blob_handle for entry in metadata_entries],
                slot_descriptors,
                slot_count=slot_count,
                num_layers=self.num_layers,
                blob_layer_stride=self.block_blob_layer_stride,
                queue_depth=self.config.xds_io_queue_depth,
                timeout=30.0,
                progress_callback=publish_native_layer,
            )
            with self._layer_read_slots_lock:
                for slot_id, memory_obj in enumerate(slot_memory_objs):
                    self._layer_read_slots[id(memory_obj)] = XDSLayerReadSlotLease(
                        stream=stream,
                        memory_obj=memory_obj,
                        slot_id=slot_id,
                        slot_count=slot_count,
                        num_layers=self.num_layers,
                        current_layer=slot_id,
                    )
            registered_leases = True
            if trace_context is not None:
                trace_context["xds_io_lock_wait_start_ns"] = time.monotonic_ns()
            with self._io_lock.read():
                io_start_ns = time.monotonic_ns()
                if trace_context is not None:
                    trace_context["xds_io_lock_wait_end_ns"] = io_start_ns
                    trace_context["xds_native_run_start_ns"] = io_start_ns
                native_stream_stats = stream.run() or {}
                if trace_context is not None:
                    trace_context["xds_native_run_end_ns"] = time.monotonic_ns()
            first_submit_ns = int(native_stream_stats.get("first_submit_ns", 0))
            last_completion_ns = int(native_stream_stats.get("last_completion_ns", 0))
            if first_submit_ns > 0:
                io_start_ns = first_submit_ns
                if trace_context is not None:
                    trace_context["xds_native_first_submit_ns"] = first_submit_ns
                    for layer_trace in trace_context["layers"]:
                        layer_trace["xds_io_start_ns"] = first_submit_ns
            if last_completion_ns > 0 and trace_context is not None:
                trace_context["xds_native_last_completion_ns"] = last_completion_ns
            if published_layers != self.num_layers:
                raise RuntimeError(
                    "xDS native stream did not publish every layer: "
                    f"published={published_layers}, expected={self.num_layers}"
                )
        except BaseException:
            cleanup_objects: list[MemoryObj] = []
            if registered_leases:
                with self._layer_read_slots_lock:
                    for memory_obj in slot_memory_objs:
                        lease = self._layer_read_slots.get(id(memory_obj))
                        if lease is None:
                            continue
                        lease.terminal = True
                        if not lease.published:
                            del self._layer_read_slots[id(memory_obj)]
                            cleanup_objects.append(memory_obj)
            else:
                cleanup_objects.extend(slot_memory_objs)
            for memory_obj in cleanup_objects:
                memory_obj.ref_count_down()
            raise

        io_end_ns = time.monotonic_ns()
        if trace_context is not None:
            trace_context["xds_io_window_layers"] = slot_count
            trace_context["xds_layer_staging_bytes"] = aligned_layer_bytes
        if self.config.get_extra_config_value("benchmark_transfer_metrics", False):
            logger.info(
                "[XDS_LAYERWISE_READ_METRICS] %s",
                json.dumps(
                    {
                        "worker_id": int(self.metadata.worker_id),
                        "bytes": total_bytes,
                        "layers": self.num_layers,
                        "blocks": len(block_keys),
                        "io_start_ns": io_start_ns,
                        "io_end_ns": io_end_ns,
                        "io_seconds": (io_end_ns - io_start_ns) / 1e9,
                        "transfer_start_ns": io_start_ns,
                        "transfer_end_ns": io_end_ns,
                        "transfer_seconds": (io_end_ns - io_start_ns) / 1e9,
                        "io_queue_depth": self.config.xds_io_queue_depth,
                        "io_window_layers": slot_count,
                        "native_window_calls": 1,
                        "persistent_layer_stream": True,
                        **native_stream_stats,
                    },
                    sort_keys=True,
                ),
            )
        return []

    def _get_readv_to_gpu(
        self,
        keys: list[CacheEngineKey],
        starts: list[int],
        ends: list[int],
        gpu_connector: Any,
        **kwargs: Any,
    ) -> Optional[BackendTransferResult]:
        """Restore eligible chunks to final KV blocks; never fall back after DMA."""
        from lmcache.v1.gpu_connector.xds_readv import (
            cuda_readv_flush_function,
            finish_readv_dma,
        )

        planner = getattr(gpu_connector, "plan_xds_readv", None)
        if (
            self.config.use_layerwise
            or not callable(planner)
            or not callable(getattr(self.client, "submit_readv", None))
        ):
            return None
        if not (len(keys) == len(starts) == len(ends)):
            raise ValueError("xDS readv batch lengths differ")
        with self._io_lock:
            if self._readv_poisoned:
                raise RuntimeError("xDS direct KV target has an unresolved DMA failure")
            with self._put_tasks_lock:
                if any(key in self._pending_objects for key in keys):
                    return None
            pinned = []
            unsafe: dict[int, int] = {}
            try:
                selected = []
                for index, (key, start, end) in enumerate(
                    zip(keys, starts, ends, strict=True)
                ):
                    with self._entries_lock:
                        entry = self._entries.get(key)
                        if entry is None:
                            continue
                        entry.pin()
                        pinned.append(entry)
                        self._entries.move_to_end(key)
                    if entry.blob_offset % self.io_unit_size:
                        return None
                    selected.append((index, entry, start, end))
                if not selected:
                    return None
                chunks = [
                    (entry.shapes, entry.dtypes, entry.fmt, start, end)
                    for _, entry, start, end in selected
                ]
                batch_planner = getattr(gpu_connector, "plan_xds_readv_batch", None)
                if callable(batch_planner):
                    plans = batch_planner(chunks, self.io_unit_size, **kwargs)
                else:
                    plans = [
                        planner(*chunk, self.io_unit_size, **kwargs) for chunk in chunks
                    ]
                if plans is None or len(plans) != len(selected):
                    return None
                items = []
                for (index, entry, _, _), plan in zip(selected, plans, strict=True):
                    if plan is None or plan.size != entry.size:
                        return None
                    items.append((index, entry, plan))
                # Full-block plans share one KV pool. Check block ownership once,
                # not every K/V/layer fragment, avoiding a large address sort.
                if len(items) > 1:
                    targets = tuple(t.data_ptr() for t in items[0][2].tensors)
                    seen_blocks: set[int] = set()
                    for _, _, plan in items:
                        if (
                            plan.tensors is not items[0][2].tensors
                            and tuple(t.data_ptr() for t in plan.tensors) != targets
                        ):
                            return None
                        if seen_blocks.intersection(plan.block_ids):
                            return None
                        seen_blocks.update(plan.block_ids)
                cuda_readv_flush_function()  # Fail before submission if unsupported.
                device = items[0][2].tensors[0].device
                pools = {id(plan.tensors): plan.tensors for _, _, plan in items}
                if any(t.device != device for pool in pools.values() for t in pool):
                    return None
                # The engine owns these destination blocks through completion.
                # Finish prior users before DMA; no GPU copy kernel is launched.
                torch.cuda.synchronize(device)
                # Target identity was checked across all plans above. Resolve
                # registrations once for the batch, before the first DMA.
                refs = []
                for tensor in items[0][2].tensors:
                    target_key = (
                        tensor.data_ptr(),
                        tensor.numel() * tensor.element_size(),
                    )
                    if target_key not in self._readv_targets:
                        ref = int(self.client.register_gpu_buffer(*target_key))
                        if ref <= 0:
                            raise RuntimeError("invalid final KV buffer reference")
                        self._readv_targets[target_key] = (tensor, ref)
                    refs.append(self._readv_targets[target_key][1])

                def submit(item: Any) -> int:
                    _, entry, plan = item
                    # All metadata and cross-chunk block ownership are validated
                    # already. Materialize this vector just in time, while prior
                    # chunks can execute; the pump drains I/O on submit errors.
                    return plan.submit(
                        self.client,
                        entry.blob_handle,
                        entry.blob_offset,
                        refs,
                    )

                pump = XDSReadRingPump(
                    self.config.xds_io_queue_depth,
                    submit,
                    self.client.wait_any_result,
                    30.0,
                )
                for item in items:
                    pump.put(item)
                pump.close_input()
                begin = time.perf_counter()
                pump.run()
                completed = [False] * len(keys)
                completed_bytes = 0
                error = pump.error
                for _ in items:
                    outcome = pump.get_completion()
                    index, entry, plan = outcome.item
                    if not outcome.safe_to_reuse:
                        unsafe[id(entry)] = unsafe.get(id(entry), 0) + 1
                        self._readv_poisoned = True
                    if outcome.error is not None:
                        error = error or outcome.error
                        continue
                    result = outcome.completion
                    if result.status or result.completed_bytes != plan.size:
                        error = error or RuntimeError(
                            "xDS direct KV read failed or was short"
                        )
                    else:
                        completed[index] = True
                        completed_bytes += plan.size
                finish_readv_dma(device)
                if error is not None:
                    raise error
                elapsed = time.perf_counter() - begin
                return self._make_transfer_result(
                    completed,
                    requested_bytes=sum(p.size for _, _, p in items),
                    completed_bytes=completed_bytes,
                    window_size=max(1, len(items)),
                    window_count=1,
                    peak_staging_bytes=0,
                    storage_seconds=elapsed,
                    gpu_seconds=0.0,
                    slot_wait_seconds=0.0,
                    short_io_chunks=0,
                    peak_inflight_io=min(len(items), self.config.xds_io_queue_depth),
                )
            finally:
                with self._entries_lock:
                    for entry in pinned:
                        retained = unsafe.get(id(entry), 0)
                        if retained:
                            unsafe[id(entry)] = retained - 1
                        else:
                            entry.unpin()
