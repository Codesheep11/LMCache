# SPDX-License-Identifier: Apache-2.0
"""CascadeKV backend using xDS service-owned CPU staging memory."""

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, Optional, Sequence, Union
import asyncio
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.logging import init_logger
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

DEFAULT_BLOB_TIMEOUT_SECS = 5.0
DEFAULT_IO_TIMEOUT_SECS = 30.0
LEGACY_CASCADEKV_SLOT_COUNT_DEFAULT = 8
CASCADEKV_READ_SLOT_COUNT_DEFAULT = 8
CASCADEKV_WRITE_SLOT_COUNT_DEFAULT = 64


def _guard_operation(method: Callable[..., Any]) -> Callable[..., Any]:
    """Reject new public operations after CascadeKV shutdown begins."""

    @wraps(method)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._active_operation():
            return method(self, *args, **kwargs)

    return guarded


def _guard_async_operation(method: Callable[..., Any]) -> Callable[..., Any]:
    """Async counterpart of :func:`_guard_operation`."""

    @wraps(method)
    async def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._active_operation():
            return await method(self, *args, **kwargs)

    return guarded


def _resolve_slot_counts(config: LMCacheEngineConfig) -> tuple[int, int]:
    """Resolve new read/write banks with compatibility for old slot configs."""
    read_slots = int(config.cascadekv_read_slot_count)
    write_slots = int(config.cascadekv_write_slot_count)
    legacy_slots = int(config.cascadekv_slot_count)
    user_set_keys = getattr(config, "_user_set_keys", set())
    legacy_explicit = "cascadekv_slot_count" in user_set_keys
    split_explicit = bool(
        {"cascadekv_read_slot_count", "cascadekv_write_slot_count"} & user_set_keys
    )
    if (legacy_explicit and not split_explicit) or (
        not user_set_keys
        and legacy_slots != LEGACY_CASCADEKV_SLOT_COUNT_DEFAULT
        and read_slots == CASCADEKV_READ_SLOT_COUNT_DEFAULT
        and write_slots == CASCADEKV_WRITE_SLOT_COUNT_DEFAULT
    ):
        return legacy_slots, legacy_slots
    return read_slots, write_slots


class CascadeKVBackend(AllocatorBackendInterface):
    """xDS storage path that stages SSD I/O through service CPU memory."""

    skip_local_cpu_writeback = True

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        xds_client: Any,
        dst_device: str = "cpu",
    ) -> None:
        super().__init__(dst_device="cpu")
        self.config = config
        self.metadata = metadata
        self.client = xds_client
        self._closed = False
        self._closing = False
        self._teardown_started = False
        self._active_operations = 0
        self._lifecycle_condition = threading.Condition()
        self._operation_local = threading.local()
        self._buffer_freed = False
        self._client_closed = False
        self._host_registered = False
        self._read_allocator_closed = False
        self._write_allocator_closed = False
        self._poisoned_error: Optional[BaseException] = None
        self._quarantined_objects: list[MemoryObj] = []
        self._quarantined_xds_requests: dict[int, MemoryObj] = {}
        self._write_scheduler: Optional[XDSAsyncWriteScheduler] = None
        self._entries: OrderedDict[CacheEngineKey, XDSBlobMetadata] = OrderedDict()
        self._entries_lock = threading.RLock()
        self._put_tasks: set[CacheEngineKey] = set()
        self._pending_objects: dict[CacheEngineKey, MemoryObj] = {}
        self._pending_transfer_events: dict[CacheEngineKey, Any] = {}
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
        self.read_slot_count, self.write_slot_count = _resolve_slot_counts(config)
        self.read_staging_size = self.read_slot_count * self.blob_size
        self.write_staging_size = self.write_slot_count * self.blob_size
        self.staging_size = self.read_staging_size + self.write_staging_size
        self.staging_buffer_ref: Optional[int] = None
        try:
            self.staging_view, self.staging_buffer_ref = xds_client.alloc_buffer(
                self.staging_size
            )
            self.staging_tensor = torch.frombuffer(
                self.staging_view,
                dtype=torch.uint8,
            )
            if self.staging_tensor.numel() != self.staging_size:
                raise RuntimeError(
                    "xDS service returned an unexpected CPU staging buffer size: "
                    f"{self.staging_tensor.numel()} != {self.staging_size}"
                )
            self.staging_base_ptr = self.staging_tensor.data_ptr()
            self._cudart = torch.cuda.cudart()
            register_status = int(
                self._cudart.cudaHostRegister(
                    self.staging_base_ptr,
                    self.staging_size,
                    0,
                )
            )
            if register_status != 0:
                error_string = self._cudart.cudaGetErrorString(register_status)
                raise RuntimeError(
                    "CascadeKV cudaHostRegister failed: "
                    f"status={register_status}, error={error_string}"
                )
            self._host_registered = True

            self.read_staging_tensor = self.staging_tensor[: self.read_staging_size]
            self.write_staging_tensor = self.staging_tensor[self.read_staging_size :]
            self.read_staging_base_ptr = self.read_staging_tensor.data_ptr()
            self.write_staging_base_ptr = self.write_staging_tensor.data_ptr()
            self.read_memory_allocator = TensorMemoryAllocator(
                self.read_staging_tensor,
                align_bytes=self.io_unit_size,
            )
            self.write_memory_allocator = TensorMemoryAllocator(
                self.write_staging_tensor,
                align_bytes=self.io_unit_size,
            )
            self.memory_allocator = self.initialize_allocator(config, metadata)
            if config.cascadekv_async_write:
                self._write_scheduler = XDSAsyncWriteScheduler(
                    write_budget_bytes=self.write_staging_size,
                    submit_write=self._submit_async_lease,
                    commit=self._commit_async_lease,
                    rollback=self._rollback_async_lease,
                )
        except BaseException:
            self._cleanup_failed_initialization()
            raise

    def _cleanup_failed_initialization(self) -> None:
        """Best-effort unwind of service and CUDA resources during construction."""
        try:
            if self._write_scheduler is not None:
                self._write_scheduler.close(timeout=DEFAULT_IO_TIMEOUT_SECS)
        except BaseException:
            logger.exception("failed to close CascadeKV scheduler after init failure")
        try:
            if self._host_registered:
                self._cudart.cudaHostUnregister(self.staging_base_ptr)
                self._host_registered = False
        except BaseException:
            logger.exception(
                "failed to unregister CascadeKV staging after init failure"
            )
        try:
            if self.staging_buffer_ref is not None and not self._buffer_freed:
                self.client.free_buffer(self.staging_buffer_ref)
                self._buffer_freed = True
        except BaseException:
            logger.exception("failed to free CascadeKV staging after init failure")
        try:
            client_close = getattr(self.client, "close", None)
            if callable(client_close) and not self._client_closed:
                client_close()
                self._client_closed = True
        except BaseException:
            logger.exception("failed to close xDS client after CascadeKV init failure")

    @contextmanager
    def _active_operation(self, allow_poison: bool = False) -> Iterator[None]:
        """Hold the backend lifecycle open for one re-entrant public operation."""
        condition = getattr(self, "_lifecycle_condition", None)
        if condition is None:
            yield
            return
        depth = int(getattr(self._operation_local, "depth", 0))
        if depth:
            self._operation_local.depth = depth + 1
            try:
                yield
            finally:
                self._operation_local.depth = depth
            return
        with condition:
            if self._closing or self._closed or self._teardown_started:
                raise RuntimeError("CascadeKVBackend is closing or closed")
            if self._poisoned_error is not None and not allow_poison:
                raise RuntimeError(
                    "CascadeKVBackend is poisoned by an incomplete transfer"
                ) from self._poisoned_error
            self._active_operations += 1
            self._operation_local.depth = 1
        try:
            yield
        finally:
            with condition:
                self._operation_local.depth = 0
                self._active_operations -= 1
                condition.notify_all()

    def __str__(self) -> str:
        return self.__class__.__name__

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> TensorMemoryAllocator:
        del config, metadata
        return self.write_memory_allocator

    def get_allocator_backend(self) -> "CascadeKVBackend":
        return self

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        return self.memory_allocator

    @_guard_operation
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        del eviction, busy_loop
        return self.memory_allocator.allocate(shapes, dtypes, fmt)

    @_guard_operation
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
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

    def calculate_chunk_budget(self) -> int:
        return self.write_slot_count

    @_guard_operation
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

    def _buffer_descriptor(self, memory_obj: MemoryObj) -> tuple[int, int]:
        offset = memory_obj.raw_data.data_ptr() - self.staging_base_ptr
        size = memory_obj.get_physical_size()
        if offset < 0 or offset + size > self.staging_size:
            raise ValueError(
                "CascadeKV memory object is outside the registered staging range"
            )
        if offset % self.io_unit_size != 0 or size % self.io_unit_size != 0:
            raise ValueError(
                "CascadeKV memory object must be I/O-unit aligned: "
                f"offset={offset}, size={size}, io_unit={self.io_unit_size}"
            )
        if size > self.blob_size:
            raise ValueError(
                "CascadeKV memory object size "
                f"{size} exceeds blob size {self.blob_size}"
            )
        return offset, size

    @staticmethod
    def _ensure_cuda_transfer_complete(
        gpu_connector: Any,
        direction: str,
        event: Any = None,
    ) -> None:
        """Prove stream completion before a staging slot can be recycled."""
        errors: list[BaseException] = []
        if event is not None:
            try:
                event.synchronize()
                return
            except BaseException as error:
                errors.append(error)
        if gpu_connector is not None:
            try:
                recovery_event = gpu_connector.record_transfer_event(direction)
                recovery_event.synchronize()
                return
            except BaseException as error:
                errors.append(error)
        try:
            torch.cuda.synchronize()
            return
        except BaseException as error:
            errors.append(error)
        raise RuntimeError(
            "could not establish CUDA transfer completion for CascadeKV slot reuse"
        ) from errors[0]

    def _quarantine_cuda_objects(
        self,
        memory_objs: Sequence[MemoryObj],
        error: BaseException,
    ) -> None:
        """Retain unsafe staging slots and reject future backend operations."""
        self._quarantined_objects.extend(memory_objs)
        self._poisoned_error = error

    def _quarantine_xds_object(
        self,
        request_id: int,
        memory_obj: MemoryObj,
        error: BaseException,
    ) -> None:
        """Retain a staging slot until its exact late xDS completion is drained."""
        self._quarantined_objects.append(memory_obj)
        self._quarantined_xds_requests[request_id] = memory_obj
        self._poisoned_error = error

    def drain_quarantined_xds_transfers(
        self,
        timeout: float = DEFAULT_IO_TIMEOUT_SECS,
    ) -> None:
        """Drain exact late xDS completions and make quarantined slots reusable."""
        with self._active_operation(allow_poison=True):
            pending = dict(self._quarantined_xds_requests)
            while pending:
                completion = self.client.wait_any_result(timeout=timeout)
                request_id = int(completion.request_id)
                memory_obj = pending.pop(request_id, None)
                if memory_obj is None:
                    raise RuntimeError(
                        "xDS returned an unknown completion while draining "
                        f"quarantined CascadeKV request {request_id}"
                    )
                memory_obj.ref_count_down()
                self._quarantined_xds_requests.pop(request_id, None)
                self._quarantined_objects = [
                    item for item in self._quarantined_objects if item is not memory_obj
                ]
            if not self._quarantined_objects:
                self._poisoned_error = None

    def _wait_for_completion(
        self,
        request_id: int,
        expected_bytes: int,
        operation: str,
    ) -> None:
        completion = self.client.wait_any_result(timeout=DEFAULT_IO_TIMEOUT_SECS)
        if int(completion.request_id) != request_id:
            raise RuntimeError(
                "unexpected xDS completion request id: "
                f"expected={request_id}, got={completion.request_id}"
            )
        self._validate_io_result(completion, expected_bytes, operation)

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

    def _write_one(self, key: CacheEngineKey, memory_obj: MemoryObj) -> bool:
        with self._entries_lock:
            if key in self._entries:
                return False

        blob_handle = int(self.client.get_blob(timeout=DEFAULT_BLOB_TIMEOUT_SECS))
        if blob_handle <= 0:
            raise RuntimeError(f"xDS returned an invalid blob handle {blob_handle}")

        try:
            buffer_offset, size = self._buffer_descriptor(memory_obj)
            with self._io_lock:
                request_id = int(
                    self.client.submit_write(
                        blob_handle,
                        self.staging_buffer_ref,
                        buffer_offset,
                        0,
                        size,
                    )
                )
                self._wait_for_completion(request_id, size, "write")

            metadata = XDSBlobMetadata(
                blob_handle=blob_handle,
                size=size,
                shapes=memory_obj.get_shapes(),
                dtypes=memory_obj.get_dtypes(),
                fmt=memory_obj.get_memory_format(),
            )
            with self._entries_lock:
                self._entries[key] = metadata
            return True
        except Exception:
            self.client.release_blob(blob_handle)
            raise

    def _write_staging_lease(self, lease: StagingLease) -> tuple[int, ...]:
        transfer_event = getattr(lease, "transfer_event", None)
        if transfer_event is not None:
            try:
                transfer_event.synchronize()
            except BaseException as transfer_error:
                try:
                    self._ensure_cuda_transfer_complete(
                        getattr(lease, "gpu_connector", None),
                        "store",
                        transfer_event,
                    )
                except BaseException as recovery_error:
                    lease.cuda_quarantined = True
                    self._quarantine_cuda_objects(lease.objects, recovery_error)
                    raise recovery_error from transfer_error
                raise

        items: list[dict[str, Any]] = []
        blob_handles: list[int] = []
        try:
            for memory_obj, expected_size in zip(
                lease.objects,
                lease.sizes,
                strict=True,
            ):
                blob_handle = int(
                    self.client.get_blob(timeout=DEFAULT_BLOB_TIMEOUT_SECS)
                )
                if blob_handle <= 0:
                    raise RuntimeError(
                        f"xDS returned an invalid blob handle {blob_handle}"
                    )
                blob_handles.append(blob_handle)
                offset, size = self._buffer_descriptor(memory_obj)
                if size != expected_size:
                    raise RuntimeError(
                        "CascadeKV staging size changed after write admission"
                    )
                items.append(
                    {
                        "blob_handle": blob_handle,
                        "offset": offset,
                        "size": size,
                    }
                )

            pump = XDSReadRingPump(
                queue_depth=self.config.cascadekv_io_queue_depth,
                submit=lambda item: int(
                    self.client.submit_write(
                        item["blob_handle"],
                        self.staging_buffer_ref,
                        item["offset"],
                        0,
                        item["size"],
                    )
                ),
                wait_any=self.client.wait_any_result,
                timeout=DEFAULT_IO_TIMEOUT_SECS,
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
                item = ring_completion.item
                completion = ring_completion.completion
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
                if all("status=0" in failure for failure in failures):
                    raise RuntimeError("short xDS write: " + "; ".join(failures))
                raise RuntimeError(
                    "CascadeKV async write failed: " + "; ".join(failures)
                )
            outcome = tuple(blob_handles)
            lease.io_outcome = outcome
            return outcome
        except BaseException:
            for blob_handle in blob_handles:
                self.client.release_blob(blob_handle)
            raise

    @staticmethod
    def _release_lease_objects(lease: StagingLease) -> None:
        if getattr(lease, "resources_released", False) or getattr(
            lease, "cuda_quarantined", False
        ):
            return
        for memory_obj in lease.objects:
            memory_obj.ref_count_down()
        lease.resources_released = True

    def _finish_lease_futures(
        self,
        lease: StagingLease,
        error: Optional[BaseException] = None,
    ) -> None:
        futures = tuple(getattr(lease, "futures", ()))
        callback = getattr(lease, "on_complete_callback", None)
        for key, future in zip(lease.keys, futures, strict=False):
            if error is not None:
                if not future.done():
                    future.set_exception(error)
                continue
            if callback is not None:
                try:
                    callback(key)
                except Exception:
                    logger.exception("CascadeKV put completion callback failed")
            if not future.done():
                future.set_result(True)

    def _submit_async_lease(self, lease: StagingLease) -> tuple[int, ...]:
        return self._write_staging_lease(lease)

    def _commit_async_lease(self, lease: StagingLease, outcome: Any) -> None:
        blob_handles = tuple(outcome)
        if len(blob_handles) != len(lease.keys):
            raise RuntimeError("CascadeKV write outcome has the wrong handle count")
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
                self._pending_transfer_events.pop(key, None)
        self._finish_lease_futures(lease)
        self._release_lease_objects(lease)

    def _rollback_async_lease(
        self,
        lease: StagingLease,
        error: BaseException,
    ) -> None:
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
                self._pending_transfer_events.pop(key, None)
        self._finish_lease_futures(lease, error=error)
        self._release_lease_objects(lease)
        logger.error(
            "[CascadeKV Async Rollback] chunks=%d bytes=%d error=%s",
            len(lease.keys),
            lease.credited_bytes,
            error,
        )

    @_guard_operation
    def flush_async_writes(
        self,
        timeout: Optional[float] = None,
    ) -> SchedulerSnapshot:
        """Wait until all admitted CascadeKV writes are durable."""
        if self._write_scheduler is None:
            raise RuntimeError("CascadeKV asynchronous writes are not enabled")
        return self._write_scheduler.drain(timeout)

    @_guard_operation
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: list[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[list[Future]]:
        del transfer_spec
        if len(keys) != len(objs):
            raise ValueError("keys and objs must have the same length")

        futures: list[Future] = [Future() for _ in keys]
        task_infos: list[tuple[int, CacheEngineKey, MemoryObj]] = []
        seen: set[CacheEngineKey] = set()
        with self._put_tasks_lock, self._entries_lock:
            for index, (key, memory_obj) in enumerate(zip(keys, objs, strict=True)):
                if key in seen or key in self._put_tasks or key in self._entries:
                    futures[index].set_result(False)
                    continue
                seen.add(key)
                self._put_tasks.add(key)
                task_infos.append((index, key, memory_obj))

        window_size = min(
            self.write_slot_count,
            int(self.config.cascadekv_write_window_size),
        )
        for window_start in range(0, len(task_infos), window_size):
            window = task_infos[window_start : window_start + window_size]
            indexes = tuple(item[0] for item in window)
            lease_keys = tuple(item[1] for item in window)
            objects = tuple(item[2] for item in window)
            sizes = tuple(memory_obj.get_physical_size() for memory_obj in objects)
            credited_bytes = sum(sizes)
            credits_acquired = False
            references_added = False
            pending_installed = False
            try:
                if self._write_scheduler is not None:
                    self._write_scheduler.acquire_credits(
                        credited_bytes,
                        timeout=float(self.config.blocking_timeout_secs),
                    )
                    credits_acquired = True
                for memory_obj in objects:
                    memory_obj.ref_count_up()
                references_added = True
                lease = StagingLease(
                    keys=lease_keys,
                    indexes=indexes,
                    objects=objects,
                    sizes=sizes,
                    credited_bytes=credited_bytes,
                )
                lease.futures = tuple(futures[index] for index in indexes)
                lease.on_complete_callback = on_complete_callback
                lease.resources_released = False
                lease.io_outcome = None
                lease.transition(LeaseState.PENDING)
                with self._put_tasks_lock:
                    for key, memory_obj in zip(lease_keys, objects, strict=True):
                        self._pending_objects[key] = memory_obj
                        self._pending_transfer_events.pop(key, None)
                pending_installed = True
                if self._write_scheduler is not None:
                    self._write_scheduler.enqueue(lease)
                else:
                    lease.transition(LeaseState.INFLIGHT)
                    outcome = self._write_staging_lease(lease)
                    self._commit_async_lease(lease, outcome)
                    lease.transition(LeaseState.COMMITTED)
            except BaseException as exc:
                if pending_installed:
                    with self._put_tasks_lock:
                        for key in lease_keys:
                            self._put_tasks.discard(key)
                            self._pending_objects.pop(key, None)
                            self._pending_transfer_events.pop(key, None)
                if references_added:
                    for memory_obj in objects:
                        memory_obj.ref_count_down()
                if credits_acquired and self._write_scheduler is not None:
                    self._write_scheduler.release_credits(credited_bytes)
                for index in indexes:
                    if not futures[index].done():
                        futures[index].set_exception(exc)
        return futures

    def _batched_submit_put_task_serial(
        self,
        keys: Sequence[CacheEngineKey],
        objs: list[MemoryObj],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> list[Future]:
        """Preserve the pre-pipeline single-I/O path for subclass fallback."""
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
                        logger.exception("storage put completion callback failed")
                future.set_result(stored)
            except Exception as error:
                future.set_exception(error)
            finally:
                with self._put_tasks_lock:
                    self._put_tasks.discard(key)
        return futures

    @_guard_async_operation
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
        await asyncio.gather(*(asyncio.wrap_future(future) for future in futures or []))

    @_guard_operation
    def batched_put_from_gpu_blocking(
        self,
        keys: Sequence[CacheEngineKey],
        starts: list[int],
        ends: list[int],
        gpu_connector: Any,
        **kwargs: Any,
    ) -> Optional[BackendWriteSubmissionResult]:
        """Gather a logical batch into bounded CPU slots for async SSD writes."""
        if not self.config.cascadekv_overlap or self._write_scheduler is None:
            return None
        if not (len(keys) == len(starts) == len(ends)):
            raise ValueError("CascadeKV streaming write inputs must have equal lengths")

        scheduler = self._write_scheduler
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
                    self._put_tasks.add(key)
                    task_infos.append(
                        {"index": index, "key": key, "start": start, "end": end}
                    )

            reserved_keys = {info["key"] for info in task_infos}

            lease_size = min(
                self.write_slot_count,
                int(self.config.cascadekv_write_window_size),
            )
            for lease_start in range(0, len(task_infos), lease_size):
                lease_infos = task_infos[lease_start : lease_start + lease_size]
                reserved_bytes = len(lease_infos) * self.blob_size
                objects: list[MemoryObj] = []
                credits_acquired = False
                pending_installed = False
                enqueued = False
                gpu_transfer_started = False
                event = None
                staging_start = time.perf_counter()
                timeout_seconds = float(self.config.blocking_timeout_secs)
                admission_deadline = time.monotonic() + timeout_seconds
                try:
                    backpressure_seconds += scheduler.acquire_credits(
                        reserved_bytes,
                        timeout=timeout_seconds,
                    )
                    credits_acquired = True
                    for info in lease_infos:
                        slot_wait_started = time.monotonic()
                        while True:
                            if (
                                self.write_memory_allocator.address_manager.get_free_size()
                                >= self.blob_size
                            ):
                                memory_obj = self.write_memory_allocator.allocate(
                                    self.metadata.get_shapes(
                                        info["end"] - info["start"]
                                    ),
                                    self.metadata.get_dtypes(),
                                )
                            else:
                                memory_obj = None
                            if memory_obj is not None:
                                break
                            remaining = admission_deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError(
                                    "timed out waiting for a CascadeKV write slot"
                                )
                            time.sleep(min(0.001, remaining))
                        backpressure_seconds += time.monotonic() - slot_wait_started
                        objects.append(memory_obj)
                    sizes = tuple(obj.get_physical_size() for obj in objects)
                    gpu_transfer_started = True
                    gpu_connector.batched_from_gpu(
                        objects,
                        [info["start"] for info in lease_infos],
                        [info["end"] for info in lease_infos],
                        **kwargs,
                    )
                    event = gpu_connector.record_transfer_event("store")
                    lease = StagingLease(
                        keys=tuple(info["key"] for info in lease_infos),
                        indexes=tuple(info["index"] for info in lease_infos),
                        objects=tuple(objects),
                        sizes=sizes,
                        credited_bytes=sum(sizes),
                        reserved_bytes=reserved_bytes,
                    )
                    lease.futures = ()
                    lease.on_complete_callback = None
                    lease.transfer_event = event
                    lease.gpu_connector = gpu_connector
                    lease.cuda_quarantined = False
                    lease.resources_released = False
                    lease.io_outcome = None
                    lease.transition(LeaseState.PENDING)
                    with self._put_tasks_lock:
                        for key, memory_obj in zip(
                            lease.keys,
                            lease.objects,
                            strict=True,
                        ):
                            self._pending_objects[key] = memory_obj
                            self._pending_transfer_events[key] = event
                    pending_installed = True
                    scheduler.enqueue(lease)
                    enqueued = True
                    reserved_keys.difference_update(lease.keys)
                    for index in lease.indexes:
                        accepted[index] = True
                    accepted_bytes += sum(sizes)
                    staging_seconds += time.perf_counter() - staging_start
                except BaseException as transfer_error:
                    recovery_error: Optional[BaseException] = None
                    if gpu_transfer_started and not enqueued:
                        try:
                            self._ensure_cuda_transfer_complete(
                                gpu_connector,
                                "store",
                                event,
                            )
                        except BaseException as error:
                            recovery_error = error
                            self._quarantine_cuda_objects(objects, error)
                    if pending_installed:
                        with self._put_tasks_lock:
                            for info in lease_infos:
                                self._put_tasks.discard(info["key"])
                                self._pending_objects.pop(info["key"], None)
                                self._pending_transfer_events.pop(info["key"], None)
                    if not enqueued and recovery_error is None:
                        for memory_obj in objects:
                            memory_obj.ref_count_down()
                        if credits_acquired:
                            scheduler.release_credits(reserved_bytes)
                    with self._put_tasks_lock:
                        for key in reserved_keys:
                            self._put_tasks.discard(key)
                    if recovery_error is not None:
                        raise recovery_error from transfer_error
                    raise

        snapshot = scheduler.snapshot()
        return BackendWriteSubmissionResult(
            accepted=tuple(accepted),
            accepted_bytes=accepted_bytes,
            durable=False,
            staging_seconds=staging_seconds,
            backpressure_seconds=backpressure_seconds,
            queue_depth=snapshot.queued_leases,
            outstanding_bytes=snapshot.outstanding_bytes,
        )

    def _make_transfer_result(
        self,
        completed: Sequence[bool],
        *,
        requested_bytes: int,
        completed_bytes: int,
        peak_staging_bytes: int,
        storage_seconds: float,
        gpu_seconds: float,
        overlap_seconds: float,
        slot_wait_seconds: float,
        short_io_chunks: int,
        peak_inflight_io: int,
    ) -> BackendTransferResult:
        return BackendTransferResult(
            completed=tuple(completed),
            requested_bytes=requested_bytes,
            completed_bytes=completed_bytes,
            device="cpu",
            window_size=self.read_slot_count,
            window_count=(len(completed) + self.read_slot_count - 1)
            // self.read_slot_count,
            peak_staging_bytes=peak_staging_bytes,
            storage_seconds=storage_seconds,
            gpu_seconds=gpu_seconds,
            overlap_seconds=overlap_seconds,
            bank_wait_seconds=slot_wait_seconds,
            short_io_chunks=short_io_chunks,
            timeout_chunks=0,
            logical_chunk_count=len(completed),
            read_slot_count=self.read_slot_count,
            write_slot_count=self.write_slot_count,
            io_queue_depth=self.config.cascadekv_io_queue_depth,
            peak_inflight_io=peak_inflight_io,
            slot_wait_seconds=slot_wait_seconds,
        )

    @_guard_operation
    def batched_get_to_gpu_blocking(
        self,
        keys: list[CacheEngineKey],
        starts: list[int],
        ends: list[int],
        gpu_connector: Any,
        **kwargs: Any,
    ) -> Optional[BackendTransferResult]:
        """Overlap bounded SSD reads with CPU-to-GPU copies by completion."""
        if not self.config.cascadekv_overlap:
            return None
        if not (len(keys) == len(starts) == len(ends)):
            raise ValueError("CascadeKV streaming read inputs must have equal lengths")

        wall_start = time.perf_counter()
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
                transfer_event = self._pending_transfer_events.get(key)
                pending_indexes.add(index)
                pending_infos.append(
                    {
                        "index": index,
                        "memory_obj": memory_obj,
                        "start": start,
                        "end": end,
                        "size": memory_obj.get_physical_size(),
                        "transfer_event": transfer_event,
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
        completed_bytes = 0
        storage_seconds = 0.0
        gpu_seconds = 0.0
        slot_wait_seconds = 0.0
        try:
            for info in pending_infos:
                gpu_start = time.perf_counter()
                load_started = False
                load_event = None
                try:
                    transfer_event = info["transfer_event"]
                    if transfer_event is not None:
                        try:
                            transfer_event.synchronize()
                        except BaseException as transfer_error:
                            try:
                                self._ensure_cuda_transfer_complete(
                                    gpu_connector,
                                    "store",
                                    transfer_event,
                                )
                            except BaseException as recovery_error:
                                info["cuda_quarantined"] = True
                                self._quarantine_cuda_objects(
                                    [info["memory_obj"]],
                                    recovery_error,
                                )
                                raise recovery_error from transfer_error
                            raise
                    load_started = True
                    gpu_connector.to_gpu(
                        info["memory_obj"],
                        info["start"],
                        info["end"],
                        **kwargs,
                    )
                    load_event = gpu_connector.record_transfer_event("load")
                    wait_start = time.perf_counter()
                    load_event.synchronize()
                    slot_wait_seconds += time.perf_counter() - wait_start
                    gpu_seconds += time.perf_counter() - gpu_start
                    completed[info["index"]] = True
                    completed_bytes += info["size"]
                except BaseException as transfer_error:
                    if load_started:
                        try:
                            self._ensure_cuda_transfer_complete(
                                gpu_connector,
                                "load",
                                load_event,
                            )
                        except BaseException as recovery_error:
                            info["cuda_quarantined"] = True
                            self._quarantine_cuda_objects(
                                [info["memory_obj"]],
                                recovery_error,
                            )
                            raise recovery_error from transfer_error
                    raise
        except BaseException:
            for info in task_infos:
                with self._entries_lock:
                    info["metadata"].unpin()
            raise
        finally:
            for info in pending_infos:
                if not info.get("cuda_quarantined", False):
                    info["memory_obj"].ref_count_down()

        if not task_infos:
            wall_seconds = time.perf_counter() - wall_start
            return self._make_transfer_result(
                completed,
                requested_bytes=requested_bytes,
                completed_bytes=completed_bytes,
                peak_staging_bytes=0,
                storage_seconds=storage_seconds,
                gpu_seconds=gpu_seconds,
                overlap_seconds=max(0.0, storage_seconds + gpu_seconds - wall_seconds),
                slot_wait_seconds=slot_wait_seconds,
                short_io_chunks=0,
                peak_inflight_io=0,
            )

        pump = XDSReadRingPump(
            queue_depth=self.config.cascadekv_io_queue_depth,
            submit=lambda entry: int(
                self.client.submit_read(
                    entry["metadata"].blob_handle,
                    self.staging_buffer_ref,
                    entry["offset"],
                    0,
                    entry["size"],
                )
            ),
            wait_any=self.client.wait_any_result,
            timeout=DEFAULT_IO_TIMEOUT_SECS,
        )

        def run_pump() -> None:
            with self._io_lock:
                pump.run()

        pump_thread = threading.Thread(target=run_pump, name="cascadekv-read-ring")
        live_entries: dict[int, dict[str, Any]] = {}
        next_task = 0
        input_closed = False
        peak_staging_bytes = 0
        submitted_count = 0
        short_io_chunks = 0

        def release_entry(entry: dict[str, Any]) -> None:
            if not entry.get("cuda_quarantined", False):
                entry["memory_obj"].ref_count_down()
            with self._entries_lock:
                entry["metadata"].unpin()

        def fill_free_slots() -> None:
            nonlocal next_task, input_closed, peak_staging_bytes, submitted_count
            while len(live_entries) < self.read_slot_count and next_task < len(
                task_infos
            ):
                info = task_infos[next_task]
                next_task += 1
                memory_obj = self.read_memory_allocator.allocate(
                    info["metadata"].shapes,
                    info["metadata"].dtypes,
                    info["metadata"].fmt,
                )
                if memory_obj is None:
                    with self._entries_lock:
                        info["metadata"].unpin()
                    continue
                try:
                    offset, size = self._buffer_descriptor(memory_obj)
                    if size != info["metadata"].size:
                        raise RuntimeError(
                            "CascadeKV allocation size does not match stored blob: "
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
                    entry = ring_completion.item
                    if entry is not None and not ring_completion.safe_to_reuse:
                        entry["cuda_quarantined"] = True
                        request_id = pump.unsafe_request_id(entry)
                        assert request_id is not None
                        self._quarantine_xds_object(
                            request_id,
                            entry["memory_obj"],
                            ring_completion.error,
                        )
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
                        load_started = True
                        load_event = None
                        try:
                            gpu_connector.to_gpu(
                                entry["memory_obj"],
                                entry["start"],
                                entry["end"],
                                **kwargs,
                            )
                            load_event = gpu_connector.record_transfer_event("load")
                            wait_start = time.perf_counter()
                            load_event.synchronize()
                        except BaseException as transfer_error:
                            if load_started:
                                try:
                                    self._ensure_cuda_transfer_complete(
                                        gpu_connector,
                                        "load",
                                        load_event,
                                    )
                                except BaseException as recovery_error:
                                    entry["cuda_quarantined"] = True
                                    self._quarantine_cuda_objects(
                                        [entry["memory_obj"]],
                                        recovery_error,
                                    )
                                    raise recovery_error from transfer_error
                            raise
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
            pump_thread.join(timeout=DEFAULT_IO_TIMEOUT_SECS)
            if pump_thread.is_alive():
                timeout_error = TimeoutError("timed out stopping CascadeKV read ring")
                for entry in live_entries.values():
                    entry["cuda_quarantined"] = True
                    self._quarantine_cuda_objects(
                        [entry["memory_obj"]],
                        timeout_error,
                    )
            else:
                for entry in live_entries.values():
                    if not entry.get("cuda_quarantined", False) and not (
                        pump.is_safe_to_reuse(entry)
                    ):
                        entry["cuda_quarantined"] = True
                        assert pump.error is not None
                        request_id = pump.unsafe_request_id(entry)
                        assert request_id is not None
                        self._quarantine_xds_object(
                            request_id,
                            entry["memory_obj"],
                            pump.error,
                        )
            for entry in live_entries.values():
                release_entry(entry)
            for info in task_infos[next_task:]:
                with self._entries_lock:
                    info["metadata"].unpin()
        if pump_thread.is_alive():
            raise TimeoutError("timed out stopping CascadeKV read ring")
        if pump.error is not None:
            raise pump.error

        wall_seconds = time.perf_counter() - wall_start
        return self._make_transfer_result(
            completed,
            requested_bytes=requested_bytes,
            completed_bytes=completed_bytes,
            peak_staging_bytes=peak_staging_bytes,
            storage_seconds=storage_seconds,
            gpu_seconds=gpu_seconds,
            overlap_seconds=max(0.0, storage_seconds + gpu_seconds - wall_seconds),
            slot_wait_seconds=slot_wait_seconds,
            short_io_chunks=short_io_chunks,
            peak_inflight_io=min(
                int(self.config.cascadekv_io_queue_depth),
                submitted_count,
            ),
        )

    @_guard_operation
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self._put_tasks_lock:
            pending_obj = self._pending_objects.get(key)
            if pending_obj is not None:
                pending_obj.ref_count_up()
                transfer_event = self._pending_transfer_events.get(key)
            else:
                transfer_event = None
        if pending_obj is not None:
            try:
                if transfer_event is not None:
                    try:
                        transfer_event.synchronize()
                    except BaseException as transfer_error:
                        try:
                            self._ensure_cuda_transfer_complete(
                                None,
                                "store",
                                transfer_event,
                            )
                        except BaseException as recovery_error:
                            self._quarantine_cuda_objects(
                                [pending_obj],
                                recovery_error,
                            )
                            raise recovery_error from transfer_error
                        raise
                return pending_obj
            except BaseException:
                if not any(item is pending_obj for item in self._quarantined_objects):
                    pending_obj.ref_count_down()
                raise

        with self._entries_lock:
            blob_metadata = self._entries.get(key)
            if blob_metadata is None:
                return None
            blob_metadata.pin()
            self._entries.move_to_end(key)

        memory_obj: Optional[MemoryObj] = None
        try:
            read_allocator = getattr(
                self,
                "read_memory_allocator",
                self.memory_allocator,
            )
            memory_obj = read_allocator.allocate(
                blob_metadata.shapes,
                blob_metadata.dtypes,
                blob_metadata.fmt,
            )
            if memory_obj is None:
                return None
            buffer_offset, size = self._buffer_descriptor(memory_obj)
            if size != blob_metadata.size:
                raise RuntimeError(
                    "CascadeKV allocation size does not match stored blob: "
                    f"allocated={size}, stored={blob_metadata.size}"
                )
            with self._io_lock:
                request_id = int(
                    self.client.submit_read(
                        blob_metadata.blob_handle,
                        self.staging_buffer_ref,
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

    @_guard_operation
    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        del location
        future: Future = Future()
        try:
            future.set_result(self.get_blocking(key))
        except Exception as exc:
            future.set_exception(exc)
        return future

    @_guard_async_operation
    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        del lookup_id
        return self.batched_contains(keys, pin)

    @_guard_async_operation
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

    @_guard_operation
    def batched_get_blocking(
        self,
        keys: list[CacheEngineKey],
    ) -> list[Optional[MemoryObj]]:
        if not hasattr(self, "read_memory_allocator"):
            return [self.get_blocking(key) for key in keys]
        with self._put_tasks_lock:
            if any(key in self._pending_objects for key in keys):
                return [self.get_blocking(key) for key in keys]

        results: list[Optional[MemoryObj]] = [None] * len(keys)
        items: list[dict[str, Any]] = []
        try:
            for index, key in enumerate(keys):
                with self._entries_lock:
                    metadata = self._entries.get(key)
                    if metadata is None:
                        continue
                    metadata.pin()
                    self._entries.move_to_end(key)
                memory_obj = self.read_memory_allocator.allocate(
                    metadata.shapes,
                    metadata.dtypes,
                    metadata.fmt,
                )
                if memory_obj is None:
                    with self._entries_lock:
                        metadata.unpin()
                    continue
                try:
                    offset, size = self._buffer_descriptor(memory_obj)
                    if size != metadata.size:
                        raise RuntimeError(
                            "CascadeKV allocation size does not match stored blob: "
                            f"allocated={size}, stored={metadata.size}"
                        )
                    items.append(
                        {
                            "index": index,
                            "metadata": metadata,
                            "memory_obj": memory_obj,
                            "offset": offset,
                            "size": size,
                        }
                    )
                except BaseException:
                    memory_obj.ref_count_down()
                    with self._entries_lock:
                        metadata.unpin()
                    raise

            pump = XDSReadRingPump(
                queue_depth=int(self.config.cascadekv_io_queue_depth),
                submit=lambda item: int(
                    self.client.submit_read(
                        item["metadata"].blob_handle,
                        self.staging_buffer_ref,
                        item["offset"],
                        0,
                        item["size"],
                    )
                ),
                wait_any=self.client.wait_any_result,
                timeout=DEFAULT_IO_TIMEOUT_SECS,
            )
            for item in items:
                pump.put(item)
            pump.close_input()
            with self._io_lock:
                pump.run()

            first_error: Optional[BaseException] = None
            for _ in items:
                ring_completion = pump.get_completion()
                item = ring_completion.item
                if ring_completion.error is not None:
                    if first_error is None:
                        first_error = ring_completion.error
                    if not ring_completion.safe_to_reuse:
                        item["cuda_quarantined"] = True
                        request_id = pump.unsafe_request_id(item)
                        assert request_id is not None
                        self._quarantine_xds_object(
                            request_id,
                            item["memory_obj"],
                            ring_completion.error,
                        )
                else:
                    try:
                        self._validate_io_result(
                            ring_completion.completion,
                            item["size"],
                            "read",
                        )
                        results[item["index"]] = item["memory_obj"]
                    except BaseException as error:
                        if first_error is None:
                            first_error = error
                if results[item["index"]] is None and not item.get(
                    "cuda_quarantined", False
                ):
                    item["memory_obj"].ref_count_down()
                with self._entries_lock:
                    item["metadata"].unpin()
            items.clear()
            if first_error is not None:
                for memory_obj in results:
                    if memory_obj is not None:
                        memory_obj.ref_count_down()
                raise first_error
            return results
        except BaseException:
            for item in items:
                if not item.get("cuda_quarantined", False):
                    item["memory_obj"].ref_count_down()
                with self._entries_lock:
                    item["metadata"].unpin()
            raise

    @_guard_operation
    def pin(self, key: CacheEngineKey) -> bool:
        with self._entries_lock:
            metadata = self._entries.get(key)
            return metadata.pin() if metadata is not None else False

    @_guard_operation
    def unpin(self, key: CacheEngineKey) -> bool:
        with self._entries_lock:
            metadata = self._entries.get(key)
            return metadata.unpin() if metadata is not None else False

    @_guard_operation
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

    @_guard_operation
    def touch_cache(self) -> None:
        with self._entries_lock:
            for key in reversed(self._keys_in_request):
                if key in self._entries:
                    self._entries.move_to_end(key)
            self._keys_in_request.clear()

    def close(self) -> None:
        with self._lifecycle_condition:
            while self._closing and not self._closed:
                self._lifecycle_condition.wait()
            if self._closed:
                return
            self._closing = True
            self._teardown_started = True
            while self._active_operations:
                self._lifecycle_condition.wait()

        scheduler_error: Optional[BaseException] = None
        try:
            if self._write_scheduler is not None:
                try:
                    self._write_scheduler.close(timeout=DEFAULT_IO_TIMEOUT_SECS)
                except TimeoutError:
                    raise
                except BaseException as error:
                    scheduler_error = error

            active_allocations = (
                self.read_memory_allocator.num_active_allocations
                + self.write_memory_allocator.num_active_allocations
            )
            if active_allocations:
                raise RuntimeError(
                    "cannot close CascadeKVBackend with active staging allocations: "
                    f"{active_allocations}"
                )

            first_release_error: Optional[BaseException] = None
            with self._entries_lock:
                entries = tuple(self._entries.items())
            for key, metadata in entries:
                try:
                    self.client.release_blob(metadata.blob_handle)
                except BaseException as error:
                    if first_release_error is None:
                        first_release_error = error
                    continue
                with self._entries_lock:
                    current = self._entries.get(key)
                    if current is metadata:
                        self._entries.pop(key)
            if first_release_error is not None:
                raise first_release_error

            if not self._read_allocator_closed:
                self.read_memory_allocator.close()
                self._read_allocator_closed = True
            if not self._write_allocator_closed:
                self.write_memory_allocator.close()
                self._write_allocator_closed = True
            if self._host_registered:
                unregister_status = int(
                    self._cudart.cudaHostUnregister(self.staging_base_ptr)
                )
                if unregister_status != 0:
                    error_string = self._cudart.cudaGetErrorString(unregister_status)
                    raise RuntimeError(
                        "CascadeKV cudaHostUnregister failed: "
                        f"status={unregister_status}, error={error_string}"
                    )
                self._host_registered = False
            if not self._buffer_freed:
                assert self.staging_buffer_ref is not None
                self.client.free_buffer(self.staging_buffer_ref)
                self._buffer_freed = True
            client_close = getattr(self.client, "close", None)
            if callable(client_close) and not self._client_closed:
                client_close()
                self._client_closed = True
        except BaseException:
            with self._lifecycle_condition:
                self._closing = False
                self._lifecycle_condition.notify_all()
            raise

        with self._lifecycle_condition:
            self._closed = True
            self._closing = False
            self._lifecycle_condition.notify_all()
        if scheduler_error is not None:
            logger.warning(
                "CascadeKV async write scheduler closed after an I/O error: %s",
                scheduler_error,
            )
