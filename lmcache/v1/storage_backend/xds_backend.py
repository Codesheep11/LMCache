# SPDX-License-Identifier: Apache-2.0
"""Direct SSD-to-GPU LMCache backend using an external xDS service."""

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Callable, Optional, Sequence
import threading

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
from lmcache.v1.storage_backend.xds_read_ring import XDSReadRingPump
from lmcache.v1.storage_backend.xds_write_scheduler import (
    LeaseState,
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
        self._put_tasks_lock = threading.RLock()
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

    def _submit_async_lease(self, lease: StagingLease) -> bool:
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
        self._finish_async_lease(lease, outcome=bool(outcome))

    def _rollback_async_lease(
        self,
        lease: StagingLease,
        error: BaseException,
    ) -> None:
        self._finish_async_lease(lease, error=error)

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
        if self._write_scheduler is not None:
            self._write_scheduler.close(timeout=30.0)
        active_allocations = self.memory_allocator.num_active_allocations
        if active_allocations:
            raise RuntimeError(
                "cannot close XDSBackend with active registered allocations: "
                f"{active_allocations}"
            )
        with self._entries_lock:
            blob_handles = [metadata.blob_handle for metadata in self._entries.values()]
            self._entries.clear()
        for blob_handle in blob_handles:
            self.client.release_blob(blob_handle)
        self.memory_allocator.close()
        self.client.unregister_buffer(self.buffer_ref)
        client_close = getattr(self.client, "close", None)
        if callable(client_close):
            client_close()
        self._closed = True
