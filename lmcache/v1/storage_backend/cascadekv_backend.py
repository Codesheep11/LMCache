# SPDX-License-Identifier: Apache-2.0
"""CascadeKV backend using xDS service-owned CPU staging memory."""

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Callable, Optional, Sequence, Union
import asyncio
import threading

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
from lmcache.v1.xds_utils import XDSBlobMetadata, align_to_io_unit


logger = init_logger(__name__)

DEFAULT_BLOB_TIMEOUT_SECS = 5.0
DEFAULT_IO_TIMEOUT_SECS = 30.0


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
        self.staging_size = config.cascadekv_slot_count * self.blob_size
        self.staging_view, self.staging_buffer_ref = xds_client.alloc_buffer(
            self.staging_size
        )
        self.staging_tensor = torch.frombuffer(
            self.staging_view,
            dtype=torch.uint8,
        )
        if self.staging_tensor.numel() != self.staging_size:
            xds_client.free_buffer(self.staging_buffer_ref)
            raise RuntimeError(
                "xDS service returned an unexpected CPU staging buffer size: "
                f"{self.staging_tensor.numel()} != {self.staging_size}"
            )
        self.staging_base_ptr = self.staging_tensor.data_ptr()
        self.memory_allocator = self.initialize_allocator(config, metadata)

    def __str__(self) -> str:
        return self.__class__.__name__

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> TensorMemoryAllocator:
        del config, metadata
        return TensorMemoryAllocator(
            self.staging_tensor,
            align_bytes=self.io_unit_size,
        )

    def get_allocator_backend(self) -> "CascadeKVBackend":
        return self

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        return self.memory_allocator

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
        return self.config.cascadekv_slot_count

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
                f"CascadeKV memory object size {size} exceeds blob size {self.blob_size}"
            )
        return offset, size

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
                        logger.exception("CascadeKV put completion callback failed")
                future.set_result(stored)
            except Exception as exc:
                future.set_exception(exc)
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
        for future in futures or []:
            future.result()

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
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

    def batched_get_blocking(
        self,
        keys: list[CacheEngineKey],
    ) -> list[Optional[MemoryObj]]:
        return [self.get_blocking(key) for key in keys]

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

    def close(self) -> None:
        if self._closed:
            return
        active_allocations = self.memory_allocator.num_active_allocations
        if active_allocations:
            raise RuntimeError(
                "cannot close CascadeKVBackend with active staging allocations: "
                f"{active_allocations}"
            )
        with self._entries_lock:
            blob_handles = [metadata.blob_handle for metadata in self._entries.values()]
            self._entries.clear()
        for blob_handle in blob_handles:
            self.client.release_blob(blob_handle)
        self.memory_allocator.close()
        self.client.free_buffer(self.staging_buffer_ref)
        client_close = getattr(self.client, "close", None)
        if callable(client_close):
            client_close()
        self._closed = True
