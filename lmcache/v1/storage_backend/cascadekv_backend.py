# SPDX-License-Identifier: Apache-2.0
"""CascadeKV backend using xDS service-owned CPU staging memory."""

# Standard
from concurrent.futures import Future
from typing import Any, Callable, Optional, Sequence, Union

# Third Party
import torch

# First Party
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
from lmcache.v1.xds_utils import align_to_io_unit


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
        del key, pin
        return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        del key
        return False

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: list[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[list[Future]]:
        del keys, objs, transfer_spec, on_complete_callback
        raise NotImplementedError("CascadeKV storage writes are not implemented yet")

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        del key
        raise NotImplementedError("CascadeKV storage reads are not implemented yet")

    def pin(self, key: CacheEngineKey) -> bool:
        del key
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        del key
        return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        del key, force
        return False

    def touch_cache(self) -> None:
        return

    def close(self) -> None:
        if self._closed:
            return
        active_allocations = self.memory_allocator.num_active_allocations
        if active_allocations:
            raise RuntimeError(
                "cannot close CascadeKVBackend with active staging allocations: "
                f"{active_allocations}"
            )
        self.memory_allocator.close()
        self.client.free_buffer(self.staging_buffer_ref)
        self._closed = True
