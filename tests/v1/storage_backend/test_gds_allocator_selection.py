# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.gds_backend import GdsBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager


class RecordingAllocator:
    """Record the format passed through the GDS allocator adapter."""

    def __init__(self) -> None:
        self.fmt = None

    def allocate(self, shapes, dtypes, fmt):
        self.fmt = fmt
        return object()


def test_pure_gds_backend_is_selected_as_allocator() -> None:
    """Pure GDS must not require a LocalCPUBackend allocator."""
    config = LMCacheEngineConfig.from_defaults()
    config.local_cpu = False
    config.gds_path = "/tmp/gds"
    gds_backend = object.__new__(GdsBackend)
    manager = object.__new__(StorageManager)
    manager.enable_pd = False
    manager.storage_backends = {"GdsBackend": gds_backend}

    assert manager._get_allocator_backend(config) is gds_backend


def test_tiered_gds_preserves_local_cpu_allocator() -> None:
    """A GDS tier with local CPU enabled keeps CPU as its allocator."""
    config = LMCacheEngineConfig.from_defaults()
    config.local_cpu = True
    config.gds_path = "/tmp/gds"
    gds_backend = object.__new__(GdsBackend)
    local_cpu_backend = object.__new__(LocalCPUBackend)
    manager = object.__new__(StorageManager)
    manager.enable_pd = False
    manager.storage_backends = {
        "LocalCPUBackend": local_cpu_backend,
        "GdsBackend": gds_backend,
    }

    assert manager._get_allocator_backend(config) is local_cpu_backend


def test_gds_allocate_normalizes_missing_memory_format() -> None:
    """Non-layerwise callers may pass None; persisted metadata must not."""
    allocator = RecordingAllocator()
    backend = object.__new__(GdsBackend)
    backend.memory_allocator = allocator
    backend.max_alloc_attempts = 1
    backend.alloc_attempt_delay_secs = 0

    result = backend.allocate(
        torch.Size([2, 1, 1]), torch.bfloat16, fmt=None, eviction=False
    )

    assert result is not None
    assert allocator.fmt is MemoryFormat.KV_2LTD
