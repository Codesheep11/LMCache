# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from types import SimpleNamespace
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.storage_manager import StorageManager


class FakeXDSClient:
    def __init__(self):
        self.metadata = SimpleNamespace(
            queue_id=0,
            worker_id=0,
            io_unit_size=4096,
            blob_size_bytes=4096,
            capacity_bytes=4096 * 16,
            gpu_buffer_size_bytes=4096 * 10,
        )
        self.closed = False

    def register_gpu_buffer(self, _ptr, _size):
        return 71

    def unregister_buffer(self, _buffer_ref):
        return

    def alloc_buffer(self, size):
        return memoryview(bytearray(size)), 72

    def free_buffer(self, _buffer_ref):
        return

    def close(self):
        self.closed = True


@pytest.fixture
def metadata():
    return LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(1, 2, 4, 1, 8),
        chunk_size=4,
        role="worker",
    )


@pytest.mark.parametrize(
    ("direct", "cascade", "expected"),
    [(True, False, "XDSBackend"), (False, True, "CascadeKVBackend")],
)
def test_factory_constructs_only_selected_xds_backend(
    direct,
    cascade,
    expected,
    metadata,
    monkeypatch,
):
    config = LMCacheEngineConfig.from_defaults()
    config.local_cpu = False
    config.max_local_cpu_size = 0
    config.xds_service_name = "xds0"
    config.xds_backend = direct
    config.cascadekv_backend = cascade
    config.xds_require_direct = direct
    fake_client = FakeXDSClient()
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.connect_xds_client",
        lambda *_args, **_kwargs: fake_client,
        raising=False,
    )

    backends = CreateStorageBackends(
        config,
        metadata,
        asyncio.new_event_loop(),
        dst_device="cpu",
    )

    assert expected in backends
    assert ({"XDSBackend", "CascadeKVBackend"} & backends.keys()) == {expected}
    backends[expected].close()
    assert fake_client.closed is True


def test_storage_manager_selects_direct_xds_allocator(metadata, monkeypatch):
    config = LMCacheEngineConfig.from_defaults()
    config.local_cpu = False
    config.max_local_cpu_size = 0
    config.xds_service_name = "xds0"
    config.xds_backend = True
    config.xds_require_direct = True
    client = FakeXDSClient()
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.connect_xds_client",
        lambda *_args, **_kwargs: client,
        raising=False,
    )
    backends = CreateStorageBackends(
        config,
        metadata,
        asyncio.new_event_loop(),
        dst_device="cpu",
    )
    manager = object.__new__(StorageManager)
    manager.enable_pd = False
    manager.storage_backends = backends

    assert manager._get_allocator_backend(config) is backends["XDSBackend"]
    backends["XDSBackend"].close()


def test_storage_manager_selects_cascadekv_allocator(metadata, monkeypatch):
    config = LMCacheEngineConfig.from_defaults()
    config.local_cpu = False
    config.max_local_cpu_size = 0
    config.xds_service_name = "xds0"
    config.cascadekv_backend = True
    config.cascadekv_slot_count = 2
    client = FakeXDSClient()
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.connect_xds_client",
        lambda *_args, **_kwargs: client,
        raising=False,
    )
    backends = CreateStorageBackends(
        config,
        metadata,
        asyncio.new_event_loop(),
        dst_device="cpu",
    )
    manager = object.__new__(StorageManager)
    manager.enable_pd = False
    manager.storage_backends = backends

    assert manager._get_allocator_backend(config) is backends["CascadeKVBackend"]
    backends["CascadeKVBackend"].close()


def test_xds_read_skips_local_cpu_writeback(monkeypatch):
    memory_obj = object()

    class FakeXDSBackend:
        skip_local_cpu_writeback = True

        def get_blocking(self, _key):
            return memory_obj

    class FakeLocalCPUBackend:
        def __init__(self):
            self.writes = []

        def submit_put_task(self, key, obj):
            self.writes.append((key, obj))

    local_backend = FakeLocalCPUBackend()
    manager = object.__new__(StorageManager)
    manager.storage_backends = OrderedDict(
        [
            ("XDSBackend", FakeXDSBackend()),
            ("LocalCPUBackend", local_backend),
        ]
    )
    manager._freeze = False
    manager._freeze_lock = threading.RLock()
    manager._bypassed_backends = set()
    manager._bypass_lock = threading.RLock()
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.LocalCPUBackend",
        FakeLocalCPUBackend,
    )
    key = CacheEngineKey("test-model", 1, 0, 1, torch.float16)

    assert manager.get(key) is memory_obj
    assert local_backend.writes == []
