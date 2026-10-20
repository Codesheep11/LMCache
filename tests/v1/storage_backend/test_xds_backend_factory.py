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

    def submit_readv(self, *_args):
        raise AssertionError("factory must not submit I/O")

    def submit_writev(self, *_args):
        raise AssertionError("factory must not submit I/O")

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


class FakeCUDART:
    def cudaHostRegister(self, _ptr, _size, _flags):
        return 0

    def cudaHostUnregister(self, _ptr):
        return 0

    def cudaGetErrorString(self, status):
        return f"fake CUDA error {status}"


class FakeServiceAllocator:
    instances = []

    def __init__(self, client, size, align_bytes, close_client=False):
        self.client = client
        self.size = size
        self.align_bytes = align_bytes
        self.close_client = close_client
        self.closed = False
        self.instances.append(self)

    def close(self):
        self.closed = True
        if self.close_client:
            self.client.close()


class FakeLegacyCascadeBackend:
    instance = None

    def __init__(
        self,
        config,
        metadata,
        loop,
        memory_allocator,
        xds_client,
        dst_device,
        lmcache_worker,
    ):
        self.config = config
        self.metadata = metadata
        self.loop = loop
        self.memory_allocator = memory_allocator
        self.client = xds_client
        self.dst_device = dst_device
        self.lmcache_worker = lmcache_worker
        FakeLegacyCascadeBackend.instance = self

    def __str__(self):
        return "CascadeKVBackend"

    def close(self):
        self.client.close()


class FailingLegacyCascadeBackend:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError("fake CascadeKV construction failure")


@pytest.fixture(autouse=True)
def fake_cudart(monkeypatch):
    runtime = FakeCUDART()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: runtime)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return runtime


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
    config.max_local_cpu_size = 8192 / 1024**3 if cascade else 0
    config.xds_service_name = "xds0"
    config.xds_backend = direct
    config.cascadekv_backend = cascade
    config.use_layerwise = False
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
    assert backends["XDSBackend"].staging_size == 0
    assert backends["XDSBackend"].gpu_tensor is None
    backends["XDSBackend"].close()


def test_storage_manager_selects_cascadekv_allocator(metadata, monkeypatch):
    config = LMCacheEngineConfig.from_defaults()
    config.local_cpu = False
    config.max_local_cpu_size = 8192 / 1024**3
    config.xds_service_name = "xds0"
    config.cascadekv_backend = True
    config.use_layerwise = False
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


def test_factory_injects_service_dma_allocator_into_cascadekv(
    metadata, monkeypatch
):
    config = LMCacheEngineConfig.from_defaults()
    config.local_cpu = False
    config.max_local_cpu_size = 8192 / 1024**3
    config.xds_service_name = "xds0"
    config.cascadekv_backend = True
    config.use_layerwise = False
    client = FakeXDSClient()
    loop = asyncio.new_event_loop()
    FakeServiceAllocator.instances.clear()
    FakeLegacyCascadeBackend.instance = None
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.connect_xds_client",
        lambda *_args, **_kwargs: client,
        raising=False,
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.XDSServiceMemoryAllocator",
        FakeServiceAllocator,
        raising=False,
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.cascadekv_backend.CascadeKVBackend",
        FakeLegacyCascadeBackend,
    )

    backends = CreateStorageBackends(config, metadata, loop, dst_device="cpu")

    assert list(backends) == ["CascadeKVBackend"]
    assert len(FakeServiceAllocator.instances) == 1
    allocator = FakeServiceAllocator.instances[0]
    assert allocator.size == 8192
    assert allocator.align_bytes == 4096
    assert allocator.close_client is True
    assert FakeLegacyCascadeBackend.instance.memory_allocator is allocator
    assert FakeLegacyCascadeBackend.instance.client is client


def test_factory_cleans_service_dma_allocator_when_cascadekv_init_fails(
    metadata, monkeypatch
):
    config = LMCacheEngineConfig.from_defaults()
    config.local_cpu = False
    config.max_local_cpu_size = 8192 / 1024**3
    config.xds_service_name = "xds0"
    config.cascadekv_backend = True
    config.use_layerwise = False
    client = FakeXDSClient()
    FakeServiceAllocator.instances.clear()
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.connect_xds_client",
        lambda *_args, **_kwargs: client,
        raising=False,
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.XDSServiceMemoryAllocator",
        FakeServiceAllocator,
        raising=False,
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.cascadekv_backend.CascadeKVBackend",
        FailingLegacyCascadeBackend,
    )

    with pytest.raises(RuntimeError, match="construction failure"):
        CreateStorageBackends(
            config,
            metadata,
            asyncio.new_event_loop(),
            dst_device="cpu",
        )

    assert FakeServiceAllocator.instances[0].closed is True
    assert client.closed is True


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
