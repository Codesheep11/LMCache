# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.xds_backend import XDSBackend


class FakeXDSClient:
    def __init__(self):
        self.metadata = SimpleNamespace(
            io_unit_size=4096,
            blob_size_bytes=4096,
            gpu_buffer_size_bytes=8192,
        )
        self.registered = []
        self.unregistered = []
        self.released_blob_handles = []
        self.calls = []
        self._next_request_id = 1
        self._last_request_id = 0
        self._pending_request_ids = []
        self._next_blob_handle = 700
        self.next_completion = None

    def register_gpu_buffer(self, ptr, size):
        self.registered.append((ptr, size))
        return 77

    def unregister_buffer(self, buffer_ref):
        self.calls.append("unregister_buffer")
        self.unregistered.append(buffer_ref)

    def get_blob(self, timeout=5.0):
        del timeout
        handle = self._next_blob_handle
        self._next_blob_handle += 1
        return handle

    def release_blob(self, blob_handle):
        self.released_blob_handles.append(blob_handle)

    def submit_read(self, *_descriptor):
        self.calls.append("submit_read")
        return self._new_request()

    def submit_write(self, *_descriptor):
        self.calls.append("submit_write")
        return self._new_request()

    def _new_request(self):
        request_id = self._next_request_id
        self._next_request_id += 1
        self._last_request_id = request_id
        self._pending_request_ids.append(request_id)
        return request_id

    def wait_any_result(self, timeout=5.0):
        del timeout
        self.calls.append("wait_any_result")
        if self.next_completion is not None:
            completion = self.next_completion
            if completion.request_id is None:
                completion.request_id = self._last_request_id
            self.next_completion = None
            return completion
        request_id = self._pending_request_ids.pop(0)
        return SimpleNamespace(
            request_id=request_id,
            status=0,
            error_code=0,
            completed_bytes=4096,
            result=0,
        )

    def close(self):
        self.calls.append("close")


@pytest.fixture
def direct_config():
    config = LMCacheEngineConfig.from_defaults()
    config.xds_service_name = "xds0"
    config.xds_backend = True
    config.xds_require_direct = True
    config.xds_read_slot_count = 1
    config.xds_write_slot_count = 1
    return config


@pytest.fixture
def direct_metadata():
    return LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(1, 2, 4, 1, 8),
        chunk_size=4,
    )


@pytest.fixture
def cache_key():
    return CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=99,
        dtype=torch.float16,
    )


def test_xds_backend_registers_exact_allocator_pool(direct_config, direct_metadata):
    client = FakeXDSClient()
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    allocator = backend.get_memory_allocator()

    assert client.registered == [
        (allocator.buffer.data_ptr(), allocator.buffer.numel())
    ]

    backend.close()
    assert client.unregistered == [backend.buffer_ref]
    assert client.calls[-2:] == ["unregister_buffer", "close"]


def test_xds_backend_rejects_descriptor_outside_registered_pool(
    direct_config,
    direct_metadata,
):
    backend = XDSBackend(direct_config, direct_metadata, FakeXDSClient(), "cpu")
    with pytest.raises(ValueError, match="registered range"):
        backend._validate_registered_range(backend.registered_size - 4096, 8192)
    backend.close()


def test_xds_backend_short_write_rolls_back_blob(
    direct_config,
    direct_metadata,
    cache_key,
):
    client = FakeXDSClient()
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    memory_obj = backend.allocate(
        direct_metadata.get_shapes(),
        direct_metadata.get_dtypes(),
    )
    assert memory_obj is not None
    client.next_completion = SimpleNamespace(
        request_id=None,
        status=0,
        error_code=0,
        completed_bytes=0,
        result=0,
    )

    futures = backend.batched_submit_put_task([cache_key], [memory_obj])

    assert futures is not None
    with pytest.raises(RuntimeError, match="short xDS write"):
        futures[0].result()
    assert not backend.contains(cache_key)
    assert client.released_blob_handles
    memory_obj.ref_count_down()
    backend.close()


def test_xds_backend_round_trip_returns_registered_pool_object(
    direct_config,
    direct_metadata,
    cache_key,
):
    client = FakeXDSClient()
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    source = backend.allocate(
        direct_metadata.get_shapes(),
        direct_metadata.get_dtypes(),
    )
    assert source is not None
    futures = backend.batched_submit_put_task([cache_key], [source])
    assert futures is not None
    assert futures[0].result() is True
    source.ref_count_down()

    result = backend.get_blocking(cache_key)

    assert result is not None
    assert result.raw_data.data_ptr() >= backend.registered_base_ptr
    result.ref_count_down()
    backend.close()


def test_xds_backend_async_write_commits_before_releasing_object(
    direct_config,
    direct_metadata,
    cache_key,
):
    direct_config.xds_async_write = True
    client = FakeXDSClient()
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    source = backend.allocate(
        direct_metadata.get_shapes(),
        direct_metadata.get_dtypes(),
    )
    assert source is not None

    futures = backend.batched_submit_put_task([cache_key], [source])

    assert futures is not None
    assert futures[0].result(timeout=1.0) is True
    assert backend.contains(cache_key)
    source.ref_count_down()
    backend.close()


def test_xds_backend_batched_read_uses_bounded_ring(
    direct_config,
    direct_metadata,
):
    direct_config.xds_io_queue_depth = 2
    client = FakeXDSClient()
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    keys = [
        CacheEngineKey("test-model", 1, 0, chunk_hash, torch.float16)
        for chunk_hash in (100, 101)
    ]
    sources = backend.batched_allocate(
        direct_metadata.get_shapes(),
        direct_metadata.get_dtypes(),
        batch_size=2,
    )
    assert sources is not None
    futures = backend.batched_submit_put_task(keys, sources)
    assert futures is not None
    assert [future.result() for future in futures] == [True, True]
    for source in sources:
        source.ref_count_down()

    results = backend.batched_get_blocking(keys)

    assert all(result is not None for result in results)
    for result in results:
        assert result is not None
        result.ref_count_down()
    backend.close()
