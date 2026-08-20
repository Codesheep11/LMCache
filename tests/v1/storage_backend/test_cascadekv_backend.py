# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cascadekv_backend import CascadeKVBackend
from lmcache.utils import CacheEngineKey


class FakeXDSClient:
    def __init__(self):
        self.metadata = SimpleNamespace(
            io_unit_size=4096,
            blob_size_bytes=4096,
        )
        self.alloc_sizes = []
        self.free_refs = []
        self.released_blob_handles = []
        self.calls = []
        self._buffers = {}
        self._next_blob_handle = 501
        self._next_request_id = 1
        self._last_request_id = 0
        self.next_completion = None

    def alloc_buffer(self, size):
        self.alloc_sizes.append(size)
        buffer_ref = 101
        view = memoryview(bytearray(size))
        self._buffers[buffer_ref] = view
        return view, buffer_ref

    def free_buffer(self, buffer_ref):
        self.calls.append("free_buffer")
        self.free_refs.append(buffer_ref)
        del self._buffers[buffer_ref]

    def get_blob(self, timeout=5.0):
        del timeout
        handle = self._next_blob_handle
        self._next_blob_handle += 1
        return handle

    def release_blob(self, blob_handle):
        self.calls.append("release_blob")
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
        return SimpleNamespace(
            request_id=self._last_request_id,
            status=0,
            error_code=0,
            completed_bytes=4096,
            result=0,
        )

    def close(self):
        self.calls.append("close")


@pytest.fixture
def cascade_config():
    config = LMCacheEngineConfig.from_defaults()
    config.xds_service_name = "xds0"
    config.cascadekv_backend = True
    config.cascadekv_slot_count = 2
    return config


@pytest.fixture
def cascade_metadata():
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
        chunk_hash=1234,
        dtype=torch.float16,
    )


def test_cascadekv_allocator_wraps_service_cpu_buffer_without_copy(
    cascade_config,
    cascade_metadata,
):
    fake_xds_client = FakeXDSClient()
    backend = CascadeKVBackend(
        cascade_config,
        cascade_metadata,
        fake_xds_client,
        dst_device="cpu",
    )

    obj = backend.allocate(
        cascade_metadata.get_shapes(),
        cascade_metadata.get_dtypes(),
        MemoryFormat.KV_2LTD,
    )

    assert obj is not None
    assert obj.raw_data.data_ptr() >= backend.staging_base_ptr
    assert obj.raw_data.data_ptr() < backend.staging_base_ptr + backend.staging_size
    assert fake_xds_client.alloc_sizes == [backend.staging_size]

    obj.ref_count_down()
    backend.close()
    assert fake_xds_client.free_refs == [backend.staging_buffer_ref]


def test_cascadekv_close_rejects_live_staging_objects(
    cascade_config,
    cascade_metadata,
):
    fake_xds_client = FakeXDSClient()
    backend = CascadeKVBackend(
        cascade_config,
        cascade_metadata,
        fake_xds_client,
        dst_device="cpu",
    )
    obj = backend.allocate(
        cascade_metadata.get_shapes(),
        cascade_metadata.get_dtypes(),
    )
    assert obj is not None

    with pytest.raises(RuntimeError, match="active staging allocations"):
        backend.close()
    assert fake_xds_client.free_refs == []

    obj.ref_count_down()
    backend.close()
    backend.close()
    assert fake_xds_client.free_refs == [backend.staging_buffer_ref]


def test_cascadekv_read_uses_service_cpu_buffer(
    cascade_config,
    cascade_metadata,
    cache_key,
):
    client = FakeXDSClient()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    source = backend.allocate(
        cascade_metadata.get_shapes(),
        cascade_metadata.get_dtypes(),
    )
    assert source is not None
    futures = backend.batched_submit_put_task([cache_key], [source])
    assert futures is not None
    assert futures[0].result() is True
    source.ref_count_down()
    client.calls.clear()

    obj = backend.batched_get_blocking([cache_key])[0]

    assert obj is not None
    assert client.calls[:2] == ["submit_read", "wait_any_result"]
    assert obj.raw_data.data_ptr() >= backend.staging_base_ptr
    obj.ref_count_down()
    backend.close()


def test_cascadekv_short_write_releases_blob_and_does_not_publish_key(
    cascade_config,
    cascade_metadata,
    cache_key,
):
    client = FakeXDSClient()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    memory_obj = backend.allocate(
        cascade_metadata.get_shapes(),
        cascade_metadata.get_dtypes(),
    )
    assert memory_obj is not None
    client.next_completion = SimpleNamespace(
        request_id=None,
        status=0,
        error_code=0,
        completed_bytes=memory_obj.get_physical_size() - 1,
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


def test_cascadekv_remove_honors_pin_and_close_order(
    cascade_config,
    cascade_metadata,
    cache_key,
):
    client = FakeXDSClient()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    memory_obj = backend.allocate(
        cascade_metadata.get_shapes(),
        cascade_metadata.get_dtypes(),
    )
    assert memory_obj is not None
    futures = backend.batched_submit_put_task([cache_key], [memory_obj])
    assert futures is not None
    assert futures[0].result() is True
    memory_obj.ref_count_down()

    assert backend.pin(cache_key) is True
    assert backend.remove(cache_key, force=False) is False
    assert backend.unpin(cache_key) is True
    assert backend.remove(cache_key, force=False) is True

    backend.close()
    backend.close()
    assert client.calls[-2:] == ["free_buffer", "close"]
    assert client.calls.count("free_buffer") == 1
    assert client.calls.count("close") == 1
