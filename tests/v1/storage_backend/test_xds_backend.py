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
            self._pending_request_ids.remove(completion.request_id)
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


class FakeTransferEvent:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class FakeStreamingGPUConnector:
    def __init__(self, backend, fail_scatter=False, fail_gather=False):
        self.backend = backend
        self.fail_scatter = fail_scatter
        self.fail_gather = fail_gather
        self.events = []
        self.scatter_ranges = []
        self.gather_ranges = []
        self.peak_active = 0

    def _record_peak(self):
        self.peak_active = max(
            self.peak_active,
            self.backend.get_memory_allocator().num_active_allocations,
        )

    def to_gpu(self, memory_obj, start, end, **kwargs):
        del memory_obj, kwargs
        self._record_peak()
        if self.fail_scatter:
            raise RuntimeError("scatter failed")
        self.scatter_ranges.append((start, end))

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        del kwargs
        self._record_peak()
        if self.fail_gather:
            raise RuntimeError("gather failed")
        self.gather_ranges.append((len(memory_objs), tuple(starts), tuple(ends)))

    def record_transfer_event(self, direction):
        assert direction in ("load", "store")
        event = FakeTransferEvent()
        self.events.append(event)
        return event


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


def _five_keys():
    return [
        CacheEngineKey("test-model", 1, 0, 200 + index, torch.float16)
        for index in range(5)
    ]


def _store_keys_sequentially(backend, metadata, keys):
    for key in keys:
        source = backend.allocate(metadata.get_shapes(), metadata.get_dtypes())
        assert source is not None
        futures = backend.batched_submit_put_task([key], [source])
        assert futures is not None
        assert futures[0].result() is True
        source.ref_count_down()


def test_xds_backend_windowed_read_bounds_registered_gpu_staging(
    direct_config,
    direct_metadata,
):
    direct_config.xds_read_slot_count = 2
    direct_config.xds_write_slot_count = 2
    direct_config.xds_io_queue_depth = 1
    client = FakeXDSClient()
    client.metadata.gpu_buffer_size_bytes = 4 * 4096
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    keys = _five_keys()
    _store_keys_sequentially(backend, direct_metadata, keys)
    connector = FakeStreamingGPUConnector(backend)
    starts = [index * 4 for index in range(5)]
    ends = [start + 4 for start in starts]

    result = backend.batched_get_to_gpu_blocking(
        keys, starts, ends, connector, slot_mapping=object()
    )

    assert result is not None
    assert result.completed == (True, True, True, True, True)
    assert connector.scatter_ranges == list(zip(starts, ends, strict=True))
    assert connector.peak_active <= 2
    assert backend.get_memory_allocator().num_active_allocations == 0
    assert result.peak_staging_bytes <= 2 * 4096
    assert result.peak_inflight_io == 1
    assert all(event.synchronized for event in connector.events)
    backend.close()


def test_xds_backend_windowed_write_bounds_registered_gpu_staging(
    direct_config,
    direct_metadata,
):
    direct_config.xds_read_slot_count = 2
    direct_config.xds_write_slot_count = 2
    direct_config.xds_io_queue_depth = 1
    client = FakeXDSClient()
    client.metadata.gpu_buffer_size_bytes = 4 * 4096
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    connector = FakeStreamingGPUConnector(backend)
    keys = _five_keys()
    starts = [index * 4 for index in range(5)]
    ends = [start + 4 for start in starts]

    result = backend.batched_put_from_gpu_blocking(
        keys, starts, ends, connector, slot_mapping=object()
    )

    assert result is not None
    assert result.completed == (True, True, True, True, True)
    assert [entry[0] for entry in connector.gather_ranges] == [2, 2, 1]
    assert connector.peak_active <= 2
    assert backend.get_memory_allocator().num_active_allocations == 0
    assert result.peak_staging_bytes <= 2 * 4096
    assert all(backend.contains(key) for key in keys)
    backend.close()


def test_xds_backend_windowed_read_short_io_releases_all_slots(
    direct_config,
    direct_metadata,
):
    client = FakeXDSClient()
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    keys = _five_keys()[:2]
    _store_keys_sequentially(backend, direct_metadata, keys)
    client.next_completion = SimpleNamespace(
        request_id=None,
        status=0,
        error_code=0,
        completed_bytes=2048,
    )
    connector = FakeStreamingGPUConnector(backend)

    result = backend.batched_get_to_gpu_blocking(
        keys, [0, 4], [4, 8], connector, slot_mapping=object()
    )

    assert result is not None
    assert result.completed == (False, True)
    assert result.short_io_chunks == 1
    assert backend.get_memory_allocator().num_active_allocations == 0
    backend.close()


def test_xds_backend_windowed_read_scatter_failure_releases_all_slots(
    direct_config,
    direct_metadata,
):
    client = FakeXDSClient()
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    keys = _five_keys()[:2]
    _store_keys_sequentially(backend, direct_metadata, keys)
    connector = FakeStreamingGPUConnector(backend, fail_scatter=True)

    with pytest.raises(RuntimeError, match="scatter failed"):
        backend.batched_get_to_gpu_blocking(
            keys, [0, 4], [4, 8], connector, slot_mapping=object()
        )

    assert backend.get_memory_allocator().num_active_allocations == 0
    backend.close()


def test_xds_backend_windowed_write_short_io_does_not_publish_blob(
    direct_config,
    direct_metadata,
):
    client = FakeXDSClient()
    client.next_completion = SimpleNamespace(
        request_id=None,
        status=0,
        error_code=0,
        completed_bytes=2048,
    )
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    connector = FakeStreamingGPUConnector(backend)
    keys = _five_keys()[:2]

    result = backend.batched_put_from_gpu_blocking(
        keys, [0, 4], [4, 8], connector, slot_mapping=object()
    )

    assert result is not None
    assert result.completed == (False, True)
    assert result.short_io_chunks == 1
    assert not backend.contains(keys[0])
    assert backend.contains(keys[1])
    assert backend.get_memory_allocator().num_active_allocations == 0
    backend.close()


def test_xds_backend_windowed_write_gather_failure_releases_all_slots(
    direct_config,
    direct_metadata,
):
    client = FakeXDSClient()
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    connector = FakeStreamingGPUConnector(backend, fail_gather=True)
    keys = _five_keys()[:2]

    with pytest.raises(RuntimeError, match="gather failed"):
        backend.batched_put_from_gpu_blocking(
            keys, [0, 4], [4, 8], connector, slot_mapping=object()
        )

    assert backend.get_memory_allocator().num_active_allocations == 0
    assert not any(backend.contains(key) for key in keys)
    backend.close()


def test_xds_backend_windowed_pool_reserves_read_and_write_slots(
    direct_config,
    direct_metadata,
):
    direct_config.xds_read_slot_count = 2
    direct_config.xds_write_slot_count = 2
    client = FakeXDSClient()
    client.metadata.gpu_buffer_size_bytes = 3 * 4096

    with pytest.raises(ValueError, match="read_slots=2, write_slots=2"):
        XDSBackend(direct_config, direct_metadata, client, "cpu")
