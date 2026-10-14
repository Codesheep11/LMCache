# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from threading import Event

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
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
        self.read_descriptors = []
        self.read_window_calls = []
        self.layer_stream_calls = []
        self.released_layer_stream_slots = []
        self.write_descriptors = []
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

    def submit_read(self, *descriptor):
        self.calls.append("submit_read")
        self.read_descriptors.append(descriptor)
        return self._new_request()

    def submit_write(self, *descriptor):
        self.calls.append("submit_write")
        self.write_descriptors.append(descriptor)
        return self._new_request()

    def read_window(
        self,
        requests,
        queue_depth=4,
        timeout=5.0,
        progress_interval=0,
        progress_callback=None,
        validate_only=False,
    ):
        del timeout
        descriptors = list(requests)
        self.calls.append("read_window")
        self.read_descriptors.extend(descriptors)
        self.read_window_calls.append((len(descriptors), queue_depth, validate_only))
        if progress_callback is not None:
            for completed_count in range(
                progress_interval, len(descriptors) + 1, progress_interval
            ):
                progress_callback(completed_count)
        return []

    def read_window_block_layers(
        self,
        blob_handles,
        descriptor_templates,
        num_layers,
        blob_layer_stride,
        queue_depth=4,
        timeout=5.0,
        progress_interval=0,
        progress_callback=None,
        validate_only=False,
    ):
        templates = list(descriptor_templates)
        block_count = len(blob_handles)
        requests = []
        for io_index in range(block_count * num_layers):
            template_index = io_index * 3
            requests.append(
                (
                    blob_handles[io_index % block_count],
                    templates[template_index],
                    templates[template_index + 1],
                    (io_index // block_count) * blob_layer_stride,
                    templates[template_index + 2],
                )
            )
        return self.read_window(
            requests,
            queue_depth=queue_depth,
            timeout=timeout,
            progress_interval=progress_interval,
            progress_callback=progress_callback,
            validate_only=validate_only,
        )

    def create_layer_read_stream(
        self,
        blob_handles,
        slot_descriptors,
        slot_count,
        num_layers,
        blob_layer_stride,
        queue_depth=4,
        timeout=5.0,
        progress_callback=None,
    ):
        del timeout
        client = self
        handles = list(blob_handles)
        descriptors = list(slot_descriptors)
        client.layer_stream_calls.append(
            (
                len(handles),
                slot_count,
                num_layers,
                queue_depth,
            )
        )

        class FakeLayerReadStream:
            def run(self):
                block_count = len(handles)
                for layer_id in range(num_layers):
                    slot_id = layer_id % slot_count
                    for block_id, blob_handle in enumerate(handles):
                        buffer_ref, buffer_offset, size = descriptors[
                            slot_id * block_count + block_id
                        ]
                        client.read_descriptors.append(
                            (
                                blob_handle,
                                buffer_ref,
                                buffer_offset,
                                layer_id * blob_layer_stride,
                                size,
                            )
                        )
                    assert progress_callback is not None
                    progress_callback(layer_id)
                return {
                    "configured_qd": queue_depth,
                    "peak_qd": queue_depth,
                    "average_qd": float(queue_depth),
                }

            def release_slot(self, layer_id):
                client.released_layer_stream_slots.append(layer_id)

        return FakeLayerReadStream()

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


class BlockingWriteClient(FakeXDSClient):
    def __init__(self):
        super().__init__()
        self.write_wait_started = Event()
        self.release_write = Event()

    def wait_any_result(self, timeout=5.0):
        self.write_wait_started.set()
        if not self.release_write.wait(timeout):
            raise TimeoutError("test write remained blocked")
        return super().wait_any_result(timeout)


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


def test_xds_layerwise_block_blob_reuses_handle_and_reads_layer_offsets():
    config = LMCacheEngineConfig.from_defaults()
    config.xds_service_name = "xds0"
    config.xds_backend = True
    config.xds_require_direct = True
    config.xds_windowed_io = True
    config.xds_io_queue_depth = 2
    config.xds_read_slot_count = 1
    config.xds_write_slot_count = 1
    config.use_layerwise = True
    config.extra_config = {"xds_layerwise_block_blob": True}
    metadata = LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(2, 2, 4, 1, 256),
        chunk_size=4,
    )
    client = FakeXDSClient()
    client.metadata.blob_size_bytes = 8192
    client.metadata.gpu_buffer_size_bytes = 32768
    backend = XDSBackend(config, metadata, client, "cpu")
    keys = [
        CacheEngineKey("test-model", 1, 0, chunk_hash, torch.float16)
        for chunk_hash in (301, 302, 303)
    ]
    objects_by_layer = []
    layer_shape = torch.Size([4, 2, 256])
    for _ in range(2):
        layer_objects = backend.batched_allocate(
            layer_shape,
            torch.float16,
            batch_size=3,
            fmt=MemoryFormat.KV_T2D,
        )
        assert layer_objects is not None
        objects_by_layer.append(layer_objects)

    class CompletionEvent:
        synchronized = False
        fail = True

        def synchronize(self) -> None:
            assert client.write_descriptors == []
            assert not any(backend.contains(key) for key in keys)
            if self.fail:
                raise RuntimeError("GPU production failed")
            self.synchronized = True

    completion_event = CompletionEvent()
    with pytest.raises(RuntimeError, match="GPU production failed"):
        backend.blockwise_batched_submit_put_task(
            keys, objects_by_layer, completion_event=completion_event
        )
    assert client.write_descriptors == []
    assert not any(backend.contains(key) for key in keys)
    completion_event.fail = False
    backend.blockwise_batched_submit_put_task(
        keys, objects_by_layer, completion_event=completion_event
    )
    assert completion_event.synchronized

    assert all(backend.contains(key) for key in keys)
    assert len({descriptor[0] for descriptor in client.write_descriptors}) == 3
    assert [descriptor[3] for descriptor in client.write_descriptors] == [
        0,
        0,
        0,
        4096,
        4096,
        4096,
    ]
    for layer_objects in objects_by_layer:
        for memory_obj in layer_objects:
            memory_obj.ref_count_down()

    published = []

    def publish(completed_count, layer_views):
        published.append((completed_count, layer_views))

    result = backend.blockwise_batched_get_blocking(keys, publish)

    assert result == []
    assert [completed for completed, _ in published] == [3, 6]
    assert [views[0].chunk_count for _, views in published] == [3, 3]
    assert [views[0].get_shape()[0] for _, views in published] == [12, 12]
    assert [descriptor[3] for descriptor in client.read_descriptors] == [
        0,
        0,
        0,
        4096,
        4096,
        4096,
    ]
    assert client.layer_stream_calls == [(3, 2, 2, 2)]
    assert client.read_window_calls == []
    for _, views in published:
        retained = backend.release_layerwise_staging(views)
        if id(views[0]) not in retained:
            views[0].ref_count_down()
    assert backend.get_memory_allocator().num_active_allocations == 0
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


def test_xds_backend_async_direct_write_uses_bounded_leases(
    direct_config,
    direct_metadata,
):
    direct_config.xds_async_write = True
    direct_config.xds_read_slot_count = 2
    direct_config.xds_write_slot_count = 2
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
    assert result.accepted == (True, True, True, True, True)
    assert result.accepted_bytes == 5 * 4096
    assert result.durable is False
    assert [entry[0] for entry in connector.gather_ranges] == [2, 2, 1]
    assert connector.peak_active <= 2
    snapshot = backend.flush_async_writes(timeout=1.0)
    assert snapshot.outstanding_bytes == 0
    assert all(backend.contains(key) for key in keys)
    assert backend.get_memory_allocator().num_active_allocations == 0
    backend.close()


def test_xds_backend_async_pending_lease_is_immediately_readable(
    direct_config,
    direct_metadata,
):
    direct_config.xds_async_write = True
    client = BlockingWriteClient()
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    store_connector = FakeStreamingGPUConnector(backend)
    load_connector = FakeStreamingGPUConnector(backend)
    key = _five_keys()[0]

    submission = backend.batched_put_from_gpu_blocking(
        [key], [0], [4], store_connector, slot_mapping=object()
    )
    assert submission is not None
    assert submission.accepted == (True,)
    assert client.write_wait_started.wait(timeout=1.0)

    result = backend.batched_get_to_gpu_blocking(
        [key], [0], [4], load_connector, slot_mapping=object()
    )

    assert result is not None
    assert result.completed == (True,)
    assert load_connector.scatter_ranges == [(0, 4)]
    assert "submit_read" not in client.calls
    client.release_write.set()
    backend.flush_async_writes(timeout=1.0)
    backend.close()


def test_xds_backend_async_failure_rolls_back_and_close_cleans_resources(
    direct_config,
    direct_metadata,
):
    direct_config.xds_async_write = True
    client = FakeXDSClient()
    client.next_completion = SimpleNamespace(
        request_id=None,
        status=0,
        error_code=0,
        completed_bytes=2048,
    )
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    connector = FakeStreamingGPUConnector(backend)
    key = _five_keys()[0]

    submission = backend.batched_put_from_gpu_blocking(
        [key], [0], [4], connector, slot_mapping=object()
    )

    assert submission is not None
    assert submission.accepted == (True,)
    with pytest.raises(RuntimeError, match="xDS async write failed"):
        backend.flush_async_writes(timeout=1.0)
    assert not backend.contains(key)
    assert backend.get_memory_allocator().num_active_allocations == 0
    assert client.released_blob_handles == [700]
    with pytest.raises(RuntimeError, match="xDS async write failed"):
        backend.close()
    assert client.unregistered == [77]
    assert "close" in client.calls


@pytest.mark.parametrize(
    "failure", ["none", "short", "timeout", "partial_timeout", "descriptor_error"]
)
@pytest.mark.parametrize("native", [False, True])
def test_direct_readv_skips_scatter_and_releases_pins(
    direct_config, direct_metadata, cache_key, monkeypatch, failure, native
):
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2

    class ReadvClient(FakeXDSClient):
        def __init__(self):
            super().__init__()
            self.metadata.blob_size_bytes = 32768
            self.metadata.gpu_buffer_size_bytes = 65536
            self.readv_calls = []
            self.sizes = {}

        def register_gpu_buffer(self, ptr, size):
            super().register_gpu_buffer(ptr, size)
            return len(self.registered)

        def submit_write(self, *descriptor):
            rid = super().submit_write(*descriptor)
            self.sizes[rid] = descriptor[-1]
            return rid

        def submit_readv(self, blob, offset, slices):
            self.readv_calls.append((blob, offset, slices))
            rid = self._new_request()
            self.sizes[rid] = sum(s[2] for s in slices) - (
                4096 if failure == "short" else 0
            )
            return rid

        def wait_any_result(self, timeout=5.0):
            if self.readv_calls and (
                failure == "timeout"
                or (
                    failure == "partial_timeout" and len(self._pending_request_ids) == 1
                )
            ):
                raise TimeoutError("injected direct KV timeout")
            result = super().wait_any_result(timeout)
            result.completed_bytes = self.sizes[result.request_id]
            return result

    direct_config.extra_config = {"xds_readv_direct": True}
    direct_config.xds_windowed_io = False
    client = ReadvClient()
    if native:
        import xds

        prepared = 0

        def prepare(ids, groups, size):
            nonlocal prepared
            prepared += 1
            if failure == "descriptor_error" and prepared == 2:
                raise MemoryError("injected descriptor allocation failure")
            return xds.prepare_indexed_readv(ids, groups, size, 4096)

        client.prepare_indexed_readv = prepare
        client.submit_readv_plan = lambda blob, offset, plan: client.submit_readv(
            blob, offset, plan.to_slices()
        )
    backend = XDSBackend(direct_config, direct_metadata, client, "cpu")
    shape = torch.Size([2, 1, 32, 256])
    obj = backend.allocate([shape], [torch.float16])
    assert backend.batched_submit_put_task([cache_key], [obj])[0].result()
    obj.ref_count_down()
    connector = VLLMPagedMemGPUConnectorV2(256, 1, layout_hints={"kv_layout": "NHD"})
    target = torch.empty((2, 8, 16, 2, 128), dtype=torch.float16, device="cuda")

    def reject_scatter(*args, **kwargs):
        raise AssertionError("direct readv must not call scatter")

    monkeypatch.setattr(connector, "to_gpu", reject_scatter)
    kwargs = dict(kvcaches=[target], slot_mapping=torch.arange(32))
    if failure == "descriptor_error":
        from lmcache.v1.gpu_connector.xds_readv import XDSReadvPlan

        original_bind = XDSReadvPlan.bind_slices
        calls = 0

        def fail_second_bind(
            plan: XDSReadvPlan, refs: list[int]
        ) -> list[tuple[int, int, int]]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise MemoryError("injected descriptor allocation failure")
            return original_bind(plan, refs)

        monkeypatch.setattr(XDSReadvPlan, "bind_slices", fail_second_bind)
        with pytest.raises(MemoryError, match="descriptor allocation"):
            backend.batched_get_to_gpu_blocking(
                [cache_key, cache_key],
                [0, 32],
                [32, 64],
                connector,
                kvcaches=[target],
                slot_mapping=torch.arange(64),
            )
        assert len(client.readv_calls) == 1
        # The already submitted sibling is drained and both pins are released.
        assert backend.remove(cache_key, force=False)
        backend.close()
        return
    if failure == "partial_timeout":
        with pytest.raises(TimeoutError):
            backend.batched_get_to_gpu_blocking(
                [cache_key, cache_key],
                [0, 32],
                [32, 64],
                connector,
                kvcaches=[target],
                slot_mapping=torch.arange(64),
            )
        assert not backend.remove(cache_key, force=False)
        failure = "none"
        client.wait_any_result()
        assert backend.unpin(cache_key)
        assert not backend.unpin(cache_key)  # Successful sibling kept no extra pin.
        backend.close()
        return
    if failure == "timeout":
        with pytest.raises(TimeoutError):
            backend.batched_get_to_gpu_blocking(
                [cache_key], [0], [32], connector, **kwargs
            )
        assert not backend.remove(cache_key, force=False)
        with pytest.raises(RuntimeError, match="unresolved DMA"):
            backend.batched_get_to_gpu_blocking(
                [cache_key], [0], [32], connector, **kwargs
            )
        # Explicitly finish the fake DMA before releasing the retained pin.
        failure = "none"
        client.wait_any_result()
        assert backend.unpin(cache_key)
        backend.close()
        return
    if failure == "short":
        with pytest.raises(RuntimeError, match="short"):
            backend.batched_get_to_gpu_blocking(
                [cache_key], [0], [32], connector, **kwargs
            )
    else:
        result = backend.batched_get_to_gpu_blocking(
            [cache_key], [0], [32], connector, **kwargs
        )
        assert result.completed == (True,) and result.completed_bytes == 32768
        assert result.peak_staging_bytes == 0
        backend.batched_get_to_gpu_blocking([cache_key], [0], [32], connector, **kwargs)
        assert len(client.registered) == 2  # Pool + reused final target.
        batch_kwargs = dict(kvcaches=[target], slot_mapping=torch.arange(64))
        result = backend.batched_get_to_gpu_blocking(
            [cache_key, cache_key], [0, 32], [32, 64], connector, **batch_kwargs
        )
        assert result.completed == (True, True) and result.completed_bytes == 65536
        before = len(client.readv_calls)
        batch_kwargs["slot_mapping"] = torch.arange(32).repeat(2)
        assert (
            backend.batched_get_to_gpu_blocking(
                [cache_key, cache_key], [0, 32], [32, 64], connector, **batch_kwargs
            )
            is None
        )
        assert len(client.readv_calls) == before
    slices = client.readv_calls[0][2]
    assert len(slices) == 4
    assert sum(length for _, _, length in slices) == 32768
    # Unsupported partial block falls back before publishing another readv.
    count = len(client.readv_calls)
    assert (
        backend.batched_get_to_gpu_blocking([cache_key], [0], [31], connector, **kwargs)
        is None
    )
    assert len(client.readv_calls) == count
    backend.close()
    assert len(client.released_blob_handles) == 1
    assert len(client.unregistered) == 2


@pytest.mark.parametrize("failure", ["none", "overlap", "cancel", "timeout", "short"])
def test_layerwise_readv_offsets_readiness_and_lifetime(failure):
    from lmcache.v1.gpu_connector.xds_layerwise import XDSReadvLayerwiseGPUConnector

    class LayerClient(FakeXDSClient):
        def __init__(self):
            super().__init__()
            self.metadata.blob_size_bytes = 65536
            self.metadata.gpu_buffer_size_bytes = 262144
            self.sizes = {}
            self.readv_calls = []

        def register_gpu_buffer(self, ptr, size):
            super().register_gpu_buffer(ptr, size)
            return len(self.registered)

        def submit_write(self, *args):
            rid = super().submit_write(*args)
            self.sizes[rid] = args[-1]
            return rid

        def submit_readv(self, blob, offset, slices):
            rid = self._new_request()
            self.readv_calls.append((blob, offset, slices))
            self.sizes[rid] = sum(x[2] for x in slices)
            return rid

        def wait_any_result(self, timeout=5.0):
            if self.readv_calls and failure == "timeout":
                raise TimeoutError("layer DMA pending")
            result = super().wait_any_result(timeout)
            result.completed_bytes = self.sizes[result.request_id]
            if self.readv_calls and failure == "short":
                result.completed_bytes -= 4096
            return result

    config = LMCacheEngineConfig.from_defaults()
    config.xds_backend = True
    config.use_layerwise = True
    config.xds_windowed_io = False
    config.xds_io_queue_depth = 2
    config.extra_config = {
        "xds_layerwise_block_blob": True,
        "xds_layerwise_readv": True,
    }
    metadata = LMCacheMetadata(
        model_name="layer-readv",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(2, 2, 32, 2, 128),
        chunk_size=32,
    )
    client = LayerClient()
    backend = XDSBackend(config, metadata, client, "cpu")
    keys = [CacheEngineKey("layer-readv", 1, 0, i, torch.float16) for i in [901, 902]]
    objects = [
        backend.batched_allocate(
            torch.Size([2, 32, 256]), torch.float16, 2, fmt=MemoryFormat.KV_2TD
        )
        for _ in range(2)
    ]
    backend.blockwise_batched_submit_put_task(keys, objects)
    for layer in objects:
        for obj in layer:
            obj.ref_count_down()
    connector = XDSReadvLayerwiseGPUConnector.from_metadata(
        metadata, False, torch.device("cuda"), layout_hints={"kv_layout": "NHD"}
    )
    targets = [
        torch.empty((2, 8, 16, 2, 128), device="cuda", dtype=torch.float16)
        for _ in range(2)
    ]
    slots = torch.arange(64)
    if failure == "overlap":
        slots[32:] = slots[:32]
    iterator = backend.prepare_layerwise_readv(
        keys, [0, 32], [32, 64], connector, kvcaches=targets, slot_mapping=slots
    )
    if failure == "overlap":
        assert iterator is None and client.readv_calls == []
    elif failure in ["short", "timeout"]:
        with pytest.raises((RuntimeError, TimeoutError)):
            next(iterator)
        if failure == "timeout":
            assert not backend.remove(keys[0], force=False)
            failure = "none"
            client.wait_any_result()
            client.wait_any_result()
            for key in keys:
                assert backend.unpin(key)
    else:
        result = next(iterator)
        assert result.completed_bytes == 65536 and result.completed == (True, True)
        assert [c[1] for c in client.readv_calls[:2]] == [0, 0]
        assert not backend.remove(keys[0], force=False)
        if failure == "cancel":
            iterator.close()
        else:
            assert next(iterator).completed_bytes == 65536
            with pytest.raises(StopIteration):
                next(iterator)
            assert [c[1] for c in client.readv_calls] == [0, 0, 32768, 32768]
    for key in keys:
        assert backend.remove(key, force=False)
    backend.close()


def test_layerwise_kv_major_fallback_keeps_chunk_layout() -> None:
    config = LMCacheEngineConfig.from_defaults()
    config.xds_backend = True
    config.use_layerwise = True
    config.xds_windowed_io = False
    config.xds_io_queue_depth = 2
    config.extra_config = {"xds_layerwise_block_blob": True}
    metadata = LMCacheMetadata(
        model_name="fallback",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(2, 2, 4, 1, 256),
        chunk_size=4,
    )
    client = FakeXDSClient()
    client.metadata.blob_size_bytes = 8192
    client.metadata.gpu_buffer_size_bytes = 65536
    backend = XDSBackend(config, metadata, client, "cpu")
    keys = [CacheEngineKey("fallback", 1, 0, i, torch.float16) for i in [1001, 1002]]
    objects = [
        backend.batched_allocate(
            torch.Size([2, 4, 256]),
            torch.float16,
            batch_size=2,
            fmt=MemoryFormat.KV_2TD,
        )
        for _ in range(2)
    ]
    backend.blockwise_batched_submit_put_task(keys, objects)
    for layer in objects:
        for obj in layer:
            obj.ref_count_down()
    published = []

    def publish(count: int, objects: list) -> None:
        assert len(objects) == 2
        assert all(obj.get_memory_format() == MemoryFormat.KV_2TD for obj in objects)
        assert all(obj.tensor.shape == (2, 4, 256) for obj in objects)
        published.append(count)
        for obj in objects:
            obj.ref_count_down()

    assert backend.blockwise_batched_get_blocking(keys, publish) == []
    assert published == [2, 4]
    for key in keys:
        assert backend.remove(key, force=False)
    backend.close()
