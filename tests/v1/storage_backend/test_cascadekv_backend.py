# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import deque
from types import SimpleNamespace
import threading

# Third Party
import pytest
import torch

# First Party
import lmcache.v1.storage_backend.cascadekv_backend as cascadekv_backend_module
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cascadekv_backend import CascadeKVBackend
from lmcache.utils import CacheEngineKey


class FakeXDSClient:
    def __init__(self, lifecycle_log=None):
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
        self.lifecycle_log = lifecycle_log
        self._pending_requests = deque()
        self.max_inflight = 0

    def alloc_buffer(self, size):
        self.alloc_sizes.append(size)
        buffer_ref = 101
        view = memoryview(bytearray(size))
        self._buffers[buffer_ref] = view
        return view, buffer_ref

    def free_buffer(self, buffer_ref):
        if self.lifecycle_log is not None:
            self.lifecycle_log.append("free_buffer")
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
        return self._new_request(_descriptor[-1])

    def submit_write(self, *_descriptor):
        self.calls.append("submit_write")
        return self._new_request(_descriptor[-1])

    def _new_request(self, size):
        request_id = self._next_request_id
        self._next_request_id += 1
        self._last_request_id = request_id
        self._pending_requests.append((request_id, size))
        self.max_inflight = max(self.max_inflight, len(self._pending_requests))
        return request_id

    def wait_any_result(self, timeout=5.0):
        del timeout
        self.calls.append("wait_any_result")
        request_id, size = self._pending_requests.popleft()
        if self.next_completion is not None:
            completion = self.next_completion
            if completion.request_id is None:
                completion.request_id = request_id
            self.next_completion = None
            return completion
        return SimpleNamespace(
            request_id=request_id,
            status=0,
            error_code=0,
            completed_bytes=size,
            result=0,
        )

    def close(self):
        if self.lifecycle_log is not None:
            self.lifecycle_log.append("close")
        self.calls.append("close")


class BlockingXDSClient(FakeXDSClient):
    def __init__(self):
        super().__init__()
        self.wait_entered = threading.Event()
        self.allow_completion = threading.Event()

    def wait_any_result(self, timeout=5.0):
        self.wait_entered.set()
        if not self.allow_completion.wait(timeout):
            raise TimeoutError("fake xDS completion was not released")
        return super().wait_any_result(timeout)


class SelectiveBlockingXDSClient(FakeXDSClient):
    def __init__(self):
        super().__init__()
        self.block_completions = False
        self.wait_entered = threading.Event()
        self.allow_completion = threading.Event()

    def wait_any_result(self, timeout=5.0):
        if self.block_completions:
            self.wait_entered.set()
            if not self.allow_completion.wait(timeout):
                raise TimeoutError("fake xDS completion was not released")
        return super().wait_any_result(timeout)


class TimeoutReadXDSClient(FakeXDSClient):
    def __init__(self):
        super().__init__()
        self.fail_wait = False

    def wait_any_result(self, timeout=5.0):
        if self.fail_wait:
            raise TimeoutError("fake xDS read timeout")
        return super().wait_any_result(timeout)


class FlakyCloseXDSClient(FakeXDSClient):
    def __init__(self):
        super().__init__()
        self.close_attempts = 0

    def close(self):
        self.close_attempts += 1
        self.calls.append("close")
        if self.close_attempts == 1:
            raise RuntimeError("fake xDS client close failure")


class FakeCUDART:
    def __init__(self, lifecycle_log=None):
        self.calls = []
        self.register_status = 0
        self.unregister_status = 0
        self.lifecycle_log = lifecycle_log

    def cudaHostRegister(self, ptr, size, flags):
        if self.lifecycle_log is not None:
            self.lifecycle_log.append("cuda_register")
        self.calls.append(("register", ptr, size, flags))
        return self.register_status

    def cudaHostUnregister(self, ptr):
        if self.lifecycle_log is not None:
            self.lifecycle_log.append("cuda_unregister")
        self.calls.append(("unregister", ptr))
        return self.unregister_status

    def cudaGetErrorString(self, status):
        return f"fake CUDA error {status}"


@pytest.fixture(autouse=True)
def fake_cudart(monkeypatch):
    runtime = FakeCUDART()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: runtime)
    return runtime


class FakeEvent:
    def __init__(self, calls):
        self.calls = calls
        self.synchronized = False

    def synchronize(self):
        self.calls.append(("event_synchronize", threading.current_thread().name))
        self.synchronized = True


class FakeGPUConnector:
    def __init__(self, store_event=None):
        self.calls = []
        self.events = []
        self.store_event = store_event

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        del kwargs
        self.calls.append(("batched_from_gpu", len(memory_objs), starts, ends))

    def to_gpu(self, memory_obj, start, end, **kwargs):
        del kwargs
        self.calls.append(("to_gpu", memory_obj, start, end))

    def record_transfer_event(self, direction):
        self.calls.append(("record_transfer_event", direction))
        event = (
            self.store_event
            if direction == "store" and self.store_event is not None
            else FakeEvent(self.calls)
        )
        self.events.append(event)
        return event


class FailingLoadGPUConnector(FakeGPUConnector):
    def to_gpu(self, memory_obj, start, end, **kwargs):
        del memory_obj, start, end, kwargs
        raise RuntimeError("fake CPU-to-GPU failure")


class ScheduledThenFailLoadGPUConnector(FakeGPUConnector):
    def to_gpu(self, memory_obj, start, end, **kwargs):
        super().to_gpu(memory_obj, start, end, **kwargs)
        raise RuntimeError("fake scheduled CPU-to-GPU failure")


class ScheduledThenFailStoreGPUConnector(FakeGPUConnector):
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        super().batched_from_gpu(memory_objs, starts, ends, **kwargs)
        raise RuntimeError("fake scheduled GPU-to-CPU failure")


class BlockingGatherGPUConnector(FakeGPUConnector):
    def __init__(self):
        super().__init__()
        self.gather_entered = threading.Event()
        self.allow_gather = threading.Event()

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        super().batched_from_gpu(memory_objs, starts, ends, **kwargs)
        self.gather_entered.set()
        if not self.allow_gather.wait(5.0):
            raise TimeoutError("fake GPU gather was not released")


class BlockingTransferEvent(FakeEvent):
    def __init__(self, calls):
        super().__init__(calls)
        self.wait_entered = threading.Event()
        self.allow_completion = threading.Event()

    def synchronize(self):
        self.wait_entered.set()
        if not self.allow_completion.wait(5.0):
            raise TimeoutError("fake CUDA transfer was not released")
        super().synchronize()


def make_cache_keys(count):
    return [
        CacheEngineKey(
            model_name="test-model",
            world_size=1,
            worker_id=0,
            chunk_hash=2000 + index,
            dtype=torch.float16,
        )
        for index in range(count)
    ]


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
    fake_cudart,
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
    assert obj.raw_data.data_ptr() >= backend.write_staging_base_ptr
    assert (
        obj.raw_data.data_ptr()
        < backend.write_staging_base_ptr + backend.write_staging_size
    )
    assert backend.read_staging_size == 2 * backend.blob_size
    assert backend.write_staging_size == 2 * backend.blob_size
    assert fake_xds_client.alloc_sizes == [backend.staging_size]
    assert fake_cudart.calls == [
        ("register", backend.staging_base_ptr, backend.staging_size, 0)
    ]

    obj.ref_count_down()
    backend.close()
    assert fake_xds_client.free_refs == [backend.staging_buffer_ref]
    assert fake_cudart.calls[-1] == ("unregister", backend.staging_base_ptr)


@pytest.mark.parametrize(
    ("slot_overrides", "expected_counts"),
    [
        ({"cascadekv_slot_count": 8}, (8, 8)),
        (
            {
                "cascadekv_slot_count": 3,
                "cascadekv_read_slot_count": 8,
                "cascadekv_write_slot_count": 64,
            },
            (8, 64),
        ),
    ],
)
def test_cascadekv_slot_resolution_uses_explicit_config_keys(
    cascade_metadata,
    slot_overrides,
    expected_counts,
):
    config = LMCacheEngineConfig.from_dict(
        {
            "xds_service_name": "xds0",
            "cascadekv_backend": True,
            **slot_overrides,
        }
    )
    client = FakeXDSClient()
    backend = CascadeKVBackend(config, cascade_metadata, client)

    assert (backend.read_slot_count, backend.write_slot_count) == expected_counts
    backend.close()


def test_cascadekv_registration_failure_frees_service_buffer(
    cascade_config,
    cascade_metadata,
    fake_cudart,
):
    client = FakeXDSClient()
    fake_cudart.register_status = 17

    with pytest.raises(RuntimeError, match="cudaHostRegister"):
        CascadeKVBackend(cascade_config, cascade_metadata, client)

    assert client.free_refs == [101]


def test_cascadekv_allocator_init_failure_unwinds_registered_service_buffer(
    cascade_config,
    cascade_metadata,
    fake_cudart,
    monkeypatch,
):
    client = FakeXDSClient()

    def fail_allocator(*_args, **_kwargs):
        raise RuntimeError("fake allocator construction failure")

    monkeypatch.setattr(
        cascadekv_backend_module,
        "TensorMemoryAllocator",
        fail_allocator,
    )

    with pytest.raises(RuntimeError, match="allocator construction"):
        CascadeKVBackend(cascade_config, cascade_metadata, client)

    assert fake_cudart.calls[-1][0] == "unregister"
    assert client.calls[-2:] == ["free_buffer", "close"]


def test_cascadekv_close_unregisters_before_service_buffer_free(
    cascade_config,
    cascade_metadata,
    monkeypatch,
):
    lifecycle = []
    runtime = FakeCUDART(lifecycle)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: runtime)
    client = FakeXDSClient(lifecycle)
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)

    backend.close()

    assert lifecycle == [
        "cuda_register",
        "cuda_unregister",
        "free_buffer",
        "close",
    ]


def test_cascadekv_unregister_failure_keeps_service_mapping_for_retry(
    cascade_config,
    cascade_metadata,
    fake_cudart,
):
    client = FakeXDSClient()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    fake_cudart.unregister_status = 19

    with pytest.raises(RuntimeError, match="cudaHostUnregister"):
        backend.close()

    assert client.free_refs == []

    fake_cudart.unregister_status = 0
    backend.close()
    assert client.free_refs == [backend.staging_buffer_ref]
    assert client.calls[-1] == "close"


def test_cascadekv_close_retry_does_not_double_free_service_buffer(
    cascade_config,
    cascade_metadata,
):
    client = FlakyCloseXDSClient()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)

    with pytest.raises(RuntimeError, match="client close failure"):
        backend.close()

    backend.close()
    assert client.free_refs == [backend.staging_buffer_ref]
    assert client.close_attempts == 2


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
    assert obj.raw_data.data_ptr() >= backend.read_staging_base_ptr
    assert (
        obj.raw_data.data_ptr()
        < backend.read_staging_base_ptr + backend.read_staging_size
    )
    obj.ref_count_down()
    backend.close()


def test_cascadekv_batched_read_uses_bounded_completion_ring(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 3
    cascade_config.cascadekv_write_slot_count = 3
    cascade_config.cascadekv_io_queue_depth = 2
    client = FakeXDSClient()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    keys = make_cache_keys(3)
    sources = [
        backend.allocate(
            cascade_metadata.get_shapes(),
            cascade_metadata.get_dtypes(),
        )
        for _ in keys
    ]
    assert all(source is not None for source in sources)
    futures = backend.batched_submit_put_task(keys, sources)
    assert futures is not None
    assert all(future.result(timeout=5.0) for future in futures)
    for source in sources:
        source.ref_count_down()
    client.calls.clear()
    client.max_inflight = 0

    results = backend.batched_get_blocking(keys)

    assert all(result is not None for result in results)
    first_wait = client.calls.index("wait_any_result")
    assert client.calls[:first_wait].count("submit_read") == 2
    assert client.max_inflight == 2
    for result in results:
        result.ref_count_down()
    backend.close()


def test_cascadekv_read_timeout_quarantines_service_owned_dma_slot(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 1
    cascade_config.cascadekv_write_slot_count = 1
    cascade_config.cascadekv_io_queue_depth = 1
    client = TimeoutReadXDSClient()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    key = make_cache_keys(1)[0]
    source = backend.allocate(
        cascade_metadata.get_shapes(),
        cascade_metadata.get_dtypes(),
    )
    assert source is not None
    futures = backend.batched_submit_put_task([key], [source])
    assert futures is not None
    assert futures[0].result(timeout=5.0) is True
    source.ref_count_down()
    client.fail_wait = True

    with pytest.raises(TimeoutError, match="fake xDS read timeout"):
        backend.batched_get_blocking([key])

    assert backend.read_memory_allocator.num_active_allocations == 1
    with pytest.raises(RuntimeError, match="poisoned"):
        backend.contains(key)

    client.fail_wait = False
    backend.drain_quarantined_xds_transfers(timeout=5.0)
    assert backend.read_memory_allocator.num_active_allocations == 0
    assert backend.contains(key)
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


def test_cascadekv_aggregates_write_submissions_before_waiting(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 2
    cascade_config.cascadekv_write_slot_count = 4
    cascade_config.cascadekv_io_queue_depth = 2
    cascade_config.cascadekv_write_window_size = 4
    client = FakeXDSClient()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    keys = make_cache_keys(3)
    objects = [
        backend.allocate(
            cascade_metadata.get_shapes(),
            cascade_metadata.get_dtypes(),
        )
        for _ in keys
    ]
    assert all(memory_obj is not None for memory_obj in objects)

    futures = backend.batched_submit_put_task(keys, objects)

    assert futures is not None
    assert all(future.result(timeout=5.0) for future in futures)
    first_wait = client.calls.index("wait_any_result")
    assert client.calls[:first_wait].count("submit_write") == 2
    assert client.max_inflight == 2
    for memory_obj in objects:
        memory_obj.ref_count_down()
    backend.close()


def test_cascadekv_streaming_write_and_read_overlap_more_chunks_than_slots(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 2
    cascade_config.cascadekv_write_slot_count = 2
    cascade_config.cascadekv_io_queue_depth = 2
    cascade_config.cascadekv_write_window_size = 2
    client = FakeXDSClient()
    connector = FakeGPUConnector()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    keys = make_cache_keys(5)
    starts = [index * 4 for index in range(len(keys))]
    ends = [start + 4 for start in starts]

    submission = backend.batched_put_from_gpu_blocking(
        keys,
        starts,
        ends,
        connector,
    )
    assert submission is not None
    assert submission.accepted == (True, True, True, True, True)
    backend.flush_async_writes(timeout=5.0)
    assert all(backend.contains(key) for key in keys)
    store_sync_threads = [
        call[1]
        for call in connector.calls
        if call[0] == "event_synchronize" and call[1] == "xds-async-write"
    ]
    assert store_sync_threads

    client.calls.clear()
    client.max_inflight = 0
    result = backend.batched_get_to_gpu_blocking(
        keys,
        starts,
        ends,
        connector,
    )

    assert result is not None
    assert result.completed == (True, True, True, True, True)
    assert result.peak_staging_bytes <= 2 * backend.blob_size
    assert result.peak_inflight_io == 2
    first_wait = client.calls.index("wait_any_result")
    assert client.calls[:first_wait].count("submit_read") == 2
    assert len([call for call in connector.calls if call[0] == "to_gpu"]) == 5
    assert all(event.synchronized for event in connector.events)
    assert backend.read_memory_allocator.num_active_allocations == 0
    backend.close()


def test_cascadekv_pending_write_is_readable_before_ssd_completion(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 2
    cascade_config.cascadekv_write_slot_count = 2
    cascade_config.cascadekv_io_queue_depth = 2
    client = BlockingXDSClient()
    connector = FakeGPUConnector()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    key = make_cache_keys(1)[0]

    submission = backend.batched_put_from_gpu_blocking(
        [key],
        [0],
        [4],
        connector,
    )
    assert submission is not None
    assert client.wait_entered.wait(5.0)
    client.calls.clear()

    pending_obj = backend.get_blocking(key)
    assert pending_obj is not None
    assert pending_obj.raw_data.data_ptr() >= backend.write_staging_base_ptr
    pending_obj.ref_count_down()

    result = backend.batched_get_to_gpu_blocking(
        [key],
        [0],
        [4],
        connector,
    )

    assert result is not None
    assert result.completed == (True,)
    assert "submit_read" not in client.calls
    client.allow_completion.set()
    backend.flush_async_writes(timeout=5.0)
    backend.close()


def test_cascadekv_pending_read_waits_for_gpu_to_cpu_event(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 1
    cascade_config.cascadekv_write_slot_count = 1
    cascade_config.cascadekv_io_queue_depth = 1
    transfer_event = BlockingTransferEvent([])
    connector = FakeGPUConnector(store_event=transfer_event)
    client = FakeXDSClient()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    key = make_cache_keys(1)[0]
    submission = backend.batched_put_from_gpu_blocking(
        [key],
        [0],
        [4],
        connector,
    )
    assert submission is not None
    assert transfer_event.wait_entered.wait(5.0)

    result_holder = []

    def read_pending():
        result_holder.append(backend.get_blocking(key))

    read_thread = threading.Thread(target=read_pending)
    read_thread.start()
    read_thread.join(timeout=0.1)
    assert read_thread.is_alive()
    assert result_holder == []

    transfer_event.allow_completion.set()
    read_thread.join(timeout=5.0)
    assert not read_thread.is_alive()
    assert result_holder[0] is not None
    result_holder[0].ref_count_down()
    backend.flush_async_writes(timeout=5.0)
    backend.close()


def test_cascadekv_streaming_admission_reserves_key_against_regular_put(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 1
    cascade_config.cascadekv_write_slot_count = 2
    cascade_config.cascadekv_io_queue_depth = 1
    client = FakeXDSClient()
    connector = BlockingGatherGPUConnector()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    key = make_cache_keys(1)[0]
    regular_obj = backend.allocate(
        cascade_metadata.get_shapes(),
        cascade_metadata.get_dtypes(),
    )
    assert regular_obj is not None
    direct_results = []

    def submit_direct():
        direct_results.append(
            backend.batched_put_from_gpu_blocking(
                [key],
                [0],
                [4],
                connector,
            )
        )

    direct_thread = threading.Thread(target=submit_direct)
    direct_thread.start()
    assert connector.gather_entered.wait(5.0)

    regular_futures = backend.batched_submit_put_task([key], [regular_obj])
    assert regular_futures is not None
    assert regular_futures[0].result(timeout=5.0) is False

    connector.allow_gather.set()
    direct_thread.join(timeout=5.0)
    assert not direct_thread.is_alive()
    assert direct_results[0] is not None
    backend.flush_async_writes(timeout=5.0)
    assert backend.contains(key)
    regular_obj.ref_count_down()
    backend.close()


def test_cascadekv_close_waits_for_active_read_and_rejects_new_operations(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 1
    cascade_config.cascadekv_write_slot_count = 1
    cascade_config.cascadekv_io_queue_depth = 1
    client = SelectiveBlockingXDSClient()
    connector = FakeGPUConnector()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    key = make_cache_keys(1)[0]
    submission = backend.batched_put_from_gpu_blocking(
        [key],
        [0],
        [4],
        connector,
    )
    assert submission is not None
    backend.flush_async_writes(timeout=5.0)
    client.block_completions = True
    read_results = []
    close_errors = []

    def read_to_gpu():
        read_results.append(
            backend.batched_get_to_gpu_blocking(
                [key],
                [0],
                [4],
                connector,
            )
        )

    def close_backend():
        try:
            backend.close()
        except BaseException as error:
            close_errors.append(error)

    read_thread = threading.Thread(target=read_to_gpu)
    read_thread.start()
    assert client.wait_entered.wait(5.0)
    close_thread = threading.Thread(target=close_backend)
    close_thread.start()
    close_thread.join(timeout=0.1)
    assert close_thread.is_alive()
    with pytest.raises(RuntimeError, match="closing or closed"):
        backend.contains(key)

    client.allow_completion.set()
    read_thread.join(timeout=5.0)
    close_thread.join(timeout=5.0)
    assert not read_thread.is_alive()
    assert not close_thread.is_alive()
    assert close_errors == []
    assert read_results[0] is not None
    assert read_results[0].completed == (True,)


def test_cascadekv_pending_reader_holds_slot_backpressure_until_release(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 1
    cascade_config.cascadekv_write_slot_count = 1
    cascade_config.cascadekv_io_queue_depth = 1
    client = BlockingXDSClient()
    connector = FakeGPUConnector()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    first_key, second_key = make_cache_keys(2)
    first_submission = backend.batched_put_from_gpu_blocking(
        [first_key],
        [0],
        [4],
        connector,
    )
    assert first_submission is not None
    assert client.wait_entered.wait(5.0)
    pending_obj = backend.get_blocking(first_key)
    assert pending_obj is not None
    client.allow_completion.set()
    backend.flush_async_writes(timeout=5.0)
    second_results = []
    second_errors = []

    def submit_second():
        try:
            second_results.append(
                backend.batched_put_from_gpu_blocking(
                    [second_key],
                    [4],
                    [8],
                    connector,
                )
            )
        except BaseException as error:
            second_errors.append(error)

    submit_thread = threading.Thread(target=submit_second)
    submit_thread.start()
    submit_thread.join(timeout=0.1)
    assert submit_thread.is_alive()

    pending_obj.ref_count_down()
    submit_thread.join(timeout=5.0)
    assert not submit_thread.is_alive()
    assert second_errors == []
    assert second_results[0] is not None
    backend.flush_async_writes(timeout=5.0)
    backend.close()


def test_cascadekv_short_read_recycles_read_slot(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 1
    cascade_config.cascadekv_write_slot_count = 1
    cascade_config.cascadekv_io_queue_depth = 1
    client = FakeXDSClient()
    connector = FakeGPUConnector()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    key = make_cache_keys(1)[0]
    submission = backend.batched_put_from_gpu_blocking(
        [key],
        [0],
        [4],
        connector,
    )
    assert submission is not None
    backend.flush_async_writes(timeout=5.0)
    client.next_completion = SimpleNamespace(
        request_id=None,
        status=0,
        error_code=0,
        completed_bytes=backend.blob_size - 1,
        result=0,
    )

    result = backend.batched_get_to_gpu_blocking(
        [key],
        [0],
        [4],
        connector,
    )

    assert result is not None
    assert result.completed == (False,)
    assert result.short_io_chunks == 1
    assert backend.read_memory_allocator.num_active_allocations == 0
    backend.close()


def test_cascadekv_gpu_load_failure_releases_slot_and_metadata_pin(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 1
    cascade_config.cascadekv_write_slot_count = 1
    cascade_config.cascadekv_io_queue_depth = 1
    client = FakeXDSClient()
    store_connector = FakeGPUConnector()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    key = make_cache_keys(1)[0]
    submission = backend.batched_put_from_gpu_blocking(
        [key],
        [0],
        [4],
        store_connector,
    )
    assert submission is not None
    backend.flush_async_writes(timeout=5.0)

    with pytest.raises(RuntimeError, match="fake CPU-to-GPU failure"):
        backend.batched_get_to_gpu_blocking(
            [key],
            [0],
            [4],
            FailingLoadGPUConnector(),
        )

    assert backend.read_memory_allocator.num_active_allocations == 0
    assert backend.remove(key, force=False) is True
    backend.close()


def test_cascadekv_scheduled_store_failure_synchronizes_before_slot_release(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 1
    cascade_config.cascadekv_write_slot_count = 1
    cascade_config.cascadekv_io_queue_depth = 1
    client = FakeXDSClient()
    connector = ScheduledThenFailStoreGPUConnector()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    key = make_cache_keys(1)[0]

    with pytest.raises(RuntimeError, match="scheduled GPU-to-CPU failure"):
        backend.batched_put_from_gpu_blocking(
            [key],
            [0],
            [4],
            connector,
        )

    assert any(call[0] == "record_transfer_event" for call in connector.calls)
    assert any(call[0] == "event_synchronize" for call in connector.calls)
    assert backend.write_memory_allocator.num_active_allocations == 0
    backend.close()


def test_cascadekv_scheduled_load_failure_synchronizes_before_slot_release(
    cascade_config,
    cascade_metadata,
):
    cascade_config.cascadekv_read_slot_count = 1
    cascade_config.cascadekv_write_slot_count = 1
    cascade_config.cascadekv_io_queue_depth = 1
    client = FakeXDSClient()
    store_connector = FakeGPUConnector()
    backend = CascadeKVBackend(cascade_config, cascade_metadata, client)
    key = make_cache_keys(1)[0]
    submission = backend.batched_put_from_gpu_blocking(
        [key],
        [0],
        [4],
        store_connector,
    )
    assert submission is not None
    backend.flush_async_writes(timeout=5.0)
    load_connector = ScheduledThenFailLoadGPUConnector()

    with pytest.raises(RuntimeError, match="scheduled CPU-to-GPU failure"):
        backend.batched_get_to_gpu_blocking(
            [key],
            [0],
            [4],
            load_connector,
        )

    assert any(call[0] == "record_transfer_event" for call in load_connector.calls)
    assert any(call[0] == "event_synchronize" for call in load_connector.calls)
    assert backend.read_memory_allocator.num_active_allocations == 0
    assert backend.remove(key, force=False) is True
    backend.close()
