# SPDX-License-Identifier: Apache-2.0
"""Public behavior tests for the whole-block CascadeKV backend."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
import asyncio
import threading
import time

import pytest
import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, XDSServiceMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cascadekv_backend import (
    BlockTable,
    CascadeKVBackend,
    SSDBlock,
)


class FakeCUDART:
    def cudaHostRegister(self, _ptr, _size, _flags):
        return 0

    def cudaHostUnregister(self, _ptr):
        return 0

    def cudaGetErrorString(self, status):
        return f"fake CUDA error {status}"


class FakeXDSClient:
    """Controllable out-of-order xDS client."""

    def __init__(self, buffer_size: int, blob_size: int = 8192):
        self.metadata = SimpleNamespace(
            io_unit_size=4096,
            blob_size_bytes=blob_size,
            capacity_bytes=blob_size * 32,
            worker_id=0,
        )
        self.buffer_size = buffer_size
        self.allow_completions = threading.Event()
        self.allow_completions.set()
        self.blocked_operations: set[str] = set()
        self.failed_operations: set[str] = set()
        self.calls: list[tuple] = []
        self.event_order: list[str] = []
        self.submissions: list[tuple[str, tuple, int]] = []
        self.released: list[int] = []
        self._pending: deque = deque()
        self._lock = threading.Lock()
        self._next_blob = 100
        self._next_request = 1
        self.first_read_completed = threading.Event()
        self.allow_more_reads = threading.Event()
        self.allow_more_reads.set()

    def alloc_buffer(self, size):
        assert size == self.buffer_size
        return memoryview(bytearray(size)), 71

    def free_buffer(self, buffer_ref):
        self.calls.append(("free_buffer", buffer_ref))

    def get_blob(self, timeout=5.0):
        del timeout
        with self._lock:
            handle = self._next_blob
            self._next_blob += 1
        self.calls.append(("get_blob", handle, threading.current_thread().name))
        return handle

    def release_blob(self, blob_handle):
        self.released.append(blob_handle)
        self.calls.append(
            ("release_blob", blob_handle, threading.current_thread().name)
        )

    def submit_read(self, *descriptor):
        return self._submit("read", descriptor)

    def submit_write(self, *descriptor):
        return self._submit("write", descriptor)

    def _submit(self, operation, descriptor):
        with self._lock:
            request_id = self._next_request
            self._next_request += 1
            completion = SimpleNamespace(
                request_id=request_id,
                status=int(operation in self.failed_operations),
                error_code=int(operation in self.failed_operations),
                completed_bytes=descriptor[-1],
            )
            self._pending.append((operation, completion))
            self.submissions.append((operation, descriptor, request_id))
            self.event_order.append(f"submit_{operation}")
            return request_id

    def poll_burst(self, max_count=64):
        if not self.allow_completions.is_set():
            return []
        with self._lock:
            completed = []
            remaining = deque()
            while self._pending and len(completed) < max_count:
                operation, completion = self._pending.pop()
                if (
                    operation == "read"
                    and self.first_read_completed.is_set()
                    and not self.allow_more_reads.is_set()
                ):
                    remaining.appendleft((operation, completion))
                    break
                if operation in self.blocked_operations:
                    remaining.appendleft((operation, completion))
                else:
                    completed.append(completion)
                    if operation == "read":
                        self.first_read_completed.set()
            self._pending.extend(remaining)
            return completed

    def wait_burst(self, max_count=64, timeout=0.00005):
        del timeout
        return self.poll_burst(max_count)


class FakeTransferEvent:
    def __init__(self, connector):
        self.connector = connector

    def query(self):
        self.connector.event_queries += 1
        return True

    def synchronize(self):
        self.connector.event_syncs += 1
        self.connector.allow_h2d_completion.wait(timeout=5)
        return None


class FakeGPUConnector:
    def __init__(self, client):
        self.client = client
        self.load_stream = object()
        self.calls = []
        self.events_recorded = 0
        self.event_queries = 0
        self.event_syncs = 0
        self.allow_h2d_completion = threading.Event()
        self.allow_h2d_completion.set()

    def to_gpu(self, _memory_obj, start, end, **_kwargs):
        self.calls.append((start, end, threading.current_thread().name))
        self.client.event_order.append("h2d")
        self.client.allow_more_reads.set()

    def record_transfer_event(self, _direction):
        self.events_recorded += 1
        return FakeTransferEvent(self)


@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCUDART())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())


def make_stack(*, layerwise=False):
    buffer_size = 256 * 1024
    config = LMCacheEngineConfig.from_defaults()
    config.local_cpu = False
    config.max_local_cpu_size = buffer_size / 1024**3
    config.xds_service_name = "xds0"
    config.cascadekv_backend = True
    config.use_layerwise = layerwise
    config.xds_max_size = 1
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
    client = FakeXDSClient(buffer_size)
    allocator = XDSServiceMemoryAllocator(client, buffer_size, 4096)
    loop = asyncio.new_event_loop()
    backend = CascadeKVBackend(config, metadata, loop, allocator, client, "cpu")
    return backend, client, loop


@pytest.fixture
def cascade_stack():
    backend, client, loop = make_stack()
    yield backend, client
    backend.close()
    loop.close()


def make_keys(count):
    return [
        CacheEngineKey("test-model", 1, 0, 1000 + index, torch.float16)
        for index in range(count)
    ]


def allocate_blocks(backend, count=1, shape=None):
    shape = shape or torch.Size([2, 2, 4, 256])
    objects = backend.batched_allocate(
        [shape], [torch.float16], count, MemoryFormat.KV_2LTD
    )
    assert objects is not None
    return objects


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not met")
        time.sleep(0.001)


def store(backend, keys, objects):
    backend.batched_submit_put_task(keys, objects)
    for memory_obj in objects:
        memory_obj.ref_count_down()


def test_layerwise_configuration_is_rejected():
    with pytest.raises(ValueError, match="use_layerwise: false"):
        make_stack(layerwise=True)


def test_write_returns_before_ssd_and_buffer_hit_is_readable(cascade_stack):
    backend, client = cascade_stack
    key = make_keys(1)[0]
    memory_obj = allocate_blocks(backend)[0]
    source_ptr = memory_obj.data_ptr
    client.allow_completions.clear()

    store(backend, [key], [memory_obj])
    wait_until(lambda: len(client.submissions) == 1)

    assert key in backend.buffer_table
    assert key not in backend.block_table
    hit = backend.batched_get_blocking([key])
    assert hit[0] is not None and hit[0].data_ptr == source_ptr
    hit[0].ref_count_down()

    client.allow_completions.set()
    wait_until(lambda: key in backend.block_table)
    assert key not in backend.buffer_table


def test_concurrent_buffer_hits_use_standard_memory_obj_lifecycle(cascade_stack):
    backend, client = cascade_stack
    key = make_keys(1)[0]
    client.allow_completions.clear()
    store(backend, [key], allocate_blocks(backend))
    wait_until(lambda: key in backend.buffer_table)
    try:
        barrier = threading.Barrier(60)
        errors = []

        def read_once():
            try:
                barrier.wait(timeout=5)
                result = backend.batched_get_blocking([key])
                assert result[0] is not None
                result[0].ref_count_down()
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=read_once) for _ in range(60)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not errors
        assert all(not thread.is_alive() for thread in threads)
        assert backend.metrics()["block_buffer_in_use"] == 1
        assert backend.remove(key)
        client.allow_completions.set()
        wait_until(lambda: backend.metrics()["block_buffer_in_use"] == 0)
    finally:
        client.allow_completions.set()


def test_write_and_read_each_submit_one_full_blob_io(cascade_stack):
    backend, client = cascade_stack
    key = make_keys(1)[0]
    shape = torch.Size([2, 2, 4, 128])
    store(backend, [key], allocate_blocks(backend, shape=shape))
    wait_until(lambda: key in backend.block_table)

    result = backend.batched_get_blocking([key])
    assert result[0] is not None and result[0].get_shape() == shape
    result[0].ref_count_down()

    writes = [item for item in client.submissions if item[0] == "write"]
    reads = [item for item in client.submissions if item[0] == "read"]
    assert len(writes) == len(reads) == 1
    assert writes[0][1][-1] == reads[0][1][-1] == backend.blob_size


def test_successful_write_commits_and_releases_source(cascade_stack):
    backend, client = cascade_stack
    key = make_keys(1)[0]
    store(backend, [key], allocate_blocks(backend))

    wait_until(lambda: key in backend.block_table)
    assert key not in backend.buffer_table
    assert backend.metrics()["buffer_blocks"] == 0
    assert all(call[2] == "CascadeKV-StorageIOEngine" for call in client.calls)


def test_remove_reinsert_rejects_stale_write_completion(cascade_stack):
    backend, client = cascade_stack
    key = make_keys(1)[0]
    client.allow_completions.clear()

    store(backend, [key], allocate_blocks(backend))
    wait_until(lambda: len(client.submissions) == 1)
    assert backend.remove(key)
    store(backend, [key], allocate_blocks(backend))
    wait_until(lambda: len(client.submissions) == 2)

    client.allow_completions.set()
    wait_until(lambda: key in backend.block_table)
    wait_until(lambda: len(client.released) == 1)
    assert backend.block_table[key].blob_handle == 101
    assert client.released == [100]


def test_write_failure_never_publishes_durable_metadata(cascade_stack):
    backend, client = cascade_stack
    key = make_keys(1)[0]
    client.failed_operations.add("write")
    store(backend, [key], allocate_blocks(backend))

    wait_until(lambda: client.released == [100])
    assert key not in backend.buffer_table
    assert key not in backend.block_table


def test_sixty_concurrent_reads_share_one_rank_engine(cascade_stack):
    backend, _client = cascade_stack
    key = make_keys(1)[0]
    store(backend, [key], allocate_blocks(backend))
    wait_until(lambda: key in backend.block_table)
    barrier = threading.Barrier(60)
    errors = []

    def read_once():
        try:
            barrier.wait(timeout=5)
            result = backend.batched_get_blocking([key])
            assert result[0] is not None
            result[0].ref_count_down()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=read_once) for _ in range(60)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert backend.metrics()["engine_peak_active_tasks"] > 1


def test_h2d_starts_from_completion_with_block_window(cascade_stack):
    backend, client = cascade_stack
    keys = make_keys(2)
    store(backend, keys, allocate_blocks(backend, count=2))
    wait_until(lambda: len(backend.block_table) == 2)
    client.first_read_completed.clear()
    client.allow_more_reads.clear()
    connector = FakeGPUConnector(client)

    result = backend.batched_get_to_gpu_blocking(keys, [0, 4], [4, 8], connector)

    assert result is not None and result.completed == (True, True)
    assert len(connector.calls) == 2
    assert connector.events_recorded == 2
    assert connector.event_queries == 0
    assert connector.event_syncs == 2
    chain = [event for event in client.event_order if event in {"submit_read", "h2d"}]
    assert chain == ["submit_read", "submit_read", "h2d", "h2d"]
    assert all(call[2] == "CascadeKV-H2DSubmitter" for call in connector.calls)


def test_direct_read_releases_each_four_block_window(cascade_stack):
    backend, client = cascade_stack
    keys = make_keys(8)
    store(backend, keys, allocate_blocks(backend, count=8))
    wait_until(lambda: len(backend.block_table) == 8)
    connector = FakeGPUConnector(client)
    connector.allow_h2d_completion.clear()
    result = []

    thread = threading.Thread(
        target=lambda: result.append(
            backend.batched_get_to_gpu_blocking(keys, [0] * 8, [4] * 8, connector)
        )
    )
    thread.start()
    wait_until(
        lambda: len([item for item in client.submissions if item[0] == "read"]) == 4
    )
    time.sleep(0.01)

    assert len([item for item in client.submissions if item[0] == "read"]) == 4
    assert backend.metrics()["block_buffer_in_use"] == 4

    connector.allow_h2d_completion.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result[0] is not None and all(result[0].completed)


def test_close_drains_pending_write_and_releases_hugepage(cascade_stack):
    backend, client = cascade_stack
    store(backend, make_keys(1), allocate_blocks(backend))
    backend.close()

    assert not backend.buffer_table
    assert not backend.block_table
    assert any(call[:2] == ("free_buffer", 71) for call in client.calls)


def make_ssd_block(handle):
    return SSDBlock(
        handle,
        torch.Size([1]),
        torch.uint8,
        MemoryFormat.KV_2LTD,
    )


def test_block_table_applies_coalesced_touches_in_exact_lru_order():
    table = BlockTable()
    keys = make_keys(3)
    for handle, key in enumerate(keys, start=100):
        assert table.publish(key, make_ssd_block(handle))

    for key in (keys[0], keys[1], keys[0]):
        block = table.acquire(key)
        assert block is not None
        assert block.release_reader() is None

    assert table.evict_oldest() == 102
    assert keys[2] not in table
    assert table.evict_oldest() == 101


def test_block_table_skips_pins_and_active_readers():
    table = BlockTable()
    keys = make_keys(3)
    for handle, key in enumerate(keys, start=200):
        assert table.publish(key, make_ssd_block(handle))

    assert table.pin(keys[0])
    reader = table.acquire(keys[1])
    assert reader is not None
    assert table.evict_oldest() == 202

    assert reader.release_reader() is None
    assert table.unpin(keys[0])
    assert table.evict_oldest() == 201
    assert table.evict_oldest() == 200


def test_block_table_ignores_stale_touch_after_same_key_is_republished():
    table = BlockTable()
    key, later_key = make_keys(2)
    old = make_ssd_block(300)
    assert table.publish(key, old)
    reader = table.acquire(key)
    assert reader is old
    assert table.remove(key) == (True, None)
    assert reader.release_reader() == 300

    assert table.publish(key, make_ssd_block(301))
    assert table.publish(later_key, make_ssd_block(302))
    assert table.evict_oldest() == 301


def test_block_table_repeated_eviction_does_constant_candidate_work():
    class CountingBlock(SSDBlock):
        probes = 0

        def try_take_for_reuse(self):
            CountingBlock.probes += 1
            return super().try_take_for_reuse()

    table = BlockTable()
    keys = make_keys(256)
    for handle, key in enumerate(keys, start=400):
        block = CountingBlock(
            handle,
            torch.Size([1]),
            torch.uint8,
            MemoryFormat.KV_2LTD,
        )
        assert table.publish(key, block)

    assert [table.evict_oldest() for _ in keys] == list(range(400, 656))
    assert CountingBlock.probes == len(keys)
