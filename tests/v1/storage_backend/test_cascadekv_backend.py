# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the legacy-compatible CascadeKV pipeline."""

# Standard
from collections import deque
from types import SimpleNamespace
import asyncio
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import XDSServiceMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.cascadekv_backend import CascadeKVBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


class FakeCUDART:
    def cudaHostRegister(self, _ptr, _size, _flags):
        return 0

    def cudaHostUnregister(self, _ptr):
        return 0

    def cudaGetErrorString(self, status):
        return f"fake CUDA error {status}"


class FakeXDSClient:
    def __init__(self, buffer_size: int):
        self.metadata = SimpleNamespace(
            io_unit_size=4096,
            blob_size_bytes=4096,
        )
        self.buffer_size = buffer_size
        self.calls = []
        self.submit_descriptors = []
        self.released_blob_handles = []
        self.closed = False
        self.allow_completions = threading.Event()
        self.allow_completions.set()
        self._pending = deque()
        self._lock = threading.Lock()
        self._next_blob_handle = 100
        self._next_request_id = 1

    def alloc_buffer(self, size):
        assert size == self.buffer_size
        return memoryview(bytearray(size)), 71

    def free_buffer(self, buffer_ref):
        self.calls.append(("free_buffer", buffer_ref))

    def get_blob(self, timeout=5.0):
        del timeout
        with self._lock:
            blob_handle = self._next_blob_handle
            self._next_blob_handle += 1
        self.calls.append(("get_blob", blob_handle))
        return blob_handle

    def release_blob(self, blob_handle):
        self.released_blob_handles.append(blob_handle)

    def submit_read(self, *descriptor):
        return self._submit("read", descriptor)

    def submit_write(self, *descriptor):
        return self._submit("write", descriptor)

    def _submit(self, operation, descriptor):
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            size = descriptor[-1]
            completion = SimpleNamespace(
                request_id=request_id,
                status=0,
                error_code=0,
                completed_bytes=size,
                result=0,
            )
            self._pending.append(completion)
            self.submit_descriptors.append((operation, descriptor, request_id))
        return request_id

    def poll_burst(self, max_count=64):
        if not self.allow_completions.is_set():
            return []
        with self._lock:
            count = min(max_count, len(self._pending))
            completions = [self._pending.pop() for _ in range(count)]
        return completions

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_cuda_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCUDART())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


@pytest.fixture
def cascade_stack():
    buffer_size = 256 * 1024
    config = LMCacheEngineConfig.from_defaults()
    config.local_cpu = False
    config.max_local_cpu_size = buffer_size / 1024**3
    config.xds_service_name = "xds0"
    config.cascadekv_backend = True
    config.xds_max_size = 1
    config.xds_write_window_size = 4
    metadata = LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(1, 2, 4, 1, 8),
        chunk_size=4,
    )
    client = FakeXDSClient(buffer_size)
    allocator = XDSServiceMemoryAllocator(client, buffer_size, 4096)
    local_cpu_backend = LocalCPUBackend(
        config,
        metadata,
        "cpu",
        None,
        memory_allocator=allocator,
    )
    loop = asyncio.new_event_loop()
    backend = CascadeKVBackend(
        config,
        metadata,
        loop,
        local_cpu_backend,
        client,
        "cpu",
    )
    yield backend, local_cpu_backend, client
    backend.close()
    local_cpu_backend.close()
    loop.close()


def make_keys(count):
    return [
        CacheEngineKey("test-model", 1, 0, 1000 + index, torch.float16)
        for index in range(count)
    ]


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not met")
        time.sleep(0.001)


def allocate_chunks(backend, count):
    chunks = [backend.allocate(torch.Size([4096]), torch.uint8) for _ in range(count)]
    assert all(chunk is not None for chunk in chunks)
    return chunks


def test_cascadekv_aggregates_write_window_before_waiting(cascade_stack):
    backend, _local_cpu_backend, client = cascade_stack
    keys = make_keys(4)
    chunks = allocate_chunks(backend, 4)
    client.allow_completions.clear()

    backend.batched_submit_put_task(keys, chunks)

    wait_until(
        lambda: len(
            [item for item in client.submit_descriptors if item[0] == "write"]
        )
        == 4
    )
    assert not backend.dict

    client.allow_completions.set()
    wait_until(lambda: len(backend.dict) == 4)
    for chunk in chunks:
        chunk.ref_count_down()


def test_cascadekv_batch_read_submits_all_requests_and_routes_out_of_order(
    cascade_stack,
):
    backend, _local_cpu_backend, client = cascade_stack
    keys = make_keys(4)
    chunks = allocate_chunks(backend, 4)
    backend.batched_submit_put_task(keys, chunks)
    wait_until(lambda: len(backend.dict) == 4)
    for chunk in chunks:
        chunk.ref_count_down()
    client.submit_descriptors.clear()

    results = backend.batched_get_blocking(keys)

    read_descriptors = [
        item for item in client.submit_descriptors if item[0] == "read"
    ]
    assert len(read_descriptors) == 4
    assert all(result is not None for result in results)
    assert len({descriptor[1][2] for descriptor in read_descriptors}) == 4
    for result in results:
        assert result is not None
        result.ref_count_down()


def test_cascadekv_pipeline_is_not_bounded_by_deprecated_slot_config(
    cascade_stack,
):
    backend, _local_cpu_backend, client = cascade_stack
    backend.config.cascadekv_slot_count = 1
    backend.config.cascadekv_read_slot_count = 1
    backend.config.cascadekv_write_slot_count = 1
    backend.config.cascadekv_io_queue_depth = 1
    backend.write_window_size = 12
    keys = make_keys(12)
    chunks = allocate_chunks(backend, 12)

    backend.batched_submit_put_task(keys, chunks)
    wait_until(lambda: len(backend.dict) == 12)
    results = backend.batched_get_blocking(keys)

    assert len([item for item in client.submit_descriptors if item[0] == "write"]) == 12
    assert len([item for item in client.submit_descriptors if item[0] == "read"]) == 12
    assert all(result is not None for result in results)
    for chunk in chunks:
        chunk.ref_count_down()
    for result in results:
        assert result is not None
        result.ref_count_down()


def test_cascadekv_callback_failure_does_not_roll_back_durable_write(
    cascade_stack,
):
    backend, _local_cpu_backend, client = cascade_stack
    key = make_keys(1)[0]
    chunk = allocate_chunks(backend, 1)[0]

    def fail_callback(_key):
        raise RuntimeError("fake callback failure")

    backend.batched_submit_put_task(
        [key],
        [chunk],
        on_complete_callback=fail_callback,
    )
    wait_until(lambda: key in backend.dict)
    wait_until(lambda: key not in backend.put_tasks)

    assert backend.contains(key)
    assert client.released_blob_handles == []
    chunk.ref_count_down()
