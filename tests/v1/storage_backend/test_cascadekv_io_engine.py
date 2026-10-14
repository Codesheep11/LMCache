# SPDX-License-Identifier: Apache-2.0
"""Concurrency and timeout tests for the rank-local CascadeKV I/O engine."""

# Standard
from collections import deque
from types import SimpleNamespace
import threading
import time

# First Party
from lmcache.v1.memory_management import XDSIORegion
from lmcache.v1.storage_backend.cascadekv_backend import (
    BlockTable,
    IODescriptor,
    StorageIOEngine,
)


class SlowProgressClient:
    """Complete one request per poll after a controllable delay."""

    def __init__(self, poll_delay: float) -> None:
        self.metadata = SimpleNamespace(
            capacity_bytes=8192,
            blob_size_bytes=8192,
            worker_id=7,
        )
        self.poll_delay = poll_delay
        self.enabled = threading.Event()
        self.enabled.set()
        self.pending: deque[tuple[int, int]] = deque()
        self.next_request = 1
        self.wait_burst_calls = 0

    def submit_read(self, *_args) -> int:
        request_id = self.next_request
        self.next_request += 1
        self.pending.append((request_id, int(_args[-1])))
        return request_id

    def poll_burst(self, _max_count: int):
        time.sleep(self.poll_delay)
        if not self.enabled.is_set() or not self.pending:
            return []
        request_id, size = self.pending.popleft()
        completion_ns = time.monotonic_ns()
        return [
            SimpleNamespace(
                request_id=request_id,
                status=0,
                error_code=0,
                completed_bytes=size,
                profile_completion_publish_ns=completion_ns,
                profile_completion_dequeue_ns=completion_ns,
            )
        ]

    def wait_burst(self, max_count: int, _timeout: float):
        self.wait_burst_calls += 1
        return self.poll_burst(max_count)


class RingLimitedClient(SlowProgressClient):
    """Expose the native request ring's non-blocking ENOSPC behavior."""

    def __init__(self, capacity: int) -> None:
        super().__init__(poll_delay=0.0001)
        self.capacity = capacity
        self.full_errors = 0

    def submit_read(self, *_args) -> int:
        if len(self.pending) >= self.capacity:
            self.full_errors += 1
            raise OSError("SPDK C-API call failed with error code: -28")
        return super().submit_read(*_args)


class PausedBurstClient(SlowProgressClient):
    """Return a full window in one burst, then pause the next window."""

    def __init__(self) -> None:
        super().__init__(poll_delay=0.0001)
        self.first_batch_done = threading.Event()

    def poll_burst(self, max_count: int):
        time.sleep(self.poll_delay)
        if not self.enabled.is_set() or not self.pending:
            return []
        pending = [
            self.pending.popleft() for _ in range(min(max_count, len(self.pending)))
        ]
        completion_ns = time.monotonic_ns()
        completions = [
            SimpleNamespace(
                request_id=request_id,
                status=0,
                error_code=0,
                completed_bytes=size,
                profile_completion_publish_ns=completion_ns,
                profile_completion_dequeue_ns=completion_ns,
            )
            for request_id, size in pending
        ]
        if not self.first_batch_done.is_set():
            self.enabled.clear()
            self.first_batch_done.set()
        return completions


class WriteBurstClient(SlowProgressClient):
    """Hold the first allocation until a complete write burst is queued."""

    def __init__(self) -> None:
        super().__init__(poll_delay=0.0001)
        self.metadata.capacity_bytes = 256 * self.metadata.blob_size_bytes
        self.all_submitted = threading.Event()
        self.first_poll = threading.Event()
        self.get_blob_calls = 0
        self.get_blobs_at_first_poll = 0

    def get_blob(self, timeout=5.0):
        assert self.all_submitted.wait(timeout=timeout)
        self.get_blob_calls += 1
        return self.get_blob_calls

    def submit_write(self, *_args):
        return self.submit_read(*_args)

    def wait_burst(self, max_count: int, timeout: float):
        if not self.first_poll.is_set():
            self.get_blobs_at_first_poll = self.get_blob_calls
            self.first_poll.set()
        return super().wait_burst(max_count, timeout)


def descriptors(count: int) -> list[IODescriptor]:
    region = XDSIORegion(buffer_ref=1, offset=0, size=4096)
    return [IODescriptor(1, region, index * 4096, 4096) for index in range(count)]


def test_timeout_measures_inactivity_not_total_batch_duration():
    client = SlowProgressClient(poll_delay=0.01)
    engine = StorageIOEngine(
        client,
        BlockTable(),
        io_timeout_secs=0.03,
    )
    results = []
    started_at = time.monotonic()

    handle = engine.submit_read(descriptors(8), results.append)
    assert handle is not None
    handle.wait(timeout=2)

    assert time.monotonic() - started_at > engine.io_timeout_secs
    assert results == [None]
    engine.close()


def test_inactivity_timeout_drains_submitted_request_before_failure():
    client = SlowProgressClient(poll_delay=0.001)
    client.enabled.clear()
    engine = StorageIOEngine(
        client,
        BlockTable(),
        io_timeout_secs=0.02,
    )
    results = []

    handle = engine.submit_read(descriptors(2), results.append)
    assert handle is not None
    time.sleep(0.04)
    assert not handle.done
    client.enabled.set()
    deadline = time.monotonic() + 2
    while not handle.done and time.monotonic() < deadline:
        time.sleep(0.001)

    assert handle.done
    assert isinstance(handle.error, TimeoutError)
    assert isinstance(results[0], TimeoutError)
    assert client.pending == deque()
    engine.close()


def test_multiple_tasks_each_fill_four_block_window():
    client = SlowProgressClient(poll_delay=0.0001)
    client.enabled.clear()
    engine = StorageIOEngine(client, BlockTable())

    first = engine.submit_read(descriptors(8), lambda _error: None)
    second = engine.submit_read(descriptors(8), lambda _error: None)
    deadline = time.monotonic() + 2
    while len(client.pending) < 8 and time.monotonic() < deadline:
        time.sleep(0.001)

    assert len(client.pending) == 8
    assert not first.done
    assert not second.done
    assert engine.peak_active_tasks == 2
    assert engine.peak_inflight == 8

    client.enabled.set()
    first.wait(timeout=2)
    second.wait(timeout=2)
    engine.close()


def test_read_window_is_configurable():
    client = SlowProgressClient(poll_delay=0.0001)
    client.enabled.clear()
    engine = StorageIOEngine(client, BlockTable(), read_block_window=6)

    handle = engine.submit_read(descriptors(8), lambda _error: None)
    deadline = time.monotonic() + 2
    while len(client.pending) < 6 and time.monotonic() < deadline:
        time.sleep(0.001)

    assert len(client.pending) == 6
    assert not handle.done

    client.enabled.set()
    handle.wait(timeout=2)
    engine.close()


def test_burst_completion_does_not_finish_task_with_descriptors_remaining():
    client = PausedBurstClient()
    engine = StorageIOEngine(client, BlockTable())

    handle = engine.submit_read(descriptors(8), lambda _error: None)
    assert client.first_batch_done.wait(timeout=2)
    deadline = time.monotonic() + 2
    while len(client.pending) < 4 and time.monotonic() < deadline:
        time.sleep(0.001)

    assert len(client.pending) == 4
    assert not handle.done

    client.enabled.set()
    handle.wait(timeout=2)
    engine.close()


def test_ring_full_is_backpressure_not_task_failure():
    client = RingLimitedClient(capacity=2)
    engine = StorageIOEngine(client, BlockTable())
    results = []

    handles = [engine.submit_read(descriptors(2), results.append) for _ in range(3)]
    for handle in handles:
        handle.wait(timeout=2)

    assert results == [None, None, None]
    assert client.full_errors > 0
    assert engine.peak_inflight == 2
    engine.close()


def test_profiling_observes_scheduler_without_limiting_submission():
    client = SlowProgressClient(poll_delay=0.0001)
    engine = StorageIOEngine(client, BlockTable(), profile=True)

    def callback(_descriptor: IODescriptor) -> None:
        time.sleep(0.0001)

    handles = [
        engine.submit_read(descriptors(8), lambda _error: None, callback)
        for _ in range(2)
    ]
    for handle in handles:
        handle.wait(timeout=2)

    metrics = engine.metrics()
    assert metrics["engine_peak_qd"] == 8
    assert metrics["engine_read_descriptor_callback_count"] == 16
    assert metrics["engine_read_completion_to_refill_count"] == 8
    assert metrics["engine_read_avg_qd"] > 0
    assert client.wait_burst_calls > 0
    engine.close()


def test_write_activation_polls_completions_between_bounded_bursts():
    client = WriteBurstClient()
    engine = StorageIOEngine(client, BlockTable())
    region = XDSIORegion(buffer_ref=1, offset=0, size=4096)
    results = []

    for _ in range(128):
        engine.submit_write(
            region,
            lambda _handle, error: results.append(error) is None,
        )
    client.all_submitted.set()

    assert client.first_poll.wait(timeout=2)
    assert client.get_blobs_at_first_poll <= 64
    deadline = time.monotonic() + 2
    while len(results) < 128 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert results == [None] * 128
    engine.close()
