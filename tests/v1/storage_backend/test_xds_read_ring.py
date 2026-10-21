# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from threading import Thread
import queue

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.xds_read_ring import XDSReadRingPump


@dataclass(frozen=True)
class FakeCompletion:
    request_id: int


class FakeClient:
    def __init__(self, completion_order=None):
        self.next_request_id = 1
        self.pending = []
        self.completion_order = queue.Queue()
        for request_id in completion_order or []:
            self.completion_order.put(request_id)
        self.peak_inflight = 0

    def submit(self, _item):
        request_id = self.next_request_id
        self.next_request_id += 1
        self.pending.append(request_id)
        self.peak_inflight = max(self.peak_inflight, len(self.pending))
        return request_id

    def wait_any_result(self, _timeout):
        try:
            request_id = self.completion_order.get_nowait()
        except queue.Empty:
            request_id = self.pending[0]
        if request_id in self.pending:
            self.pending.remove(request_id)
        return FakeCompletion(request_id)


class FailingWaitClient(FakeClient):
    def wait_any_result(self, _timeout):
        raise TimeoutError("xDS completion timed out")


def _run_pump(pump, items):
    for item in items:
        pump.put(item)
    pump.close_input()
    worker = Thread(target=pump.run)
    worker.start()
    return worker


def test_read_ring_bounds_inflight_requests_and_reuses_slots():
    client = FakeClient()
    pump = XDSReadRingPump(2, client.submit, client.wait_any_result, 1.0)
    worker = _run_pump(pump, range(8))

    results = [pump.get_completion() for _ in range(8)]
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert client.peak_inflight == 2
    assert [result.item for result in results] == list(range(8))
    assert all(result.error is None for result in results)


def test_read_ring_maps_out_of_order_completions_to_original_items():
    client = FakeClient(completion_order=[2, 1])
    pump = XDSReadRingPump(2, client.submit, client.wait_any_result, 1.0)
    worker = _run_pump(pump, ["first", "second"])

    results = [pump.get_completion() for _ in range(2)]
    worker.join(timeout=1.0)

    assert [result.item for result in results] == ["second", "first"]


def test_read_ring_reports_unknown_completion_after_draining_inflight():
    client = FakeClient(completion_order=[999, 2, 1])
    pump = XDSReadRingPump(2, client.submit, client.wait_any_result, 1.0)
    worker = _run_pump(pump, ["first", "second", "not-submitted"])

    results = [pump.get_completion() for _ in range(3)]
    worker.join(timeout=1.0)

    failures = [result for result in results if result.error is not None]
    assert client.next_request_id == 3
    assert len(failures) == 1
    assert "unknown completion request 999" in str(failures[0].error)


def test_read_ring_wait_failure_marks_submitted_items_unsafe_for_reuse():
    client = FailingWaitClient()
    pump = XDSReadRingPump(2, client.submit, client.wait_any_result, 1.0)
    worker = _run_pump(pump, ["first", "second", "not-submitted"])

    worker.join(timeout=1.0)

    assert not worker.is_alive()
    results = [pump.get_completion() for _ in range(3)]
    assert {result.item for result in results} == {
        "first",
        "second",
        "not-submitted",
    }
    assert all(isinstance(result.error, TimeoutError) for result in results)
    safety = {result.item: result.safe_to_reuse for result in results}
    assert safety == {
        "first": False,
        "second": False,
        "not-submitted": True,
    }


def test_batch_submission_and_completion_preserve_depth_and_order():
    client = FakeClient()
    batches = []

    def submit_batch(items):
        batches.append(list(items))
        return [client.submit(item) for item in items]

    def wait_burst(limit, timeout):
        return [
            client.wait_any_result(timeout)
            for _ in range(min(limit, len(client.pending)))
        ]

    pump = XDSReadRingPump(
        3,
        client.submit,
        client.wait_any_result,
        1.0,
        submit_batch=submit_batch,
        wait_burst=wait_burst,
    )
    worker = _run_pump(pump, range(8))
    worker.join(timeout=1)
    assert not worker.is_alive()
    results = [pump.get_completion() for _ in range(8)]
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7]]
    assert client.peak_inflight == 3
    assert [r.item for r in results] == list(range(8))
    assert all(r.error is None for r in results)


def test_batch_partial_submit_failure_drains_accepted_ids():
    client = FakeClient()

    def submit_batch(items):
        return [client.submit(items[0]), OSError("submit failed"), OSError("cancelled")]

    pump = XDSReadRingPump(
        3, client.submit, client.wait_any_result, 1.0, submit_batch=submit_batch
    )
    worker = _run_pump(pump, ["accepted", "failed", "cancelled", "queued"])
    worker.join(timeout=1)
    assert not worker.is_alive()
    results = [pump.get_completion() for _ in range(4)]
    assert client.pending == []
    assert results[0].item == "accepted" and results[0].error is None
    assert all(r.safe_to_reuse for r in results)
    assert all(r.error is not None for r in results[1:])


def test_batch_callback_exception_retains_unknown_dma_targets():
    client = FakeClient()

    def submit_batch(items):
        client.submit(items[0])
        raise MemoryError("result allocation failed")

    pump = XDSReadRingPump(
        2, client.submit, client.wait_any_result, 1.0, submit_batch=submit_batch
    )
    worker = _run_pump(pump, ["maybe-live", "unknown", "queued"])
    worker.join(timeout=1)
    assert not worker.is_alive()
    results = [pump.get_completion() for _ in range(3)]
    assert {r.item: r.safe_to_reuse for r in results} == {
        "maybe-live": False,
        "unknown": False,
        "queued": True,
    }


def test_batch_timeout_retains_all_pending_targets():
    client = FakeClient()
    pump = XDSReadRingPump(
        2,
        client.submit,
        client.wait_any_result,
        1.0,
        submit_batch=lambda items: [client.submit(x) for x in items],
        wait_burst=lambda limit, timeout: [],
    )
    worker = _run_pump(pump, ["a", "b", "queued"])
    worker.join(timeout=1)
    assert not worker.is_alive()
    results = [pump.get_completion() for _ in range(3)]
    assert {r.item: r.safe_to_reuse for r in results} == {
        "a": False,
        "b": False,
        "queued": True,
    }
    assert all(isinstance(r.error, TimeoutError) for r in results)


def test_batch_waits_for_group_before_refilling_even_with_single_completions():
    client = FakeClient()
    batches = []

    def submit_batch(items):
        assert client.pending == []
        batches.append(list(items))
        return [client.submit(item) for item in items]

    pump = XDSReadRingPump(
        3,
        client.submit,
        client.wait_any_result,
        1.0,
        submit_batch=submit_batch,
        wait_burst=lambda limit, timeout: [client.wait_any_result(timeout)],
    )
    worker = _run_pump(pump, range(8))
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7]]
    assert [pump.get_completion().item for _ in range(8)] == list(range(8))


def test_batch_respects_sdk_maximum_with_large_queue_depth():
    client = FakeClient()
    sizes = []

    def submit_batch(items):
        sizes.append(len(items))
        assert len(items) <= 64
        return [client.submit(item) for item in items]

    def wait_burst(limit, timeout):
        assert limit <= 64
        return [client.wait_any_result(timeout) for _ in range(limit)]

    pump = XDSReadRingPump(
        128,
        client.submit,
        client.wait_any_result,
        1.0,
        submit_batch=submit_batch,
        wait_burst=wait_burst,
    )
    worker = _run_pump(pump, range(130))
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert sizes == [64, 64, 2]
    assert [pump.get_completion().item for _ in range(130)] == list(range(130))


@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("batched", [False, True])
def test_synchronous_items_preserve_depth_and_completion_mapping(
    depth: int, batched: bool
) -> None:
    client = FakeClient(completion_order=[2, 1] if depth > 1 else None)
    pump = XDSReadRingPump(
        depth,
        client.submit,
        client.wait_any_result,
        1.0,
        submit_batch=(lambda items: [client.submit(x) for x in items])
        if batched
        else None,
    )
    results = pump.run_items(list(range(9)))
    assert sorted(r.item for r in results) == list(range(9))
    assert all(r.error is None and r.safe_to_reuse for r in results)
    assert client.peak_inflight == depth and not client.pending
    assert pump.error is None
    with pytest.raises(RuntimeError, match="closed"):
        pump.put("late")


@pytest.mark.parametrize("batched", [False, True])
def test_synchronous_timeout_retains_only_submitted_targets(batched: bool) -> None:
    client = FailingWaitClient()
    pump = XDSReadRingPump(
        2,
        client.submit,
        client.wait_any_result,
        1.0,
        submit_batch=(lambda items: [client.submit(x) for x in items])
        if batched
        else None,
    )
    results = pump.run_items(["a", "b", "queued"])
    assert {r.item: r.safe_to_reuse for r in results} == {
        "a": False,
        "b": False,
        "queued": True,
    }
    assert isinstance(pump.error, TimeoutError)
    assert not pump.is_safe_to_reuse("a") and not pump.is_safe_to_reuse("b")
    assert pump.unsafe_request_id("a") == 1 and pump.unsafe_request_id("b") == 2


def test_synchronous_submit_error_drains_accepted_requests() -> None:
    client = FakeClient()

    def submit(item: str) -> int:
        if item == "bad":
            raise OSError("submit failed")
        return client.submit(item)

    pump = XDSReadRingPump(2, submit, client.wait_any_result, 1.0)
    results = pump.run_items(["accepted", "bad", "queued"])
    assert not client.pending and isinstance(pump.error, OSError)
    assert results[0].item == "accepted" and results[0].error is None
    assert all(r.safe_to_reuse for r in results)
    assert {r.item for r in results[1:]} == {"bad", "queued"}


def test_synchronous_unknown_completion_preserves_terminal_error() -> None:
    client = FakeClient(completion_order=[999, 2, 1])
    pump = XDSReadRingPump(2, client.submit, client.wait_any_result, 1.0)
    results = pump.run_items(["a", "b"])
    assert not client.pending and "unknown completion" in str(pump.error)
    assert {r.item for r in results if r.item is not None} == {"a", "b"}
    assert any(r.item is None and r.error is pump.error for r in results)


def test_synchronous_batch_exception_retains_unknown_dma_targets() -> None:
    client = FakeClient()

    def submit_batch(items: list[str]) -> list[int]:
        client.submit(items[0])
        raise MemoryError("unknown submitted IDs")

    pump = XDSReadRingPump(
        2, client.submit, client.wait_any_result, 1.0, submit_batch=submit_batch
    )
    results = pump.run_items(["maybe-live", "unknown", "queued"])
    assert {r.item: r.safe_to_reuse for r in results} == {
        "maybe-live": False,
        "unknown": False,
        "queued": True,
    }
    assert isinstance(pump.error, MemoryError)


def test_synchronous_input_rejects_mixed_modes_without_losing_queued_item() -> None:
    client = FakeClient()
    pump = XDSReadRingPump(1, client.submit, client.wait_any_result, 1.0)
    pump.put("original")
    with pytest.raises(RuntimeError, match="fresh"):
        pump.run_items(["new"])
    pump.close_input()
    pump.run()
    assert pump.get_completion().item == "original"


def test_synchronous_empty_input_completes_without_submissions() -> None:
    client = FakeClient()
    pump = XDSReadRingPump(1, client.submit, client.wait_any_result, 1.0)
    assert pump.run_items([]) == [] and pump.error is None
    assert not client.pending
