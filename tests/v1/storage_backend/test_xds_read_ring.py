# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from threading import Thread
import queue

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
