# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time

# First Party
from lmcache.v1.storage_backend.cascadekv_completion import (
    CascadeKVCompletionPump,
)


@dataclass(frozen=True)
class FakeCompletion:
    request_id: int
    status: int = 0
    error_code: int = 0
    completed_bytes: int = 4096
    result: int = 0


class FakePollClient:
    def __init__(self) -> None:
        self._items: deque[FakeCompletion] = deque()
        self._lock = threading.Lock()

    def publish(self, *items: FakeCompletion) -> None:
        with self._lock:
            self._items.extend(items)

    def poll_burst(self, max_count: int = 64) -> list[FakeCompletion]:
        with self._lock:
            count = min(max_count, len(self._items))
            return [self._items.popleft() for _ in range(count)]


def test_completion_pump_routes_out_of_order_ids_to_groups() -> None:
    client = FakePollClient()
    pump = CascadeKVCompletionPump(client)
    first = pump.create_group()
    second = pump.create_group()

    pump.submit(first, "read-a", lambda: 11)
    pump.submit(second, "write-b", lambda: 22)
    client.publish(FakeCompletion(22), FakeCompletion(11))

    assert second.wait_next(1.0).context == "write-b"
    assert first.wait_next(1.0).context == "read-a"
    pump.close()


def test_wait_all_returns_registration_order() -> None:
    client = FakePollClient()
    pump = CascadeKVCompletionPump(client)
    group = pump.create_group()

    pump.submit(group, "first", lambda: 31)
    pump.submit(group, "second", lambda: 32)
    client.publish(FakeCompletion(32), FakeCompletion(31))

    assert [item.context for item in group.wait_all(1.0)] == ["first", "second"]
    pump.close()


def test_wait_next_times_out_without_cancelling_request() -> None:
    client = FakePollClient()
    pump = CascadeKVCompletionPump(client)
    group = pump.create_group()
    pump.submit(group, "late", lambda: 41)

    started = time.monotonic()
    try:
        group.wait_next(0.01)
    except TimeoutError:
        pass
    else:
        raise AssertionError("wait_next did not time out")
    assert time.monotonic() - started < 0.5

    client.publish(FakeCompletion(41))
    assert group.wait_next(1.0).context == "late"
    pump.close()


def test_pending_count_tracks_unconsumed_completions() -> None:
    client = FakePollClient()
    pump = CascadeKVCompletionPump(client)
    group = pump.create_group()

    pump.submit(group, "first", lambda: 51)
    pump.submit(group, "second", lambda: 52)
    client.publish(FakeCompletion(51), FakeCompletion(52))

    assert len(group.wait_all(1.0)) == 2

    # Both requests completed before the caller began consuming them.  They
    # must still count as pending work for a wait_next()-driven overlap loop.
    assert group.pending_count == 2
    assert group.wait_next(1.0).context == "first"
    assert group.pending_count == 1
    assert group.wait_next(1.0).context == "second"
    assert group.pending_count == 0
    pump.close()


def test_completion_module_does_not_use_futures() -> None:
    source = Path(
        "/home/yangxc/workspace/LMCache-v043-xds/"
        "lmcache/v1/storage_backend/cascadekv_completion.py"
    ).read_text()

    assert "concurrent.futures" not in source
    assert "Future" not in source
