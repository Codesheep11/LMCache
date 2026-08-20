# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.xds_write_scheduler import (
    LeaseState,
    StagingLease,
    XDSAsyncWriteScheduler,
)


def _make_lease(name="item", size=4096):
    lease = StagingLease(
        keys=(name,),
        indexes=(0,),
        objects=(object(),),
        sizes=(size,),
        credited_bytes=size,
    )
    lease.transition(LeaseState.PENDING)
    return lease


def test_write_scheduler_applies_credit_backpressure():
    scheduler = XDSAsyncWriteScheduler(write_budget_bytes=4096)
    scheduler.acquire_credits(4096, timeout=0.1)
    started = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as executor:
        def wait_for_credit():
            started.set()
            return scheduler.acquire_credits(4096, timeout=1.0)

        waiter = executor.submit(wait_for_credit)
        assert started.wait(0.1)
        assert not waiter.done()
        scheduler.release_credits(4096)
        assert waiter.result() >= 0

    scheduler.release_credits(4096)
    scheduler.close()


def test_write_scheduler_commits_and_releases_each_lease_once():
    committed = []
    scheduler = XDSAsyncWriteScheduler(
        write_budget_bytes=4096,
        submit_write=lambda lease: lease.keys[0],
        commit=lambda lease, outcome: committed.append((lease.keys[0], outcome)),
    )
    scheduler.acquire_credits(4096, timeout=0.1)
    lease = _make_lease()
    scheduler.enqueue(lease)

    snapshot = scheduler.drain(timeout=1.0)

    assert lease.state is LeaseState.COMMITTED
    assert committed == [("item", "item")]
    assert snapshot.outstanding_bytes == 0
    assert snapshot.committed_leases == 1
    scheduler.close()


def test_write_scheduler_rolls_back_inflight_and_queued_once_on_failure():
    first_started = threading.Event()
    allow_failure = threading.Event()
    rolled_back = []

    def submit_write(lease):
        first_started.set()
        assert allow_failure.wait(1.0)
        raise RuntimeError(f"write failed for {lease.keys[0]}")

    scheduler = XDSAsyncWriteScheduler(
        write_budget_bytes=8192,
        submit_write=submit_write,
        rollback=lambda lease, _error: rolled_back.append(lease.keys[0]),
    )
    scheduler.acquire_credits(8192, timeout=0.1)
    scheduler.enqueue(_make_lease("first"))
    assert first_started.wait(0.2)
    scheduler.enqueue(_make_lease("queued"))
    allow_failure.set()

    with pytest.raises(RuntimeError, match="write failed for first"):
        scheduler.drain(timeout=1.0)

    assert rolled_back == ["first", "queued"]
    assert scheduler.snapshot().outstanding_bytes == 0
    with pytest.raises(RuntimeError, match="write failed for first"):
        scheduler.close()
