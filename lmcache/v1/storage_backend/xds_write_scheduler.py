# SPDX-License-Identifier: Apache-2.0
"""Credit-based asynchronous write scheduler for xDS staging slots."""

from __future__ import annotations

# Standard
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Hashable, Optional
import threading
import time


class LeaseState(Enum):
    GATHERING = auto()
    PENDING = auto()
    INFLIGHT = auto()
    COMMITTED = auto()
    FAILED = auto()
    CANCELLED = auto()


_ALLOWED_TRANSITIONS = {
    LeaseState.GATHERING: {LeaseState.PENDING, LeaseState.FAILED},
    LeaseState.PENDING: {
        LeaseState.INFLIGHT,
        LeaseState.FAILED,
        LeaseState.CANCELLED,
    },
    LeaseState.INFLIGHT: {
        LeaseState.COMMITTED,
        LeaseState.FAILED,
        LeaseState.CANCELLED,
    },
    LeaseState.COMMITTED: set(),
    LeaseState.FAILED: set(),
    LeaseState.CANCELLED: set(),
}


@dataclass
class StagingLease:
    keys: tuple[Hashable, ...]
    indexes: tuple[int, ...]
    objects: tuple[Any, ...]
    sizes: tuple[int, ...]
    credited_bytes: int
    reserved_bytes: Optional[int] = None
    state: LeaseState = LeaseState.GATHERING
    credits_released: bool = False

    def __post_init__(self) -> None:
        count = len(self.keys)
        if not (count == len(self.indexes) == len(self.objects) == len(self.sizes)):
            raise ValueError("lease fields must have matching lengths")
        if self.credited_bytes < 0 or any(size < 0 for size in self.sizes):
            raise ValueError("lease byte sizes must be non-negative")
        if sum(self.sizes) != self.credited_bytes:
            raise ValueError("credited_bytes must equal the sum of lease sizes")
        if self.reserved_bytes is None:
            self.reserved_bytes = self.credited_bytes
        if self.reserved_bytes < self.credited_bytes:
            raise ValueError("reserved_bytes must cover all credited lease bytes")

    def transition(self, new_state: LeaseState) -> None:
        if new_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(
                f"invalid lease transition {self.state.name} -> {new_state.name}"
            )
        self.state = new_state


@dataclass(frozen=True)
class SchedulerSnapshot:
    write_budget_bytes: int
    available_bytes: int
    outstanding_bytes: int
    queued_leases: int
    inflight_leases: int
    committed_leases: int
    failed_leases: int
    accepting: bool
    sticky_error: Optional[BaseException]


class XDSAsyncWriteScheduler:
    def __init__(
        self,
        write_budget_bytes: int,
        submit_write: Optional[Callable[[StagingLease], Any]] = None,
        commit: Optional[Callable[[StagingLease, Any], None]] = None,
        rollback: Optional[Callable[[StagingLease, BaseException], None]] = None,
    ) -> None:
        if write_budget_bytes <= 0:
            raise ValueError("write_budget_bytes must be positive")
        self._write_budget_bytes = write_budget_bytes
        self._available_bytes = write_budget_bytes
        self._queue: deque[StagingLease] = deque()
        self._inflight: Optional[StagingLease] = None
        self._committed_leases = 0
        self._failed_leases = 0
        self._accepting = True
        self._stopping = False
        self._sticky_error: Optional[BaseException] = None
        self._condition = threading.Condition()
        self._submit_write = submit_write
        self._commit = commit or (lambda _lease, _outcome: None)
        self._rollback = rollback or (lambda _lease, _error: None)
        self._dispatcher: Optional[threading.Thread] = None
        if submit_write is not None:
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name="xds-async-write",
                daemon=True,
            )
            self._dispatcher.start()

    def _raise_if_unhealthy_locked(self) -> None:
        if self._sticky_error is not None:
            raise self._sticky_error

    def acquire_credits(self, size: int, timeout: Optional[float]) -> float:
        if size < 0:
            raise ValueError("credit size must be non-negative")
        if size > self._write_budget_bytes:
            raise ValueError(
                f"requested {size} bytes exceeds write budget "
                f"{self._write_budget_bytes}"
            )
        started = time.monotonic()
        deadline = None if timeout is None else started + timeout
        with self._condition:
            while self._available_bytes < size:
                self._raise_if_unhealthy_locked()
                if not self._accepting:
                    raise RuntimeError("xDS async write scheduler is closed")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for xDS write credits")
                self._condition.wait(remaining)
            self._raise_if_unhealthy_locked()
            if not self._accepting:
                raise RuntimeError("xDS async write scheduler is closed")
            self._available_bytes -= size
        return time.monotonic() - started

    def release_credits(self, size: int) -> None:
        if size < 0:
            raise ValueError("credit size must be non-negative")
        with self._condition:
            if self._available_bytes + size > self._write_budget_bytes:
                raise RuntimeError("released more xDS write credits than acquired")
            self._available_bytes += size
            self._condition.notify_all()

    def _release_lease_credits_locked(self, lease: StagingLease) -> None:
        if lease.credits_released:
            raise RuntimeError("xDS write lease credits were already released")
        assert lease.reserved_bytes is not None
        self._available_bytes += lease.reserved_bytes
        if self._available_bytes > self._write_budget_bytes:
            raise RuntimeError("released more xDS write credits than acquired")
        lease.credits_released = True

    def enqueue(self, lease: StagingLease) -> None:
        with self._condition:
            self._raise_if_unhealthy_locked()
            if not self._accepting:
                raise RuntimeError("xDS async write scheduler is closed")
            if self._dispatcher is None:
                raise RuntimeError("xDS write scheduler has no submit callback")
            if lease.state is not LeaseState.PENDING:
                raise RuntimeError("only a PENDING lease can be enqueued")
            assert lease.reserved_bytes is not None
            outstanding = self._write_budget_bytes - self._available_bytes
            if lease.reserved_bytes > outstanding:
                raise RuntimeError("lease does not own enough acquired credits")
            self._queue.append(lease)
            self._condition.notify_all()

    def _fail_lease(self, lease: StagingLease, error: BaseException) -> None:
        self._rollback(lease, error)
        with self._condition:
            lease.transition(LeaseState.FAILED)
            self._failed_leases += 1
            self._release_lease_credits_locked(lease)
            self._condition.notify_all()

    def _dispatch_loop(self) -> None:
        assert self._submit_write is not None
        while True:
            with self._condition:
                while not self._stopping and not self._queue:
                    self._condition.wait()
                if self._stopping:
                    return
                lease = self._queue.popleft()
                lease.transition(LeaseState.INFLIGHT)
                self._inflight = lease

            try:
                outcome = self._submit_write(lease)
                self._commit(lease, outcome)
            except BaseException as error:
                self._fail_lease(lease, error)
                with self._condition:
                    self._inflight = None
                    self._sticky_error = error
                    self._accepting = False
                    queued = list(self._queue)
                    self._queue.clear()
                    self._condition.notify_all()
                for queued_lease in queued:
                    self._fail_lease(queued_lease, error)
                continue

            with self._condition:
                lease.transition(LeaseState.COMMITTED)
                self._committed_leases += 1
                self._release_lease_credits_locked(lease)
                self._inflight = None
                self._condition.notify_all()

    def snapshot(self) -> SchedulerSnapshot:
        with self._condition:
            return SchedulerSnapshot(
                write_budget_bytes=self._write_budget_bytes,
                available_bytes=self._available_bytes,
                outstanding_bytes=self._write_budget_bytes - self._available_bytes,
                queued_leases=len(self._queue),
                inflight_leases=int(self._inflight is not None),
                committed_leases=self._committed_leases,
                failed_leases=self._failed_leases,
                accepting=self._accepting,
                sticky_error=self._sticky_error,
            )

    def check_healthy(self) -> None:
        with self._condition:
            self._raise_if_unhealthy_locked()

    def drain(self, timeout: Optional[float] = None) -> SchedulerSnapshot:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while (
                self._queue
                or self._inflight is not None
                or self._available_bytes != self._write_budget_bytes
            ):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out draining xDS async writes")
                self._condition.wait(remaining)
            self._raise_if_unhealthy_locked()
        return self.snapshot()

    def close(self, timeout: Optional[float] = None) -> SchedulerSnapshot:
        with self._condition:
            self._accepting = False
            self._condition.notify_all()
        error: Optional[BaseException] = None
        snapshot: Optional[SchedulerSnapshot] = None
        try:
            snapshot = self.drain(timeout)
        except BaseException as exc:
            error = exc
        finally:
            with self._condition:
                self._stopping = True
                self._condition.notify_all()
            if self._dispatcher is not None:
                self._dispatcher.join(timeout)
                if self._dispatcher.is_alive() and error is None:
                    error = TimeoutError("timed out stopping xDS write dispatcher")
        if error is not None:
            raise error
        assert snapshot is not None
        return snapshot
