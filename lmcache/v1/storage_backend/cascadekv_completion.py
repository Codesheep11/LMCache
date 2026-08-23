# SPDX-License-Identifier: Apache-2.0
"""Request-ID completion routing for the CascadeKV xDS service path."""

# Standard
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar
import threading
import time


ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class CascadeKVCompletedRequest(Generic[ContextT]):
    """One completed xDS request paired with its caller-owned context."""

    request_id: int
    context: ContextT
    completion: Any


class CascadeKVRequestGroup(Generic[ContextT]):
    """Collect completion records for one logical CascadeKV operation."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._contexts: dict[int, ContextT] = {}
        self._registration_order: list[int] = []
        self._completion_order: deque[int] = deque()
        self._results: dict[int, CascadeKVCompletedRequest[ContextT]] = {}
        self._consumed_count = 0

    @property
    def pending_count(self) -> int:
        """Return the number of registered requests not yet consumed."""
        with self._condition:
            return len(self._registration_order) - self._consumed_count

    def add(self, request_id: int, context: ContextT) -> None:
        """Register a submitted request and its caller context."""
        with self._condition:
            if request_id in self._contexts:
                raise ValueError(f"duplicate xDS request id {request_id}")
            self._contexts[request_id] = context
            self._registration_order.append(request_id)

    def complete(self, request_id: int, completion: Any) -> None:
        """Publish one completion to waiters in device completion order."""
        with self._condition:
            if request_id not in self._contexts:
                raise KeyError(f"unknown xDS request id {request_id}")
            if request_id in self._results:
                raise ValueError(f"duplicate completion for xDS request {request_id}")
            result = CascadeKVCompletedRequest(
                request_id,
                self._contexts[request_id],
                completion,
            )
            self._results[request_id] = result
            self._completion_order.append(request_id)
            self._condition.notify_all()

    def wait_next(self, timeout: float) -> CascadeKVCompletedRequest[ContextT]:
        """Wait for and consume the next device-ordered completion."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._completion_order:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for xDS completion")
                self._condition.wait(remaining)
            request_id = self._completion_order.popleft()
            self._consumed_count += 1
            return self._results[request_id]

    def wait_all(
        self, timeout: float
    ) -> list[CascadeKVCompletedRequest[ContextT]]:
        """Wait for every request and return registration-ordered results."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self._results) != len(self._registration_order):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for xDS completion group")
                self._condition.wait(remaining)
            return [self._results[item] for item in self._registration_order]


class CascadeKVCompletionPump:
    """Own the single completion consumer for one rank-local xDS client."""

    def __init__(self, client: Any, poll_count: int = 64) -> None:
        self.client = client
        self.poll_count = poll_count
        self._client_lock = threading.RLock()
        self._owners_lock = threading.Lock()
        self._owners: dict[int, CascadeKVRequestGroup[Any]] = {}
        self._running = threading.Event()
        self._running.set()
        self._work_available = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="CascadeKV-CompletionPump",
            daemon=True,
        )
        self._thread.start()

    def create_group(self) -> CascadeKVRequestGroup[Any]:
        """Create an empty logical request group."""
        return CascadeKVRequestGroup()

    def submit(
        self,
        group: CascadeKVRequestGroup[ContextT],
        context: ContextT,
        submitter: Callable[[], int],
    ) -> int:
        """Submit and register a request atomically against completion polling."""
        with self._client_lock:
            request_id = submitter()
            group.add(request_id, context)
            with self._owners_lock:
                if request_id in self._owners:
                    raise ValueError(f"duplicate xDS request id {request_id}")
                self._owners[request_id] = group
        self._work_available.set()
        return request_id

    def call_client(self, operation: Callable[[], ContextT]) -> ContextT:
        """Run one short non-completion client operation safely."""
        with self._client_lock:
            return operation()

    def close(self) -> None:
        """Stop the completion poller without changing request semantics."""
        self._running.clear()
        self._work_available.set()
        self._thread.join(timeout=5.0)

    def _poll_loop(self) -> None:
        while self._running.is_set():
            with self._owners_lock:
                has_work = bool(self._owners)
            if not has_work:
                self._work_available.wait(timeout=0.1)
                self._work_available.clear()
                continue

            with self._client_lock:
                completions = self.client.poll_burst(self.poll_count)
            if not completions:
                self._work_available.wait(timeout=0.0005)
                self._work_available.clear()
                continue

            for completion in completions:
                request_id = int(completion.request_id)
                with self._owners_lock:
                    owner = self._owners.pop(request_id, None)
                if owner is not None:
                    owner.complete(request_id, completion)
