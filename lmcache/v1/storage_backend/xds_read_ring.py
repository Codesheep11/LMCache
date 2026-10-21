# SPDX-License-Identifier: Apache-2.0
"""Bounded request pump for xDS read completions."""

# Standard
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional
import queue
import threading
import time


@dataclass(frozen=True)
class ReadRingCompletion:
    item: Any
    completion: Optional[Any] = None
    error: Optional[BaseException] = None
    safe_to_reuse: bool = True


class XDSReadRingPump:
    """Submit and reap reads on one thread with a bounded queue depth."""

    _SENTINEL = object()

    def __init__(
        self,
        queue_depth: int,
        submit: Callable[[Any], int],
        wait_any: Callable[[float], Any],
        timeout: float,
        submit_batch: Optional[Callable[[list[Any]], list[int | BaseException]]] = None,
        wait_burst: Optional[Callable[[int, float], list[Any]]] = None,
    ) -> None:
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        self._queue_depth = queue_depth
        self._submit = submit
        self._submit_batch = submit_batch
        self._wait_burst = wait_burst
        self._wait_any = wait_any
        self._timeout = timeout
        self._submissions: queue.Queue[Any] = queue.Queue()
        self._completions: queue.Queue[ReadRingCompletion] = queue.Queue()
        self._state_lock = threading.Lock()
        self._input_closed = False
        self._ready_items: Optional[Iterator[Any]] = None
        self._ready_completions: Optional[list[ReadRingCompletion]] = None
        self._failed = threading.Event()
        self._terminal_error: Optional[BaseException] = None
        self._unsafe_item_ids: set[int] = set()
        self._unsafe_request_ids: dict[int, int] = {}

    def put(self, item: Any) -> None:
        with self._state_lock:
            if self._input_closed:
                raise RuntimeError("xDS read ring input is closed")
            if self._failed.is_set():
                raise RuntimeError("xDS read ring has failed")
            self._submissions.put(item)

    def close_input(self) -> None:
        with self._state_lock:
            if self._input_closed:
                return
            self._input_closed = True
            self._submissions.put(self._SENTINEL)

    def get_completion(self) -> ReadRingCompletion:
        return self._completions.get()

    @property
    def error(self) -> Optional[BaseException]:
        return self._terminal_error

    def is_safe_to_reuse(self, item: Any) -> bool:
        """Return whether xDS can no longer DMA into the item's buffer."""
        with self._state_lock:
            return id(item) not in self._unsafe_item_ids

    def unsafe_request_id(self, item: Any) -> Optional[int]:
        """Return the still-owned xDS request ID for an unsafe item."""
        with self._state_lock:
            return self._unsafe_request_ids.get(id(item))

    def run_items(self, items: Sequence[Any]) -> list[ReadRingCompletion]:
        """Run a complete input sequence synchronously without queue handoffs.

        The pump must be fresh, with no queued or closed input. The rolling
        queue-depth limit, error draining and unsafe-target tracking are the
        same as run(). Results are returned directly instead of being placed
        on the completion queue. Terminal errors may have item=None; callers
        must also inspect error. Raises RuntimeError for mixed input modes.
        """
        ready = (*items, self._SENTINEL)
        results: list[ReadRingCompletion] = []
        with self._state_lock:
            if (
                self._input_closed
                or self._failed.is_set()
                or not self._submissions.empty()
            ):
                raise RuntimeError("synchronous xDS input requires a fresh read ring")
            self._input_closed = True
            self._ready_items = iter(ready)
            self._ready_completions = results
        try:
            self.run()
        finally:
            self._ready_items = None
            self._ready_completions = None
        return results

    def run(self) -> None:
        """Run the rolling single path, or bounded groups for a batch submitter.

        Batch callbacks return one ID or exception per item. Accepted IDs are
        drained before failed items are returned. An unexpected batch exception
        retains every potentially submitted target as unsafe. A group completes
        before the next is submitted; callers choose a suitable queue depth.
        """
        if self._submit_batch is not None:
            self._run_batch()
            return
        pending: dict[int, Any] = {}
        input_drained = False
        first_error: Optional[BaseException] = None
        failed_items: list[tuple[Any, bool]] = []

        while True:
            if first_error is None and not input_drained:
                while len(pending) < self._queue_depth:
                    try:
                        item = self._take_submission(block=not pending)
                    except queue.Empty:
                        break
                    if item is self._SENTINEL:
                        input_drained = True
                        break
                    try:
                        request_id = int(self._submit(item))
                    except BaseException as error:
                        first_error = error
                        failed_items.append((item, True))
                        self._failed.set()
                        break
                    if request_id in pending:
                        existing_item = pending.pop(request_id)
                        first_error = RuntimeError(
                            f"xDS reused pending request ID {request_id}"
                        )
                        failed_items.extend(((existing_item, False), (item, False)))
                        with self._state_lock:
                            self._unsafe_item_ids.update((id(existing_item), id(item)))
                            self._unsafe_request_ids.update(
                                {
                                    id(existing_item): request_id,
                                    id(item): request_id,
                                }
                            )
                        self._failed.set()
                        break
                    pending[request_id] = item

            if pending:
                try:
                    completion = self._wait_any(self._timeout)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                        self._failed.set()
                    unsafe_items = tuple(pending.values())
                    failed_items.extend((item, False) for item in unsafe_items)
                    with self._state_lock:
                        self._unsafe_item_ids.update(id(item) for item in unsafe_items)
                        self._unsafe_request_ids.update(
                            {
                                id(item): request_id
                                for request_id, item in pending.items()
                            }
                        )
                    pending.clear()
                    continue
                item = pending.pop(int(completion.request_id), None)
                if item is None:
                    if first_error is None:
                        first_error = RuntimeError(
                            "xDS returned unknown completion request "
                            f"{completion.request_id}"
                        )
                        self._failed.set()
                    continue
                self._publish_completion(
                    ReadRingCompletion(item=item, completion=completion)
                )
                continue

            if first_error is not None:
                while True:
                    try:
                        item = self._take_submission(block=False)
                    except queue.Empty:
                        break
                    if item is not self._SENTINEL:
                        failed_items.append((item, True))
                self._terminal_error = first_error
                if not failed_items:
                    self._publish_completion(
                        ReadRingCompletion(item=None, error=first_error)
                    )
                else:
                    for item, safe_to_reuse in failed_items:
                        self._publish_completion(
                            ReadRingCompletion(
                                item=item,
                                error=first_error,
                                safe_to_reuse=safe_to_reuse,
                            )
                        )
                return
            if input_drained:
                return

    def _wait_batch_results(self, count: int) -> list[Any]:
        results = []
        deadline = time.monotonic() + self._timeout
        while len(results) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("xDS batch completion timed out")
            part = (
                self._wait_burst(count - len(results), remaining)
                if self._wait_burst is not None
                else [self._wait_any(remaining)]
            )
            if not part:
                raise TimeoutError("xDS batch completion timed out")
            results.extend(part)
        return results

    def _run_batch(self) -> None:
        pending: dict[int, Any] = {}
        input_drained = False
        first_error: Optional[BaseException] = None
        failed_items: list[tuple[Any, bool]] = []

        while True:
            if first_error is None and not input_drained:
                batch = []
                capacity = min(64, self._queue_depth) - len(pending)
                if self._submit_batch is None:
                    capacity = min(capacity, 1)
                while len(batch) < capacity:
                    try:
                        item = self._take_submission(block=not pending and not batch)
                    except queue.Empty:
                        break
                    if item is self._SENTINEL:
                        input_drained = True
                        break
                    batch.append(item)
                try:
                    if self._submit_batch is not None and batch:
                        outcomes = self._submit_batch(batch)
                        if len(outcomes) != len(batch):
                            raise RuntimeError("batch result count differs from inputs")
                    elif batch:
                        outcomes = [int(self._submit(batch[0]))]
                    else:
                        outcomes = []
                except BaseException as error:
                    first_error = error
                    # A batch callback that raises may have already submitted
                    # part of the batch without returning IDs. Retain targets.
                    safe = self._submit_batch is None
                    failed_items.extend((item, safe) for item in batch)
                    if not safe:
                        with self._state_lock:
                            self._unsafe_item_ids.update(map(id, batch))
                    self._failed.set()
                    outcomes = []
                for item, outcome in zip(batch, outcomes, strict=False):
                    if isinstance(outcome, BaseException):
                        first_error = first_error or outcome
                        failed_items.append((item, True))
                        self._failed.set()
                        continue
                    request_id = int(outcome)
                    if request_id in pending:
                        existing_item = pending.pop(request_id)
                        first_error = RuntimeError(
                            f"xDS reused pending request ID {request_id}"
                        )
                        failed_items.extend(((existing_item, False), (item, False)))
                        with self._state_lock:
                            self._unsafe_item_ids.update((id(existing_item), id(item)))
                            self._unsafe_request_ids.update(
                                {
                                    id(existing_item): request_id,
                                    id(item): request_id,
                                }
                            )
                        self._failed.set()
                        continue
                    pending[request_id] = item

            if (
                first_error is None
                and not input_drained
                and len(pending) < min(64, self._queue_depth)
            ):
                if self._ready_items is not None or not self._submissions.empty():
                    continue

            if pending:
                try:
                    completions = self._wait_batch_results(len(pending))
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                        self._failed.set()
                    unsafe_items = tuple(pending.values())
                    failed_items.extend((item, False) for item in unsafe_items)
                    with self._state_lock:
                        self._unsafe_item_ids.update(id(item) for item in unsafe_items)
                        self._unsafe_request_ids.update(
                            {
                                id(item): request_id
                                for request_id, item in pending.items()
                            }
                        )
                    pending.clear()
                    continue
                for completion in completions:
                    item = pending.pop(int(completion.request_id), None)
                    if item is None:
                        if first_error is None:
                            first_error = RuntimeError(
                                "xDS returned unknown completion request "
                                f"{completion.request_id}"
                            )
                            self._failed.set()
                        continue
                    self._publish_completion(
                        ReadRingCompletion(item=item, completion=completion)
                    )
                continue

            if first_error is not None:
                while True:
                    try:
                        item = self._take_submission(block=False)
                    except queue.Empty:
                        break
                    if item is not self._SENTINEL:
                        failed_items.append((item, True))
                self._terminal_error = first_error
                if not failed_items:
                    self._publish_completion(
                        ReadRingCompletion(item=None, error=first_error)
                    )
                else:
                    for item, safe_to_reuse in failed_items:
                        self._publish_completion(
                            ReadRingCompletion(
                                item=item,
                                error=first_error,
                                safe_to_reuse=safe_to_reuse,
                            )
                        )
                return
            if input_drained:
                return

    def _take_submission(self, block: bool) -> Any:
        if self._ready_items is None:
            return self._submissions.get(block=block)
        try:
            return next(self._ready_items)
        except StopIteration:
            raise queue.Empty from None

    def _publish_completion(self, completion: ReadRingCompletion) -> None:
        if self._ready_completions is None:
            self._completions.put(completion)
        else:
            self._ready_completions.append(completion)
