# SPDX-License-Identifier: Apache-2.0
"""Bounded request pump for xDS read completions."""

# Standard
from dataclasses import dataclass
from typing import Any, Callable, Optional
import queue
import threading


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
    ) -> None:
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        self._queue_depth = queue_depth
        self._submit = submit
        self._wait_any = wait_any
        self._timeout = timeout
        self._submissions: queue.Queue[Any] = queue.Queue()
        self._completions: queue.Queue[ReadRingCompletion] = queue.Queue()
        self._state_lock = threading.Lock()
        self._input_closed = False
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

    def run(self) -> None:
        pending: dict[int, Any] = {}
        input_drained = False
        first_error: Optional[BaseException] = None
        failed_items: list[tuple[Any, bool]] = []

        while True:
            if first_error is None and not input_drained:
                while len(pending) < self._queue_depth:
                    try:
                        item = self._submissions.get(block=not pending)
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
                self._completions.put(
                    ReadRingCompletion(item=item, completion=completion)
                )
                continue

            if first_error is not None:
                while True:
                    try:
                        item = self._submissions.get_nowait()
                    except queue.Empty:
                        break
                    if item is not self._SENTINEL:
                        failed_items.append((item, True))
                self._terminal_error = first_error
                if not failed_items:
                    self._completions.put(
                        ReadRingCompletion(item=None, error=first_error)
                    )
                else:
                    for item, safe_to_reuse in failed_items:
                        self._completions.put(
                            ReadRingCompletion(
                                item=item,
                                error=first_error,
                                safe_to_reuse=safe_to_reuse,
                            )
                        )
                return
            if input_drained:
                return
