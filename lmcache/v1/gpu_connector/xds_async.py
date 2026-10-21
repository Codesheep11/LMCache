# SPDX-License-Identifier: Apache-2.0
"""Single-owner asynchronous xDS restores with explicit completion reporting."""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class XDSAsyncLoader:
    """Serialize I/O on one thread while the serving thread continues running.

    Submit, poll and close are called by the serving thread. Submitted callables
    must finish DMA and visibility fencing before returning. There is deliberately
    no cancellation: the scheduler must retain destination blocks until completion.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="xds-load"
        )
        self._pending: dict[str, Future[Any]] = {}
        self._closed = False

    def submit(self, request_id: str, load: Callable[..., Any], *args: Any) -> None:
        """Queue a restore; reject duplicates and submission after close."""
        if self._closed:
            raise RuntimeError("xDS loader is closed")
        if request_id in self._pending:
            raise ValueError(f"duplicate xDS restore: {request_id}")
        self._pending[request_id] = self._executor.submit(load, *args)

    def poll(self) -> set[str]:
        """Return completed IDs once; propagate errors without reporting readiness."""
        ready = set()
        for request_id, future in list(self._pending.items()):
            if future.done():
                future.result()
                ready.add(request_id)
                del self._pending[request_id]
        return ready

    def close(self) -> None:
        """Drain all queued restores before KV registrations can be released."""
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
