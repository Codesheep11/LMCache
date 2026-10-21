# SPDX-License-Identifier: Apache-2.0
"""Asynchronous restores retain ownership until fenced completion."""

# Standard
from threading import Event, Thread

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector.xds_async import XDSAsyncLoader


def test_serial_owner_nonblocking_poll_and_close_drain() -> None:
    loader = XDSAsyncLoader()
    entered, release, second, closed = Event(), Event(), Event(), Event()

    def load() -> None:
        entered.set()
        assert release.wait(5)

    loader.submit("first", load)
    assert entered.wait(5)
    loader.submit("second", second.set)
    assert loader.poll() == set()
    assert not second.is_set()
    with pytest.raises(ValueError, match="duplicate"):
        loader.submit("first", load)

    def close() -> None:
        loader.close()
        closed.set()

    closer = Thread(target=close)
    closer.start()
    try:
        assert not closed.wait(0.02)
    finally:
        release.set()
        closer.join(5)
    assert closed.is_set() and second.is_set()
    assert loader.poll() == {"first", "second"}
    assert loader.poll() == set()
    with pytest.raises(RuntimeError, match="closed"):
        loader.submit("third", load)


def test_failed_dma_is_never_reported_ready() -> None:
    loader = XDSAsyncLoader()

    def fail() -> None:
        raise RuntimeError("unresolved DMA")

    loader.submit("unsafe", fail)
    loader.close()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unresolved DMA"):
            loader.poll()


def test_completed_request_does_not_wait_for_next_restore() -> None:
    loader = XDSAsyncLoader()
    entered, release = Event(), Event()

    def slow() -> None:
        entered.set()
        assert release.wait(5)

    loader.submit("ready", lambda: None)
    loader.submit("pending", slow)
    try:
        assert entered.wait(5)
        assert loader.poll() == {"ready"}
        assert loader.poll() == set()
    finally:
        release.set()
        loader.close()
    assert loader.poll() == {"pending"}
