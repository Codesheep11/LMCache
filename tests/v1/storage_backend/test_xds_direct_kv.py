# SPDX-License-Identifier: Apache-2.0
"""Payload-free read/write contract and completion ownership regression tests."""

# Standard
from collections import deque
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch
import ctypes

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.xds_readv import XDSReadvPlan
from lmcache.v1.storage_backend.xds_backend import XDSBackend


class VectorClient:
    """Emulate registered-memory DMA without a backend-owned payload buffer."""

    def __init__(self) -> None:
        self.metadata = SimpleNamespace(
            io_unit_size=4096,
            blob_size_bytes=8192,
            gpu_buffer_size_bytes=0,
            capacity_bytes=65536,
        )
        self.buffers = {}
        self.blobs = {}
        self.pending = deque()
        self.released = []
        self.unregistered = []
        self.sequence = 0
        self.failure = None
        self.entered = Event()
        self.allow = Event()
        self.allow.set()
        self.on_complete = lambda: None

    def register_gpu_buffer(self, pointer: int, size: int) -> int:
        ref = len(self.buffers) + 1
        self.buffers[ref] = (pointer, size)
        return ref

    def unregister_buffer(self, ref: int) -> None:
        self.unregistered.append(ref)

    def get_blob(self, timeout: float = 5) -> int:
        handle = len(self.blobs) + 100
        self.blobs[handle] = b""
        return handle

    def release_blob(self, handle: int) -> None:
        self.released.append(handle)

    def submit_writev(self, handle: int, offset: int, slices: list) -> int:
        assert offset == 0
        payload = b"".join(
            ctypes.string_at(self.buffers[r][0] + o, n) for r, o, n in slices
        )
        self.blobs[handle] = payload
        self.sequence += 1
        self.pending.append(
            SimpleNamespace(
                request_id=self.sequence,
                status=0,
                error_code=0,
                completed_bytes=len(payload),
            )
        )
        return self.sequence

    def submit_readv(self, handle: int, offset: int, slices: list) -> int:
        payload = self.blobs[handle][offset:]
        cursor = 0
        for ref, start, size in slices:
            ctypes.memmove(
                self.buffers[ref][0] + start, payload[cursor : cursor + size], size
            )
            cursor += size
        self.sequence += 1
        self.pending.append(
            SimpleNamespace(
                request_id=self.sequence, status=0, error_code=0, completed_bytes=cursor
            )
        )
        return self.sequence

    def wait_any_result(self, timeout: float = 30) -> SimpleNamespace:
        self.entered.set()
        assert self.allow.wait(5)
        self.on_complete()
        if self.failure == "timeout":
            raise TimeoutError("pending DMA")
        result = self.pending.popleft()
        if self.failure == "short":
            result.completed_bytes -= 4096
        return result


class Connector:
    def __init__(self, tensors: tuple) -> None:
        self.tensors = tensors
        self.pool = SimpleNamespace(
            tensors=tensors,
            owner=None,
            buffer_refs=None,
            closed=False,
            registration_ranges=tuple(
                (t.data_ptr(), t.numel() * t.element_size()) for t in tensors
            ),
        )
        self.fences = 0
        self.supported = True

    def initialize_xds_readv_pool(self, caches: tuple) -> SimpleNamespace:
        assert caches is self.tensors
        return self.pool

    def plan_xds_readv(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("batched planner required")

    def plan_xds_readv_batch(
        self, chunks: list, io_unit: int, **kwargs: object
    ) -> list | None:
        if not self.supported:
            return None
        block = kwargs["block"]
        return [
            XDSReadvPlan(
                self.tensors,
                ((0, block * 4096, 4096), (1, block * 4096, 4096)),
                8192,
                (block,),
                self.pool.registration_ranges,
                self.pool.buffer_refs,
            )
            for _ in chunks
        ]

    def prepare_xds_readv_dma(self, device: torch.device) -> None:
        self.fences += 1

    def finish_xds_readv_dma(self, device: torch.device) -> None:
        self.fences += 1


@pytest.fixture
def stack():
    config = LMCacheEngineConfig.from_defaults()
    config.xds_backend = True
    config.xds_windowed_io = False
    config.local_cpu = False
    config.max_local_cpu_size = 0
    client = VectorClient()
    metadata = SimpleNamespace(
        kv_shape=(1, 2, 16, 1, 128),
        get_shapes=lambda n: [torch.Size((2, 1, n, 128))],
        get_dtypes=lambda: [torch.float16],
    )
    tensors = tuple(
        torch.arange(8192, dtype=torch.float16).reshape(4, 16, 1, 128) for _ in range(2)
    )
    connector = Connector(tensors)
    with patch(
        "lmcache.v1.storage_backend.xds_backend.torch.empty",
        side_effect=AssertionError("backend allocated payload memory"),
    ):
        backend = XDSBackend(config, metadata, client, "cpu")
    backend.register_kv_caches(connector, tensors)
    key = CacheEngineKey("test", 1, 0, 99, torch.float16)
    return backend, client, connector, key


def test_round_trip_uses_final_pages_and_no_payload_allocator(stack):
    backend, client, connector, key = stack
    assert len(client.buffers) == 2  # Only serving tensors, no staging registration.
    assert backend.staging_size == 0
    assert not hasattr(backend.config, "xds_read_slot_count")
    assert not hasattr(backend.config, "xds_write_slot_count")
    assert backend.calculate_chunk_budget() == 8
    expected = tuple(t[1].clone() for t in connector.tensors)
    result = backend.batched_put_from_gpu_blocking([key], [0], [16], connector, block=1)
    assert result.completed == (True,) and result.peak_staging_bytes == 0
    assert result.read_slot_count == 0 and result.write_slot_count == 0
    for t in connector.tensors:
        t[2].fill_(-7)
    result = backend.batched_get_to_gpu_blocking([key], [0], [16], connector, block=2)
    assert result.completed == (True,) and result.peak_staging_bytes == 0
    assert result.read_slot_count == 0 and result.write_slot_count == 0
    for tensor, saved in zip(connector.tensors, expected, strict=True):
        assert torch.equal(tensor[2], saved)
    with pytest.raises(RuntimeError, match="forbids staging"):
        backend.allocate([torch.Size((1,))], [torch.float16])
    backend.close()
    assert len(client.unregistered) == 2


def test_write_blocks_and_does_not_publish_before_completion(stack):
    backend, client, connector, key = stack
    client.allow.clear()
    results = []
    thread = Thread(
        target=lambda: results.append(
            backend.batched_put_from_gpu_blocking([key], [0], [16], connector, block=1)
        )
    )
    thread.start()
    assert client.entered.wait(5)
    assert thread.is_alive() and not backend.contains(key)
    client.allow.set()
    thread.join(5)
    assert not thread.is_alive() and len(results) == 1 and backend.contains(key)
    backend.close()


@pytest.mark.parametrize("failure", ["short", "timeout"])
def test_failed_write_never_publishes_and_retains_unknown_dma(stack, failure):
    backend, client, connector, key = stack
    client.failure = failure
    with pytest.raises((RuntimeError, TimeoutError)):
        backend.batched_put_from_gpu_blocking([key], [0], [16], connector, block=1)
    assert not backend.contains(key)
    if failure == "timeout":
        assert not client.released and not client.unregistered
        with pytest.raises(RuntimeError, match="unresolved DMA"):
            backend.close()
        assert connector.pool.tensors
    else:
        assert len(client.released) == 1
        backend.close()


def test_unsupported_geometry_fails_before_io_and_never_stages(stack):
    backend, client, connector, key = stack
    connector.supported = False
    with pytest.raises(ValueError, match="unsupported KV"):
        backend.batched_put_from_gpu_blocking([key], [0], [16], connector, block=1)
    assert not client.blobs
    backend.close()


def test_submit_failure_drains_accepted_writes_and_rolls_back_all_keys(stack):
    backend, client, connector, first = stack
    second = CacheEngineKey("test", 1, 0, 100, torch.float16)
    backend.config.xds_io_queue_depth = 2
    original = client.submit_writev
    calls = []

    def submit(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("rejected submit")
        return original(*args)

    client.submit_writev = submit
    with pytest.raises(RuntimeError, match="rejected submit"):
        backend.batched_put_from_gpu_blocking(
            [first, second], [0, 16], [16, 32], connector, block=1
        )
    assert not client.pending
    assert len(client.released) == 2
    assert not backend.contains(first) and not backend.contains(second)
    backend.close()


@pytest.mark.parametrize("option", ["xds_async_write", "store_async", "use_layerwise"])
def test_direct_mode_rejects_incompatible_lifetimes(stack, option):
    backend, client, connector, key = stack
    setattr(backend.config, option, True)
    with pytest.raises(ValueError, match="synchronous stores"):
        XDSBackend(backend.config, backend.metadata, VectorClient(), "cpu")
    backend.close()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("failure", [None, "short", "timeout"])
def test_compact_restore_preserves_payload_and_failure_ownership(
    stack: tuple, native: bool, bound: bool, failure: str | None
) -> None:
    from lmcache.v1.gpu_connector.xds_readv import (
        XDSPagedKVGeometry,
        XDSPagedReadvBatch,
    )

    backend, client, connector, first = stack
    backend.config.xds_io_queue_depth = 2
    second = CacheEngineKey("test", 1, 0, 100, torch.float16)
    keys = [first, second]
    expected = tuple(t[1].clone() for t in connector.tensors)
    backend.batched_put_from_gpu_blocking(keys, [0, 16], [16, 32], connector, block=1)
    geometry = XDSPagedKVGeometry(
        connector.tensors,
        "SEPARATE",
        "cpu",
        torch.float16,
        1,
        4,
        16,
        128,
        4096,
        512,
        4096,
    )

    def planner(chunks: list, io_unit: int, **kwargs: object) -> XDSPagedReadvBatch:
        assert kwargs["xds_compact_batch"] is True and len(chunks) == 2
        return XDSPagedReadvBatch(
            geometry,
            (2, 3),
            (0, 1, 2),
            (8192, 8192),
            connector.pool.registration_ranges,
            connector.pool.buffer_refs if bound else None,
        )

    connector.plan_xds_readv_batch = planner
    for t in connector.tensors:
        t[2:].fill_(-7)
    groups_seen = []
    if native:

        def prepare(ids: tuple, groups: tuple, block_bytes: int) -> list:
            groups_seen.append(groups)
            return [
                (ref, base + block * stride, block_bytes)
                for ref, base, stride in groups
                for block in ids
            ]

        client.prepare_indexed_readv = prepare
        client.submit_readv_plan = client.submit_readv
    client.failure = failure
    if failure:
        with pytest.raises((RuntimeError, TimeoutError)):
            backend.batched_get_to_gpu_blocking(keys, [0, 16], [16, 32], connector)
        if failure == "timeout":
            assert not client.unregistered
            with pytest.raises(RuntimeError, match="unresolved DMA"):
                backend.close()
        else:
            assert not client.pending
            backend.close()
    else:
        result = backend.batched_get_to_gpu_blocking(keys, [0, 16], [16, 32], connector)
        assert result.completed == (True, True) and result.completed_bytes == 16384
        assert result.peak_staging_bytes == 0 and len(client.buffers) == 2
        for t, saved in zip(connector.tensors, expected, strict=True):
            assert torch.equal(t[2], saved) and torch.equal(t[3], saved)
        backend.close()
    if native:
        assert len(groups_seen) == 2 and groups_seen[0] is groups_seen[1]
