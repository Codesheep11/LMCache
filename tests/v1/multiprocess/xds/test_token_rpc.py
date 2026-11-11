# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import replace
from types import SimpleNamespace

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules import xds_metadata
from lmcache.v1.multiprocess.modules.xds_metadata import XDSMetadataModule
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.xds_catalog import XDSExtent, XDSLease
from lmcache.v1.multiprocess.xds_protocol import (
    BindWorker,
    ClaimRange,
    Complete,
    Register,
    SchedulerLookup,
    TokenTransfer,
)


def test_token_prefix_isolation_rank_binding_and_partial_release() -> None:
    module = XDSMetadataModule(
        SimpleNamespace(chunk_size=4, token_hasher=TokenHasher(4))
    )
    scheduler = IPCCacheServerKey(
        "model", 2, None, tuple(range(12)), 0, 12, "request", num_kv_readers=1
    )
    for rank in range(2):
        worker = 100 + rank
        module.register(
            msgspec.msgpack.encode(
                Register(
                    worker, tuple(XDSExtent(worker, i * 4096, 4096) for i in range(3))
                )
            )
        )
        module.bind_worker(
            msgspec.msgpack.encode(BindWorker(worker, "model", 2, rank, 4096))
        )
        key = replace(scheduler, worker_id=rank)
        writes = msgspec.msgpack.decode(
            module.reserve_tokens(msgspec.msgpack.encode(TokenTransfer(worker, key))),
            type=tuple[XDSLease, ...],
        )
        for write in writes:
            module.publish(msgspec.msgpack.encode(Complete(worker, write.ticket)))
    assert (
        msgspec.msgpack.decode(module.lookup_tokens(msgspec.msgpack.encode(scheduler)))
        == 3
    )
    salted = replace(scheduler, request_id="salt", cache_salt="other")
    assert (
        msgspec.msgpack.decode(module.lookup_tokens(msgspec.msgpack.encode(salted)))
        == 0
    )
    module.release_tokens(msgspec.msgpack.encode(replace(scheduler, end=4)))
    for rank in range(2):
        key = replace(scheduler, worker_id=rank, start=4)
        leases = msgspec.msgpack.decode(
            module.claim_tokens(msgspec.msgpack.encode(TokenTransfer(100 + rank, key))),
            type=tuple[XDSLease, ...],
        )
        assert len(leases) == 2
        with pytest.raises(ValueError, match="different connection"):
            module.reserve_tokens(msgspec.msgpack.encode(TokenTransfer(999, key)))
        for lease in leases:
            module.release_read(
                msgspec.msgpack.encode(Complete(100 + rank, lease.ticket))
            )
    module.cancel_lookup(msgspec.msgpack.encode("request"))
    module.cancel_lookup(msgspec.msgpack.encode("salt"))
    for rank in range(2):
        module.unregister(msgspec.msgpack.encode(100 + rank))
    assert (
        msgspec.msgpack.decode(module.lookup_tokens(msgspec.msgpack.encode(scheduler)))
        == 0
    )


def test_scheduler_reaping_releases_only_unclaimed_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(xds_metadata, "monotonic", lambda: now[0])
    module = XDSMetadataModule(
        SimpleNamespace(chunk_size=4, token_hasher=TokenHasher(4))
    )
    module.register(
        msgspec.msgpack.encode(
            Register(
                100,
                (
                    XDSExtent(100, 0, 4096),
                    XDSExtent(100, 4096, 4096),
                ),
            )
        )
    )
    module.bind_worker(msgspec.msgpack.encode(BindWorker(100, "model", 1, 0, 4096)))
    scheduler = IPCCacheServerKey(
        "model", 1, None, tuple(range(8)), 0, 8, "orphan", num_kv_readers=1
    )
    worker = replace(scheduler, worker_id=0)
    writes = msgspec.msgpack.decode(
        module.reserve_tokens(msgspec.msgpack.encode(TokenTransfer(100, worker))),
        type=tuple[XDSLease, ...],
    )
    for write in writes:
        module.publish(msgspec.msgpack.encode(Complete(100, write.ticket)))
    owned = msgspec.msgpack.encode(SchedulerLookup(-7, scheduler))
    assert msgspec.msgpack.decode(module.lookup_owned_tokens(owned)) == 2
    module.touch_instance(-7)
    with pytest.raises(ValueError, match="already owned"):
        module.lookup_owned_tokens(
            msgspec.msgpack.encode(SchedulerLookup(-8, scheduler))
        )
    claimed = msgspec.msgpack.decode(
        module.claim_range(
            msgspec.msgpack.encode(ClaimRange(100, "orphan", "model", 1, 0, 0, 4))
        ),
        type=tuple[XDSLease, ...],
    )
    assert len(claimed) == 1
    now[0] = 11
    assert module.reap_stale_instances(10, 1000) == [-7]
    assert module.tracked_instance_count() == 1
    # The second chunk's orphaned pin is gone; the first still belongs to DMA.
    assert msgspec.msgpack.decode(module.clear_idle(msgspec.msgpack.encode(None))) == 1
    module.release_read(msgspec.msgpack.encode(Complete(100, claimed[0].ticket)))
    assert msgspec.msgpack.decode(module.clear_idle(msgspec.msgpack.encode(None))) == 1
    module.cancel_lookup(msgspec.msgpack.encode("orphan"))


def test_scheduler_heartbeat_preserves_long_queued_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(xds_metadata, "monotonic", lambda: now[0])
    module = XDSMetadataModule(
        SimpleNamespace(chunk_size=4, token_hasher=TokenHasher(4))
    )
    key = IPCCacheServerKey(
        "model", 1, None, (1, 2, 3, 4), 0, 4, "queued", num_kv_readers=1
    )
    module.lookup_owned_tokens(msgspec.msgpack.encode(SchedulerLookup(-7, key)))
    now[0] = 99
    assert module.reap_stale_instances(10, 100) == []
    module.touch_instance(-7)
    now[0] = 108
    assert module.reap_stale_instances(10, 100) == []
    now[0] = 109
    assert module.reap_stale_instances(10, 100) == [-7]
    assert module.tracked_instance_count() == 0


def test_scheduler_and_worker_epoch_namespaces_do_not_overlap() -> None:
    module = XDSMetadataModule(
        SimpleNamespace(chunk_size=4, token_hasher=TokenHasher(4))
    )
    with pytest.raises(ValueError, match="nonnegative"):
        module.register(msgspec.msgpack.encode(Register(-7, (XDSExtent(1, 0, 4096),))))
    key = IPCCacheServerKey(
        "model", 1, None, (1, 2, 3, 4), 0, 4, "request", num_kv_readers=1
    )
    with pytest.raises(ValueError, match="negative"):
        module.lookup_owned_tokens(msgspec.msgpack.encode(SchedulerLookup(7, key)))
    assert module.tracked_instance_count() == 0


@pytest.mark.parametrize("start,end", [(0, 12), (4, 16), (8, 24)])
def test_indexed_claim_ranges_and_lifetime(start: int, end: int) -> None:
    module = XDSMetadataModule(
        SimpleNamespace(chunk_size=4, token_hasher=TokenHasher(4))
    )
    scheduler = IPCCacheServerKey(
        "model",
        2,
        None,
        tuple(range(end + 4)),
        start,
        end + 4,
        "indexed",
        num_kv_readers=1,
        cache_salt="tenant",
    )
    for rank in range(2):
        worker = 100 + rank
        module.register(
            msgspec.msgpack.encode(
                Register(
                    worker,
                    tuple(
                        XDSExtent(worker, i * 4096, 4096)
                        for i in range((end - start) // 4)
                    ),
                )
            )
        )
        module.bind_worker(
            msgspec.msgpack.encode(BindWorker(worker, "model", 2, rank, 4096))
        )
        writes = msgspec.msgpack.decode(
            module.reserve_tokens(
                msgspec.msgpack.encode(
                    TokenTransfer(worker, replace(scheduler, worker_id=rank, end=end))
                )
            ),
            type=tuple[XDSLease, ...],
        )
        for write in writes:
            module.publish(msgspec.msgpack.encode(Complete(worker, write.ticket)))
    assert (
        msgspec.msgpack.decode(
            module.lookup_owned_tokens(
                msgspec.msgpack.encode(SchedulerLookup(-1, scheduler))
            )
        )
        == (end - start) // 4
    )
    request = ClaimRange(100, "indexed", "model", 2, 0, start, start + 4)
    for invalid in (
        replace(request, model_name="other"),
        replace(request, world_size=1),
        replace(request, worker_id=-1),
        replace(request, worker_id=2),
        replace(request, worker=999),
        replace(request, start=start + 1),
        replace(request, end=end + 4),
        replace(request, end=start),
    ):
        with pytest.raises(ValueError):
            module.claim_range(msgspec.msgpack.encode(invalid))
    first = msgspec.msgpack.decode(
        module.claim_range(msgspec.msgpack.encode(request)), type=tuple[XDSLease, ...]
    )
    # An overlapping batch must not consume the as-yet unclaimed tail.
    with pytest.raises(KeyError):
        module.claim_range(msgspec.msgpack.encode(replace(request, end=end)))
    tail = msgspec.msgpack.decode(
        module.claim_range(
            msgspec.msgpack.encode(replace(request, start=start + 4, end=end))
        ),
        type=tuple[XDSLease, ...],
    )
    assert len(first + tail) == (end - start) // 4
    module.cancel_lookup(msgspec.msgpack.encode("indexed"))
    with pytest.raises(KeyError):
        module.claim_range(
            msgspec.msgpack.encode(replace(request, worker=101, worker_id=1))
        )
    # Cancellation frees rank 1 but retains every rank 0 DMA lease.
    assert (
        msgspec.msgpack.decode(module.clear_idle(msgspec.msgpack.encode(None)))
        == (end - start) // 4
    )
    for lease in first + tail:
        module.release_read(msgspec.msgpack.encode(Complete(100, lease.ticket)))
    assert (
        msgspec.msgpack.decode(module.clear_idle(msgspec.msgpack.encode(None)))
        == (end - start) // 4
    )
    module.close()


@pytest.mark.parametrize("expiry", ["worker", "scheduler", "close", "cancel"])
def test_indexed_claim_rejects_invalidated_lookup(
    expiry: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(xds_metadata, "monotonic", lambda: now[0])
    module = XDSMetadataModule(
        SimpleNamespace(chunk_size=4, token_hasher=TokenHasher(4))
    )
    key = IPCCacheServerKey(
        "model", 1, 0, (1, 2, 3, 4), 0, 4, "epoch", num_kv_readers=1
    )
    module.register(msgspec.msgpack.encode(Register(100, (XDSExtent(1, 0, 4096),))))
    module.bind_worker(msgspec.msgpack.encode(BindWorker(100, "model", 1, 0, 4096)))
    writes = msgspec.msgpack.decode(
        module.reserve_tokens(msgspec.msgpack.encode(TokenTransfer(100, key))),
        type=tuple[XDSLease, ...],
    )
    module.publish(msgspec.msgpack.encode(Complete(100, writes[0].ticket)))
    module.lookup_owned_tokens(
        msgspec.msgpack.encode(SchedulerLookup(-1, replace(key, worker_id=None)))
    )
    if expiry in ("worker", "scheduler"):
        module.touch_instance(100 if expiry == "worker" else -1)
        now[0] = 11
        module.reap_stale_instances(10, 1000)
    elif expiry == "close":
        module.close()
    else:
        module.cancel_lookup(msgspec.msgpack.encode("epoch"))
    with pytest.raises(KeyError):
        module.claim_range(
            msgspec.msgpack.encode(ClaimRange(100, "epoch", "model", 1, 0, 0, 4))
        )
    module.close()


@pytest.mark.parametrize("chunks", [1, 3, 257, 1200])
def test_mla_shared_prefix_has_independent_replica_pins(chunks: int) -> None:
    module = XDSMetadataModule(
        SimpleNamespace(chunk_size=4, token_hasher=TokenHasher(4))
    )
    key = IPCCacheServerKey(
        "mla",
        1,
        0,
        tuple(range(chunks * 4)),
        0,
        chunks * 4,
        "mla-request",
        num_kv_readers=4,
    )
    module.register(
        msgspec.msgpack.encode(
            Register(
                100, tuple(XDSExtent(1, i * 4096, 4096, 999) for i in range(chunks))
            )
        )
    )
    module.bind_worker(
        msgspec.msgpack.encode(BindWorker(100, "mla", 1, 0, 4096, 0, 4, True))
    )
    writes = msgspec.msgpack.decode(
        module.reserve_tokens(msgspec.msgpack.encode(TokenTransfer(100, key))),
        type=tuple[XDSLease, ...],
    )
    for lease in writes:
        module.publish(msgspec.msgpack.encode(Complete(100, lease.ticket)))
    # A cohort missing three readers is a miss, not a partially pinned hit.
    scheduler = replace(key, worker_id=None)
    assert (
        msgspec.msgpack.decode(module.lookup_tokens(msgspec.msgpack.encode(scheduler)))
        == 0
    )
    for rank in range(1, 4):
        module.register(msgspec.msgpack.encode(Register(100 + rank, (), True)))
        module.bind_worker(
            msgspec.msgpack.encode(
                BindWorker(100 + rank, "mla", 1, 0, 4096, rank, 4, False)
            )
        )
        with pytest.raises(ValueError, match="Read-only"):
            module.reserve_tokens(
                msgspec.msgpack.encode(TokenTransfer(100 + rank, key))
            )
    assert (
        msgspec.msgpack.decode(module.lookup_tokens(msgspec.msgpack.encode(scheduler)))
        == chunks
    )
    batches = [
        msgspec.msgpack.decode(
            module.claim_range(
                msgspec.msgpack.encode(
                    ClaimRange(100 + rank, key.request_id, "mla", 1, 0, 0, chunks * 4)
                )
            ),
            type=tuple[XDSLease, ...],
        )
        for rank in range(4)
    ]
    assert len({lease.ticket for batch in batches for lease in batch}) == chunks * 4
    assert all(
        [x.extent for x in batch] == [x.extent for x in writes] for batch in batches
    )
    module.cancel_lookup(msgspec.msgpack.encode(key.request_id))
    for rank, batch in enumerate(batches):
        assert (
            msgspec.msgpack.decode(module.clear_idle(msgspec.msgpack.encode(None))) == 0
        )
        with pytest.raises(RuntimeError, match="outstanding"):
            module.unregister(msgspec.msgpack.encode(100))
        for lease in batch:
            module.release_read(
                msgspec.msgpack.encode(Complete(100 + rank, lease.ticket))
            )
    assert (
        msgspec.msgpack.decode(module.clear_idle(msgspec.msgpack.encode(None)))
        == chunks
    )
    for rank in range(4):
        module.unregister(msgspec.msgpack.encode(100 + rank))
    module.close()
