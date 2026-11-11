# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.xds_catalog import XDSCatalog, XDSExtent


def key(index: int, rank: int = 1, salt: str = "") -> ObjectKey:
    """Build independent content identities for public contract tests."""
    return ObjectKey(index.to_bytes(4, "big"), "model", rank, cache_salt=salt)


def catalog() -> XDSCatalog:
    """Create two workers with one independently owned extent each."""
    result = XDSCatalog()
    result.register(10, (XDSExtent(100, 0, 4096),))
    result.register(20, (XDSExtent(200, 0, 4096),))
    return result


def test_partial_write_never_becomes_a_tp_hit() -> None:
    c = catalog()
    a = c.reserve_write(10, key(1), 4096)
    b = c.reserve_write(20, key(1, 2), 4096)
    assert a is not None and b is not None
    objects = ((10, key(1)), (20, key(1, 2)))
    assert c.acquire_reads(objects) is None
    c.publish(10, a.ticket)
    assert c.acquire_reads(objects) is None
    c.publish(20, b.ticket)
    leases = c.acquire_reads(objects)
    assert leases is not None
    assert [lease.extent.blob for lease in leases] == [100, 200]
    assert c.reserve_write(10, key(2), 4096) is None
    for (worker, _), lease in zip(objects, leases, strict=True):
        c.release_read(worker, lease.ticket)
    assert c.reserve_write(10, key(2), 4096) is not None


def test_failed_lookup_does_not_leave_partial_pins() -> None:
    c = catalog()
    write = c.reserve_write(10, key(1), 4096)
    assert write is not None
    c.publish(10, write.ticket)
    assert c.acquire_reads(((10, key(1)), (20, key(1, 2)))) is None
    assert c.reserve_write(10, key(2), 4096) is not None


def test_duplicate_writers_and_stale_completions() -> None:
    c = catalog()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(lambda _: c.reserve_write(10, key(1), 4096), range(32))
        )
    writes = [lease for lease in results if lease is not None]
    assert len(writes) == 1
    lease = writes[0]
    with pytest.raises(KeyError):
        c.publish(20, lease.ticket)
    c.abort_write(10, lease.ticket)
    replacement = c.reserve_write(10, key(2), 4096)
    assert replacement is not None
    assert replacement.extent == lease.extent
    with pytest.raises(KeyError):
        c.publish(10, lease.ticket)
    assert c.acquire_reads(((10, key(1)),)) is None
    assert c.acquire_reads(((10, key(2)),)) is None


def test_each_reader_must_finish_before_eviction() -> None:
    c = catalog()
    write = c.reserve_write(10, key(1), 4096)
    assert write is not None
    c.publish(10, write.ticket)
    leases = c.acquire_reads(((10, key(1)), (10, key(1))))
    assert leases is not None
    c.release_read(10, leases[0].ticket)
    assert c.reserve_write(10, key(2), 4096) is None
    with pytest.raises(RuntimeError):
        c.unregister(10)
    c.release_read(10, leases[1].ticket)
    c.unregister(10)
    assert c.acquire_reads(((10, key(1)),)) is None


def test_dead_worker_capacity_is_not_reassigned() -> None:
    c = catalog()
    write = c.reserve_write(10, key(1), 4096)
    assert write is not None
    c.forget_dead_worker(10)
    c.register(11, (XDSExtent(300, 0, 4096),))
    with pytest.raises(KeyError):
        c.publish(10, write.ticket)
    replacement = c.reserve_write(11, key(1), 4096)
    assert replacement is not None and replacement.extent.blob == 300


def test_size_and_salt_isolation() -> None:
    c = catalog()
    assert c.reserve_write(10, key(1), 8192) is None
    write = c.reserve_write(10, key(1, salt="a"), 4096)
    assert write is not None
    c.publish(10, write.ticket)
    assert c.acquire_reads(((10, key(1, salt="b")),)) is None


def test_lru_skips_pinned_objects() -> None:
    c = XDSCatalog()
    c.register(10, (XDSExtent(100, 0, 4096), XDSExtent(100, 4096, 4096)))
    for index in (1, 2):
        write = c.reserve_write(10, key(index), 4096)
        assert write is not None
        c.publish(10, write.ticket)
    read = c.acquire_reads(((10, key(1)),))
    assert read is not None
    assert c.reserve_write(10, key(3), 4096) is not None
    assert c.acquire_reads(((10, key(2)),)) is None
    c.release_read(10, read[0].ticket)


@pytest.mark.parametrize(
    "extents",
    [(), (XDSExtent(1, 0, 0),), (XDSExtent(1, 0, 4096), XDSExtent(1, 2048, 4096))],
)
def test_invalid_capacity_is_rejected(extents: tuple[XDSExtent, ...]) -> None:
    with pytest.raises(ValueError):
        XDSCatalog().register(10, extents)


def test_lookup_claim_survives_scheduler_cancellation() -> None:
    c = catalog()
    for worker, rank in ((10, 1), (20, 2)):
        write = c.reserve_write(worker, key(1, rank), 4096)
        assert write is not None
        c.publish(worker, write.ticket)
    chunks = (((10, key(1)), (20, key(1, 2))), ((10, key(2)), (20, key(2, 2))))
    assert c.lookup_prefix("request", chunks) == 1
    with pytest.raises(ValueError):
        c.lookup_prefix("request", chunks)
    with pytest.raises(KeyError):
        c.claim_reads("request", 10, (key(1), key(2)))
    read = c.claim_reads("request", 10, (key(1),))[0]
    c.cancel_lookup("request")
    c.cancel_lookup("request")
    assert c.reserve_write(10, key(2), 4096) is None
    assert c.reserve_write(20, key(2, 2), 4096) is not None
    c.release_read(10, read.ticket)
    assert c.reserve_write(10, key(2), 4096) is not None


def test_prefix_stops_at_missing_shard_without_pinning_tail() -> None:
    c = catalog()
    write = c.reserve_write(10, key(1), 4096)
    assert write is not None
    c.publish(10, write.ticket)
    assert c.lookup_prefix("miss", (((10, key(1)), (20, key(1, 2))),)) == 0
    assert c.reserve_write(10, key(2), 4096) is not None
    c.cancel_lookup("miss")


def test_dead_worker_cancels_unclaimed_other_rank_pins() -> None:
    c = catalog()
    for worker, rank in ((10, 1), (20, 2)):
        write = c.reserve_write(worker, key(1, rank), 4096)
        assert write is not None
        c.publish(worker, write.ticket)
    assert c.lookup_prefix("dead", (((10, key(1)), (20, key(1, 2))),)) == 1
    c.forget_dead_worker(10)
    with pytest.raises(KeyError):
        c.claim_reads("dead", 20, (key(1, 2),))
    assert c.reserve_write(20, key(2, 2), 4096) is not None


def test_claim_cancel_race_never_recycles_claimed_extent() -> None:
    for _ in range(32):
        c = catalog()
        write = c.reserve_write(10, key(1), 4096)
        assert write is not None
        c.publish(10, write.ticket)
        assert c.lookup_prefix("race", (((10, key(1)),),)) == 1
        with ThreadPoolExecutor(max_workers=2) as executor:
            claim = executor.submit(c.claim_reads, "race", 10, (key(1),))
            cancel = executor.submit(c.cancel_lookup, "race")
            cancel.result()
            try:
                reads = claim.result()
            except KeyError:
                assert c.reserve_write(10, key(2), 4096) is not None
            else:
                assert c.reserve_write(10, key(2), 4096) is None
                c.release_read(10, reads[0].ticket)
                assert c.reserve_write(10, key(2), 4096) is not None


def test_reaping_claimed_rank_preserves_other_inflight_read() -> None:
    c = catalog()
    for worker, rank in ((10, 1), (20, 2)):
        write = c.reserve_write(worker, key(1, rank), 4096)
        assert write is not None
        c.publish(worker, write.ticket)
    assert c.lookup_prefix("reap", (((10, key(1)), (20, key(1, 2))),)) == 1
    c.claim_reads("reap", 10, (key(1),))
    read = c.claim_reads("reap", 20, (key(1, 2),))[0]
    c.forget_dead_worker(10)
    assert c.reserve_write(20, key(2, 2), 4096) is None
    c.release_read(20, read.ticket)
    assert c.reserve_write(20, key(2, 2), 4096) is not None
    # Reaping also removes the request even when all pins were already claimed.
    assert c.lookup_prefix("reap", ()) == 0
    c.cancel_lookup("reap")


def test_clear_preserves_pending_writes_and_claimed_reads() -> None:
    c = XDSCatalog()
    c.register(10, tuple(XDSExtent(100, i * 4096, 4096) for i in range(3)))
    writes = [c.reserve_write(10, key(i), 4096) for i in range(3)]
    assert all(write is not None for write in writes)
    c.publish(10, writes[0].ticket)
    c.publish(10, writes[1].ticket)
    assert c.lookup_prefix("clear", (((10, key(0)),),)) == 1
    read = c.claim_reads("clear", 10, (key(0),))[0]
    assert c.clear_idle() == 1
    assert c.acquire_reads(((10, key(1)),)) is None
    c.publish(10, writes[2].ticket)
    assert c.clear_idle() == 1
    c.release_read(10, read.ticket)
    assert c.clear_idle() == 1
    c.cancel_lookup("clear")
    c.unregister(10)


def test_batch_completion_validates_every_ticket_before_mutation() -> None:
    c = XDSCatalog()
    c.register(10, (XDSExtent(100, 0, 4096), XDSExtent(100, 4096, 4096)))
    writes = tuple(c.reserve_write(10, key(i), 4096) for i in range(2))
    assert all(write is not None for write in writes)
    tickets = tuple(write.ticket for write in writes)
    with pytest.raises(KeyError):
        c.publish_many(10, (tickets[0], "unknown"))
    assert c.acquire_reads(((10, key(0)),)) is None
    with pytest.raises(ValueError):
        c.publish_many(10, (tickets[0], tickets[0]))
    c.publish_many(10, tickets)
    reads = c.acquire_reads(((10, key(0)), (10, key(1))))
    assert reads is not None
    with pytest.raises(KeyError):
        c.release_many(10, (reads[0].ticket, "unknown"))
    assert c.clear_idle() == 0
    c.release_many(10, tuple(read.ticket for read in reads))
    assert c.clear_idle() == 2
    c.unregister(10)


def test_shared_owner_death_does_not_invalidate_other_read_completion() -> None:
    catalog = XDSCatalog()
    catalog.register(1, (XDSExtent(1, 0, 4096),))
    catalog.register_reader(2)
    key = ObjectKey(b"hash", "model", 0)
    write = catalog.reserve_write(1, key, 4096)
    catalog.publish(1, write.ticket)
    assert catalog.lookup_shared_prefix("shared", (((1, 2, key),),)) == 1
    read = catalog.claim_reads("shared", 2, (key,))[0]
    catalog.forget_dead_worker(1)
    catalog.release_read(2, read.ticket)
    catalog.unregister(2)


@pytest.mark.parametrize("reader_count", [1, 2, 4])
def test_shared_partial_claim_and_cancel_keep_reader_pins(reader_count: int) -> None:
    """Partial hits and failed claims must preserve independent reader ownership."""
    c = XDSCatalog()
    c.register(1, tuple(XDSExtent(100, i * 4096, 4096) for i in range(3)))
    for reader in range(2, reader_count + 1):
        c.register_reader(reader)
    writes = [c.reserve_write(1, key(i), 4096) for i in range(3)]
    assert all(lease is not None for lease in writes)
    for lease in writes[:2]:
        assert lease is not None
        c.publish(1, lease.ticket)
    chunks = tuple(
        tuple((1, reader, key(i)) for reader in range(1, reader_count + 1))
        for i in range(3)
    )
    assert c.lookup_shared_prefix("a", chunks) == 2
    assert c.lookup_shared_prefix("b", chunks) == 2
    with pytest.raises(KeyError):
        c.claim_reads("a", reader_count, (key(0), key(2)))
    claimed = c.claim_reads("a", reader_count, (key(0),))
    with pytest.raises(KeyError):
        c.claim_reads("a", reader_count, (key(0),))
    c.release_lookup_keys("a", (key(0),))
    c.cancel_lookup("a")
    c.cancel_lookup("b")
    assert c.clear_idle() == 1
    c.release_read(reader_count, claimed[0].ticket)
    assert c.clear_idle() == 1
    with pytest.raises(KeyError):
        c.release_read(reader_count, claimed[0].ticket)
    pending = writes[2]
    assert pending is not None
    c.abort_write(1, pending.ticket)
    for reader in range(1, reader_count + 1):
        c.unregister(reader)
