# SPDX-License-Identifier: Apache-2.0
"""Metadata leases for SSD extents owned by an engine worker's xDS connection.

No HBM pointers, IPC handles, or KV payloads enter this catalog. A connection
epoch is represented by a fresh worker ID; an ID must never be reused after
disconnect. All methods are thread safe. Transport and liveness integration
are separate from this state machine.
"""

# Standard
from collections import OrderedDict
from dataclasses import dataclass, field
from secrets import token_hex
from threading import RLock

# First Party
from lmcache.v1.distributed.api import ObjectKey


@dataclass(frozen=True)
class XDSExtent:
    """Owner-held SSD range with an optional read-only export token; sizes are bytes."""

    blob: int
    offset: int
    size: int
    read_token: int = 0


@dataclass(frozen=True)
class XDSLease:
    """Opaque 128-bit random completion ticket and its reserved SSD range."""

    ticket: str
    extent: XDSExtent


@dataclass
class _Entry:
    extent: XDSExtent
    ready: bool = False
    readers: set[str] = field(default_factory=set)


@dataclass
class _Worker:
    # The worker retains its xDS blob leases until all catalog leases drain.
    free: list[XDSExtent]
    entries: OrderedDict[ObjectKey, _Entry] = field(default_factory=OrderedDict)
    writes: dict[str, ObjectKey] = field(default_factory=dict)
    reads: dict[str, _Entry] = field(default_factory=dict)


@dataclass
class _Lookup:
    workers: frozenset[int]
    # Reader-local maps avoid allocating (reader, key) tuples on every claim.
    pins: dict[int, dict[ObjectKey, XDSLease]] = field(default_factory=dict)


class XDSCatalog:
    """Bounded per-worker LRU with explicit write publication and read pins.

    A timed-out RPC or heartbeat is not evidence of DMA completion. Callers
    must drain I/O before aborting a write or releasing a read. A dead worker
    can instead be forgotten: its extents are discarded, never reassigned to
    another connection. xDS remains responsible for connection teardown.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._workers: dict[int, _Worker] = {}
        # Only unclaimed pins live here. Claimed pins belong to worker DMA.
        self._lookups: dict[str, _Lookup] = {}

    def register(self, worker: int, extents: tuple[XDSExtent, ...]) -> None:
        """Register nonoverlapping worker-owned capacity once per connection.

        Args:
            worker: Unique connection epoch, not a reusable rank or PID.
            extents: Nonempty SSD ranges; the caller validates xDS alignment.

        Raises:
            ValueError: For duplicate workers, empty or overlapping capacity.
        """
        if not extents:
            raise ValueError("Worker capacity cannot be empty")
        previous: XDSExtent | None = None
        for extent in sorted(extents, key=lambda e: (e.blob, e.offset)):
            if extent.blob <= 0 or extent.offset < 0 or extent.size <= 0:
                raise ValueError("Invalid SSD extent")
            if (
                previous is not None
                and previous.blob == extent.blob
                and previous.offset + previous.size > extent.offset
            ):
                raise ValueError("Overlapping SSD extents")
            previous = extent
        with self._lock:
            if worker in self._workers:
                raise ValueError("Worker already registered")
            self._workers[worker] = _Worker(list(extents))

    def register_reader(self, worker: int) -> None:
        """Register a read-only connection without allocating SSD capacity.

        Args:
            worker: Fresh connection epoch.

        Raises:
            ValueError: If the epoch is already registered.
        """
        with self._lock:
            if worker in self._workers:
                raise ValueError("Worker already registered")
            self._workers[worker] = _Worker([])

    def reserve_write(self, worker: int, key: ObjectKey, size: int) -> XDSLease | None:
        """Reserve an exact-size extent, evicting the oldest unpinned entry.

        Args:
            worker: Registered connection epoch.
            key: Content identity including model, rank, group and salt.
            size: Serialized KV object size in bytes.

        Returns:
            A completion lease, or None if the key already exists (including
            an unfinished write) or no unpinned capacity of this size exists.

        Raises:
            KeyError: If the worker is not registered.
            ValueError: If size is not positive.
        """
        if size <= 0:
            raise ValueError("Object size must be positive")
        with self._lock:
            state = self._workers[worker]
            if key in state.entries:
                return None
            extent = next((e for e in state.free if e.size == size), None)
            if extent is not None:
                state.free.remove(extent)
            else:
                victim = next(
                    (
                        k
                        for k, entry in state.entries.items()
                        if entry.ready
                        and not entry.readers
                        and entry.extent.size == size
                    ),
                    None,
                )
                if victim is None:
                    return None
                extent = state.entries.pop(victim).extent
            ticket = token_hex(16)
            state.entries[key] = _Entry(extent)
            state.writes[ticket] = key
            return XDSLease(ticket, extent)

    def publish(self, worker: int, ticket: str) -> None:
        """Publish a write only after successful completion of all its DMA.

        Args:
            worker: Registered connection epoch.
            ticket: Outstanding write ticket returned by reserve_write.

        Raises:
            KeyError: For stale, already completed, or foreign tickets/workers.
        """
        with self._lock:
            state = self._workers[worker]
            key = state.writes.pop(ticket)
            state.entries[key].ready = True
            state.entries.move_to_end(key)

    def abort_write(self, worker: int, ticket: str) -> None:
        """Discard a failed write after the worker has drained its DMA.

        Args:
            worker: Registered connection epoch.
            ticket: Outstanding write ticket; RPC timeout alone cannot abort it.

        Raises:
            KeyError: For an unknown worker or write ticket.
        """
        with self._lock:
            state = self._workers[worker]
            key = state.writes.pop(ticket)
            state.free.append(state.entries.pop(key).extent)

    def acquire_reads(
        self, objects: tuple[tuple[int, ObjectKey], ...]
    ) -> tuple[XDSLease, ...] | None:
        """Atomically pin all requested objects, including across TP workers.

        Args:
            objects: Ordered (connection epoch, object key) pairs.

        Returns:
            Matching read leases, or None if any worker/key is absent or has
            not completed its write. A miss acquires no leases. Empty input
            returns an empty tuple. Pins remain until release_read, including
            between scheduler lookup and worker submission.
        """
        with self._lock:
            entries = []
            for worker, key in objects:
                state = self._workers.get(worker)
                if state is None:
                    return None
                entry = state.entries.get(key)
                if entry is None or not entry.ready:
                    return None
                entries.append((state, key, entry))
            leases = []
            for state, key, entry in entries:
                ticket = token_hex(16)
                entry.readers.add(ticket)
                state.reads[ticket] = entry
                state.entries.move_to_end(key)
                leases.append(XDSLease(ticket, entry.extent))
            return tuple(leases)

    def release_read(self, worker: int, ticket: str) -> None:
        """Unpin an unused lookup lease or a fully completed read.

        Args:
            worker: Registered connection epoch.
            ticket: Outstanding read ticket returned by acquire_reads.

        Raises:
            KeyError: For an unknown worker or read ticket.
        """
        with self._lock:
            state = self._workers[worker]
            entry = state.reads.pop(ticket)
            entry.readers.remove(ticket)

    def publish_many(self, worker: int, tickets: tuple[str, ...]) -> None:
        """Publish an entire completed batch atomically after DMA drains.

        Args:
            worker: Registered connection epoch.
            tickets: Distinct outstanding write tickets; an empty batch is valid.

        Raises:
            KeyError: For any unknown ticket/worker, without publishing any write.
            ValueError: For duplicate tickets.
        """
        if len(set(tickets)) != len(tickets):
            raise ValueError("Duplicate completion tickets")
        with self._lock:
            state = self._workers[worker]
            for ticket in tickets:
                if ticket not in state.writes:
                    raise KeyError(ticket)
            for ticket in tickets:
                self.publish(worker, ticket)

    def release_many(self, worker: int, tickets: tuple[str, ...]) -> None:
        """Release a completed read batch, validating every ticket first.

        Args:
            worker: Registered connection epoch.
            tickets: Distinct read tickets whose DMA has drained.

        Raises:
            KeyError: For any unknown ticket/worker, without releasing any pin.
            ValueError: For duplicate tickets.
        """
        if len(set(tickets)) != len(tickets):
            raise ValueError("Duplicate completion tickets")
        with self._lock:
            state = self._workers[worker]
            for ticket in tickets:
                if ticket not in state.reads:
                    raise KeyError(ticket)
            for ticket in tickets:
                self.release_read(worker, ticket)

    def lookup_prefix(
        self, lookup: str, chunks: tuple[tuple[tuple[int, ObjectKey], ...], ...]
    ) -> int:
        """Pin the longest complete prefix across all requested ranks/groups.

        Args:
            lookup: Unique scheduler request epoch, retained until cancellation.
            chunks: Token-ordered chunks, each containing all required shards.
                Keys must be distinct across the entire request.

        Returns:
            Number of consecutive fully available chunks. A missing shard pins
            none of its chunk; later chunks are not inspected.

        Raises:
            ValueError: For a reused lookup ID, empty chunk, or duplicate key.
        """
        objects = [obj for chunk in chunks for obj in chunk]
        if not lookup or any(not chunk for chunk in chunks):
            raise ValueError("Lookup ID and each chunk must be nonempty")
        if len(set(objects)) != len(objects):
            raise ValueError("Duplicate objects in lookup")
        with self._lock:
            if lookup in self._lookups:
                raise ValueError("Lookup already exists")
            pins: dict[int, dict[ObjectKey, XDSLease]] = {}
            self._lookups[lookup] = _Lookup(frozenset(w for w, _ in objects), pins)
            hits = 0
            for chunk in chunks:
                leases = self.acquire_reads(chunk)
                if leases is None:
                    break
                for (worker, key), lease in zip(chunk, leases, strict=True):
                    pins.setdefault(worker, {})[key] = lease
                hits += 1
            return hits

    def lookup_shared_prefix(
        self, lookup: str, chunks: tuple[tuple[tuple[int, int, ObjectKey], ...], ...]
    ) -> int:
        """Pin a complete prefix, with one independent ticket per physical reader.

        Generate independent random tickets in one batch and index pins by
        reader, preserving per-reader claims and cancellation ownership.

        Args:
            lookup: Unique scheduler request epoch.
            chunks: Token-ordered (SSD owner, reader epoch, canonical key) groups.
                A reader/key pair must appear only once in the whole request.

        Returns:
            Complete matched chunk count. Missing owners/readers/shards stop the
            prefix without retaining any pins for the incomplete chunk.

        Raises:
            ValueError: Duplicate request, empty chunk or duplicate reader/key.
        """
        objects = [obj for chunk in chunks for obj in chunk]
        identities = [(reader, key) for _, reader, key in objects]
        if not lookup or any(not chunk for chunk in chunks):
            raise ValueError("Lookup ID and each chunk must be nonempty")
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate reader objects in lookup")
        with self._lock:
            if lookup in self._lookups:
                raise ValueError("Lookup already exists")
            pins: dict[int, dict[ObjectKey, XDSLease]] = {
                reader: {} for reader in {reader for _, reader, _ in objects}
            }
            workers = frozenset(
                w for owner, reader, _ in objects for w in (owner, reader)
            )
            self._lookups[lookup] = _Lookup(workers, pins)
            hits = 0
            tickets = ""
            ticket_offset = 0
            for chunk in chunks:
                entries = []
                for owner, reader, key in chunk:
                    source = self._workers.get(owner)
                    target = self._workers.get(reader)
                    if source is None or target is None:
                        return hits
                    entry = source.entries.get(key)
                    if entry is None or not entry.ready:
                        return hits
                    entries.append((source, target, reader, key, entry))
                if hits == 0:
                    tickets = token_hex(16 * len(objects))
                for source, target, reader, key, entry in entries:
                    ticket = tickets[ticket_offset : ticket_offset + 32]
                    ticket_offset += 32
                    entry.readers.add(ticket)
                    target.reads[ticket] = entry
                    source.entries.move_to_end(key)
                    pins[reader][key] = XDSLease(ticket, entry.extent)
                hits += 1
            return hits

    def claim_reads(
        self, lookup: str, worker: int, keys: tuple[ObjectKey, ...]
    ) -> tuple[XDSLease, ...]:
        """Transfer exact lookup pins from its reader map without an eviction gap.

        Args:
            lookup: Request epoch supplied to lookup_prefix.
            worker: Connection epoch claiming its independently pinned read tickets.
            keys: Ordered keys for this transfer; duplicate keys are invalid.

        Returns:
            Leases in key order. The worker must release them after DMA drains.
            Scheduler cancellation cannot release a claimed lease.

        Raises:
            KeyError: If the lookup or any requested pin is absent/claimed.
                Failure leaves all remaining pins unchanged.
            ValueError: If keys contains duplicates or is empty.
        """
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Claim requires distinct, nonempty keys")
        with self._lock:
            pins = self._lookups[lookup].pins[worker]
            leases = tuple(pins[key] for key in keys)
            for key in keys:
                del pins[key]
            return leases

    def release_lookup_keys(self, lookup: str, keys: tuple[ObjectKey, ...]) -> None:
        """Release unclaimed pins in a scheduler range; claimed DMA stays pinned.

        Args:
            lookup: Request epoch; absent epochs are ignored.
            keys: Content keys to release across their owning workers.
        """
        selected = set(keys)
        with self._lock:
            state = self._lookups.get(lookup)
            if state is None:
                return
            for worker, pins in state.pins.items():
                for key in selected.intersection(pins):
                    self.release_read(worker, pins.pop(key).ticket)

    def cancel_lookup(self, lookup: str) -> None:
        """Release only scheduler-owned pins and forget the request epoch.

        Args:
            lookup: Request epoch; missing epochs are harmless.

        Claimed pins remain live until their worker reports DMA completion.
        This operation is safe to repeat when cancelling a request.
        """
        with self._lock:
            state = self._lookups.pop(lookup, None)
            if state is not None:
                for worker, pins in state.pins.items():
                    for lease in pins.values():
                        self.release_read(worker, lease.ticket)

    def clear_idle(self) -> int:
        """Evict completed unpinned entries, preserving all outstanding I/O.

        Returns:
            Number of removed cache objects. Worker-owned capacity remains
            registered and can be reused by subsequent writes.
        """
        removed = 0
        with self._lock:
            for state in self._workers.values():
                for key, entry in tuple(state.entries.items()):
                    if entry.ready and not entry.readers:
                        state.free.append(entry.extent)
                        del state.entries[key]
                        removed += 1
        return removed

    def unregister(self, worker: int) -> None:
        """Forget a gracefully drained worker before it releases its blobs.

        Args:
            worker: Registered connection epoch.

        Raises:
            KeyError: If the worker is not registered.
            RuntimeError: If read or write leases remain outstanding.
        """
        with self._lock:
            state = self._workers[worker]
            if (
                state.reads
                or state.writes
                or any(e.readers for e in state.entries.values())
            ):
                raise RuntimeError("Worker still has outstanding I/O leases")
            self._cancel_worker_lookups(worker)
            del self._workers[worker]

    def forget_dead_worker(self, worker: int) -> None:
        """Invalidate all metadata without recycling a dead connection's SSD.

        Args:
            worker: Connection epoch to invalidate; unknown epochs are no-ops.

        This never frees blobs or transfers capacity to another worker. Late
        completions fail because the epoch no longer exists. A replacement
        worker must obtain fresh xDS leases and register a fresh epoch.
        """
        with self._lock:
            self._cancel_worker_lookups(worker)
            self._workers.pop(worker, None)

    def _cancel_worker_lookups(self, worker: int) -> None:
        """Cancel affected lookups while holding the catalog lock."""
        affected = [
            lookup for lookup, state in self._lookups.items() if worker in state.workers
        ]
        for lookup in affected:
            self.cancel_lookup(lookup)
