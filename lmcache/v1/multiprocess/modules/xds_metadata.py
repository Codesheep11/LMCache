# SPDX-License-Identifier: Apache-2.0
"""LMCache MP extension exposing metadata leases without importing HBM."""

# Standard
from dataclasses import dataclass
from threading import RLock
from time import monotonic

# Third Party
import msgspec

# First Party
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleBuildContext,
    server_module_handler,
)
from lmcache.v1.multiprocess.xds_catalog import XDSCatalog
from lmcache.v1.multiprocess.xds_protocol import (
    AcquireReads,
    BindWorker,
    ClaimRange,
    ClaimReads,
    Complete,
    CompleteMany,
    LookupPrefix,
    Register,
    ReserveWrite,
    SchedulerLookup,
    TokenTransfer,
)


def build_server_modules(ctx: ExtServerModuleBuildContext) -> "XDSMetadataModule":
    """Build the metadata extension for LMCache's --server-module loader.

    Args:
        ctx: Extension build context supplied by the MP server.

    Returns:
        A metadata-only module; no xDS connections or device resources open.
    """
    return XDSMetadataModule(ctx.server_context)


@dataclass
class _WorkerLiveness:
    last_seen: float
    ping_proven: bool = False


@dataclass(frozen=True)
class _TokenLookup:
    model_name: str
    world_size: int
    start: int
    end: int
    ranks: dict[int, tuple[int, tuple[ObjectKey, ...]]]


class XDSMetadataModule(InstanceLivenessTarget):
    """Version-1 lease RPCs; worker ownership is enforced by XDSCatalog.

    The standard MP heartbeat reaper invalidates dead connection epochs without
    recycling their extents. Token lookups pin complete rank sets; drained
    workers can reannounce their existing capacity after metadata loss.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._catalog = XDSCatalog()
        self._lock = RLock()
        self._workers: dict[int, _WorkerLiveness] = {}
        self._registrations: dict[int, Register] = {}
        self._schedulers: dict[int, _WorkerLiveness] = {}
        self._lookup_owners: dict[str, int] = {}
        # Canonical hit keys live exactly as long as their scheduler lookup.
        self._token_lookups: dict[str, _TokenLookup] = {}
        self._bindings: dict[tuple[str, int, int], dict[int, BindWorker]] = {}

    @property
    def context(self) -> MPCacheServerContext:
        """Return the MP context for module composition."""
        return self._ctx

    def report_status(self) -> dict[str, object]:
        """Report the supported protocol and data-path owner."""
        return {
            "xds_metadata": {
                "protocol": 1,
                "io_owner": "worker",
                "workers": self.tracked_instance_count(),
            }
        }

    def close(self) -> None:
        """Invalidate metadata without freeing worker-owned device resources."""
        with self._lock:
            for worker in self._workers:
                self._catalog.forget_dead_worker(worker)
            self._workers.clear()
            self._registrations.clear()
            self._bindings.clear()
            self._schedulers.clear()
            self._lookup_owners.clear()
            self._token_lookups.clear()

    def touch_instance(self, instance_id: int) -> None:
        """Refresh a registered worker on PING; unknown epochs are ignored."""
        with self._lock:
            state = self._workers.get(instance_id)
            if state is None:
                state = self._schedulers.get(instance_id)
            if state is not None:
                state.last_seen = monotonic()
                state.ping_proven = True

    def tracked_instance_count(self) -> int:
        """Return the number of connection epochs tracked by the reaper."""
        with self._lock:
            return len(self._workers) + len(self._schedulers)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Invalidate silent workers using MP's heartbeat and startup budgets.

        Args:
            reap_timeout_s: Silence budget after the first PING.
            registration_grace_s: Startup budget for workers which never pinged.

        Returns:
            Reaped connection epochs. No blobs are freed or reassigned: a
            heartbeat timeout does not prove that pending DMA has stopped.
        """
        with self._lock:
            now = monotonic()
            stale = [
                worker
                for worker, state in self._workers.items()
                if now - state.last_seen
                >= (reap_timeout_s if state.ping_proven else registration_grace_s)
            ]
            for worker in stale:
                self._catalog.forget_dead_worker(worker)
                del self._workers[worker]
                self._forget_binding(worker)
            expired_schedulers = [
                owner
                for owner, state in self._schedulers.items()
                if now - state.last_seen
                >= (reap_timeout_s if state.ping_proven else registration_grace_s)
            ]
            for owner in expired_schedulers:
                for lookup, lookup_owner in tuple(self._lookup_owners.items()):
                    if lookup_owner == owner:
                        self._catalog.cancel_lookup(lookup)
                        del self._lookup_owners[lookup]
                        self._token_lookups.pop(lookup, None)
                del self._schedulers[owner]
            return stale + expired_schedulers

    @server_module_handler("xds.v1.register")
    def register(self, payload: bytes) -> bytes:
        """Decode Register, reserve capacity, and return MessagePack nil.

        Raises msgspec.ValidationError or ValueError for invalid capacity.
        """
        request = msgspec.msgpack.decode(payload, type=Register)
        if request.worker < 0:
            raise ValueError("Worker epoch must be nonnegative")
        with self._lock:
            if request.read_only:
                if request.extents:
                    raise ValueError("Read-only worker must not announce SSD capacity")
                self._catalog.register_reader(request.worker)
            else:
                self._catalog.register(request.worker, request.extents)
            self._workers[request.worker] = _WorkerLiveness(monotonic())
            self._registrations[request.worker] = request
        return msgspec.msgpack.encode(None)

    @server_module_handler("xds.v1.ensure_registered")
    def ensure_registered(self, payload: bytes) -> bytes:
        """Reannounce drained worker capacity after a metadata connection loss.

        Decode Register and return nil. An identical live registration keeps
        all entries and pins; missing metadata is recreated empty. Changed
        capacity for a live worker raises ValueError. The caller must serialize
        this operation against DMA and retain the original xDS connection.
        """
        request = msgspec.msgpack.decode(payload, type=Register)
        with self._lock:
            existing = self._registrations.get(request.worker)
            if existing is None:
                return self.register(payload)
            if existing != request:
                raise ValueError("Recovery capacity differs from live registration")
            self._touch_traffic(request.worker)
        return msgspec.msgpack.encode(None)

    @server_module_handler("xds.v1.reserve_write")
    def reserve_write(self, payload: bytes) -> bytes:
        """Decode ReserveWrite and return a MessagePack XDSLease or nil.

        Invalid requests and unknown workers raise validation or key errors.
        """
        request = msgspec.msgpack.decode(payload, type=ReserveWrite)
        self._touch_traffic(request.worker)
        return msgspec.msgpack.encode(
            self._catalog.reserve_write(request.worker, request.key, request.size)
        )

    @server_module_handler("xds.v1.publish")
    def publish(self, payload: bytes) -> bytes:
        """Decode Complete and publish successful DMA; return MessagePack nil.

        Raises KeyError for stale or foreign write tickets.
        """
        request = msgspec.msgpack.decode(payload, type=Complete)
        self._touch_traffic(request.worker)
        self._catalog.publish(request.worker, request.ticket)
        return msgspec.msgpack.encode(None)

    @server_module_handler("xds.v1.abort_write")
    def abort_write(self, payload: bytes) -> bytes:
        """Decode Complete, discard a drained write, and return MessagePack nil.

        The worker must drain all DMA first. Stale tickets raise KeyError.
        """
        request = msgspec.msgpack.decode(payload, type=Complete)
        self._touch_traffic(request.worker)
        self._catalog.abort_write(request.worker, request.ticket)
        return msgspec.msgpack.encode(None)

    @server_module_handler("xds.v1.acquire_reads")
    def acquire_reads(self, payload: bytes) -> bytes:
        """Decode AcquireReads and return MessagePack read leases or nil.

        A miss pins nothing. Invalid payloads raise msgspec.ValidationError.
        """
        request = msgspec.msgpack.decode(payload, type=AcquireReads)
        return msgspec.msgpack.encode(self._catalog.acquire_reads(request.objects))

    @server_module_handler("xds.v1.lookup_prefix")
    def lookup_prefix(self, payload: bytes) -> bytes:
        """Decode LookupPrefix and return the pinned complete chunk count.

        Duplicate request epochs or objects raise ValueError. No KV payloads
        are transferred and scheduler traffic does not refresh worker liveness.
        """
        request = msgspec.msgpack.decode(payload, type=LookupPrefix)
        return msgspec.msgpack.encode(
            self._catalog.lookup_prefix(request.lookup, request.chunks)
        )

    @server_module_handler("xds.v1.claim_reads")
    def claim_reads(self, payload: bytes) -> bytes:
        """Decode ClaimReads and transfer pinned leases to the owning worker.

        Unknown or previously claimed pins raise KeyError without partial claims.
        """
        request = msgspec.msgpack.decode(payload, type=ClaimReads)
        self._touch_traffic(request.worker)
        return msgspec.msgpack.encode(
            self._catalog.claim_reads(request.lookup, request.worker, request.keys)
        )

    @server_module_handler("xds.v1.cancel_lookup")
    def cancel_lookup(self, payload: bytes) -> bytes:
        """Decode a request epoch and release its unclaimed pins, returning nil.

        Worker-owned DMA leases remain pinned. Unknown epochs are harmless.
        """
        lookup = msgspec.msgpack.decode(payload, type=str)
        with self._lock:
            self._catalog.cancel_lookup(lookup)
            self._lookup_owners.pop(lookup, None)
            self._token_lookups.pop(lookup, None)
        return msgspec.msgpack.encode(None)

    @server_module_handler("xds.v1.release_read")
    def release_read(self, payload: bytes) -> bytes:
        """Decode Complete and unpin drained read DMA; return MessagePack nil.

        Stale or foreign read tickets raise KeyError.
        """
        request = msgspec.msgpack.decode(payload, type=Complete)
        self._touch_traffic(request.worker)
        self._catalog.release_read(request.worker, request.ticket)
        return msgspec.msgpack.encode(None)

    @server_module_handler("xds.v1.unregister")
    def unregister(self, payload: bytes) -> bytes:
        """Decode a worker ID and forget drained capacity; return MessagePack nil.

        Outstanding leases raise RuntimeError; an unknown worker raises KeyError.
        """
        worker = msgspec.msgpack.decode(payload, type=int)
        with self._lock:
            self._catalog.unregister(worker)
            del self._workers[worker]
            self._forget_binding(worker)
        return msgspec.msgpack.encode(None)

    @server_module_handler("xds.v1.bind_worker")
    def bind_worker(self, payload: bytes) -> bytes:
        """Bind a physical reader to a model/KV rank; allow exactly one writer.

        Payload is BindWorker. Returns nil; invalid geometry raises ValueError.
        An identical binding is idempotent for recovery retries. Existing
        connection epochs must unregister before a replacement binds.
        """
        request = msgspec.msgpack.decode(payload, type=BindWorker)
        if (
            request.world_size <= 0
            or not 0 <= request.worker_id < request.world_size
            or request.chunk_bytes <= 0
            or request.reader_id < 0
            or request.num_readers <= 0
        ):
            raise ValueError("Invalid worker geometry")
        rank = ObjectKey.ComputeKVRank(
            world_size=request.world_size,
            global_rank=request.worker_id,
            local_world_size=request.world_size,
            local_rank=request.worker_id,
        )
        identity = (request.model_name, request.world_size, rank)
        with self._lock:
            self._touch_traffic(request.worker)
            group = self._bindings.get(identity, {})
            if group.get(request.reader_id) == request:
                return msgspec.msgpack.encode(None)
            if request.reader_id in group:
                raise ValueError("Model rank reader already bound")
            if any(
                b.worker == request.worker
                for g in self._bindings.values()
                for b in g.values()
            ):
                raise ValueError("Worker epoch already bound")
            if request.writer == self._registrations[request.worker].read_only:
                raise ValueError("Writer binding differs from registered capacity")
            if group and (
                len(group) >= request.num_readers
                or any(
                    b.num_readers != request.num_readers
                    or b.chunk_bytes != request.chunk_bytes
                    for b in group.values()
                )
                or (request.writer and any(b.writer for b in group.values()))
            ):
                raise ValueError("Conflicting replica geometry or writer")
            self._bindings.setdefault(identity, {})[request.reader_id] = request
        return msgspec.msgpack.encode(None)

    @server_module_handler("xds.v1.lookup_owned_tokens")
    def lookup_owned_tokens(self, payload: bytes) -> bytes:
        """Pin tokens under a negative scheduler epoch refreshed by PING.

        Decode SchedulerLookup and return a chunk count. Duplicate request IDs
        and nonnegative scheduler IDs raise ValueError. Reaping releases only
        unclaimed pins; worker-owned DMA tickets retain their lifetime.
        """
        request = msgspec.msgpack.decode(payload, type=SchedulerLookup)
        if request.scheduler >= 0:
            raise ValueError("Scheduler epoch must be negative")
        with self._lock:
            lookup = request.key.request_id
            if lookup in self._lookup_owners:
                raise ValueError("Lookup request already owned")
            result = self._lookup_token_key(request.key)
            state = self._schedulers.setdefault(
                request.scheduler, _WorkerLiveness(monotonic())
            )
            state.last_seen = monotonic()
            self._lookup_owners[lookup] = request.scheduler
            return result

    @server_module_handler("xds.v1.lookup_tokens")
    def lookup_tokens(self, payload: bytes) -> bytes:
        """Hash a scheduler IPCCacheServerKey and pin its complete rank prefix.

        Returns a chunk count. Missing worker bindings or incomplete replica
        cohorts are a miss. Shared MLA readers each receive independent pins.
        """
        key = msgspec.msgpack.decode(payload, type=IPCCacheServerKey)
        return self._lookup_token_key(key)

    @server_module_handler("xds.v1.reserve_tokens")
    def reserve_tokens(self, payload: bytes) -> bytes:
        """Reserve worker-owned extents for each chunk in a TokenTransfer.

        Returns one optional write lease per chunk; existing keys or exhausted
        capacity produce nil entries. Only completed writes may be published.
        """
        request = msgspec.msgpack.decode(payload, type=TokenTransfer)
        keys = self._token_keys(request.key)
        with self._lock:
            binding = self._transfer_binding(request)
            if not binding.writer:
                raise ValueError("Read-only replica cannot reserve writes")
            leases = tuple(
                self._catalog.reserve_write(request.worker, key, binding.chunk_bytes)
                for key in keys
            )
        return msgspec.msgpack.encode(leases)

    @server_module_handler("xds.v1.claim_tokens")
    def claim_tokens(self, payload: bytes) -> bytes:
        """Claim the scheduler's exact token-range pins for their owning worker.

        Returns ordered read leases. Missing pins fail atomically with KeyError.
        """
        request = msgspec.msgpack.decode(payload, type=TokenTransfer)
        keys = self._token_keys(request.key)
        with self._lock:
            self._transfer_binding(request)
            leases = self._catalog.claim_reads(
                request.key.request_id, request.worker, tuple(keys)
            )
        return msgspec.msgpack.encode(leases)

    @server_module_handler("xds.v1.claim_range")
    def claim_range(self, payload: bytes) -> bytes:
        """Claim indexed scheduler pins without retransmitting or hashing tokens.

        Args:
            payload: MessagePack ClaimRange referencing a successful token lookup.

        Returns:
            MessagePack ordered read leases; ownership transfers to the worker.

        Raises:
            ValueError: Wrong model, rank, epoch or unaligned/out-of-hit range.
            KeyError: Expired lookup or missing/already claimed pins. A failed
                claim never transfers only part of the requested range.
        """
        request = msgspec.msgpack.decode(payload, type=ClaimRange)
        with self._lock:
            lookup = self._token_lookups[request.lookup]
            chunk_size = self._ctx.chunk_size
            if (
                request.model_name != lookup.model_name
                or request.world_size != lookup.world_size
                or not 0 <= request.worker_id < lookup.world_size
                or not lookup.start <= request.start < request.end <= lookup.end
                or request.start % chunk_size
                or request.end % chunk_size
            ):
                raise ValueError("Claim identity or range differs from pinned lookup")
            if request.worker not in lookup.ranks:
                raise ValueError("Claim belongs to a different connection epoch")
            rank, keys = lookup.ranks[request.worker]
            if rank != request.worker_id:
                raise ValueError("Claim belongs to a different KV rank")
            worker = request.worker
            self._touch_traffic(worker)
            first = (request.start - lookup.start) // chunk_size
            last = (request.end - lookup.start) // chunk_size
            leases = self._catalog.claim_reads(request.lookup, worker, keys[first:last])
        return msgspec.msgpack.encode(leases)

    @server_module_handler("xds.v1.release_tokens")
    def release_tokens(self, payload: bytes) -> bytes:
        """Release only unclaimed pins in an aligned scheduler token range.

        Payload is IPCCacheServerKey; returns nil. Worker DMA pins are retained.
        """
        key = msgspec.msgpack.decode(payload, type=IPCCacheServerKey)
        keys = self._token_keys(key)
        self._catalog.release_lookup_keys(key.request_id, tuple(keys))
        return msgspec.msgpack.encode(None)

    @server_module_handler("xds.v1.clear_idle")
    def clear_idle(self, payload: bytes) -> bytes:
        """Decode nil and evict only completed, unpinned cache objects.

        Returns the removed object count. Forced eviction is not supported.
        """
        msgspec.msgpack.decode(payload, type=type(None))
        return msgspec.msgpack.encode(self._catalog.clear_idle())

    @server_module_handler("xds.v1.publish_many")
    def publish_many(self, payload: bytes) -> bytes:
        """Decode CompleteMany and atomically publish successful DMA writes.

        Invalid tickets fail the whole batch. Returns MessagePack nil.
        """
        request = msgspec.msgpack.decode(payload, type=CompleteMany)
        self._touch_traffic(request.worker)
        self._catalog.publish_many(request.worker, request.tickets)
        return msgspec.msgpack.encode(None)

    @server_module_handler("xds.v1.release_many")
    def release_many(self, payload: bytes) -> bytes:
        """Decode CompleteMany and release a drained read batch atomically.

        Invalid tickets leave all pins in place. Returns MessagePack nil.
        """
        request = msgspec.msgpack.decode(payload, type=CompleteMany)
        self._touch_traffic(request.worker)
        self._catalog.release_many(request.worker, request.tickets)
        return msgspec.msgpack.encode(None)

    def _token_keys(self, key: IPCCacheServerKey) -> list[ObjectKey]:
        if key.num_kv_readers <= 0 or key.world_size <= 0:
            raise ValueError("xDS requires explicit positive KV readers and world size")
        if (
            key.start < 0
            or key.end < key.start
            or key.end > len(key.token_ids)
            or key.start % self._ctx.chunk_size
            or key.end % self._ctx.chunk_size
        ):
            raise ValueError("Token range must contain complete cache chunks")
        hashes = self._ctx.token_hasher.compute_chunk_hashes(
            list(key.token_ids), start=key.start, end=key.end
        )
        return ipc_key_to_object_keys(key, hashes, [0])[0]

    def _transfer_binding(self, request: TokenTransfer) -> BindWorker:
        key = request.key
        if key.worker_id is None or not 0 <= key.worker_id < key.world_size:
            raise ValueError("Transfer requires a valid worker rank")
        rank = ObjectKey.ComputeKVRank(
            world_size=key.world_size,
            global_rank=key.worker_id,
            local_world_size=key.world_size,
            local_rank=key.worker_id,
        )
        group = self._bindings[(key.model_name, key.world_size, rank)]
        binding = next((b for b in group.values() if b.worker == request.worker), None)
        if binding is None:
            raise ValueError("Transfer belongs to a different connection epoch")
        if binding.num_readers != key.num_kv_readers:
            raise ValueError("Transfer reader count differs from binding")
        self._touch_traffic(request.worker)
        return binding

    def _forget_binding(self, worker: int) -> None:
        self._registrations.pop(worker, None)
        for lookup, index in tuple(self._token_lookups.items()):
            if worker in index.ranks:
                del self._token_lookups[lookup]
        for identity, group in tuple(self._bindings.items()):
            for reader_id, binding in tuple(group.items()):
                if binding.worker == worker:
                    del group[reader_id]
            if not group:
                del self._bindings[identity]

    def _touch_traffic(self, worker: int) -> None:
        with self._lock:
            self._workers[worker].last_seen = monotonic()

    def _lookup_token_key(self, key: IPCCacheServerKey) -> bytes:
        """Resolve each rank cohort once, then pin its token-ordered reader prefix."""
        if key.worker_id is not None:
            raise ValueError("Scheduler lookup must address every rank")
        keys = self._token_keys(key)
        with self._lock:
            chunks = []
            worker_ranks: dict[int, int] = {}
            cohorts: dict[int, tuple[int, tuple[BindWorker, ...]]] = {}
            # Bindings are stable under this lock and identical across chunks.
            for obj in keys[: key.world_size]:
                group = self._bindings.get(
                    (key.model_name, key.world_size, obj.kv_rank), {}
                )
                readers = tuple(group.values())
                writers = [b for b in readers if b.writer]
                if (
                    len(readers) != key.num_kv_readers
                    or len(writers) != 1
                    or any(b.num_readers != key.num_kv_readers for b in readers)
                ):
                    return msgspec.msgpack.encode(0)
                cohorts[obj.kv_rank] = (writers[0].worker, readers)
                for binding in readers:
                    worker_ranks[binding.worker] = binding.worker_id
            for offset in range(0, len(keys), key.world_size):
                chunk = []
                for obj in keys[offset : offset + key.world_size]:
                    owner, readers = cohorts[obj.kv_rank]
                    for binding in readers:
                        chunk.append((owner, binding.worker, obj))
                chunks.append(tuple(chunk))
            if key.num_kv_readers == 1:
                hits = self._catalog.lookup_prefix(
                    key.request_id,
                    tuple(
                        tuple((reader, obj) for _, reader, obj in chunk)
                        for chunk in chunks
                    ),
                )
            else:
                hits = self._catalog.lookup_shared_prefix(key.request_id, tuple(chunks))
            ranks: dict[int, tuple[int, tuple[ObjectKey, ...]]] = {}
            if hits:
                for position, (_, worker, _) in enumerate(chunks[0]):
                    ranks[worker] = (
                        worker_ranks[worker],
                        tuple(chunk[position][2] for chunk in chunks[:hits]),
                    )
                self._token_lookups[key.request_id] = _TokenLookup(
                    key.model_name,
                    key.world_size,
                    key.start,
                    key.start + hits * self._ctx.chunk_size,
                    ranks,
                )
        return msgspec.msgpack.encode(hits)
