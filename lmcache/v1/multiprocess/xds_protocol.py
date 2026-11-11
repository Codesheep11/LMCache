# SPDX-License-Identifier: Apache-2.0
"""Versioned metadata-only messages for the worker-owned xDS MP path."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.xds_catalog import XDSExtent


@dataclass(frozen=True)
class Register:
    """Announce capacity leased by one worker's xDS connection."""

    worker: int
    extents: tuple[XDSExtent, ...]
    read_only: bool = False


@dataclass(frozen=True)
class ReserveWrite:
    """Reserve SSD capacity for one rank's fully described KV object."""

    worker: int
    key: ObjectKey
    size: int


@dataclass(frozen=True)
class Complete:
    """Complete an outstanding metadata lease after draining local DMA."""

    worker: int
    ticket: str


@dataclass(frozen=True)
class AcquireReads:
    """Atomically pin all listed ranks; a partial hit acquires nothing."""

    objects: tuple[tuple[int, ObjectKey], ...]


@dataclass(frozen=True)
class LookupPrefix:
    """Pin consecutive complete chunks for one scheduler request epoch."""

    lookup: str
    chunks: tuple[tuple[tuple[int, ObjectKey], ...], ...]


@dataclass(frozen=True)
class ClaimReads:
    """Hand scheduler pins to the worker that owns the corresponding blobs."""

    lookup: str
    worker: int
    keys: tuple[ObjectKey, ...]


@dataclass(frozen=True)
class BindWorker:
    """Bind a physical reader epoch to a logical KV rank and its writer cohort."""

    worker: int
    model_name: str
    world_size: int
    worker_id: int
    chunk_bytes: int
    reader_id: int = 0
    num_readers: int = 1
    writer: bool = True


@dataclass(frozen=True)
class TokenTransfer:
    """Resolve engine token keys using the server's canonical token hasher."""

    worker: int
    key: IPCCacheServerKey


@dataclass(frozen=True)
class CompleteMany:
    """Complete a drained batch with one metadata round trip."""

    worker: int
    tickets: tuple[str, ...]


@dataclass(frozen=True)
class SchedulerLookup:
    """Associate a token lookup with a heartbeat-tracked scheduler epoch."""

    scheduler: int
    key: IPCCacheServerKey


@dataclass(frozen=True)
class ClaimRange:
    """Claim an aligned subrange of an existing scheduler request's pinned prefix.

    Token identity (including salt) is fixed by lookup; request IDs must be
    unique within the scheduler session. Worker identifies a connection epoch.
    """

    worker: int
    lookup: str
    model_name: str
    world_size: int
    worker_id: int
    start: int
    end: int
