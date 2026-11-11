# SPDX-License-Identifier: Apache-2.0
"""One SSD reader per MLA shard, followed by direct final-page broadcasts."""

# Standard
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol
import os

# Third Party
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ReplicaGroups:
    """Existing replica groups; group creation and lifetime belong to the engine.

    The control group must be CPU/Gloo; the data group is NCCL or HCCL with
    identical membership and rank order. One group contains exactly one logical
    KV shard's physical readers. DCP/multi-server subsets require explicit groups.
    """

    data: dist.ProcessGroup
    control: dist.ProcessGroup


class PendingWork(Protocol):
    """Device collective completion handle."""

    def wait(self) -> bool:
        """Order completion before the consumer stream proceeds."""
        ...


@dataclass(frozen=True)
class ReplicaPlan:
    """CPU-only logical request and local final-page geometry."""

    identity: tuple[object, ...]
    lengths: tuple[int, ...]
    skip: int
    geometry: tuple[tuple[tuple[int, ...], str], ...]
    writer: bool
    blocks: tuple[int, ...]


@dataclass(frozen=True)
class BroadcastRun:
    """Contiguous final-page views containing corresponding KV on every rank."""

    starts: tuple[int, ...]
    blocks: int


def plan_broadcast_runs(
    block_maps: Sequence[Sequence[int]], start: int, end: int
) -> list[BroadcastRun]:
    """Coalesce corresponding final pages independently of logical page order.

    Args:
        block_maps: Unique, nonnegative final-page IDs for the same logical
            blocks on every rank; alias validation belongs to request admission.
        start: Inclusive logical block position, after any protected prefix.
        end: Exclusive logical block position within the completed SSD batch.

    Returns:
        Deterministically ordered contiguous views. Each selected logical block
        appears exactly once on every rank. Only descriptors are reordered;
        neither payload buffers nor copies are created. Runs never cross batch
        boundaries or include protected/unselected pages.

    Raises:
        ValueError: Empty maps, mismatched lengths, or invalid interval.

    Sorting by rank zero's addresses exposes all possible adjacent source pages.
    A run extends only if every other rank's corresponding address also advances
    by one. This handles shared reverse/permuted order without assuming that
    different ranks have identical block IDs or identical relative layouts.
    """
    if not block_maps or any(len(m) != len(block_maps[0]) for m in block_maps):
        raise ValueError("Replica block maps must have equal lengths")
    if not 0 <= start <= end <= len(block_maps[0]):
        raise ValueError("Invalid replica block interval")
    if start == end:
        return []
    order = sorted(range(start, end), key=block_maps[0].__getitem__)
    runs = []
    first = previous = order[0]
    length = 1
    for current in order[1:]:
        if all(m[current] == m[previous] + 1 for m in block_maps):
            length += 1
        else:
            runs.append(BroadcastRun(tuple(m[first] for m in block_maps), length))
            first = current
            length = 1
        previous = current
    runs.append(BroadcastRun(tuple(m[first] for m in block_maps), length))
    return runs


class ReplicaBroadcast:
    """Pipeline leader SSD reads with NCCL/HCCL writes into final KV views.

    No KV payload staging allocation is made. Small CPU control messages align
    requests and block maps. At most one broadcast batch is outstanding while
    a dedicated reader prefetches up to two disjoint batches. Callers retain pages
    and leases on any error; communication failures are fatal to this transfer context.
    """

    def __init__(
        self,
        groups: ReplicaGroups,
        tensors: tuple[torch.Tensor, ...],
        writer: bool,
        readers: int,
        synchronize: Callable[[], None],
    ) -> None:
        self.groups = groups
        self.tensors = tensors
        self.writer = writer
        self.synchronize = synchronize
        self.readers = readers
        self.rank = dist.get_rank(groups.data)
        if (
            dist.get_world_size(groups.data) != readers
            or dist.get_world_size(groups.control) != readers
            or dist.get_rank(groups.control) != self.rank
        ):
            raise ValueError(
                "Replica groups must contain exactly this KV shard's readers"
            )
        backend = str(dist.get_backend(groups.data)).lower()
        expected = {"cuda": "nccl", "npu": "hccl"}.get(tensors[0].device.type)
        if (
            backend != expected
            or str(dist.get_backend(groups.control)).lower() != "gloo"
        ):
            raise ValueError(
                "Replica broadcast requires NCCL/CUDA or HCCL/NPU plus Gloo control"
            )
        self.batch_bytes = int(
            os.environ.get("LMCACHE_XDS_MLA_BATCH_BYTES", str(64 << 20))
        )
        if self.batch_bytes <= 0:
            raise ValueError("LMCACHE_XDS_MLA_BATCH_BYTES must be positive")
        # Retain all views/work on failure; the enclosing context also retains HBM.
        self.pending: list[tuple[PendingWork, torch.Tensor]] = []

    def restore(
        self,
        identity: tuple[object, ...],
        chunks: list[list[int]],
        skip: int,
        read: Callable[[int, int, list[int]], None],
    ) -> None:
        """Restore one ordered range, with only the writer invoking SSD I/O.

        Args:
            identity: Request epoch/model/shard/range; must match on every rank.
            chunks: Local final block IDs grouped by SSD chunk.
            skip: Protected prefix block count; identical across replicas.
            read: Leader-only callback that drains a chunk interval's SSD reads.
                Runs serially on a dedicated host thread, using already registered
                pages; it must not depend on caller thread-local device state.
                At most two batches are admitted ahead of the consumer. The
                reader is joined before return, including on collective errors.

        Raises:
            RuntimeError: Replica requests differ or any leader read fails.
            ValueError: Local page maps alias or have incompatible geometry.
        """
        blocks = [b for chunk in chunks for b in chunk]
        geometry = tuple((tuple(t.shape[1:]), str(t.dtype)) for t in self.tensors)
        local = ReplicaPlan(
            identity,
            tuple(map(len, chunks)),
            skip,
            geometry,
            self.writer,
            tuple(blocks),
        )
        gathered = [local] * self.readers
        dist.all_gather_object(gathered, local, group=self.groups.control)
        if any(
            (item.identity, item.lengths, item.skip, item.geometry)
            != (local.identity, local.lengths, local.skip, local.geometry)
            for item in gathered
        ):
            raise RuntimeError("MLA replica request, skip or plane geometry mismatch")
        writers = [rank for rank, item in enumerate(gathered) if item.writer]
        if len(writers) != 1:
            raise RuntimeError("MLA replica group must have exactly one SSD reader")
        maps = [item.blocks for item in gathered]
        if any(len(set(m)) != len(m) for m in maps):
            raise ValueError("MLA replica destination pages must not alias")
        source = dist.get_global_rank(self.groups.data, writers[0])
        self.synchronize()
        chunk_bytes = sum(t.stride(0) * t.element_size() for t in self.tensors) * len(
            chunks[0]
        )
        batch_chunks = max(1, self.batch_bytes // chunk_bytes)
        batches = []
        remaining_skip = skip
        for first in range(0, len(chunks), batch_chunks):
            last = min(first + batch_chunks, len(chunks))
            batch_skips = []
            for chunk in chunks[first:last]:
                take = min(remaining_skip, len(chunk))
                batch_skips.append(take)
                remaining_skip -= take
            batches.append((first, last, batch_skips))
        # Only the leader owns this executor/client. Device collectives remain
        # on the calling thread; read-ahead touches disjoint final pages only.
        executor = ThreadPoolExecutor(max_workers=1) if self.writer else None
        reads: deque[Future[None]] = deque()
        try:
            if executor is not None:
                for args in batches[:2]:
                    reads.append(executor.submit(read, *args))
            position = 0
            remaining_skip = skip
            for index, (first, last, _) in enumerate(batches):
                count = sum(map(len, chunks[first:last]))
                protected = min(remaining_skip, count)
                remaining_skip -= protected
                error = None
                if executor is not None:
                    try:
                        reads.popleft().result()
                    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    if error is None and index + 2 < len(batches):
                        reads.append(executor.submit(read, *batches[index + 2]))
                self.agree(error)
                self.drain()
                runs = plan_broadcast_runs(maps, position + protected, position + count)
                for tensor in self.tensors:
                    for run in runs:
                        view = tensor.narrow(0, run.starts[self.rank], run.blocks)
                        work = dist.broadcast(
                            view, src=source, group=self.groups.data, async_op=True
                        )
                        self.pending.append((work, view))
                position += count
        finally:
            if executor is not None:
                # Cancel queued work and join active reads before the caller can
                # leave the transfer lock. Uncertain DMA still retains its pins.
                executor.shutdown(wait=True, cancel_futures=True)
        self.drain()
        self.synchronize()

    def agree(self, error: str | None) -> None:
        """Propagate any rank's preparation/I/O error before data collectives.

        Args:
            error: Local failure description, or None when ready.

        Raises:
            RuntimeError: Any replica reported a failure. No new broadcasts start.
        """
        statuses: list[str | None] = [None] * self.readers
        dist.all_gather_object(statuses, error, group=self.groups.control)
        if any(status is not None for status in statuses):
            raise RuntimeError(
                f"MLA replica preparation or SSD read failed: {statuses}"
            )

    def drain(self) -> None:
        """Wait for queued communication without dropping views on failure."""
        for work, _ in self.pending:
            work.wait()
        self.pending.clear()
