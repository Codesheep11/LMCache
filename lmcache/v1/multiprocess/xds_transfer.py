# SPDX-License-Identifier: Apache-2.0
"""Worker-owned xDS I/O shared by CUDA and Ascend platform adapters."""

# Standard
from collections import deque
from collections.abc import Iterator, Sequence
from threading import RLock
import os

# Third Party
import torch
import xds

# First Party
from lmcache.utils import EngineType, init_logger
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.worker_transfer import TransferContext
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.xds_catalog import XDSExtent, XDSLease
from lmcache.v1.multiprocess.xds_client import xds_rpc
from lmcache.v1.multiprocess.xds_cuda_pool import KVPool, RegisteredCUDAKVPool
from lmcache.v1.multiprocess.xds_protocol import (
    BindWorker,
    ClaimRange,
    CompleteMany,
    Register,
    TokenTransfer,
)
from lmcache.v1.multiprocess.xds_replica import ReplicaBroadcast, ReplicaGroups

logger = init_logger(__name__)


class XDSWorkerTransferContext(TransferContext):
    """Own one xDS connection and fixed HBM registrations in the vLLM worker.

    The initial implementation executes synchronous direct I/O; payloads never
    traverse the metadata server or an intermediate CPU/device buffer.
    """

    def __init__(
        self,
        instance_id: int,
        req_client: RequestClient,
        worker_id: int,
        *,
        reader_id: int | None = None,
        io_world_size: int | None = None,
        num_readers: int = 1,
        writer: bool = True,
        replica_groups: ReplicaGroups | None = None,
    ) -> None:
        super().__init__(instance_id, req_client)
        self._worker_id = worker_id
        self._reader_id = worker_id if reader_id is None else reader_id
        self._io_world_size = io_world_size
        self._num_readers = num_readers
        self._writer = writer
        self._replica_groups = replica_groups
        self._replica: ReplicaBroadcast | None = None
        self._imported_blobs: set[int] = set()
        if num_readers <= 0:
            raise ValueError("Reader count must be positive")
        self._client = xds.XDSClient(
            xds.XDSClientConfig(
                service_name=os.environ.get("LMCACHE_XDS_SERVICE", "xds-yxc"),
                require_direct=True,
            )
        )
        self._pool: KVPool | None = None
        self._blobs: list[int] = []
        self._timeout = 60.0
        self._io_failed = False
        self._transfer_lock = RLock()

    def register(
        self,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register owned NHD planes once and announce exact-size SSD extents.

        Dense NHD and MLA latent/RoPE planes share one block address space.
        Hybrid, recurrent, and sliding-window groups raise ValueError.
        """
        with self._transfer_lock:
            if self._pool is not None or self._is_closed():
                raise RuntimeError("xDS worker is already registered or closed")
            if len(engine_group_infos) > 1 or any(
                g.recurrent_state or g.sw_size_tokens != -1 for g in engine_group_infos
            ):
                raise ValueError(
                    "xDS worker currently requires one dense attention group"
                )
            tensors = self.prepare_planes(kv_caches)
            self._tokens_per_block = tensors[0].shape[1]
            if any(t.shape[:2] != tensors[0].shape[:2] for t in tensors):
                raise ValueError("KV planes must share block and token geometry")
            if (
                engine_group_infos
                and engine_group_infos[0].tokens_per_block != self._tokens_per_block
            ):
                raise ValueError(
                    "Physical NHD token dimension differs from engine block size"
                )
            self._device = tensors[0].device
            self._timeout = mq_timeout
            self._blocks_in_chunk = blocks_in_chunk
            self._chunk_tokens = blocks_in_chunk * self._tokens_per_block
            self.configure_device(self._device)
            self._client.config = xds.XDSClientConfig(
                service_name=os.environ.get("LMCACHE_XDS_SERVICE", "xds-yxc"),
                rank_id=self._reader_id,
                world_size=world_size
                if self._io_world_size is None
                else self._io_world_size,
                require_direct=True,
            )
            self._client.connect()
            self._pool = self.create_pool(tuple(tensors))
            self._tensors = tuple(tensors)
            mode = os.environ.get("LMCACHE_XDS_MLA_READ_MODE", "leader")
            if mode not in ("leader", "direct"):
                raise ValueError("LMCACHE_XDS_MLA_READ_MODE must be leader or direct")
            if self._num_readers > 1 and mode == "leader":
                if self._replica_groups is None:
                    raise ValueError("MLA leader reads require explicit replica groups")
                self._replica = ReplicaBroadcast(
                    self._replica_groups,
                    self._tensors,
                    self._writer,
                    self._num_readers,
                    self.synchronize_device,
                )
            sizes = {plane.block_bytes for plane in self._pool.planes}
            # Keep the indexed fast path for equal-width K/V; MLA planes can
            # have distinct strides and are packed without padding on SSD.
            self._block_bytes = next(iter(sizes)) if len(sizes) == 1 else 0
            self._bytes_per_block = sum(p.block_bytes for p in self._pool.planes)
            self._read_groups = tuple(
                (p.buffer_ref, 0, p.block_bytes) for p in self._pool.planes
            )
            self._queue_depth = int(os.environ.get("LMCACHE_XDS_QUEUE_DEPTH", "4"))
            if not 1 <= self._queue_depth <= 64:
                raise ValueError("LMCACHE_XDS_QUEUE_DEPTH must be between 1 and 64")
            self._chunk_bytes = (
                sum(p.block_bytes for p in self._pool.planes) * blocks_in_chunk
            )
            blob_bytes = self._client.get_blob_size()
            unit = self._client.get_io_unit_size()
            if self._chunk_bytes > blob_bytes or any(
                p.block_bytes % unit for p in self._pool.planes
            ):
                raise ValueError(
                    "KV chunk must fit one xDS blob and each block must be I/O aligned"
                )
            count = int(os.environ.get("LMCACHE_XDS_BLOB_COUNT", "8"))
            if count <= 0:
                raise ValueError("LMCACHE_XDS_BLOB_COUNT must be positive")
            extents: list[XDSExtent] = []
            for _ in range(count if self._writer else 0):
                blob = self._client.get_blob()
                self._blobs.append(blob)
                token = (
                    self._client.export_blob_read(blob) if self._num_readers > 1 else 0
                )
                extents.extend(
                    XDSExtent(blob, offset, self._chunk_bytes, token)
                    for offset in range(
                        0, blob_bytes - self._chunk_bytes + 1, self._chunk_bytes
                    )
                )
            self._registration = Register(
                self._instance_id, tuple(extents), not self._writer
            )
            self._binding = BindWorker(
                self._instance_id,
                model_name,
                world_size,
                self._worker_id,
                self._chunk_bytes,
                self._reader_id,
                self._num_readers,
                self._writer,
            )
            self._submit_registration(
                lambda: xds_rpc(
                    self._req_client,
                    "register",
                    self._registration,
                    type(None),
                )
            ).result(mq_timeout)
            xds_rpc(
                self._req_client,
                "bind_worker",
                self._binding,
                type(None),
            ).result(mq_timeout)

    def recover_registration(self) -> bool:
        """Reannounce drained capacity without reconnecting or registering HBM.

        Retains cached metadata after transient transport loss; a restarted
        server begins with an empty catalog. Raises RuntimeError if closing or
        DMA completion is uncertain. RPC failures propagate for heartbeat retry.
        """
        with self._transfer_lock:
            if self._pool is None or self._is_closed() or self._io_failed:
                raise RuntimeError("Cannot recover an undrained or closed xDS worker")
            xds_rpc(
                self._req_client, "ensure_registered", self._registration, type(None)
            ).result(self._timeout)
            xds_rpc(self._req_client, "bind_worker", self._binding, type(None)).result(
                self._timeout
            )
            return True

    def unregister(self) -> MessagingFuture | None:
        """Forget metadata only when every synchronous transfer has completed."""
        with self._transfer_lock:
            if self._io_failed:
                raise RuntimeError(
                    "xDS I/O completion is uncertain; "
                    "retain registration until worker exits"
                )
            self._mark_closed()
            return self._submit_unregistration(
                lambda: xds_rpc(
                    self._req_client, "unregister", self._instance_id, type(None)
                )
            )

    def create_recorded_event(self) -> None:
        """Producer/consumer fencing is performed synchronously in submit."""
        return None

    def submit_store(
        self,
        request_id: str,
        key: IPCCacheServerKey,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: object,
        blocks_in_chunk: int,
    ) -> MessagingFuture[bool]:
        """Gather final HBM pages directly into consecutive SSD chunk extents.

        Raises:
            ValueError: If the token range is not complete chunk aligned or
                block IDs do not match the registered pool geometry.
            RuntimeError: If metadata returns a different number of chunks.
            TimeoutError: If metadata or I/O completion times out.
        """
        with self._transfer_lock:
            chunks = self._blocks(key, block_ids, blocks_in_chunk)
            assert self._pool is not None
            if not self._writer:
                raise ValueError("Read-only replica cannot store KV")
            leases = xds_rpc(
                self._req_client,
                "reserve_tokens",
                TokenTransfer(self._instance_id, key),
                tuple[XDSLease | None, ...],
            ).result(self._timeout)
            if len(chunks) != len(leases):
                raise RuntimeError("Metadata chunk count differs from KV geometry")
            self.synchronize_device()
            self._io_failed = True
            pending: deque[int] = deque()
            for blocks, lease in zip(chunks, leases, strict=True):
                if lease is None:
                    continue
                slices = [
                    (p.buffer_ref, block * p.block_bytes, p.block_bytes)
                    for p in self._pool.planes
                    for block in blocks
                ]
                request = self._client.submit_writev(
                    lease.extent.blob, lease.extent.offset, slices
                )
                pending.append(request)
                if len(pending) >= self._queue_depth:
                    self._client.wait(pending.popleft(), timeout=self._timeout)
            while pending:
                self._client.wait(pending.popleft(), timeout=self._timeout)
            xds_rpc(
                self._req_client,
                "publish_many",
                CompleteMany(
                    self._instance_id,
                    tuple(lease.ticket for lease in leases if lease is not None),
                ),
                type(None),
            ).result(self._timeout)
            self._io_failed = False
            logger.debug(
                "xDS store request=%s chunks=%d bytes=%d",
                request_id,
                sum(lease is not None for lease in leases),
                sum(lease.extent.size for lease in leases if lease is not None),
            )
            future = MessagingFuture[bool]()
            future.set_result(True)
            return future

    def submit_retrieve(
        self,
        request_id: str,
        key: IPCCacheServerKey,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: object,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture[bool]:
        """Restore scheduler-pinned extents directly into the requested KV pages.

        Claim the scheduler's indexed range without sending tokens again.
        Validate the entire chunk count before submitting any DMA.

        Raises:
            ValueError: For a skip not aligned to a physical KV block, a skip
                outside the requested range, incompatible block geometry,
                or a key for a different worker/shared-reader layout.
            RuntimeError: If metadata returns a different number of chunks.
            TimeoutError: If metadata or I/O completion times out; unresolved
                DMA retains HBM registration and blob ownership.
        """
        with self._transfer_lock:
            error = None
            chunks: list[list[int]] = []
            leases: tuple[XDSLease, ...] = ()
            try:
                chunks, leases = self._prepare_restore(
                    key, block_ids, blocks_in_chunk, skip_first_n_tokens
                )
            except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as exc:
                if self._replica is None:
                    raise
                error = f"{type(exc).__name__}: {exc}"
            if self._replica is not None:
                self._io_failed = True
                self._replica.agree(error)
            self._io_failed = True
            if self._replica is not None:
                identity = (
                    key.request_id,
                    key.model_name,
                    key.world_size,
                    self._worker_id,
                    key.start,
                    key.end,
                )
                self._replica.restore(
                    identity,
                    chunks,
                    skip_first_n_tokens // self._tokens_per_block,
                    lambda first, last, skips: self.read_chunks(
                        chunks[first:last], leases[first:last], skips
                    ),
                )
                restored_bytes = (
                    (key.end - key.start - skip_first_n_tokens)
                    // self._tokens_per_block
                    * self._bytes_per_block
                )
            else:
                for lease in leases:
                    blob = lease.extent.blob
                    if blob not in self._blobs and blob not in self._imported_blobs:
                        self._client.import_blob_read(blob, lease.extent.read_token)
                        self._imported_blobs.add(blob)
                remaining_skip = skip_first_n_tokens // self._tokens_per_block
                skips = []
                for blocks in chunks:
                    skip = min(remaining_skip, len(blocks))
                    remaining_skip -= skip
                    skips.append(skip)
                self.synchronize_device()
                self.read_chunks(chunks, leases, skips)
                restored_bytes = sum(
                    (len(blocks) - skip) * self._bytes_per_block
                    for blocks, skip in zip(chunks, skips, strict=True)
                )
            for blob in tuple(self._imported_blobs):
                self._client.unimport_blob_read(blob)
                self._imported_blobs.remove(blob)
            xds_rpc(
                self._req_client,
                "release_many",
                CompleteMany(
                    self._instance_id, tuple(lease.ticket for lease in leases)
                ),
                type(None),
            ).result(self._timeout)
            self.synchronize_device()
            self._io_failed = False
            logger.debug(
                "xDS restore request=%s chunks=%d bytes=%d",
                request_id,
                len(leases),
                restored_bytes,
            )
            future = MessagingFuture[bool]()
            future.set_result(True)
            return future

    def flush_inflight_stores(self) -> None:
        """Synchronous writes leave no deferred store work."""

    def close(self) -> None:
        """Release drained blobs and registrations before final HBM is freed."""
        with self._transfer_lock:
            self._mark_closed()
            if self._io_failed:
                raise RuntimeError("Cannot free HBM while xDS completion is uncertain")
            for blob in tuple(self._imported_blobs):
                self._client.unimport_blob_read(blob)
                self._imported_blobs.remove(blob)
            for blob in tuple(self._blobs):
                self._client.release_blob(blob)
                self._blobs.remove(blob)
            if self._pool is not None:
                self._pool.close()
                self._pool = None
            self._client.close()
            self._tensors = ()
            self._replica = None

    def prepare_planes(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """Return CUDA block-major views without allocating KV payload buffers.

        Supports MLA [blocks, tokens, width] and separate or packed NHD K/V.
        Non-contiguous layouts are rejected by the fixed registration pool.
        """
        planes: list[torch.Tensor] = []
        for value in kv_caches.values():
            if isinstance(value, (tuple, list)) and len(value) == 2:
                planes.extend(value)
            elif isinstance(value, torch.Tensor) and value.ndim == 3:
                planes.append(value)
            elif (
                isinstance(value, torch.Tensor)
                and value.ndim == 5
                and value.shape[0] == 2
            ):
                planes.extend(value.unbind(0))
            else:
                raise ValueError("xDS CUDA requires block-major MLA or NHD K/V")
        if not planes or any(t.ndim not in (3, 4) for t in planes):
            raise ValueError("xDS CUDA requires nonempty block-major KV planes")
        return tuple(planes)

    def configure_device(self, device: torch.device) -> None:
        """Verify the worker owns a CUDA allocation before registration."""
        if device.type != "cuda":
            raise ValueError("CUDA xDS context requires CUDA KV pages")

    def create_pool(self, tensors: tuple[torch.Tensor, ...]) -> KVPool:
        """Register final CUDA pages once; the pool retains owning tensors."""
        return RegisteredCUDAKVPool(self._client, tensors)

    def synchronize_device(self) -> None:
        """Fence device users of final pages before external DMA or release."""
        torch.cuda.synchronize(self._device)

    def read_chunks(
        self, chunks: list[list[int]], leases: tuple[XDSLease, ...], skips: list[int]
    ) -> None:
        """Read a bounded chunk batch and drain every submitted SSD operation.

        Args:
            chunks: Destination block IDs per chunk.
            leases: Pinned SSD extents in matching order.
            skips: Number of protected prefix blocks in each chunk.

        Raises:
            TimeoutError: DMA completion is uncertain; the caller retains pages.
        """
        pending: deque[int] = deque()
        for blocks, lease, skip in zip(chunks, leases, skips, strict=True):
            for request in self._submit_chunk_reads(blocks, lease, skip):
                pending.append(request)
                if len(pending) >= self._queue_depth:
                    self._client.wait(pending.popleft(), timeout=self._timeout)
        while pending:
            self._client.wait(pending.popleft(), timeout=self._timeout)

    def _prepare_restore(
        self,
        key: IPCCacheServerKey,
        block_ids: list[list[int]],
        blocks_in_chunk: int,
        skip_first_n_tokens: int,
    ) -> tuple[list[list[int]], tuple[XDSLease, ...]]:
        """Validate and claim a rank before the collective readiness handshake."""
        chunks = self._blocks(key, block_ids, blocks_in_chunk)
        if (
            not 0 <= skip_first_n_tokens <= key.end - key.start
            or skip_first_n_tokens % self._tokens_per_block
        ):
            raise ValueError("Skipped prefix must be block aligned and within range")
        if key.worker_id != self._worker_id or key.num_kv_readers != self._num_readers:
            raise ValueError("Retrieve requires this worker's KV rank and reader count")
        leases = xds_rpc(
            self._req_client,
            "claim_range",
            ClaimRange(
                self._instance_id,
                key.request_id,
                key.model_name,
                key.world_size,
                self._worker_id,
                key.start,
                key.end,
            ),
            tuple[XDSLease, ...],
        ).result(self._timeout)
        if len(chunks) != len(leases):
            raise RuntimeError("Metadata chunk count differs from KV geometry")
        return chunks, leases

    def _blocks(
        self, key: IPCCacheServerKey, block_ids: list[list[int]], blocks_in_chunk: int
    ) -> list[list[int]]:
        if self._pool is None or self._is_closed() or self._io_failed:
            raise RuntimeError("xDS worker is not ready")
        if len(block_ids) != 1 or blocks_in_chunk != self._blocks_in_chunk:
            raise ValueError("Unexpected KV block group geometry")
        if (
            key.start < 0
            or key.end <= key.start
            or key.end > len(key.token_ids)
            or key.start % self._chunk_tokens
            or key.end % self._chunk_tokens
        ):
            raise ValueError("Token range must contain complete aligned KV chunks")
        count = (key.end - key.start) // self._chunk_tokens
        blocks = block_ids[0]
        if len(blocks) != count * blocks_in_chunk:
            raise ValueError("KV block count does not match token range")
        if any(b < 0 or b >= self._pool.planes[0].num_blocks for b in blocks):
            raise ValueError("KV block outside registered pool")
        return [
            blocks[i : i + blocks_in_chunk]
            for i in range(0, len(blocks), blocks_in_chunk)
        ]

    def _submit_chunk_reads(
        self, blocks: list[int], lease: XDSLease, skip: int
    ) -> Iterator[int]:
        if skip == len(blocks):
            return
        if skip == 0:
            if self._block_bytes:
                plan = self._client.prepare_indexed_readv(
                    blocks, self._read_groups, self._block_bytes
                )
                yield self._client.submit_readv_plan(
                    lease.extent.blob, lease.extent.offset, plan
                )
            else:
                slices = [
                    (ref, block * stride, stride)
                    for ref, _, stride in self._read_groups
                    for block in blocks
                ]
                yield self._client.submit_readv(
                    lease.extent.blob, lease.extent.offset, slices
                )
            return
        # Each plane has its own SSD stride and skipped prefix.
        plane_offset = lease.extent.offset
        for buffer_ref, _, block_bytes in self._read_groups:
            offset = plane_offset + skip * block_bytes
            slices = [
                (buffer_ref, block * block_bytes, block_bytes)
                for block in blocks[skip:]
            ]
            yield self._client.submit_readv(lease.extent.blob, offset, slices)
            plane_offset += len(blocks) * block_bytes
