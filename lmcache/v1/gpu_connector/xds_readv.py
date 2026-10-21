# SPDX-License-Identifier: Apache-2.0
"""Aligned, layout-preserving SSD reads into vLLM's final KV blocks."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional
import ctypes

# Third Party
import numpy as np
import torch

# First Party
from lmcache.v1.memory_management import MemoryFormat


@dataclass(frozen=True)
class _PagedSlices(Sequence[tuple[int, int, int]]):
    layers: int
    blocks: int
    block_bytes: int
    block_ids: tuple[int, ...]
    layout: str

    def __len__(self) -> int:
        return 2 * self.layers * len(self.block_ids)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        group, block_index = divmod(index, len(self.block_ids))
        kv, layer = divmod(group, self.layers)
        block = self.block_ids[block_index]
        if self.layout == "SEPARATE":
            return 2 * layer + kv, block * self.block_bytes, self.block_bytes
        offset = kv * self.blocks + block if self.layout == "2NB" else block * 2 + kv
        return layer, offset * self.block_bytes, self.block_bytes

    def bind(self, refs: Sequence[int]) -> list[tuple[int, int, int]]:
        """Construct final registered-buffer tuples in source byte order."""
        # Compute each block offset once, not once per layer. Construct only the
        # final SDK tuples; no intermediate (layer, offset, length) objects.
        result = []
        for kv in range(2):
            if self.layout == "SEPARATE":
                for layer in range(self.layers):
                    result.extend(
                        (
                            refs[2 * layer + kv],
                            block * self.block_bytes,
                            self.block_bytes,
                        )
                        for block in self.block_ids
                    )
                continue
            offsets = [
                (
                    (kv * self.blocks + block)
                    if self.layout == "2NB"
                    else (block * 2 + kv)
                )
                * self.block_bytes
                for block in self.block_ids
            ]
            for ref in refs:
                result.extend((ref, offset, self.block_bytes) for offset in offsets)
        return result


@dataclass(frozen=True)
class XDSReadvPlan:
    """Owned targets and a source-ordered, possibly lazy destination sequence.

    Plans carry fresh block mappings on every retrieval. Registration ranges
    and optional buffer_refs may be shared for the lifetime of a fixed KV pool.
    """

    tensors: tuple[torch.Tensor, ...]
    slices: Sequence[tuple[int, int, int]]
    size: int
    block_ids: tuple[int, ...]
    registration_ranges: Optional[tuple[tuple[int, int], ...]] = None
    buffer_refs: Optional[tuple[int, ...]] = None

    def submit(
        self,
        client: Any,
        handle: int,
        offset: int,
        refs: Sequence[int],
    ) -> int:
        """Submit compact indexed geometry when supported, else ordinary readv.

        Uses a fresh mapping for each retrieval. The caller retains tensors,
        target-block ownership and registrations until DMA completion. Raises
        ValueError for a mismatched reference count; propagates SDK failures.
        """
        if len(refs) != len(self.tensors):
            raise ValueError("readv reference count differs from tensor count")
        native = self.prepare_native(client, refs)
        if native is not None:
            return int(client.submit_readv_plan(handle, offset, native))
        return int(client.submit_readv(handle, offset, self.bind_slices(refs)))

    def prepare_native(self, client: Any, refs: Sequence[int]) -> Any:
        """Build an SDK plan, or return None for an unsupported SDK/layout.

        Raises on invalid references or descriptor preparation failure. Does
        not submit I/O; the caller owns target buffers through completion.
        """
        if len(refs) != len(self.tensors):
            raise ValueError("readv reference count differs from tensor count")
        prepare = getattr(client, "prepare_indexed_readv", None)
        submit_plan = getattr(client, "submit_readv_plan", None)
        if (
            isinstance(self.slices, _PagedSlices)
            and callable(prepare)
            and callable(submit_plan)
            and getattr(client, "supports_native_readv", True)
        ):
            slices = self.slices
            groups = []
            for kv in range(2):
                if slices.layout == "SEPARATE":
                    groups.extend(
                        (refs[2 * layer + kv], 0, slices.block_bytes)
                        for layer in range(slices.layers)
                    )
                    continue
                base = kv * slices.blocks if slices.layout == "2NB" else kv
                stride = 1 if slices.layout == "2NB" else 2
                groups.extend(
                    (ref, base * slices.block_bytes, stride * slices.block_bytes)
                    for ref in refs
                )
            return prepare(slices.block_ids, groups, slices.block_bytes)
        return None

    @staticmethod
    def submit_batch(
        client: Any,
        requests: Sequence[tuple["XDSReadvPlan", int, int, Sequence[int]]],
    ) -> list[int | BaseException]:
        """Prepare and submit independent plans, preserving per-request errors.

        Each input is (plan, handle, offset, references). Preparation failures
        return errors for the whole batch before any I/O is submitted. Accepted
        request IDs must be drained even if another entry failed submission.
        Requires a client advertising supports_readv_batch and native plans.
        """
        try:
            prepared = []
            for plan, handle, offset, refs in requests:
                native = plan.prepare_native(client, refs)
                if native is None:
                    raise ValueError("batch requires native paged readv plans")
                prepared.append((handle, offset, native))
        except Exception as error:
            return [error] * len(requests)
        return client.submit_readv_plans(prepared)

    def bind_slices(self, refs: Sequence[int]) -> list[tuple[int, int, int]]:
        """Materialize SDK descriptors using one registered reference per tensor.

        Raises ValueError if the reference count does not match the target pool.
        Descriptor allocation is deferred until submission; no plan is cached
        across retrievals, where slot ownership may have changed.
        """
        if len(refs) != len(self.tensors):
            raise ValueError("readv reference count differs from tensor count")
        if isinstance(self.slices, _PagedSlices):
            return self.slices.bind(refs)
        return [(refs[i], offset, length) for i, offset, length in self.slices]


def plan_paged_readv_batch(
    tensors: list[torch.Tensor],
    chunks: Sequence[
        tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int, int]
    ],
    slots: torch.Tensor,
    io_unit: int,
    layout: str,
    device_type: str = "cuda",
    registration_ranges: Optional[tuple[tuple[int, int], ...]] = None,
    buffer_refs: Optional[tuple[int, ...]] = None,
) -> Optional[list[XDSReadvPlan]]:
    """Validate a whole restore before DMA, sharing target-pool validation.

    Each chunk supplies shapes, dtypes, format, start and end token positions.
    Return lazy plans in chunk order, or None for unsupported layouts, invalid
    metadata, unaligned spans or overlapping target blocks, including overlap
    across chunks. The caller owns tensors and slot assignments until completion.
    No descriptors or validation results are reused across calls. An optional
    registration_ranges tuple carries ranges validated by the connector for
    this call, so the backend need not repeat platform-specific validation.
    Optional buffer_refs are attached during construction, avoiding a second
    plan allocation for an already registered pool. Raises ValueError if their
    count differs from the number of target tensors.
    """
    if not chunks:
        return []
    if buffer_refs is not None and len(buffer_refs) != len(tensors):
        raise ValueError("readv reference count differs from tensor count")
    if (
        not tensors
        or io_unit <= 0
        or slots.ndim != 1
        or layout not in ("2NB", "NB2", "SEPARATE")
        or device_type not in ("cuda", "npu")
    ):
        return None
    separate = layout == "SEPARATE"
    if separate and len(tensors) % 2:
        return None
    layers = len(tensors) // 2 if separate else len(tensors)
    first = tensors[0]
    if any(
        t.device.type != device_type
        or t.ndim != (4 if separate else 5)
        or not t.is_contiguous()
        or t.dtype != first.dtype
        or t.device != first.device
        or t.shape != first.shape
        for t in tensors
    ):
        return None
    if separate:
        blocks, block_size, heads, head_size = first.shape
    else:
        axis = 0 if layout == "2NB" else 1
        if first.shape[axis] != 2:
            return None
        blocks = first.shape[1 if axis == 0 else 0]
        block_size, heads, head_size = first.shape[2:]
    if min(blocks, block_size, heads, head_size) <= 0:
        return None
    element_size = first.element_size()
    block_bytes = block_size * heads * head_size * element_size
    if block_bytes % io_unit:
        return None
    spans = sorted(
        (t.data_ptr(), t.data_ptr() + t.numel() * element_size) for t in tensors
    )
    registration_alignment = max(io_unit, 4096) if device_type == "npu" else io_unit
    if any(
        a % registration_alignment or (b - a) % registration_alignment for a, b in spans
    ):
        return None
    if any(b > c for (_, b), (c, _) in zip(spans, spans[1:], strict=False)):
        return None
    for shapes, dtypes, fmt, start, end in chunks:
        if (
            fmt != MemoryFormat.KV_2LTD
            or len(shapes) != 1
            or len(dtypes) != 1
            or dtypes[0] != first.dtype
            or start < 0
            or end <= start
            or end > slots.numel()
            or (end - start) % block_size
            or tuple(shapes[0]) != (2, layers, end - start, heads * head_size)
        ):
            return None
    # One device-to-host synchronization, if the caller did not provide the CPU
    # mapping. Transfer only the span used by this batch, not the whole KV pool.
    base = min(chunk[3] for chunk in chunks)
    limit = max(chunk[4] for chunk in chunks)
    slot_values = slots[base:limit].detach().cpu()
    if slot_values.dtype == torch.int64:
        # Validate only selected spans: gaps may belong to other requests.
        # NumPy keeps 30K+ slot IDs in a compact CPU array instead of allocating
        # a Python integer for every token and a range list for every block.
        slot_array = slot_values.numpy()
        rows = np.concatenate(
            [
                slot_array[start - base : end - base].reshape(-1, block_size)
                for _, _, _, start, end in chunks
            ],
            axis=0,
        )
        first_slots = rows[:, 0]
        if (
            np.any(first_slots < 0)
            or np.any(first_slots % block_size)
            or np.any(first_slots // block_size >= blocks)
            or not np.all(
                rows == first_slots[:, None] + np.arange(block_size, dtype=np.int64)
            )
        ):
            return None
        all_block_ids = (first_slots // block_size).tolist()
        if len(set(all_block_ids)) != len(all_block_ids):
            return None
    else:
        # Preserve the generic planner's existing behavior for other dtypes.
        slot_list = slot_values.tolist()
        seen: set[int] = set()
        all_block_ids = []
        for _, _, _, start, end in chunks:
            for i in range(start - base, end - base, block_size):
                slot = slot_list[i]
                if not isinstance(slot, int) or slot < 0 or slot % block_size:
                    return None
                block = slot // block_size
                if (
                    block >= blocks
                    or block in seen
                    or slot_list[i : i + block_size]
                    != list(range(slot, slot + block_size))
                ):
                    return None
                seen.add(block)
                all_block_ids.append(block)
    owned_tensors = tuple(tensors)
    plans = []
    cursor = 0
    for shapes, _, _, start, end in chunks:
        count = (end - start) // block_size
        ids = tuple(all_block_ids[cursor : cursor + count])
        cursor += count
        plans.append(
            XDSReadvPlan(
                owned_tensors,
                _PagedSlices(layers, blocks, block_bytes, ids, layout),
                shapes[0].numel() * element_size,
                ids,
                registration_ranges,
                buffer_refs,
            )
        )
    return plans


def plan_paged_readv(
    tensors: list[torch.Tensor],
    shapes: list[torch.Size],
    dtypes: list[torch.dtype],
    fmt: MemoryFormat,
    slots: torch.Tensor,
    start: int,
    end: int,
    io_unit: int,
    layout: str,
) -> Optional[XDSReadvPlan]:
    """Plan one chunk; see plan_paged_readv_batch for validation and ownership."""
    plans = plan_paged_readv_batch(
        tensors, [(shapes, dtypes, fmt, start, end)], slots, io_unit, layout
    )
    return plans[0] if plans else None


def plan_layerwise_readv_batch(
    tensors: list[torch.Tensor],
    chunks: Sequence[
        tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int, int]
    ],
    slots: torch.Tensor,
    io_unit: int,
    layout: str,
    device_type: str = "cuda",
    registration_ranges: Optional[tuple[tuple[int, int], ...]] = None,
) -> Optional[list[list[XDSReadvPlan]]]:
    """Plan KV_2TD blobs as layer-major reads without changing source byte order.

    Validate all layers and all destination blocks before returning. Return None
    for legacy token-major metadata or unsupported spans/layouts. The result is
    indexed by layer, then chunk. Caller retains target ownership through DMA.
    """
    if not tensors or (layout == "SEPARATE" and len(tensors) % 2):
        return None
    layers = len(tensors) // 2 if layout == "SEPARATE" else len(tensors)
    expanded = []
    for shapes, dtypes, fmt, start, end in chunks:
        if fmt != MemoryFormat.KV_2TD or len(shapes) != 1:
            return None
        shape = shapes[0]
        if len(shape) != 3 or shape[0] != 2 or shape[1] != end - start:
            return None
        expanded.append(
            (
                [torch.Size([2, layers, end - start, shape[2]])],
                dtypes,
                MemoryFormat.KV_2LTD,
                start,
                end,
            )
        )
    plans = plan_paged_readv_batch(
        tensors, expanded, slots, io_unit, layout, device_type, registration_ranges
    )
    if plans is None:
        return None
    result = []
    for layer in range(layers):
        indices = (2 * layer, 2 * layer + 1) if layout == "SEPARATE" else (layer,)
        targets = tuple(tensors[i] for i in indices)
        ranges = (
            tuple(registration_ranges[i] for i in indices)
            if registration_ranges is not None
            else None
        )
        layer_plans = []
        for plan in plans:
            slices = plan.slices
            assert isinstance(slices, _PagedSlices)
            layer_plans.append(
                XDSReadvPlan(
                    targets,
                    _PagedSlices(
                        1,
                        slices.blocks,
                        slices.block_bytes,
                        slices.block_ids,
                        slices.layout,
                    ),
                    plan.size // layers,
                    plan.block_ids,
                    ranges,
                )
            )
        result.append(layer_plans)
    return result


@lru_cache(maxsize=1)
def cuda_readv_flush_function() -> Any:
    """Resolve the CUDA driver visibility fence, raising if unavailable."""
    driver = ctypes.CDLL("libcuda.so.1")
    function = driver.cuFlushGPUDirectRDMAWrites
    function.argtypes = [ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    return function


def finish_readv_dma(device: torch.device) -> None:
    """Make completed GPUDirect writes visible before subsequent GPU work.

    Raises RuntimeError on a CUDA driver error. Does not launch a copy kernel.
    The caller must first wait for every relevant storage completion.
    """
    with torch.cuda.device(device):
        status = cuda_readv_flush_function()(0, 100)  # CURRENT_CTX, TO_OWNER
        if status:
            raise RuntimeError(f"cuFlushGPUDirectRDMAWrites failed: {status}")
