# SPDX-License-Identifier: Apache-2.0
"""Opt-in KV-major layerwise storage for aligned xDS direct reads."""

# Standard
from typing import Any, Iterator, Optional, Sequence

# Third Party
import torch

# First Party
import lmcache.c_ops as lmc_ops
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemLayerwiseGPUConnector
from lmcache.v1.gpu_connector.utils import discover_gpu_kv_format
from lmcache.v1.gpu_connector.xds_readv import XDSReadvPlan, plan_layerwise_readv_batch
from lmcache.v1.memory_management import MemoryFormat, MemoryObj


class XDSReadvLayerwiseGPUConnector(VLLMPagedMemLayerwiseGPUConnector):
    """Persist each layer as KV_2TD; use the existing kernel for stores/fallback.

    Selected only by xds_layerwise_readv. Old KV_T2D entries can still be loaded
    by the fallback kernel. MLA and blending must use their original connectors.
    """

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Return the explicit [K/V, token, hidden] on-disk tensor shape."""
        return torch.Size([2, num_tokens, self.hidden_dim_size])

    def plan_layerwise_xds_readv(
        self,
        chunks: Sequence[
            tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int, int]
        ],
        io_unit: int,
        **kwargs: Any,
    ) -> Optional[list[list[XDSReadvPlan]]]:
        """Return validated layer/chunk plans, or None before any direct DMA."""
        self.initialize_kvcaches_ptr(**kwargs)
        targets = self.kvcaches
        slots = kwargs.get("slot_mapping_cpu", kwargs.get("slot_mapping"))
        if (
            self.use_mla
            or not isinstance(targets, list)
            or len(targets) != self.num_layers
            or not isinstance(slots, torch.Tensor)
            or any(kwargs.get("vllm_cached_tokens", 0) > c[3] for c in chunks)
        ):
            return None
        fmt = discover_gpu_kv_format(
            targets, EngineType.VLLM, layout_hints=self.layout_hints
        )
        layout = {
            lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS: "2NB",
            lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS: "NB2",
        }.get(fmt)
        if layout is None:
            return None
        return plan_layerwise_readv_batch(targets, chunks, slots, io_unit, layout)

    def batched_from_gpu(
        self,
        memory_objs: list[list[MemoryObj]],
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> Iterator[None]:
        """Store KV-major chunks with one existing gather kernel per chunk.

        Yield once per layer and once after final synchronization, matching the
        layerwise connector protocol. Source block ownership belongs to caller.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        slots = kwargs["slot_mapping"]
        fmt = discover_gpu_kv_format(
            self.kvcaches, EngineType.VLLM, layout_hints=self.layout_hints
        )
        current = torch.cuda.current_stream()
        for layer, objects in enumerate(memory_objs):
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current)
                for obj, start, end in zip(objects, starts, ends, strict=True):
                    if obj.tensor is None or obj.tensor.shape != self.get_shape(
                        end - start
                    ):
                        raise ValueError("xDS layerwise KV-major store shape mismatch")
                    obj.metadata.fmt = MemoryFormat.KV_2TD
                    lmc_ops.single_layer_kv_transfer(
                        obj.tensor,
                        self.kvcaches[layer],
                        slots[start:end],
                        lmc_ops.TransferDirection.D2H,
                        fmt,
                        token_major=False,
                    )
            yield
            if kwargs.get("sync", False):
                self.store_stream.synchronize()
        yield

    def batched_to_gpu(
        self,
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> Iterator[None]:
        """Fallback for unsupported direct layouts and legacy token-major data.

        Consume one list of MemoryObjs per layer via send(). Kernel dispatch uses
        each object's explicit format; no stored data is silently reinterpreted.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        slots = kwargs["slot_mapping"]
        fmt = discover_gpu_kv_format(
            self.kvcaches, EngineType.VLLM, layout_hints=self.layout_hints
        )
        current = torch.cuda.current_stream()
        for layer in range(self.num_layers):
            objects = yield
            if kwargs.get("sync", False):
                current.wait_stream(self.load_stream)
            if len(objects) == 1 and getattr(objects[0], "chunk_count", 0) == len(
                starts
            ):
                objects_and_slots = [
                    (
                        objects[0],
                        torch.cat(
                            [slots[a:b] for a, b in zip(starts, ends, strict=True)]
                        ),
                    )
                ]
            else:
                objects_and_slots = [
                    (obj, slots[a:b])
                    for obj, a, b in zip(objects, starts, ends, strict=True)
                ]
            with torch.cuda.stream(self.load_stream):
                for obj, mapping in objects_and_slots:
                    if obj.metadata.fmt not in (
                        MemoryFormat.KV_2TD,
                        MemoryFormat.KV_T2D,
                    ):
                        raise ValueError("unsupported xDS layerwise source format")
                    lmc_ops.single_layer_kv_transfer(
                        obj.tensor,
                        self.kvcaches[layer],
                        mapping,
                        lmc_ops.TransferDirection.H2D,
                        fmt,
                        token_major=obj.metadata.fmt == MemoryFormat.KV_T2D,
                    )
                events = kwargs.get("h2d_copy_completion_events")
                if events is not None:
                    event = torch.cuda.Event()
                    event.record(self.load_stream)
                    events.append(event)
        yield
        if kwargs.get("sync", False):
            current.wait_stream(self.load_stream)
        yield
