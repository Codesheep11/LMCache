# SPDX-License-Identifier: Apache-2.0
"""Opt-in scheduler-visible asynchronous direct xDS KV restores for vLLM V1."""

# Standard
from dataclasses import replace
from typing import Any

# Third Party
import torch

# First Party
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    ReqMeta,
    extract_request_configs,
)
from lmcache.v1.gpu_connector.xds_async import XDSAsyncLoader


class XDSAsyncMetadata(LMCacheConnectorMetadata):
    """Normal store metadata plus loads whose blocks are reserved by the scheduler."""

    def __init__(self, requests: list[ReqMeta], loads: list[ReqMeta]) -> None:
        self.requests = requests
        self.loads = loads


class XDSAsyncConnector(LMCacheConnectorV1Dynamic):
    """Restore independent requests in the background and report per-rank readiness.

    Select with kv_connector='XDSAsyncConnector' and kv_connector_module_path=
    'lmcache.integration.vllm.xds_async_connector'. Requires CUDA, a single KV
    cache group, text-only prompts and non-layerwise, non-MLA direct xDS I/O.
    The synchronous connector remains available. vLLM aggregates readiness across
    all ranks and retains blocks for aborted in-flight loads until completion.
    """

    def __init__(
        self, vllm_config: Any, role: Any, kv_cache_config: Any = None
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        config = self._lmcache_engine.config
        if (
            not config.xds_backend
            or config.use_layerwise
            or not config.get_extra_config_value("xds_direct_kv_io", True)
            or config.enable_blending
            or config.enable_async_loading
        ):
            raise ValueError("XDSAsyncConnector requires non-layerwise direct xDS I/O")
        self._block_size = vllm_config.cache_config.block_size
        self._pending_loads: dict[str, ReqMeta] = {}
        self._loader: XDSAsyncLoader | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register fixed CUDA targets and create the single I/O owner thread."""
        if not kv_caches or any(t.device.type != "cuda" for t in kv_caches.values()):
            raise ValueError(
                "asynchronous direct xDS currently requires CUDA KV tensors"
            )
        super().register_kv_caches(kv_caches)
        self._loader = XDSAsyncLoader()

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Reserve destination blocks without scheduling model execution."""
        if getattr(request, "mm_features", None):
            raise ValueError(
                "asynchronous direct xDS currently requires text-only prompts"
            )
        count = self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        return count, count is not None and count > 0

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        """Snapshot target blocks; later allocation calls do not reload."""
        adapter = self._lmcache_engine
        adapter.update_state_after_alloc(request, num_external_tokens)
        if num_external_tokens <= 0:
            return
        groups = blocks.get_block_ids()
        if len(groups) != 1:
            raise ValueError("asynchronous direct xDS requires one KV cache group")
        spec = adapter.load_specs[request.request_id]
        count = spec.lmcache_cached_tokens
        ids = torch.tensor(groups[0], dtype=torch.long)
        slots = (
            ids[:, None] * self._block_size + torch.arange(self._block_size)
        ).flatten()[:count]
        if len(slots) != count:
            raise ValueError("allocated KV blocks do not cover the xDS restore")
        self._pending_loads[request.request_id] = ReqMeta(
            req_id=request.request_id,
            token_ids=list(request.all_token_ids[:count]),
            slot_mapping=slots,
            load_spec=replace(spec, can_load=True),
            request_configs=extract_request_configs(request.sampling_params),
        )
        # Keep the cached-token accounting for normal store metadata, but never
        # restore again when vLLM schedules the remaining prompt after readiness.
        spec.can_load = False
        spec.allow_compact_metadata = adapter.kv_role != "kv_producer"

    def build_connector_meta(self, scheduler_output: Any) -> XDSAsyncMetadata:
        """Dispatch reserved loads even without a forward or after an abort."""
        normal = super().build_connector_meta(scheduler_output)
        loads = list(self._pending_loads.values())
        self._pending_loads.clear()
        return XDSAsyncMetadata(normal.requests, loads)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Enqueue restores without holding up other requests' model execution."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, XDSAsyncMetadata)
        for request in metadata.loads:
            assert self._loader is not None
            self._loader.submit(request.req_id, self._load, request)
        super().start_load_kv(forward_context, **kwargs)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[None, set[str] | None]:
        """Report a receive only after its DMA, visibility fence and validation finish.

        Aborts do not cancel DMA. vLLM keeps their blocks until this same receive
        completion arrives. Errors propagate instead of releasing unsafe targets.
        """
        ready = self._loader.poll() if self._loader is not None else set()
        return None, ready or None

    def shutdown(self) -> None:
        """Drain outstanding I/O before the adapter unregisters serving KV storage."""
        if self._loader is not None:
            self._loader.close()
        super().shutdown()

    def _load(self, request: ReqMeta) -> None:
        adapter = self._lmcache_engine
        engine = adapter.lmcache_engine
        assert engine is not None and request.load_spec is not None
        caches = list(adapter.kv_caches.values())
        spec = request.load_spec
        with torch.cuda.device(caches[0].device):
            mask = torch.ones(len(request.token_ids), dtype=torch.bool)
            start = (
                spec.vllm_cached_tokens
                // adapter.config.chunk_size
                * adapter.config.chunk_size
            )
            mask[:start] = False
            restored = engine.retrieve(
                request.token_ids,
                mask,
                kvcaches=caches,
                slot_mapping=request.slot_mapping.to(caches[0].device),
                slot_mapping_cpu=request.slot_mapping,
                vllm_cached_tokens=spec.vllm_cached_tokens,
                request_configs=request.request_configs,
                req_id=request.req_id,
            )
            if (
                int(restored.sum().item())
                < spec.lmcache_cached_tokens - spec.vllm_cached_tokens
            ):
                raise RuntimeError(
                    f"incomplete asynchronous xDS restore: {request.req_id}"
                )
            engine.lookup_unpin(request.req_id)
