# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.health_monitor.base import HealthMonitor

# Standard
import asyncio
import gc
import json
import multiprocessing
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    LayerCacheEngineKey,
    _lmcache_nvtx_annotate,
    compress_slot_mapping,
    convert_tokens_to_list,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.memory_management import CuFileMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import (  # noqa: E501
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.storage_backend.transfer_result import (
    BackendTransferResult,
    BackendWriteSubmissionResult,
)
from lmcache.v1.system_detection import NUMADetector, NUMAMapping
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    SegmentTokenDatabase,
    TokenDatabase,
)

logger = init_logger(__name__)

# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]


class CacheEngineEndSignal:
    pass


class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.token_database = token_database
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        # save_only_first_rank only works when use mla
        self.save_only_first_rank = (
            self.config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )

        if self.save_only_first_rank and self.gpu_connector is not None:
            self.broadcast_stream = (
                self.gpu_connector.load_stream
                if hasattr(self.gpu_connector, "load_stream")
                else torch.cuda.Stream()
            )

        self.enable_controller = config.enable_controller

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        lmcache_worker_ids = config.get_lmcache_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        # lmcache_worker_ids is empty means start on all workers
        if (
            self.enable_controller
            and self.metadata.role != "scheduler"
            and (not lmcache_worker_ids or metadata.worker_id in lmcache_worker_ids)
        ):
            self.lmcache_worker = LMCacheWorker(config, metadata, self)
        else:
            self.lmcache_worker = None
            logger.info(
                "LMCacheWorker is not initialized (related configs: "
                "enable_controller: %s, role: %s, worker_id: %s, worker_ids: %s).",
                self.enable_controller,
                self.metadata.role,
                self.metadata.worker_id,
                lmcache_worker_ids,
            )

        self.async_loading = config.enable_async_loading
        self.event_manager = EventManager()

        self.use_layerwise = config.use_layerwise

        # TODO: support save_only_first_rank when use layerwise
        # if use_layerwise is True, all ranks will initialize the storage_manager
        # if save_only_first_rank is False, all ranks will initialize
        # the storage_manager
        # if save_only_first_rank is True, only the first rank and
        # lookup server workers will initialize the storage_manager
        self.storage_manager: Optional[StorageManager] = None

        # KV events
        self.kv_events_enabled = False
        self.kv_events_enabled = config.enable_kv_events
        if self.kv_events_enabled:
            self.kv_events: List[CacheStoreEvent] = []
            logger.info("KV events are enabled.")
        else:
            logger.info("KV events are disabled.")

        # HACK: remove this in the future
        # NOTE (Jiayi): This is currently used to support
        # dropping the kv cache from the buffer in PD backend
        # at decoder.
        self.remove_after_retrieve = config.enable_pd and config.pd_role == "receiver"

        # asymmetric store/retrieve location can be specified
        # this is typically used (but not limited) in PD system
        self.store_location = config.store_location
        self.retrieve_locations = config.retrieve_locations

        self.num_layers = metadata.kv_shape[0]
        self.fmt = None
        # The direct xDS backend persists a whole multi-layer token block in
        # one GPU-resident blob.  It therefore uses the connector's native
        # multi-layer layout even though LMCache's layerwise API is disabled.
        if (config.xds_backend or config.cascadekv_backend) and not self.use_layerwise:
            self.fmt = MemoryFormat.KV_2LTD
        if self.use_layerwise:
            if metadata.use_mla:
                self.fmt = MemoryFormat.KV_MLA_FMT
            elif config.enable_blending or config.get_extra_config_value(
                "xds_layerwise_readv", False
            ):
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D
        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT

        # NOTE(ApostaC): we haven't support lookup-cache yet
        self.lookup_cache: dict[CacheEngineKey, Any] = {}

        # lookup_id -> {location -> [pinned keys]}
        self.lookup_pins: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        InitializeUsageContext(config, metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        # Initialize PinMonitor singleton with config
        PinMonitor.GetOrCreate(config)

        self.post_inited = False

        # Flag to control KVCache Check logging (can be toggled via API)
        self.kvcache_check_log_enabled = False

        gc.collect()
        if not config.py_enable_gc:
            gc.disable()

        # Health monitor reference (injected by LMCacheManager)
        self._health_monitor: Optional["HealthMonitor"] = None

        # Flag to indicate if initialization failed (irrecoverable error)
        self._init_failed = False

    def set_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """
        Set the health monitor reference.

        This is called by LMCacheManager after creating the HealthMonitor
        to inject the reference into the engine.

        Args:
            health_monitor: The HealthMonitor instance from LMCacheManager
        """
        self._health_monitor = health_monitor

    def is_healthy(self) -> bool:
        """
        Check if the LMCache system is healthy.

        This method returns False if:
        - Initialization failed (irrecoverable error)
        - HealthMonitor reports unhealthy

        If no health monitor is set and initialization succeeded,
        it returns True (assume healthy).

        Returns:
            bool: True if healthy, False otherwise
        """
        if self._init_failed:
            return False
        if self._health_monitor is not None:
            return self._health_monitor.is_healthy()
        return True

    def _get_req_id(self, kwargs: dict) -> str:
        """Extracts request ID from kwargs for logging."""
        return kwargs.get("req_id", "unspecified")

    def mark_init_failed(self, reason: str = "") -> None:
        """
        Mark the engine as having failed initialization.

        This is called by LMCacheManager when an irrecoverable error occurs
        during initialization or post_init. Once marked, is_healthy() will
        always return False, causing the system to fall back to recomputation.

        Args:
            reason: Optional reason string for logging
        """
        self._init_failed = True
        if reason:
            logger.error("LMCacheEngine marked as init failed: %s", reason)
        else:
            logger.error("LMCacheEngine marked as init failed")

    def post_init(self, **kwargs) -> None:
        if not self.post_inited:
            logger.info("Post initializing LMCacheEngine")
            lookup_server_worker_ids = self.config.get_lookup_server_worker_ids(
                self.metadata.use_mla, self.metadata.world_size
            )
            if (
                self.lmcache_worker is not None
                or self.use_layerwise
                or not self.save_only_first_rank
                or self.metadata.is_first_rank()
                or len(lookup_server_worker_ids) == 0
                or self.metadata.worker_id in lookup_server_worker_ids
            ):
                logger.info(
                    f"Initialize storage manager on rank {self.metadata.worker_id}, "
                    f"use layerwise: {self.use_layerwise},"
                    f"save only first rank: {self.save_only_first_rank}"
                )
                async_lookup_server = kwargs.get("async_lookup_server", None)
                self.storage_manager = StorageManager(
                    self.config,
                    self.metadata,
                    event_manager=self.event_manager,
                    lmcache_worker=self.lmcache_worker,
                    async_lookup_server=async_lookup_server,
                )
            self.post_inited = True

    def freeze(self, enabled: bool) -> None:
        """
        Set the freeze mode for the cache engine.

        When freeze mode is enabled:
        - All store operations will be skipped (no new data stored)
        - Only local_cpu backend will be used for retrieval
        - No admit/evict messages will be generated
        This protects the local_cpu hot cache from changes.

        Args:
            enabled (bool): Whether to enable freeze mode
        """
        if self.storage_manager is not None:
            self.storage_manager.set_freeze(enabled)

    def is_frozen(self) -> bool:
        """
        Get the current freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_frozen()
        return False

    def set_hot_cache(self, enabled: bool) -> None:
        """
        Dynamically enable or disable the LocalCPUBackend hot cache.

        When disabled, the existing hot cache entries will be cleared
        and no new data will be written to the hot cache.

        Args:
            enabled (bool): Whether to enable hot cache
        """
        if self.storage_manager is not None:
            self.storage_manager.set_hot_cache(enabled)

    def is_hot_cache_enabled(self) -> bool:
        """
        Get the current hot cache status of LocalCPUBackend.

        Returns:
            bool: True if hot cache is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_hot_cache_enabled()
        return False

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store operation")
            return

        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )

        if self._is_passive():
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
            return

        assert self.storage_manager is not None

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        # Initialize num_to_store_tokens to avoid reference before assignment
        num_to_store_tokens = 0

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device="cuda"
            )

        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=False,
        )

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store operation for %d tokens",
                num_to_store_tokens,
            )
            return

        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts: List[int] = []
        ends: List[int] = []
        keys: List[CacheEngineKey] = []

        tot_kv_size = 0
        tot_token_num = 0

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        with store_stats.profile_process_tokens():
            prev_key = 0
            for start, end, key in self.token_database.process_tokens(
                tokens,
                hashes,
                offsets,
                mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)
                num_tokens = end - start
                starts.append(start)
                ends.append(end)
                keys.append(key)

                # Create KV event
                if self.kv_events_enabled:
                    stored_event = CacheStoreEvent(
                        block_hashes=[key.chunk_hash],
                        parent_block_hash=None if start == 0 else prev_key,
                        token_ids=[],
                        block_size=num_tokens,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    if tokens is not None:
                        stored_event.token_ids = convert_tokens_to_list(
                            tokens,
                            start,
                            end,
                        )
                        if isinstance(tokens, torch.Tensor):
                            stored_event.medium = tokens.device
                    elif hashes is not None:
                        stored_event.token_ids = hashes[start : end + 1]
                    logger.debug(
                        (
                            "Added kv cache event '%s' to kv cache events queue"
                            % stored_event
                        )
                    )
                    self.kv_events.append(stored_event)
                    prev_key = key.chunk_hash

        if not keys:
            return

        with store_stats.profile_from_gpu():
            direct_transfer = self.storage_manager.batched_put_from_gpu(
                keys=keys,
                starts=starts,
                ends=ends,
                gpu_connector=self.gpu_connector,
                location=self.store_location,
                **kwargs,
            )
        if isinstance(direct_transfer, BackendTransferResult):
            if len(direct_transfer.completed) != len(keys):
                raise RuntimeError(
                    "backend transfer result length does not match key count"
                )
            tot_token_num = sum(
                end - start
                for completed, start, end in zip(
                    direct_transfer.completed, starts, ends, strict=True
                )
                if completed
            )
            tot_kv_size = direct_transfer.completed_bytes
        elif isinstance(direct_transfer, BackendWriteSubmissionResult):
            if len(direct_transfer.accepted) != len(keys):
                raise RuntimeError(
                    "backend write submission length does not match key count"
                )
            tot_token_num = sum(
                end - start
                for accepted, start, end in zip(
                    direct_transfer.accepted, starts, ends, strict=True
                )
                if accepted
            )
            tot_kv_size = direct_transfer.accepted_bytes
        else:
            direct_transfer = None

        if direct_transfer is not None:
            self.stats_monitor.on_store_finished(store_stats, tot_token_num)
            tot_time = store_stats.time_to_store()
            logger.info(
                "[req_id=%s] Stored %d out of total %d tokens through direct "
                "backend. size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s",
                req_id,
                tot_token_num,
                num_to_store_tokens,
                tot_kv_size / 1024**3,
                tot_time * 1000,
                tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            )
            return

        memory_objs: List[MemoryObj] = []
        for start, end in zip(starts, ends, strict=True):
            num_tokens = end - start
            memory_obj = self.storage_manager.allocate(
                self.metadata.get_shapes(num_tokens),
                self.metadata.get_dtypes(),
                busy_loop=self.config.get_extra_config_value("force_store_wait", False),
                fmt=self.fmt,
            )
            if memory_obj is None:
                logger.warning(
                    "Local cpu memory under pressure so choosing to store only "
                    "%d total chunks of KV cache.",
                    len(memory_objs),
                )
                break
            memory_objs.append(memory_obj)
            tot_kv_size += memory_obj.get_size()
            tot_token_num += num_tokens

        if not memory_objs:
            return
        if len(memory_objs) != len(keys):
            keys = keys[: len(memory_objs)]
            starts = starts[: len(memory_objs)]
            ends = ends[: len(memory_objs)]

        with store_stats.profile_from_gpu():
            self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)

        with store_stats.profile_put():
            transfer_spec = kwargs.get("transfer_spec", None)
            # TODO: we implicitly rely on batched_put to call ref_count_down
            # this management should be done in a cleaner way
            self.storage_manager.batched_put(
                keys,
                memory_objs,
                transfer_spec=transfer_spec,
                location=self.store_location,
            )

        self.stats_monitor.on_store_finished(
            store_stats,
            tot_token_num,
        )
        tot_time = store_stats.time_to_store()

        logger.info(
            "[req_id=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms",
            req_id,
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
            store_stats.put_time * 1000,
        )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store_layer operation")
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Layerwise store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=True,
        )

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store_layer for %d tokens",
                num_to_store_tokens,
            )
            # Still need to yield to avoid StopIteration
            for layer_id in range(self.num_layers):
                yield
            return

        starts = []
        ends = []
        keys = []
        memory_objs = []
        layer_major_shapes: list[torch.Size] = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        use_layer_major_store_allocation = (
            bool(
                self.config.get_extra_config_value(
                    "cascadekv_layer_major_store_allocation", False
                )
            )
            and not self.config.cascadekv_backend
        )
        block_blob_layout = bool(
            self.config.cascadekv_backend
            or self.config.get_extra_config_value("xds_layerwise_block_blob", False)
        )
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        prev_key = 0
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = (
                None if block_blob_layout else key.split_layers(self.num_layers)
            )
            lookup_key = key if block_blob_layout else keys_multi_layer[0]
            if self.storage_manager.contains(lookup_key, self.retrieve_locations):
                continue

            # Allocate the memory object
            num_tokens = end - start
            kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

            if use_layer_major_store_allocation:
                memory_objs_multi_layer = []
            else:
                memory_objs_multi_layer = self.storage_manager.batched_allocate(
                    kv_shape_single_layer,
                    kv_dtype,
                    batch_size=self.num_layers,
                    fmt=self.fmt,
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                )

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(key if block_blob_layout else keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            layer_major_shapes.append(kv_shape_single_layer)
            tot_token_num += num_tokens

            # Create KV event
            if self.kv_events_enabled and tokens is not None:
                stored_event = CacheStoreEvent(
                    block_hashes=[key.chunk_hash],
                    parent_block_hash=None if start == 0 else prev_key,
                    token_ids=[],
                    block_size=num_tokens,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                )
                if tokens is not None:
                    stored_event.token_ids = convert_tokens_to_list(
                        tokens,
                        start,
                        end,
                    )
                    if isinstance(tokens, torch.Tensor):
                        stored_event.medium = tokens.device
                logger.debug(
                    f"Added kv cache event '{stored_event}' to kv cache events queue"
                )
                self.kv_events.append(stored_event)
                prev_key = key.chunk_hash

        if use_layer_major_store_allocation and keys:
            allocated_objects: list[MemoryObj] = []
            allocation_failed = False
            run_start = 0
            while run_start < len(layer_major_shapes) and not allocation_failed:
                run_end = run_start + 1
                while (
                    run_end < len(layer_major_shapes)
                    and layer_major_shapes[run_end] == layer_major_shapes[run_start]
                ):
                    run_end += 1
                run_size = run_end - run_start
                for _layer_id in range(self.num_layers):
                    layer_objects = self.storage_manager.batched_allocate(
                        layer_major_shapes[run_start],
                        kv_dtype,
                        batch_size=run_size,
                        fmt=self.fmt,
                        busy_loop=self.config.get_extra_config_value(
                            "force_store_wait", False
                        ),
                    )
                    if layer_objects is None:
                        allocation_failed = True
                        break
                    allocated_objects.extend(layer_objects)
                    for chunk_offset, memory_obj in enumerate(layer_objects):
                        memory_objs[run_start + chunk_offset].append(memory_obj)
                run_start = run_end

            if allocation_failed:
                for memory_obj in allocated_objects:
                    memory_obj.ref_count_down()
                logger.warning(
                    "Local cpu memory under pressure while allocating "
                    "layer-major CascadeKV staging; skipping this store."
                )
                starts.clear()
                ends.clear()
                keys.clear()
                memory_objs.clear()
                tot_token_num = 0

        if keys:
            # Transpose payload objects into layer-major format. Block-blob
            # keys remain one per token block and are never split by layer.
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            if not block_blob_layout:
                keys = [list(row) for row in zip(*keys, strict=False)]

            # Calculate total KV size for logging
            tot_kv_size = sum(
                mo.get_size() for layer_objs in memory_objs for mo in layer_objs
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

            t_start = time.perf_counter()
            mem_obj_generator = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )

            next(mem_obj_generator)

            for layer_id in range(self.num_layers):
                yield
                next(mem_obj_generator)
                if not block_blob_layout:
                    self.storage_manager.batched_put(
                        keys[layer_id],
                        memory_objs[layer_id],
                        location=self.store_location,
                    )
            if block_blob_layout:
                d2h_event = self.gpu_connector.record_transfer_event("store")
                self.storage_manager.blockwise_batched_put(
                    keys,
                    memory_objs,
                    location=self.store_location,
                    completion_event=d2h_event,
                )

            tot_time = time.perf_counter() - t_start
            logger.info(
                "[req_id=%s] Stored %d out of total %d tokens. "
                "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s",
                req_id,
                tot_token_num,
                len(tokens),
                tot_kv_size / 1024**3,
                tot_time * 1000,
                tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            )
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        yield

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        fine_profile_enabled = bool(
            self.config.get_extra_config_value("retrieve_fine_profile", False)
        )
        fine_profile: Optional[dict[str, Any]] = None
        if fine_profile_enabled:
            fine_profile = {
                "schema": 1,
                "worker_id": self.metadata.worker_id,
                "request_start_ns": time.monotonic_ns(),
            }

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve operation")
            return torch.zeros(len(tokens), dtype=torch.bool)

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)
        if fine_profile is not None:
            fine_profile["req_id"] = req_id

        tot_kv_size = 0

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="retrieve",
            kwargs=kwargs,
            token_count=num_required_tokens,
            require_req_id=True,
        )

        transfer_start_ns = time.monotonic_ns()
        retrieve_stats = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[ProcessedChunk] = []
        if not self._is_passive():
            with retrieve_stats.profile_process_tokens():
                if fine_profile is not None:
                    fine_profile["process_start_ns"] = time.monotonic_ns()
                    process_kwargs = dict(kwargs)
                    process_kwargs["_lmcache_retrieve_profile"] = fine_profile
                else:
                    process_kwargs = kwargs
                if self.async_loading:
                    reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                        tokens,
                        mask,
                        ret_mask,
                        **process_kwargs,
                    )
                else:
                    reordered_chunks, tot_kv_size = self._process_tokens_internal(
                        tokens,
                        mask,
                        ret_mask,
                        **process_kwargs,
                    )
                if fine_profile is not None:
                    fine_profile["process_end_ns"] = time.monotonic_ns()

        if self.save_only_first_rank:
            with retrieve_stats.profile_broadcast():
                with torch.cuda.stream(self.broadcast_stream):
                    self._broadcast_or_receive_memory_objs(
                        reordered_chunks,
                        ret_mask,
                    )

                # if self.gpu_connector has load_stream, self.broadcast_stream is equals
                # to self.gpu_connector.load_stream, the broadcast and to_gpu operation
                # will execute sequentially within the stream.
                # if self.gpu_connector does not have load_stream, self.broadcast_stream
                # is created by torch.cuda.Stream(), we need to synchronize broadcast
                # operation, and then process to_cpu operation.
                if not hasattr(self.gpu_connector, "load_stream"):
                    self.broadcast_stream.synchronize()

        # NOTE(Jiayi): memory_obj doesn't have to be a pinned
        # cpu tensor for the sake of performance.
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        if len(reordered_chunks) > 0:
            with retrieve_stats.profile_to_gpu():
                _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
                transfer_start_event = None
                if fine_profile is not None:
                    fine_profile["h2d_host_start_ns"] = time.monotonic_ns()
                    transfer_start_event = self.gpu_connector.record_transfer_event(
                        "load", enable_timing=True
                    )
                self.gpu_connector.batched_to_gpu(
                    list(memory_objs), list(starts), list(ends), **kwargs
                )
                if fine_profile is not None:
                    transfer_end_event = self.gpu_connector.record_transfer_event(
                        "load", enable_timing=True
                    )
                    transfer_end_event.synchronize()
                    fine_profile["h2d_host_end_ns"] = time.monotonic_ns()
                    assert transfer_start_event is not None
                    fine_profile["h2d_cuda_ms"] = transfer_start_event.elapsed_time(
                        transfer_end_event
                    )

        # TODO(Jiayi): Remove the following for loop with batched operations
        # TODO(Jiayi): Need to refactor the `remove_after_retrieve` logic.
        if fine_profile is not None:
            fine_profile["release_start_ns"] = time.monotonic_ns()
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                assert self.storage_manager is not None
                self.storage_manager.remove(key, self.retrieve_locations)
            if not self.async_loading:
                memory_obj.ref_count_down()
        if fine_profile is not None:
            fine_profile["release_end_ns"] = time.monotonic_ns()

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(
            retrieve_stats,
            retrieved_tokens,
        )
        transfer_end_ns = time.monotonic_ns()
        if fine_profile is not None:
            fine_profile.update(
                {
                    "request_end_ns": time.monotonic_ns(),
                    "bytes": tot_kv_size,
                    "chunk_count": len(reordered_chunks),
                    "retrieved_tokens": int(retrieved_tokens),
                }
            )
            logger.info(
                "[LMCACHE_RETRIEVE_FINE_PROFILE] %s",
                json.dumps(fine_profile, separators=(",", ":")),
            )
        onload_time = retrieve_stats.time_to_retrieve()
        # The retrieved may be larger than the need_to_load
        # Example (page_size=16, chunk_size=256):
        #
        # chunks:  [0..255]                [256..511]
        # pages:   [0..15]...[240..255]    [256..271][272..287] ...
        #
        # num_computed_tokens = 288 => vLLM already has [0..287] (18 pages)
        # LMCache hit_prefix_tokens = 512 => cache covers [0..511] (2 chunks)
        #
        # Skip chunk 1, retrieve chunk 2, overwrite [256..287] (32-token overlap)
        # need_to_load: 512 - 288 = 224 tokens
        # retrieved: 256 tokens
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d required tokens "
                "(from %d total tokens). size: %.4f gb, "
                "cost %.4f ms, throughput: %.4f GB/s;",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
                tot_kv_size / 1024**3,
                onload_time * 1000,
                tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
            )
            if self.config.get_extra_config_value("benchmark_transfer_metrics", False):
                logger.info(
                    "[LMCACHE_SSD_GPU_METRICS] %s",
                    json.dumps(
                        {
                            "worker_id": self.metadata.worker_id,
                            "mode": "direct",
                            "bytes": tot_kv_size,
                            "seconds": onload_time,
                            "transfer_start_ns": transfer_start_ns,
                            "transfer_end_ns": transfer_end_ns,
                            "retrieved_tokens": int(retrieved_tokens),
                        },
                        separators=(",", ":"),
                    ),
                )
        return ret_mask

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """
        cache_retrieve_start_ns = time.monotonic_ns()

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve_layer operation")
            yield torch.zeros(len(tokens), dtype=torch.bool)
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []
        first_layer_keys = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        block_blob_layout = bool(
            self.config.cascadekv_backend
            or self.config.get_extra_config_value("xds_layerwise_block_blob", False)
        )

        location = None
        key_scan_start_ns = time.monotonic_ns()
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)

            first_layer_key = key if block_blob_layout else key.get_first_layer()

            # NOTE: Only check the first layer
            if current_location := self.storage_manager.contains(
                first_layer_key, self.retrieve_locations
            ):
                if location is None:
                    location = current_location
                else:
                    # TODO(Jiayi): Support multi-location retrieval in the future
                    assert location == current_location, (
                        "All retrieved keys should be from the same location "
                        "when use layerwise retrieval."
                        "Please support multi-location retrieval in the future."
                    )
            else:
                break

            starts.append(start)
            ends.append(end)
            keys.append(key)
            first_layer_keys.append(first_layer_key)

            ret_mask[start:end] = True
        key_scan_end_ns = time.monotonic_ns()

        transfer_start = time.perf_counter()
        transfer_start_ns = time.monotonic_ns()
        tot_kv_size = 0
        pipeline_h2d_complete_ns: Optional[int] = None
        pipeline_gpu_ready_complete_ns: Optional[int] = None
        timeline_trace: Optional[dict[str, Any]] = None
        if keys:
            layer_key_build_start_ns = time.monotonic_ns()
            keys_layer_major = None
            if not block_blob_layout:
                keys_layer_major = [first_layer_keys]
                keys_layer_major.extend(
                    [LayerCacheEngineKey.from_parent(key, layer_id) for key in keys]
                    for layer_id in range(1, self.num_layers)
                )
            layer_key_build_end_ns = time.monotonic_ns()
            if self.config.get_extra_config_value("cascadekv_tp_read_barrier", False):
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            timeline_enabled = bool(
                self.config.get_extra_config_value(
                    "cascadekv_layerwise_timeline_trace", False
                )
            )
            if timeline_enabled:
                timeline_trace = {
                    "schema": 2,
                    "trace_id": (f"{self.metadata.worker_id}-{transfer_start_ns}"),
                    "worker_id": self.metadata.worker_id,
                    "req_id": req_id,
                    "cache_retrieve_start_ns": cache_retrieve_start_ns,
                    "request_start_ns": transfer_start_ns,
                    "input_tokens": len(tokens),
                    "required_tokens": int(num_required_tokens),
                    "chunk_count": len(keys),
                    "key_scan_start_ns": key_scan_start_ns,
                    "key_scan_end_ns": key_scan_end_ns,
                    "layer_key_build_start_ns": layer_key_build_start_ns,
                    "layer_key_build_end_ns": layer_key_build_end_ns,
                    "layers": [
                        {"layer": layer_id} for layer_id in range(self.num_layers)
                    ],
                }
            if block_blob_layout and self.config.get_extra_config_value(
                "xds_layerwise_readv", False
            ):
                direct_layers = self.storage_manager.prepare_layerwise_readv(
                    keys, starts, ends, self.gpu_connector, location=location, **kwargs
                )
                if direct_layers is not None:
                    try:
                        yield torch.sum(ret_mask)
                        layer_count = 0
                        for result in direct_layers:
                            if not all(result.completed):
                                raise RuntimeError(
                                    "incomplete xDS direct layer restore"
                                )
                            layer_count += 1
                            tot_kv_size += result.completed_bytes
                            yield None
                        if layer_count != self.num_layers:
                            raise RuntimeError("xDS direct layer count mismatch")
                    finally:
                        direct_layers.close()
                    retrieved_tokens = torch.sum(ret_mask)
                    self.stats_monitor.on_retrieve_finished(
                        monitor_req_id, retrieved_tokens
                    )
                    if self.config.get_extra_config_value(
                        "benchmark_transfer_metrics", False
                    ):
                        logger.info(
                            "[LMCACHE_SSD_GPU_METRICS] %s",
                            json.dumps(
                                {
                                    "worker_id": int(self.metadata.worker_id),
                                    "mode": "layerwise_readv",
                                    "bytes": tot_kv_size,
                                    "seconds": time.perf_counter() - transfer_start,
                                    "retrieved_tokens": int(retrieved_tokens),
                                }
                            ),
                        )
                    yield ret_mask
                    return
            if block_blob_layout:
                get_generator = self.storage_manager.blockwise_batched_get(
                    keys,
                    self.num_layers,
                    location=location,
                    trace_context=timeline_trace,
                )
            else:
                assert keys_layer_major is not None
                get_generator = self.storage_manager.layerwise_batched_get(
                    keys_layer_major,
                    location=location,
                    trace_context=timeline_trace,
                )
            # Start the storage producer before initializing the GPU consumer.
            # The first Future is returned immediately, so metadata lookup and
            # staging preparation can overlap connector/stream setup without
            # changing the per-layer publication contract.
            first_layer_task = next(get_generator)

            assert_layerwise_gpu_connector(self.gpu_connector)

            direct_xds_layerwise = bool(
                self.config.xds_backend
                and self.config.get_extra_config_value(
                    "xds_layerwise_block_blob", False
                )
            )
            eager_source_release = bool(
                direct_xds_layerwise or self.config.cascadekv_backend
            )
            gpu_producer_enabled = bool(
                self.config.cascadekv_backend
                or self.config.get_extra_config_value(
                    "cascadekv_layerwise_gpu_producer", direct_xds_layerwise
                )
            )
            h2d_start_read_ahead_layers = 1
            h2d_max_submitted_ahead_layers = self.num_layers
            h2d_max_inflight_layers = 0
            prefetched_layer_tasks = [first_layer_task]
            if gpu_producer_enabled:
                h2d_start_read_ahead_layers = max(
                    1,
                    min(
                        self.num_layers,
                        int(
                            self.config.get_extra_config_value(
                                "cascadekv_h2d_start_read_ahead_layers", 1
                            )
                        ),
                    ),
                )
                configured_h2d_ahead = int(
                    self.config.get_extra_config_value(
                        "cascadekv_h2d_max_submitted_ahead_layers", 0
                    )
                )
                if configured_h2d_ahead > 0:
                    h2d_max_submitted_ahead_layers = max(
                        1, min(self.num_layers, configured_h2d_ahead)
                    )
                configured_h2d_inflight = int(
                    self.config.get_extra_config_value(
                        "cascadekv_h2d_max_inflight_layers", 0
                    )
                )
                if configured_h2d_inflight > 0:
                    h2d_max_inflight_layers = max(
                        1, min(self.num_layers, configured_h2d_inflight)
                    )
                if timeline_trace is not None:
                    timeline_trace["h2d_max_submitted_ahead_layers"] = (
                        h2d_max_submitted_ahead_layers
                    )
                    timeline_trace["h2d_max_inflight_layers"] = h2d_max_inflight_layers
                # Keep the transfer layer-granular, but optionally let the SSD
                # producer establish a short lead before H2D begins.  H2D can
                # then consume that lead while later layers are still being read,
                # avoiding the worst startup contention without aggregating GPU
                # transfers or delaying a layer on GPU compute.
                for _ in range(1, h2d_start_read_ahead_layers):
                    prefetched_layer_tasks.append(next(get_generator))
            consumer_kwargs = kwargs
            h2d_copy_completion_events: list[Any] = []
            if gpu_producer_enabled:
                # The serving thread waits on the exact CUDA event for the next
                # layer below. Avoid inserting wait_stream edges for every
                # prefetched layer into vLLM's compute stream.
                consumer_kwargs = dict(kwargs)
                consumer_kwargs["sync"] = False
                consumer_kwargs["h2d_scatter_overlap"] = bool(
                    self.config.get_extra_config_value(
                        "cascadekv_layerwise_h2d_scatter_overlap", False
                    )
                )
                if timeline_trace is not None:
                    consumer_kwargs["layerwise_timeline_trace"] = timeline_trace
                consumer_kwargs["h2d_copy_completion_events"] = (
                    h2d_copy_completion_events
                )
                self.gpu_connector.set_transfer_stream_priority(
                    "load",
                    int(
                        self.config.get_extra_config_value(
                            "cascadekv_h2d_stream_priority", -1
                        )
                    ),
                )
            mem_obj_consumer = self.gpu_connector.batched_to_gpu(
                starts, ends, **consumer_kwargs
            )
            next(mem_obj_consumer)

            def release_completed_layer(memory_objs: list[MemoryObj]) -> None:
                pooled_ids = self.storage_manager.release_layerwise_memory_objs(
                    memory_objs,
                    location=location,
                )
                for memory_obj in memory_objs:
                    if id(memory_obj) not in pooled_ids:
                        memory_obj.ref_count_down()
                    if memory_obj.is_pinned:
                        memory_obj.unpin()

            def submit_layer(
                layer_id: int,
            ) -> tuple[Any, list[MemoryObj], Optional[dict[str, Any]], Any]:
                nonlocal tot_kv_size
                task = (
                    prefetched_layer_tasks[layer_id]
                    if layer_id < len(prefetched_layer_tasks)
                    else next(get_generator)
                )
                assert task is not None
                layer_trace = (
                    timeline_trace["layers"][layer_id]
                    if timeline_trace is not None
                    else None
                )
                if layer_trace is not None:
                    layer_trace["future_wait_start_ns"] = time.monotonic_ns()
                mem_objs_layer = task.result()
                if layer_trace is not None:
                    layer_trace["future_wait_end_ns"] = time.monotonic_ns()
                layer_bytes = sum(mem_obj.get_size() for mem_obj in mem_objs_layer)
                tot_kv_size += layer_bytes
                transfer_start_event = None
                if layer_trace is not None:
                    layer_trace["bytes"] = layer_bytes
                    layer_trace["h2d_submit_start_ns"] = time.monotonic_ns()
                    transfer_start_event = self.gpu_connector.record_transfer_event(
                        "load", enable_timing=True
                    )
                mem_obj_consumer.send(mem_objs_layer)
                transfer_event = self.gpu_connector.record_transfer_event(
                    "load", enable_timing=layer_trace is not None
                )
                if layer_trace is not None:
                    layer_trace["h2d_submit_end_ns"] = time.monotonic_ns()
                return (
                    transfer_event,
                    mem_objs_layer,
                    layer_trace,
                    transfer_start_event,
                )

            def complete_layer(
                transfer: tuple[Any, list[MemoryObj], Optional[dict[str, Any]], Any],
            ) -> None:
                (
                    completed_event,
                    completed_memory_objs,
                    completed_layer_trace,
                    completed_start_event,
                ) = transfer
                if eager_source_release:
                    # The producer publishes this transfer only after its
                    # scatter event has completed and its source staging has
                    # been released. Synchronizing the same event again on the
                    # model thread only adds a CUDA API and host scheduling
                    # round trip.
                    assert not completed_memory_objs
                    return
                sync_start_ns = time.monotonic_ns()
                completed_event.synchronize()
                sync_end_ns = time.monotonic_ns()
                if completed_layer_trace is not None:
                    completed_layer_trace["h2d_sync_start_ns"] = sync_start_ns
                    completed_layer_trace["h2d_complete_ns"] = sync_end_ns
                    assert completed_start_event is not None
                    completed_layer_trace["h2d_cuda_ms"] = (
                        completed_start_event.elapsed_time(completed_event)
                    )
                release_completed_layer(completed_memory_objs)

            if gpu_producer_enabled:
                layer_transfers: list[
                    tuple[Any, list[MemoryObj], Optional[dict[str, Any]], Any]
                ] = []
                producer_errors: list[BaseException] = []
                producer_condition = threading.Condition()
                cuda_device = torch.cuda.current_device()
                consumer_completed_layers = 0
                h2d_pacing_wait_ns = 0

                def run_gpu_producer() -> None:
                    nonlocal h2d_pacing_wait_ns
                    nonlocal pipeline_h2d_complete_ns
                    nonlocal pipeline_gpu_ready_complete_ns
                    try:
                        torch.cuda.set_device(cuda_device)
                        with torch.inference_mode():
                            if h2d_start_read_ahead_layers > 1:
                                if timeline_trace is not None:
                                    timeline_trace["h2d_lead_wait_start_ns"] = (
                                        time.monotonic_ns()
                                    )
                                    timeline_trace["h2d_lead_layers"] = (
                                        h2d_start_read_ahead_layers
                                    )
                                prefetched_layer_tasks[-1].result()
                                if timeline_trace is not None:
                                    timeline_trace["h2d_lead_wait_end_ns"] = (
                                        time.monotonic_ns()
                                    )
                            for producer_layer_id in range(self.num_layers):
                                pace_wait_start_ns = time.monotonic_ns()
                                producer_layer_trace = (
                                    timeline_trace["layers"][producer_layer_id]
                                    if timeline_trace is not None
                                    else None
                                )
                                if producer_layer_trace is not None:
                                    producer_layer_trace["pacing_wait_start_ns"] = (
                                        pace_wait_start_ns
                                    )
                                if eager_source_release:
                                    # Drain every SSD-ready layer into final
                                    # paged KV immediately.  The exact scatter
                                    # event remains the compute-side readiness
                                    # fence, while CPU/GPU source staging can be
                                    # returned without waiting for model
                                    # progress.  This prevents prefetched
                                    # requests from exhausting a bounded source
                                    # pool ahead of the scheduled request.
                                    if producer_layer_trace is not None:
                                        producer_layer_trace["pacing_wait_kind"] = (
                                            "eager_source_release"
                                        )
                                elif h2d_max_inflight_layers > 0:
                                    # Keep the GPU load pipeline work-conserving
                                    # without coupling it to model progress. Once
                                    # all staging slots are occupied, wait only
                                    # for the oldest H2D+scatter event that makes
                                    # one slot reusable, then immediately submit
                                    # the next layer.
                                    completed_index = (
                                        producer_layer_id - h2d_max_inflight_layers
                                    )
                                    if completed_index >= 0:
                                        if producer_layer_trace is not None:
                                            producer_layer_trace["pacing_wait_kind"] = (
                                                "scatter_buffer_reuse"
                                            )
                                            producer_layer_trace[
                                                "pacing_wait_for_layer"
                                            ] = completed_index
                                        with producer_condition:
                                            completed_event = layer_transfers[
                                                completed_index
                                            ][0]
                                        if not direct_xds_layerwise:
                                            completed_event.synchronize()
                                    elif producer_layer_trace is not None:
                                        producer_layer_trace["pacing_wait_kind"] = (
                                            "initial_free_slot"
                                        )
                                else:
                                    if producer_layer_trace is not None:
                                        producer_layer_trace["pacing_wait_kind"] = (
                                            "consumer_window"
                                        )
                                    with producer_condition:
                                        producer_condition.wait_for(
                                            lambda layer_id=producer_layer_id: (
                                                layer_id
                                                < consumer_completed_layers
                                                + h2d_max_submitted_ahead_layers
                                            )
                                        )
                                pace_wait_end_ns = time.monotonic_ns()
                                h2d_pacing_wait_ns += (
                                    pace_wait_end_ns - pace_wait_start_ns
                                )
                                if producer_layer_trace is not None:
                                    producer_layer_trace["pacing_wait_end_ns"] = (
                                        pace_wait_end_ns
                                    )
                                transfer = submit_layer(producer_layer_id)
                                if eager_source_release:
                                    # Once this layer's scatter event completes,
                                    # its source is no longer needed by compute
                                    # and can be returned immediately. Keep the
                                    # event for the model thread, but clear the
                                    # objects to avoid a second release there.
                                    sync_start_ns = time.monotonic_ns()
                                    transfer[0].synchronize()
                                    sync_end_ns = time.monotonic_ns()
                                    if producer_layer_trace is not None:
                                        producer_layer_trace["h2d_sync_start_ns"] = (
                                            sync_start_ns
                                        )
                                        producer_layer_trace["h2d_complete_ns"] = (
                                            sync_end_ns
                                        )
                                        assert transfer[3] is not None
                                        producer_layer_trace["h2d_cuda_ms"] = transfer[
                                            3
                                        ].elapsed_time(transfer[0])
                                    if producer_layer_trace is not None:
                                        producer_layer_trace[
                                            "staging_release_start_ns"
                                        ] = time.monotonic_ns()
                                    release_completed_layer(transfer[1])
                                    if producer_layer_trace is not None:
                                        producer_layer_trace[
                                            "staging_release_end_ns"
                                        ] = time.monotonic_ns()
                                    transfer = (
                                        transfer[0],
                                        [],
                                        transfer[2],
                                        transfer[3],
                                    )
                                with producer_condition:
                                    layer_transfers.append(transfer)
                                    producer_condition.notify_all()
                            if timeline_trace is not None:
                                timeline_trace["h2d_all_submitted_ns"] = (
                                    time.monotonic_ns()
                                )
                            if len(h2d_copy_completion_events) != self.num_layers:
                                raise RuntimeError(
                                    "layerwise connector did not publish one H2D "
                                    "copy completion event per layer: "
                                    f"expected={self.num_layers}, "
                                    f"actual={len(h2d_copy_completion_events)}"
                                )
                            if eager_source_release:
                                # Every eager transfer was synchronized once
                                # before publication so its source staging
                                # could be reused safely. Both final events are
                                # already complete here.
                                pipeline_h2d_complete_ns = time.monotonic_ns()
                                pipeline_gpu_ready_complete_ns = (
                                    pipeline_h2d_complete_ns
                                )
                            else:
                                h2d_copy_completion_events[-1].synchronize()
                                pipeline_h2d_complete_ns = time.monotonic_ns()
                                layer_transfers[-1][0].synchronize()
                                pipeline_gpu_ready_complete_ns = time.monotonic_ns()
                            if timeline_trace is not None:
                                timeline_trace["pipeline_h2d_complete_ns"] = (
                                    pipeline_h2d_complete_ns
                                )
                            if timeline_trace is not None:
                                timeline_trace["pipeline_gpu_ready_complete_ns"] = (
                                    pipeline_gpu_ready_complete_ns
                                )
                    except BaseException as error:
                        with producer_condition:
                            producer_errors.append(error)
                    finally:
                        with producer_condition:
                            producer_condition.notify_all()

                gpu_producer = threading.Thread(
                    target=run_gpu_producer,
                    name=(f"cascadekv-h2d-producer-{self.metadata.worker_id}"),
                    daemon=True,
                )
                gpu_producer.start()

                # The first step reports the lookup result. Every later vLLM
                # callback waits only for the exact layer event while the
                # producer continues filling the load stream independently.
                yield torch.sum(ret_mask)

                for layer_id in range(self.num_layers):
                    consumer_submit_wait_start_ns = time.monotonic_ns()
                    with producer_condition:
                        producer_condition.wait_for(
                            lambda awaited_layer=layer_id: (
                                len(layer_transfers) > awaited_layer
                                or bool(producer_errors)
                            )
                        )
                        if producer_errors:
                            raise RuntimeError(
                                "CascadeKV H2D producer failed"
                            ) from producer_errors[0]
                    if timeline_trace is not None:
                        layer_trace = timeline_trace["layers"][layer_id]
                        layer_trace["consumer_submit_wait_start_ns"] = (
                            consumer_submit_wait_start_ns
                        )
                        layer_trace["consumer_submit_wait_end_ns"] = time.monotonic_ns()
                    complete_layer(layer_transfers[layer_id])
                    with producer_condition:
                        consumer_completed_layers = layer_id + 1
                        producer_condition.notify_all()
                    if layer_id < self.num_layers - 1:
                        yield None
                gpu_producer.join()
                if timeline_trace is not None:
                    timeline_trace["h2d_pacing_wait_ns"] = h2d_pacing_wait_ns
            else:
                pending_layer_transfers = deque()
                for layer_id in range(self.num_layers):
                    if layer_id == 0:
                        # SGLang needs the retrieved-token count on the first step.
                        yield torch.sum(ret_mask)
                    else:
                        yield None
                    pending_layer_transfers.append(submit_layer(layer_id))

                    # Keep exactly two layer buffers: while layer N+1 is loading,
                    # layer N may still be consumed by the GPU.
                    if len(pending_layer_transfers) >= 2:
                        complete_layer(pending_layer_transfers.popleft())

                while pending_layer_transfers:
                    complete_layer(pending_layer_transfers.popleft())
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        yield None

        # synchronize the last layer
        next(mem_obj_consumer)
        if self.config.get_extra_config_value("benchmark_transfer_metrics", False):
            torch.cuda.current_stream().synchronize()
        transfer_seconds = time.perf_counter() - transfer_start
        transfer_end_ns = time.monotonic_ns()
        pipeline_seconds = (
            (pipeline_h2d_complete_ns - transfer_start_ns) / 1_000_000_000
            if pipeline_h2d_complete_ns is not None
            else transfer_seconds
        )
        gpu_ready_seconds = (
            (pipeline_gpu_ready_complete_ns - transfer_start_ns) / 1_000_000_000
            if pipeline_gpu_ready_complete_ns is not None
            else transfer_seconds
        )
        retrieved_tokens = torch.sum(ret_mask)
        if timeline_trace is not None:
            timeline_trace["bytes"] = tot_kv_size
            timeline_trace["request_end_ns"] = transfer_end_ns
            timeline_trace["retrieved_tokens"] = int(retrieved_tokens)
            logger.info(
                "[LMCACHE_LAYERWISE_TIMELINE] %s",
                json.dumps(timeline_trace, separators=(",", ":")),
            )
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d out of total %d tokens",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
            )
            if self.config.get_extra_config_value("benchmark_transfer_metrics", False):
                logger.info(
                    "[LMCACHE_SSD_GPU_METRICS] %s",
                    json.dumps(
                        {
                            "worker_id": self.metadata.worker_id,
                            "mode": "layerwise",
                            "endpoint": (
                                "h2d_copy_complete"
                                if pipeline_h2d_complete_ns is not None
                                else "consumer_complete"
                            ),
                            "bytes": tot_kv_size,
                            "seconds": pipeline_seconds,
                            "gpu_ready_seconds": gpu_ready_seconds,
                            "consumer_seconds": transfer_seconds,
                            "retrieved_tokens": int(retrieved_tokens),
                        },
                        separators=(",", ":"),
                    ),
                )

        yield ret_mask

    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: the input tokens,
        with shape [seq_len]

        :param Optional[List[int]] hashes: the input hashes, with length [num_chunks]
        :param Optional[List[int]] offsets: the offsets of each chunk,
        with length [num_chunks]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param Optional[str] lookup_id: The lookup ID to
            associate with the lookup. When pin is true, this argument is
            required to be not None.

        :param bool pin: If True, pin the KV cache in the storage.

        :param Optional[dict] request_configs: the configs of the request.

        :return: An int indicating how many prefix tokens exist inside LMCache.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping lookup operation")
            return 0

        assert self.storage_manager is not None

        if tokens is not None:
            lookup_stats = self.stats_monitor.on_lookup_request(len(tokens))
        else:
            assert offsets is not None
            assert hashes is not None
            lookup_stats = self.stats_monitor.on_lookup_request(sum(offsets))

        if search_range is None:
            search_range = self.retrieve_locations

        res = 0
        try:
            chunk_info_iterator = self.token_database.process_tokens(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
            )

            # TODO: support batched_contains when layerwise is enabled
            if self.use_layerwise:
                block_blob_layout = bool(
                    self.config.cascadekv_backend
                    or self.config.get_extra_config_value(
                        "xds_layerwise_block_blob", False
                    )
                )
                for start, end, key in chunk_info_iterator:
                    assert isinstance(key, CacheEngineKey)

                    # A CascadeKV block blob has exactly one owner key and one
                    # handle for all layers.  Other layerwise backends retain
                    # the legacy one-key-per-layer lookup contract.
                    lookup_keys = (
                        [key]
                        if block_blob_layout
                        else key.split_layers(self.num_layers)
                    )
                    expected_hits = 1 if block_blob_layout else self.num_layers

                    hit_chunks, block_mapping = self.storage_manager.batched_contains(
                        lookup_keys,
                        search_range,
                        pin,
                    )
                    # Only all layers are hit and hit in one location,
                    # we consider this key as a hit
                    if hit_chunks == expected_hits and len(block_mapping) == 1:
                        if pin:
                            assert lookup_id is not None, (
                                "lookup_id is required when pin is True"
                            )
                            location = next(iter(block_mapping.keys()))
                            self.lookup_pins[lookup_id][location].extend(lookup_keys)
                        res = end
                        continue
                    return res
            else:
                chunk_info_list = []
                keys = []
                for chunk_info in chunk_info_iterator:
                    assert isinstance(chunk_info[2], CacheEngineKey)
                    start, end, _ = chunk_info
                    chunk_info_list.append(chunk_info)
                    # chunk_info contains (start, end, key)
                    # chunk_info[2] is the key
                    keys.append(chunk_info[2])
                # hit chunks by prefix matching
                hit_chunks, block_mapping = self.storage_manager.batched_contains(
                    keys, search_range, pin
                )
                if pin and block_mapping:
                    assert lookup_id is not None, (
                        "lookup_id is required when pin is True"
                    )
                    self.lookup_pins[lookup_id] = block_mapping
                for idx, (start, end, key) in enumerate(chunk_info_list):
                    if idx < hit_chunks:
                        res = end
                        continue
                    return res

            # all tokens where found, return the maximal end
            return res
        finally:
            self.stats_monitor.on_lookup_finished(lookup_stats, res)
            # vllm lookup sets pin to True
            if pin:
                # touch_cache is tightly coupled with batched_contains
                self.storage_manager.touch_cache()

    @_lmcache_nvtx_annotate
    def move(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
    ) -> int:
        """
        Perform cross-node move of the KV cache.
        """
        assert self.storage_manager is not None

        num_tokens = self.lookup(
            tokens,
            search_range=[old_position],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[old_position]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=old_position,
        )
        assert None not in memory_objs, "Failed to get memory objects to move"
        logger.debug(
            f"Trying to send {len(memory_objs)} memory objects to {new_position}"
        )

        # TODO: reduce loops
        token_dim = memory_objs[0].meta.fmt.token_dim()  # type: ignore
        offsets = [m.meta.shape[token_dim] for m in memory_objs]  # type: ignore

        transfer_spec = {
            "target_peer_init_url": new_position[0],
            "offsets": offsets,
        }

        logger.info(self.storage_manager.storage_backends)
        p2p_backend = self.storage_manager.storage_backends["P2PBackend"]

        future = asyncio.run_coroutine_threadsafe(
            p2p_backend.async_batched_submit_put_task(
                keys,
                memory_objs,  # type: ignore
                transfer_spec=transfer_spec,
            ),
            self.storage_manager.loop,
        )

        future.result()

        if not do_copy:
            self.storage_manager.batched_remove(keys, locations=[old_position])

        logger.debug(f"Moving {num_tokens} token from {old_position} to {new_position}")
        return num_tokens

    # TODO(Jiayi): Add layerwise support.
    @_lmcache_nvtx_annotate
    def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> None:
        """
        An async version of lookup + prefetch.

        There are three categories of backends:
        (1) sync lookup + sync retrieval (e.g., cpu)
        (2) sync lookup + async retrieval (e.g., disk)
        (3) async lookup + async retrieval (e.g., p2p)
        """
        assert self.storage_manager is not None

        keys: list[CacheEngineKey] = []
        cum_chunk_lengths = [0]

        if search_range is None:
            search_range = self.retrieve_locations

        # TODO(Jiayi): make token database able to return list.
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            keys.append(key)
            cum_chunk_lengths.append(end)

        asyncio.run_coroutine_threadsafe(
            self.storage_manager.async_lookup_and_prefetch(
                lookup_id, keys, cum_chunk_lengths, search_range, pin
            ),
            self.storage_manager.loop,
        )

    def cleanup_memory_objs(self, lookup_id: str) -> None:
        """
        Cleanup memory objects allocated during prefetch for an aborted lookup.

        Called by the scheduler when it determines that an aborted lookup
        has finished its prefetch tasks.
        """
        try:
            # Get the completed future from event_manager
            if (
                self.event_manager.get_event_status(EventType.LOADING, lookup_id)
                != EventStatus.DONE
            ):
                logger.debug(
                    "No completed event found for lookup_id=%s to clean up.", lookup_id
                )
                return
            future = self.event_manager.pop_event(EventType.LOADING, lookup_id)

            # Get memory objects from the future result
            memory_objs = future.result()
            # Flatten nested lists (each backend returns a list of chunks)
            memory_objs_flat = [mm for m in memory_objs for mm in m]

            # Release each memory object
            for key, memory_obj in memory_objs_flat:
                try:
                    logger.debug("Releasing memory object for lookup_id=%s", lookup_id)
                    memory_obj.unpin()
                    memory_obj.ref_count_down()
                except Exception as e:
                    logger.error(f"Error releasing memory object: {e}")
        except Exception as e:
            logger.error(
                f"Error during cleanup_memory_objs for lookup_id={lookup_id}: {e}"
            )

    # TODO(Jiayi): Need to handle the case where `tokens=None`.
    # In this case, we compress all tokens.
    # TODO(Jiayi): support other compression methods.
    @_lmcache_nvtx_annotate
    def compress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported compression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        serializer, _ = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )
        assert None not in memory_objs, (
            "LMCacheEngine.compress: Failed to get memory objects to compress"
        )

        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def decompress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported decompression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        _, deserializer = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("there are no tokens to decompress.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )

        assert None not in compressed_memory_objs, (
            "LMCacheEngine.compress: Failed to get compressed "
            "memory objects to decompress"
        )

        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def lookup_unpin(self, lookup_id: str) -> None:
        if lookup_id in self.lookup_pins:
            assert self.storage_manager is not None
            for location, keys in self.lookup_pins.pop(lookup_id).items():
                self.storage_manager.batched_unpin(keys, [location])

        elif (
            self.async_loading is not None
            and self.event_manager.get_event_status(EventType.LOADING, lookup_id)
            != EventStatus.NOT_FOUND
        ):
            self.cleanup_memory_objs(lookup_id)

    @_lmcache_nvtx_annotate
    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        # TODO: need to clear by request_configs
        if self.save_only_first_rank:
            if self.metadata.is_first_rank():
                num_removed = self._clear(tokens, locations, request_configs)
                return num_removed
            else:
                return 0
        return self._clear(tokens, locations, request_configs)

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.kv_events_enabled and (events := self.kv_events):
            self.kv_events = []
            return events
        return []

    def _clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        assert self.storage_manager is not None
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    @_lmcache_nvtx_annotate
    def health(
        self,
    ) -> int:
        """
        Check the health of the cache engine.
        return: 0 if healthy, otherwise the error code
        """
        assert self.storage_manager is not None
        return 0 if self.storage_manager.memcheck() else -1

    def close(self) -> None:
        """Close the cache engine and free all the resources"""
        logger.info("Closing LMCacheEngine...")

        if self.lmcache_worker is not None:
            try:
                logger.info("Closing lmcache_worker...")
                self.lmcache_worker.close()
                logger.info("lmcache_worker closed successfully")
            except Exception as e:
                logger.error(f"Error closing lmcache_worker: {e}")

        try:
            logger.info("Closing storage_manager...")
            if self.storage_manager is not None:
                self.storage_manager.close()
            logger.info("storage_manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing storage_manager: {e}")

        logger.info("LMCacheEngine closed.")

    def _async_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """
        This function is used to get the memory objects from the event manager.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert "req_id" in kwargs, "req_id is required for async loading"
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        tot_kv_size = 0
        chunks: List[ProcessedChunk] = []
        future = self.event_manager.get_event_future(
            EventType.LOADING, kwargs["req_id"]
        )
        # As mentioned in async_lookup_and_prefetch(), the future.result()
        # is key data pair for each chunk in each tier. So extract the key
        # and memory object pairs to memory_obj_map
        try:
            keyed_memory_objs = future.result()
            memory_obj_map: dict[CacheEngineKey, MemoryObj] = {}
        except Exception as e:
            logger.error(f"Error popping event for request {kwargs['req_id']}: {e}")
            return [], 0

        for backend_results in keyed_memory_objs:
            for key, memory_obj in backend_results:
                memory_obj_map[key] = memory_obj

        # TODO(Jiayi): hashing inside `process_tokens` can be skipped.
        used_keys: set[CacheEngineKey] = set()
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            memory_obj = memory_obj_map.get(key)
            if memory_obj is None:
                # returned chunks are expected to be contiguous.
                # break at the first missing chunk.
                break
            chunks.append((key, memory_obj, start, end))
            tot_kv_size += memory_obj.get_size()
            ret_mask[start:end] = True
            used_keys.add(key)

        # NOTE: free the memory objects that are not hit.
        for key, mem_obj in memory_obj_map.items():
            if key not in used_keys:
                mem_obj.ref_count_down()

        return chunks, tot_kv_size

    def _process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """Process tokens and populate the reordered lists.

        This function is used to process tokens and populate the reordered lists.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert self.storage_manager is not None

        tot_kv_size = 0
        reordered_chunks: List[ProcessedChunk] = []
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        fine_profile = kwargs.get("_lmcache_retrieve_profile")
        if fine_profile is not None:
            fine_profile["chunk_scan_start_ns"] = time.monotonic_ns()
        chunk_infos = []
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            chunk_infos.append((key, start, end))
        if fine_profile is not None:
            fine_profile["chunk_scan_end_ns"] = time.monotonic_ns()

        # block_mapping: location -> [(CacheEngineKey, start, end)]
        if fine_profile is not None:
            fine_profile["mapping_start_ns"] = time.monotonic_ns()
        if (
            "req_id" in kwargs
            and kwargs["req_id"] in self.lookup_pins
            and len(self.lookup_pins[kwargs["req_id"]]) == 1
        ):
            location = next(iter(self.lookup_pins[kwargs["req_id"]].keys()))
            block_mapping = {location: chunk_infos}
        else:
            block_mapping = self.storage_manager.get_block_mapping(chunk_infos)
        if fine_profile is not None:
            fine_profile["mapping_end_ns"] = time.monotonic_ns()
            fine_profile["storage_calls"] = []

        last_failed_block_start = None
        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            assert self.gpu_connector is not None
            direct_transfer = self.storage_manager.batched_get_to_gpu(
                keys=keys,
                starts=[start for _, start, _ in blocks],
                ends=[end for _, _, end in blocks],
                gpu_connector=self.gpu_connector,
                location=location,
                **kwargs,
            )
            if isinstance(direct_transfer, BackendTransferResult):
                if len(direct_transfer.completed) != len(blocks):
                    raise RuntimeError(
                        "backend transfer result length does not match block count"
                    )
                tot_kv_size += direct_transfer.completed_bytes
                for (_, start, end), completed in zip(
                    blocks, direct_transfer.completed, strict=True
                ):
                    if completed:
                        ret_mask[start:end] = True
                    elif (
                        last_failed_block_start is None
                        or last_failed_block_start < start
                    ):
                        last_failed_block_start = start
                continue

            storage_start_ns = time.monotonic_ns()
            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=location,
            )
            storage_end_ns = time.monotonic_ns()
            if fine_profile is not None:
                fine_profile["storage_calls"].append(
                    {
                        "location": location,
                        "keys": len(keys),
                        "start_ns": storage_start_ns,
                        "end_ns": storage_end_ns,
                    }
                )

            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "The cache block is in the storage, but it can't be retrieved"
                    )
                    if (
                        last_failed_block_start is None
                        or last_failed_block_start < start
                    ):
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()
                ret_mask[start:end] = True

        if last_failed_block_start is not None:
            ret_mask[last_failed_block_start:] = False

            reordered_chunks = [
                (key, memory_obj, start, end)
                for key, memory_obj, start, end in reordered_chunks
                if end < last_failed_block_start
            ]
        return reordered_chunks, tot_kv_size

    def _broadcast_or_receive_memory_objs(
        self,
        reordered_chunks,
        ret_mask,
    ):
        """
        Handles broadcasting or receiving memory objects in a distributed environment.

        This function implements the communication logic where:
        - The first rank (coordinator) broadcasts memory objects and metadata to others
        - Other ranks receive and reconstruct the memory objects

        Parameters:
        reordered_chunks: List of tuples containing [key, memory object, start, end]
        ret_mask: Boolean mask indicating which positions have been processed

        Side Effects:
        - On first rank:
          * Broadcasts chunk count and each chunk's combined metadata
          * Broadcasts tensor data
        - On other ranks:
          * Receives chunk data and populates reordered_chunks
          * Updates ret_mask to mark received positions as True
        """
        if self.metadata.is_first_rank():
            # Broadcast total chunk count
            chunk_count = len(reordered_chunks)
            self.broadcast_object_fn(chunk_count, self.metadata.first_rank)

            # Broadcast each chunk's data
            for key, memory_obj, start, end in reordered_chunks:
                # Combine (start, end) and metadata into single broadcast
                metadata_dict = memory_obj.metadata.to_dict()
                combined_metadata = (start, end, metadata_dict)
                self.broadcast_object_fn(combined_metadata, self.metadata.first_rank)

                # Broadcast tensor data
                raw_tensor = memory_obj.raw_tensor
                assert raw_tensor is not None
                tensor_to_broadcast = raw_tensor.to(f"cuda:{self.metadata.worker_id}")
                self.broadcast_fn(tensor_to_broadcast, self.metadata.first_rank)
        else:
            # Receive total chunk count
            chunk_count = self.broadcast_object_fn(None, self.metadata.first_rank)
            if chunk_count is None:
                logger.warning(
                    f"rank={self.metadata.worker_id} received None chunk_count"
                )
                return

            # Fill reordered_chunks with received data
            for _ in range(chunk_count):
                # Receive combined metadata (start, end, metadata_dict)
                combined_metadata = self.broadcast_object_fn(
                    None, self.metadata.first_rank
                )
                if combined_metadata is None:
                    logger.warning(
                        f"rank={self.metadata.worker_id} "
                        "received None combined_metadata"
                    )
                    break
                start, end, metadata_dict = combined_metadata
                ret_mask[start:end] = True

                # Create tensor and receive data
                metadata = MemoryObjMetadata.from_dict(metadata_dict)
                local_rank = self.metadata.worker_id % torch.cuda.device_count()
                raw_tensor = torch.empty(
                    torch.Size([metadata.get_size()]),
                    dtype=torch.uint8,
                    device=f"cuda:{local_rank}",
                )
                self.broadcast_fn(raw_tensor, self.metadata.first_rank)

                # Create temporary memory object (key not needed for other ranks)
                memory_obj = TensorMemoryObj(
                    raw_data=raw_tensor, metadata=metadata, parent_allocator=None
                )
                reordered_chunks.append((None, memory_obj, start, end))

    def _is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()

    def _get_slot_mapping_list(
        self,
        slot_mapping: Optional[Union[torch.Tensor, List[int]]],
    ) -> Optional[List[int]]:
        """
        Convert slot_mapping to list if it's a tensor, otherwise return as is.

        :param slot_mapping: The slot_mapping to convert,
            can be a torch.Tensor or List[int], or None
        :type slot_mapping: Optional[Union[torch.Tensor, List[int]]]
        :return: The slot_mapping as a List[int], or None if input is None
        :rtype: Optional[List[int]]
        """
        if slot_mapping is None:
            return None
        if isinstance(slot_mapping, torch.Tensor):
            return slot_mapping.tolist()
        # At this point, slot_mapping must be List[int]
        return slot_mapping

    def _log_kvcache_for_check(
        self,
        operation: str,
        kwargs: dict,
        token_count: int,
        require_req_id: bool = False,
    ) -> None:
        """
        Helper method to log KVCache Check information.

        This method centralizes the KVCache Check logging logic that was
        duplicated in multiple methods.

        Args:
            operation: The operation being performed (e.g., "Store", "retrieve")
            kwargs: The keyword arguments containing slot_mapping and req_id
            token_count: The number of tokens involved in the operation
            require_req_id: Whether req_id must be present (default: False)
        """
        if not self.kvcache_check_log_enabled:
            return

        slot_mapping = kwargs.get("slot_mapping")
        if slot_mapping is None:
            return

        if require_req_id:
            req_id = kwargs.get("req_id")
            if req_id is None:
                return
        else:
            req_id = kwargs.get("req_id", "unspecified")

        # Convert slot_mapping to list if it's a tensor
        slot_mapping_list = self._get_slot_mapping_list(slot_mapping)
        # slot_mapping_list should not be None when slot_mapping is not None
        assert slot_mapping_list is not None

        logger.info(
            "[KVCache Check] %s request %s, tokens=%d, slot_mapping: %s",
            operation,
            req_id,
            token_count,
            compress_slot_mapping(slot_mapping_list),
        )


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    # TODO(Jiayi): Please remove this helper function in the future.
    # Currently, it's only used for testing.
    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> MemoryAllocatorInterface:
        # NOTE: should remove this function after fixing the unit tests:
        # raise RuntimeError("_Create_memory_allocator is deprecated!")
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )

        if enable_nixl_storage:
            # TODO(Jiayi): weird to import from transfer utils.
            # First Party
            from lmcache.v1.transfer_channel.transfer_utils import (
                get_correct_device,
            )

            corrected_device = get_correct_device(
                config.nixl_buffer_device,
                metadata.worker_id,
            )

            buffer = torch.empty(
                config.nixl_buffer_size,
                dtype=torch.uint8,
                device=corrected_device,
            )

            if corrected_device == "cpu":
                torch.cuda.cudart().cudaHostRegister(
                    buffer.data_ptr(), config.nixl_buffer_size, 0
                )
            else:
                logger.info(f"Setting cuda device to {corrected_device} ")
                torch.cuda.set_device(corrected_device)

            return PagedTensorMemoryAllocator(
                buffer,
                [torch.Size(metadata.kv_shape)],
                [metadata.kv_dtype],
                MemoryFormat.KV_2LTD,
            )

        if config.gds_path is not None:
            assert config.cufile_buffer_size is not None
            return CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2)

        max_local_cpu_size = config.max_local_cpu_size
        # save_only_first_rank only works when use mla
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank and metadata.is_first_rank():
            # Only the first rank will save the cache,
            # so we need to set it lager than other ranks
            first_rank_max_local_cpu_size = (
                config.extra_config.get(
                    "first_rank_max_local_cpu_size", max_local_cpu_size
                )
                if config.extra_config
                else max_local_cpu_size
            )
            return MixedMemoryAllocator(
                int(first_rank_max_local_cpu_size * 1024**3),
                numa_mapping=numa_mapping,
            )
        return MixedMemoryAllocator(
            int(max_local_cpu_size * 1024**3),
            numa_mapping=numa_mapping,
        )

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> TokenDatabase:
        if config.enable_blending:
            return SegmentTokenDatabase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            numa_mapping = NUMADetector.get_numa_mapping(config)
            logger.info(f"NUMA mapping for instance {instance_id}: {numa_mapping}")
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(
                metadata,
                log_interval=10,
                config=config,
            )

            engine = LMCacheEngine(
                config,
                metadata,
                token_database,
                gpu_connector,
                broadcast_fn,
                broadcast_object_fn,
            )

            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (
                cls._cfgs[instance_id] != config
                or cls._metadatas[instance_id] != metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata."
                )
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        logger.info(f"Destroying LMCacheEngine instance: {instance_id}")

        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            try:
                logger.info("Shutting down stats logger...")
                stat_logger.shutdown()
                logger.info("Stats logger shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down stats logger: {e}")

            engine = cls._instances[instance_id]
            try:
                logger.info("Closing cache engine...")
                engine.close()
                logger.info("Cache engine closed successfully")
            except Exception as e:
                logger.error(f"Error closing cache engine: {e}")

            try:
                logger.info("Cleaning up instance dictionaries...")
                cls._instances.pop(instance_id, None)
                cls._cfgs.pop(instance_id, None)
                cls._metadatas.pop(instance_id, None)
                cls._stat_loggers.pop(instance_id, None)
                logger.info("Instance dictionaries cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up instances: {e}")

            try:
                logger.info("Destroying stats monitor...")
                LMCStatsMonitor.DestroyInstance()
                logger.info("Stats monitor destroyed successfully")
            except Exception as e:
                logger.error(f"Error destroying stats monitor: {e}")

            logger.info(f"LMCacheEngine instance {instance_id} destroyed")
        else:
            logger.warning(f"Instance {instance_id} not found for destruction")
