# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import asyncio
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, SpdkBlobMetadata
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import (
    GPUMemoryAllocator,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.spdk_utils import align_size_to_io_unit
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.evictor import LRUEvictor, PutStatus

if TYPE_CHECKING:
    from lmcache.v1.cache_controller.worker import LMCacheWorker

# Local
import spdk_controller as spdk

logger = init_logger(__name__)


class SpdkDirectP2PBackend(StorageBackendInterface):
    # StorageManager uses these capability flags instead of class-name branches.
    is_allocator_backend = True
    skip_cpu_writeback = True

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
        lmcache_worker=None,
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        assert dst_device.startswith("cuda")
        super().__init__(dst_device)

        if not spdk.dp2p_is_enabled():
            raise RuntimeError(
                "SPDK direct-p2p is not enabled. Check peer BDF config and SPDK init."
            )

        assert isinstance(memory_allocator, GPUMemoryAllocator), (
            "SpdkDirectP2PBackend requires a GPUMemoryAllocator-compatible allocator."
        )

        self.memory_allocator = memory_allocator
        assert hasattr(self.memory_allocator, "base_pointer"), (
            "SpdkDirectP2PBackend requires allocator.base_pointer."
        )

        self.gpu_reg_handle = spdk.register_gpu_buffer(
            self.memory_allocator.base_pointer,
            self.memory_allocator.tensor.numel()
            * self.memory_allocator.tensor.element_size(),
        )
        self.loop = loop
        self.lookup_server = lookup_server
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.blob_acquire_timeout_secs = config.spdk_blob_acquire_timeout_secs
        self.io_timeout_secs = config.spdk_io_timeout_secs
        self.trace_io = os.environ.get("LMCACHE_TRACE_IO", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        self.dict: OrderedDict[CacheEngineKey, SpdkBlobMetadata] = OrderedDict()
        self.evictor = LRUEvictor(max_cache_size=config.spdk_max_size)
        self.usage = 0

        self.dict_lock = threading.RLock()
        self.usage_lock = threading.RLock()
        self.put_tasks_lock = threading.RLock()
        self.prefetch_lock = threading.RLock()

        spdk.init_blob_pool()

        self.put_tasks: dict[CacheEngineKey, MemoryObj] = {}
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}
        self.keys_in_request: List[CacheEngineKey] = []

        self.cumulative_read_bytes = 0
        self.cumulative_read_time = 0.0
        self.stats_lock = threading.Lock()

    def __str__(self):
        return self.__class__.__name__

    def _allocate_memory_obj(self, shape, dtype, fmt) -> Optional[MemoryObj]:
        return self.memory_allocator.allocate(shape, dtype, fmt)

    def _batched_allocate_memory_objs(
        self, shape, dtype, batch_size: int, fmt
    ) -> Optional[List[MemoryObj]]:
        return self.memory_allocator.batched_allocate(shape, dtype, batch_size, fmt)

    def _get_io_ptr(self, memory_obj: MemoryObj) -> int:
        assert memory_obj.tensor is not None
        return int(memory_obj.tensor.data_ptr())

    def _get_storage_size(self, memory_obj: MemoryObj) -> int:
        storage_size = memory_obj.get_physical_size()
        aligned_size = align_size_to_io_unit(storage_size)
        if storage_size != aligned_size:
            raise ValueError(
                "SPDK direct-p2p backend requires memory objects to be io_unit aligned: "
                f"physical_size={storage_size}, aligned_size={aligned_size}"
            )
        return storage_size

    def _wait_for_io(self, future: Future):
        return future.result(timeout=self.io_timeout_secs)

    def _update_usage(self, delta: int) -> None:
        with self.usage_lock:
            self.usage += delta
            self.stats_monitor.update_local_storage_usage(self.usage)

    def _update_evictor_usage(self, delta: int) -> None:
        self.evictor.current_cache_size += delta

    def _log_io_trace(
        self,
        op: str,
        sizes: List[int],
        elapsed_secs: float,
        success: bool,
    ) -> None:
        if not self.trace_io:
            return

        print(
            "[LMCACHE_IO_TRACE] "
            f"backend={self.__class__.__name__} "
            f"op={op} "
            f"reqs={len(sizes)} "
            f"total_bytes={sum(sizes)} "
            f"sizes={sizes} "
            f"elapsed_ms={elapsed_secs * 1000.0:.3f} "
            f"success={int(success)}",
            flush=True,
        )

    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
    ) -> Optional[MemoryObj]:
        return self._allocate_memory_obj(shape, dtype, fmt)

    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
    ) -> Optional[List[MemoryObj]]:
        return self._batched_allocate_memory_objs(shape, dtype, batch_size, fmt)

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.put_tasks_lock:
            if key in self.put_tasks:
                return True

        with self.dict_lock:
            if key not in self.dict:
                return False
            if pin:
                self.dict[key].pin()
                self.keys_in_request.append(key)
            return True

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_tasks_lock:
            return key in self.put_tasks

    def pin(self, key: CacheEngineKey) -> bool:
        with self.dict_lock:
            if key not in self.dict:
                return False
            self.dict[key].pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.dict_lock:
            if key not in self.dict:
                return False
            self.dict[key].unpin()
            return True

    def remove(
        self,
        key: CacheEngineKey,
        free_obj: bool = True,
        update_evictor_usage: bool = True,
    ) -> bool:
        with self.dict_lock:
            if key not in self.dict:
                return False
            metadata = self.dict.pop(key)

        self._update_usage(-metadata.size)
        if update_evictor_usage:
            self._update_evictor_usage(-metadata.size)

        spdk.release_blob(metadata.blob_handle)

        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, "spdk")
            )
        return True

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj, blob_handle: int) -> None:
        storage_size = self._get_storage_size(memory_obj)
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt

        has_stored = False
        delete_size = 0
        with self.dict_lock:
            if key in self.dict:
                old_metadata = self.dict.pop(key)
                spdk.release_blob(old_metadata.blob_handle)
                delete_size = old_metadata.size
                has_stored = True
            self.dict[key] = SpdkBlobMetadata(
                blob_handle, storage_size, shape, dtype, fmt, False
            )

        if has_stored:
            self._update_usage(-delete_size)
            self._update_evictor_usage(-delete_size)
            logger.warning("Key %s already exists in SPDK direct-p2p backend.", key)

        if self.lmcache_worker is not None and not has_stored:
            self.lmcache_worker.put_msg(
                KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, "spdk")
            )

    def touch_cache(self):
        with self.dict_lock:
            for key in reversed(self.keys_in_request):
                if key in self.dict:
                    self.evictor.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def _perform_batch_write(self, tasks: List[Tuple[CacheEngineKey, MemoryObj]]):
        if not tasks:
            return

        keys = [task[0] for task in tasks]
        memory_objs = [task[1] for task in tasks]
        blob_handles = []
        usage_added = 0
        io_sizes: List[int] = []

        try:
            blob_handles = [
                spdk.get_blob(timeout=self.blob_acquire_timeout_secs)
                for _ in keys
            ]
            total_size = 0
            requests = []
            for blob_handle, memory_obj in zip(blob_handles, memory_objs, strict=False):
                storage_size = self._get_storage_size(memory_obj)
                total_size += storage_size
                io_sizes.append(storage_size)
                requests.append(
                    (blob_handle, self._get_io_ptr(memory_obj), 0, storage_size)
                )

            self._update_usage(total_size)
            usage_added = total_size

            io_start = time.perf_counter()
            try:
                self._wait_for_io(
                    spdk.write_batch_async(
                        requests,
                    )
                )
            except Exception:
                self._log_io_trace(
                    "write_batch", io_sizes, time.perf_counter() - io_start, False
                )
                raise
            else:
                self._log_io_trace(
                    "write_batch", io_sizes, time.perf_counter() - io_start, True
                )

            for key, memory_obj, blob_handle in zip(
                keys, memory_objs, blob_handles, strict=False
            ):
                self.insert_key(key, memory_obj, blob_handle)
                memory_obj.ref_count_down()

        except Exception as e:
            logger.error("SPDK direct-p2p batch write failed: %s", e, exc_info=True)
            if usage_added > 0:
                self._update_usage(-usage_added)
                self._update_evictor_usage(-usage_added)
            for handle in blob_handles:
                spdk.release_blob(handle)
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()
        finally:
            with self.put_tasks_lock:
                for key in keys:
                    self.put_tasks.pop(key, None)

    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Optional[Future]:
        self.batched_submit_put_task([key], [memory_obj])
        return None

    def batched_submit_put_task(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> Optional[List[Future]]:
        if not keys:
            return None

        keys_to_process = []
        objs_to_process = []
        with self.put_tasks_lock, self.dict_lock:
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                if key in self.put_tasks or key in self.dict:
                    continue
                keys_to_process.append(key)
                objs_to_process.append(memory_obj)

        if not keys_to_process:
            return None

        total_storage_size = sum(
            self._get_storage_size(memory_obj) for memory_obj in objs_to_process
        )
        with self.dict_lock:
            evict_keys, put_status = self.evictor.update_on_put(
                self.dict, total_storage_size
            )

        if put_status == PutStatus.ILLEGAL:
            logger.warning(
                "Batch write failed: total size %d is too large for the cache.",
                total_storage_size,
            )
            return None

        if evict_keys:
            for evict_key in evict_keys:
                self.remove(evict_key, update_evictor_usage=False)
            if self.lookup_server is not None:
                self.lookup_server.batched_remove(evict_keys)

        tasks_to_process = []
        with self.put_tasks_lock:
            for key, memory_obj in zip(keys_to_process, objs_to_process, strict=False):
                memory_obj.ref_count_up()
                self.put_tasks[key] = memory_obj
                tasks_to_process.append((key, memory_obj))

        self._perform_batch_write(tasks_to_process)

        return None

    def _record_read_stats(self, read_bytes: int, read_time: float) -> None:
        with self.stats_lock:
            self.cumulative_read_bytes += read_bytes
            self.cumulative_read_time += read_time
            logger.info(
                "current average read bandwidth: %.2f MB/s",
                (self.cumulative_read_bytes / self.cumulative_read_time) / (1024 * 1024),
            )

    def _perform_batched_spdk_read(
        self, spdk_tasks_info: List[Dict[str, Any]]
    ) -> Tuple[List[Optional[MemoryObj]], int]:
        if not spdk_tasks_info:
            return [], 0

        batch_requests = []
        allocated_memory_objs: List[Optional[MemoryObj]] = []
        request_indexes: List[int] = []
        total_request_bytes = 0
        io_sizes: List[int] = []

        for idx, task_info in enumerate(spdk_tasks_info):
            memory_obj = self._allocate_memory_obj(
                task_info["shape"], task_info["dtype"], task_info["fmt"]
            )
            if memory_obj is None:
                logger.warning("GPU allocation failed during batched direct-p2p read.")
                allocated_memory_objs.append(None)
                continue

            storage_size = self._get_storage_size(memory_obj)
            if storage_size != task_info["storage_size"]:
                raise ValueError(
                    "Allocated memory object storage size does not match blob size: "
                    f"allocated={storage_size}, blob={task_info['storage_size']}"
                )
            allocated_memory_objs.append(memory_obj)
            batch_requests.append(
                (
                    task_info["blob_handle"],
                    self._get_io_ptr(memory_obj),
                    0,
                    storage_size,
                )
            )
            request_indexes.append(idx)
            total_request_bytes += storage_size
            io_sizes.append(storage_size)

        if not batch_requests:
            return allocated_memory_objs, 0

        future = spdk.read_batch_async(
            batch_requests,
        )
        try:
            io_start = time.perf_counter()
            self._wait_for_io(future)
            self._log_io_trace(
                "read_batch", io_sizes, time.perf_counter() - io_start, True
            )
            return allocated_memory_objs, total_request_bytes
        except Exception as e:
            self._log_io_trace(
                "read_batch", io_sizes, time.perf_counter() - io_start, False
            )
            logger.error(
                "SPDK direct-p2p batch read failed or timed out: %s",
                e,
                exc_info=True,
            )
            for idx in request_indexes:
                memory_obj = allocated_memory_objs[idx]
                if memory_obj is not None:
                    memory_obj.ref_count_down()
                    allocated_memory_objs[idx] = None
            return allocated_memory_objs, 0

    def batched_get_blocking(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        if not keys:
            return []

        results: Dict[CacheEngineKey, Optional[MemoryObj]] = {key: None for key in keys}
        remaining_keys = []

        with self.put_tasks_lock:
            for key in keys:
                if key in self.put_tasks:
                    memory_obj = self.put_tasks[key]
                    memory_obj.ref_count_up()
                    results[key] = memory_obj
                else:
                    remaining_keys.append(key)

        keys_after_prefetch = []
        prefetch_futures = []
        with self.prefetch_lock:
            for key in remaining_keys:
                if key in self.prefetch_tasks:
                    prefetch_futures.append((key, self.prefetch_tasks[key]))
                else:
                    keys_after_prefetch.append(key)

        for key, future in prefetch_futures:
            try:
                results[key] = future.result(timeout=self.io_timeout_secs)
            except Exception as e:
                logger.error(
                    "Waiting for direct-p2p prefetch task for %s failed: %s",
                    key,
                    e,
                    exc_info=True,
                )
                keys_after_prefetch.append(key)

        spdk_tasks_info = []
        with self.dict_lock:
            for key in keys_after_prefetch:
                if key not in self.dict:
                    continue
                self.evictor.update_on_hit(key, self.dict)
                metadata = self.dict[key]
                spdk_tasks_info.append(
                    {
                        "key": key,
                        "blob_handle": metadata.blob_handle,
                        "storage_size": metadata.size,
                        "dtype": metadata.dtype,
                        "shape": metadata.shape,
                        "fmt": metadata.fmt,
                    }
                )

        if spdk_tasks_info:
            start_time = time.time()
            spdk_results, total_read_bytes = self._perform_batched_spdk_read(
                spdk_tasks_info
            )
            read_time = time.time() - start_time
            if total_read_bytes > 0:
                self._record_read_stats(total_read_bytes, read_time)
            for task_info, result_obj in zip(spdk_tasks_info, spdk_results, strict=False):
                results[task_info["key"]] = result_obj

        return [results.get(key) for key in keys]

    async def async_load_bytes_from_spdk(
        self, key: CacheEngineKey, blob_handle: int, storage_size: int, dtype, shape, fmt
    ) -> Optional[MemoryObj]:
        memory_obj = self._allocate_memory_obj(shape, dtype, fmt)
        if memory_obj is None:
            logger.debug("GPU allocation failed during async direct-p2p load.")
            return None

        allocated_storage_size = self._get_storage_size(memory_obj)
        if allocated_storage_size != storage_size:
            raise ValueError(
                "Allocated memory object storage size does not match blob size: "
                f"allocated={allocated_storage_size}, blob={storage_size}"
            )

        io_start = time.perf_counter()
        try:
            await asyncio.wait_for(
                asyncio.wrap_future(
                    spdk.read_async(
                        blob_handle, self._get_io_ptr(memory_obj), 0, storage_size
                    )
                ),
                timeout=self.io_timeout_secs,
            )
        except Exception:
            self._log_io_trace(
                "read_prefetch", [storage_size], time.perf_counter() - io_start, False
            )
            raise
        else:
            self._log_io_trace(
                "read_prefetch", [storage_size], time.perf_counter() - io_start, True
            )

        with self.dict_lock:
            if key in self.dict:
                self.dict[key].unpin()

        return memory_obj

    def _remove_prefetch_task(self, key: CacheEngineKey, future: Future):
        with self.prefetch_lock:
            self.prefetch_tasks.pop(key, None)

        try:
            future.result()
        except Exception as e:
            logger.warning("Prefetch task for %s failed: %s", key, e)
            with self.dict_lock:
                if key in self.dict:
                    self.dict[key].unpin()

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        with self.prefetch_lock:
            if key in self.prefetch_tasks:
                return self.prefetch_tasks[key]

            with self.dict_lock:
                if key not in self.dict:
                    return None
                self.dict[key].pin()
                metadata = self.dict[key]
                blob_info = {
                    "key": key,
                    "blob_handle": metadata.blob_handle,
                    "storage_size": metadata.size,
                    "dtype": metadata.dtype,
                    "shape": metadata.shape,
                    "fmt": metadata.fmt,
                }

            future = asyncio.run_coroutine_threadsafe(
                self.async_load_bytes_from_spdk(
                    blob_info["key"],
                    blob_info["blob_handle"],
                    blob_info["storage_size"],
                    blob_info["dtype"],
                    blob_info["shape"],
                    blob_info["fmt"],
                ),
                self.loop,
            )
            future.add_done_callback(lambda f: self._remove_prefetch_task(key, f))
            self.prefetch_tasks[key] = future
            return future

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        return self.submit_prefetch_task(key)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.dict_lock:
            if key not in self.dict:
                return None

            self.evictor.update_on_hit(key, self.dict)
            metadata = self.dict[key]

        memory_obj = self._allocate_memory_obj(
            metadata.shape, metadata.dtype, metadata.fmt
        )
        if memory_obj is None:
            logger.debug("GPU allocation failed during direct-p2p get_blocking.")
            return None

        storage_size = self._get_storage_size(memory_obj)
        if storage_size != metadata.size:
            raise ValueError(
                "Allocated memory object storage size does not match blob size: "
                f"allocated={storage_size}, blob={metadata.size}"
            )

        start_time = time.time()
        future = spdk.read_async(
            metadata.blob_handle,
            self._get_io_ptr(memory_obj),
            0,
            metadata.size,
        )
        try:
            io_start = time.perf_counter()
            self._wait_for_io(future)
            self._log_io_trace(
                "read_blocking",
                [metadata.size],
                time.perf_counter() - io_start,
                True,
            )
            self._record_read_stats(metadata.size, time.time() - start_time)
            return memory_obj
        except Exception as e:
            self._log_io_trace(
                "read_blocking",
                [metadata.size],
                time.perf_counter() - io_start,
                False,
            )
            logger.error(
                "SPDK direct-p2p read failed or timed out: %s", e, exc_info=True
            )
            memory_obj.ref_count_down()
            return None

    def close(self) -> None:
        logger.info("Closing SPDK direct-p2p backend...")

        if self.lookup_server is not None:
            with self.dict_lock:
                keys_to_remove = list(self.dict.keys())
            if keys_to_remove:
                self.lookup_server.batched_remove(keys_to_remove)

        if self.gpu_reg_handle is not None:
            try:
                spdk.unregister_gpu_buffer(self.gpu_reg_handle)
            except Exception:
                logger.warning("Failed to unregister SPDK direct-p2p GPU buffer.")
            finally:
                self.gpu_reg_handle = None

        spdk.unload()
        logger.info("SPDK direct-p2p backend closed.")
