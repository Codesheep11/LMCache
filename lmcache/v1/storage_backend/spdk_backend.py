# Standard
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, wait
from typing import Any, Dict, List, Optional, Tuple
import threading
import asyncio
import os
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
from lmcache.v1.memory_management import GPUMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.spdk_utils import align_size_to_io_unit
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.evictor import LRUEvictor, PutStatus
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.cache_controller.worker import LMCacheWorker

# Local
import spdk_controller as spdk

logger = init_logger(__name__)

DEFAULT_BLOB_ACQUIRE_TIMEOUT_SECS = 5.0
DEFAULT_IO_TIMEOUT_SECS = 30.0


class SpdkBlobBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        self.dict: OrderedDict[CacheEngineKey, SpdkBlobMetadata] = OrderedDict()
        self.dst_device = dst_device
        self.lookup_server = lookup_server
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0
        self.bdev_name = config.bdev_name
        self.sock_path = config.rpc_addr
        self.blob_acquire_timeout_secs = DEFAULT_BLOB_ACQUIRE_TIMEOUT_SECS
        self.io_timeout_secs = DEFAULT_IO_TIMEOUT_SECS
        self.trace_io = os.environ.get("LMCACHE_TRACE_IO", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.cuda_device = (
            torch.device(dst_device)
            if torch.cuda.is_available() and str(dst_device).startswith("cuda")
            else None
        )
        self.io_unit_size = spdk.get_io_unit_size()
        requested_overlap_chunk_kb = getattr(
            config, "xds_chunked_size_kb", 4096
        )
        self.chunked_gpu_overlap_requested = getattr(
            config, "xds_enable_chunked_overlap", False
        )
        self.enable_chunked_gpu_overlap = (
            self.chunked_gpu_overlap_requested
            and self.cuda_device is not None
            and self.io_unit_size > 0
        )
        # self.enable_chunked_gpu_overlap = False
        self.chunked_gpu_overlap_size = align_size_to_io_unit(
            max(int(requested_overlap_chunk_kb) * 1024, max(self.io_unit_size, 1))
        )
        if self.chunked_gpu_overlap_requested and self.cuda_device is None:
            logger.warning(
                "xds_enable_chunked_overlap is enabled, but CUDA is unavailable. "
                "Falling back to the original SPDK read path."
            )
        
        xds_max_size = config.xds_max_size
        self.evictor = LRUEvictor(max_cache_size=xds_max_size)

        self.dict_lock = threading.RLock()
        self.usage_lock = threading.RLock()
        
        self.write_window_size = getattr(config, 'spdk_write_window_size', 64)
        self.idle_flush_time = getattr(config, 'spdk_idle_flush_time', 0.5)
        self.max_flush_delay = getattr(config, 'spdk_max_flush_delay', 1.0) 
        self.read_bios = True
        self.concurrency = True

        spdk.init_blob_pool()
        self.put_tasks: dict[CacheEngineKey, MemoryObj] = {}
        self.put_tasks_lock = threading.RLock()
        
        self.write_buffer: List[Tuple[CacheEngineKey, MemoryObj]] = []
        self.flush_condition = threading.Condition(self.put_tasks_lock)
        
        self.last_write_timestamp: float = 0.0
        self.oldest_item_timestamp: float = 0.0

        self.running = threading.Event()
        self.running.set()
        self.flush_thread = threading.Thread(
            target=self._flush_loop,
            name="SPDK-FlushThread",
            daemon=True
        )
        self.flush_thread.start()

        self.keys_in_request: List[CacheEngineKey] = []
        
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}
        self.prefetch_lock = threading.RLock()

        self.cumulative_read_bytes = 0
        self.cumulative_read_time = 0.0
        self.stats_lock = threading.Lock()
        blob_size = spdk.get_blob_size_in_bytes()
        self.sleep_time = blob_size / (19 * 1024**3) / 2
        logger.info(f"SPDK blob size: {blob_size} bytes, sleep time between reads: {self.sleep_time:.6f} seconds")

    def _allocate_memory_obj(self, shape, dtype, fmt) -> Optional[MemoryObj]:
        return self.local_cpu_backend.allocate(shape, dtype, fmt)

    def _get_io_ptr(self, memory_obj: MemoryObj) -> int:
        return memory_obj.meta.address

    def _get_storage_size(self, memory_obj: MemoryObj) -> int:
        storage_size = memory_obj.get_physical_size()
        aligned_size = align_size_to_io_unit(storage_size)
        if storage_size != aligned_size:
            raise ValueError(
                "SPDK backend requires memory objects to be io_unit aligned: "
                f"physical_size={storage_size}, aligned_size={aligned_size}"
            )
        return storage_size

    def _wait_for_io(self, future: Future):
        return future.result(timeout=self.io_timeout_secs)

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

    def _should_use_chunked_gpu_overlap(self, storage_size: int) -> bool:
        del storage_size
        return self.enable_chunked_gpu_overlap

    def _should_prefer_blob_gpu_overlap(
        self, spdk_tasks_info: List[Dict[str, Any]]
    ) -> bool:
        # Force batched reads to use blob-granularity overlap so the batch path
        # does not regress to chunk-level scheduling overhead while we evaluate
        # overlap effectiveness. Single-blob reads still use chunk granularity.
        return self.enable_chunked_gpu_overlap and len(spdk_tasks_info) > 1

    def _allocate_host_memory_obj(
        self, shape, dtype, fmt, storage_size: int
    ) -> Optional[MemoryObj]:
        memory_obj = self._allocate_memory_obj(shape, dtype, fmt)
        if memory_obj is None:
            return None

        allocated_storage_size = self._get_storage_size(memory_obj)
        if allocated_storage_size != storage_size:
            raise ValueError(
                "Allocated memory object storage size does not match blob size: "
                f"allocated={allocated_storage_size}, blob={storage_size}"
            )
        return memory_obj

    def _allocate_device_memory_obj(
        self, shape, dtype, fmt, storage_size: int
    ) -> Optional[MemoryObj]:
        if self.cuda_device is None:
            return None

        allocator = GPUMemoryAllocator(
            storage_size,
            device=self.cuda_device,
            align_bytes=self.io_unit_size,
        )
        memory_obj = allocator.allocate(shape, dtype, fmt)
        if memory_obj is None:
            return None

        allocated_storage_size = self._get_storage_size(memory_obj)
        if allocated_storage_size != storage_size:
            raise ValueError(
                "Allocated GPU memory object storage size does not match blob size: "
                f"allocated={allocated_storage_size}, blob={storage_size}"
            )
        return memory_obj

    def _read_blob_to_host_blocking(
        self,
        blob_handle: int,
        storage_size: int,
        dtype,
        shape,
        fmt,
        trace_op: str,
    ) -> Optional[MemoryObj]:
        memory_obj = self._allocate_host_memory_obj(shape, dtype, fmt, storage_size)
        if memory_obj is None:
            logger.debug("Memory allocation failed during spdk host read.")
            return None

        future = spdk.read_async(
            blob_handle,
            self._get_io_ptr(memory_obj),
            0,
            storage_size,
        )
        io_start = time.perf_counter()
        try:
            self._wait_for_io(future)
            self._log_io_trace(
                trace_op, [storage_size], time.perf_counter() - io_start, True
            )
            return memory_obj
        except Exception as e:
            self._log_io_trace(
                trace_op, [storage_size], time.perf_counter() - io_start, False
            )
            logger.error(
                f"SPDK host read operation failed or timed out: {e}",
                exc_info=True,
            )
            memory_obj.ref_count_down()
            return None

    def _read_blob_to_device_chunked_blocking(
        self,
        blob_handle: int,
        storage_size: int,
        dtype,
        shape,
        fmt,
        trace_op: str,
    ) -> Optional[MemoryObj]:
        staging_memory_obj = self._allocate_host_memory_obj(
            shape, dtype, fmt, storage_size
        )
        if staging_memory_obj is None:
            logger.debug("Memory allocation failed during SPDK staged host read.")
            return None

        output_memory_obj = self._allocate_device_memory_obj(
            shape, dtype, fmt, storage_size
        )
        if output_memory_obj is None:
            logger.warning(
                "GPU allocation failed for chunked SPDK overlap path. "
                "Falling back to the original host read path."
            )
            staging_memory_obj.ref_count_down()
            return self._read_blob_to_host_blocking(
                blob_handle, storage_size, dtype, shape, fmt, trace_op
            )

        staging_tensor = staging_memory_obj.tensor
        output_tensor = output_memory_obj.tensor
        assert staging_tensor is not None
        assert output_tensor is not None

        staging_bytes = staging_tensor.view(torch.uint8).flatten()
        output_bytes = output_tensor.view(torch.uint8).flatten()
        logical_bytes = output_memory_obj.get_size()
        copy_stream = torch.cuda.Stream(device=self.cuda_device)
        io_sizes: List[int] = []
        io_start = time.perf_counter()

        try:
            chunk_reads: List[Tuple[int, int, Future]] = []
            # Queue all chunk reads first so SPDK can keep queue depth, then drain
            # completions while GPU copies run asynchronously on the copy stream.
            for offset in range(0, storage_size, self.chunked_gpu_overlap_size):
                chunk_size = min(self.chunked_gpu_overlap_size, storage_size - offset)
                io_sizes.append(chunk_size)
                chunk_reads.append(
                    (
                        offset,
                        chunk_size,
                        spdk.read_async(
                            blob_handle,
                            self._get_io_ptr(staging_memory_obj) + offset,
                            offset,
                            chunk_size,
                        ),
                    )
                )

            for offset, chunk_size, future in chunk_reads:
                self._wait_for_io(future)

                remaining_logical_bytes = max(0, logical_bytes - offset)
                copy_size = min(chunk_size, remaining_logical_bytes)
                if copy_size > 0:
                    with torch.cuda.stream(copy_stream):
                        output_bytes[offset : offset + copy_size].copy_(
                            staging_bytes[offset : offset + copy_size],
                            non_blocking=True,
                        )

            copy_stream.synchronize()
            self._log_io_trace(
                trace_op, io_sizes, time.perf_counter() - io_start, True
            )
            return output_memory_obj
        except Exception as e:
            try:
                copy_stream.synchronize()
            except Exception:
                logger.debug(
                    "Chunked SPDK overlap stream synchronize failed during error handling.",
                    exc_info=True,
                )
            self._log_io_trace(
                trace_op, io_sizes, time.perf_counter() - io_start, False
            )
            logger.error(
                f"SPDK chunked staged read operation failed or timed out: {e}",
                exc_info=True,
            )
            output_memory_obj.ref_count_down()
            return None
        finally:
            staging_memory_obj.ref_count_down()

    async def _read_blob_to_device_chunked_async(
        self,
        blob_handle: int,
        storage_size: int,
        dtype,
        shape,
        fmt,
        trace_op: str,
    ) -> Optional[MemoryObj]:
        staging_memory_obj = self._allocate_host_memory_obj(
            shape, dtype, fmt, storage_size
        )
        if staging_memory_obj is None:
            logger.debug("Memory allocation failed during async SPDK staged host read.")
            return None

        output_memory_obj = self._allocate_device_memory_obj(
            shape, dtype, fmt, storage_size
        )
        if output_memory_obj is None:
            staging_memory_obj.ref_count_down()
            raise RuntimeError(
                "GPU allocation failed for chunked SPDK overlap path"
            )

        staging_tensor = staging_memory_obj.tensor
        output_tensor = output_memory_obj.tensor
        assert staging_tensor is not None
        assert output_tensor is not None

        staging_bytes = staging_tensor.view(torch.uint8).flatten()
        output_bytes = output_tensor.view(torch.uint8).flatten()
        logical_bytes = output_memory_obj.get_size()
        copy_stream = torch.cuda.Stream(device=self.cuda_device)
        io_sizes: List[int] = []
        io_start = time.perf_counter()

        try:
            chunk_reads: List[Tuple[int, int, Future]] = []
            # Queue all chunk reads first so SPDK can keep queue depth, then drain
            # completions while GPU copies run asynchronously on the copy stream.
            for offset in range(0, storage_size, self.chunked_gpu_overlap_size):
                chunk_size = min(self.chunked_gpu_overlap_size, storage_size - offset)
                io_sizes.append(chunk_size)
                chunk_reads.append(
                    (
                        offset,
                        chunk_size,
                        spdk.read_async(
                            blob_handle,
                            self._get_io_ptr(staging_memory_obj) + offset,
                            offset,
                            chunk_size,
                        ),
                    )
                )

            for offset, chunk_size, future in chunk_reads:
                await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=self.io_timeout_secs,
                )

                remaining_logical_bytes = max(0, logical_bytes - offset)
                copy_size = min(chunk_size, remaining_logical_bytes)
                if copy_size > 0:
                    with torch.cuda.stream(copy_stream):
                        output_bytes[offset : offset + copy_size].copy_(
                            staging_bytes[offset : offset + copy_size],
                            non_blocking=True,
                        )

            copy_stream.synchronize()
            self._log_io_trace(
                trace_op, io_sizes, time.perf_counter() - io_start, True
            )
            return output_memory_obj
        except Exception:
            try:
                copy_stream.synchronize()
            except Exception:
                logger.debug(
                    "Chunked SPDK overlap stream synchronize failed during async error handling.",
                    exc_info=True,
                )
            self._log_io_trace(
                trace_op, io_sizes, time.perf_counter() - io_start, False
            )
            output_memory_obj.ref_count_down()
            raise
        finally:
            staging_memory_obj.ref_count_down()

    def _flush_loop(self):
        while self.running.is_set():
            tasks_to_process = []
            with self.flush_condition:
                self.flush_condition.wait(timeout=0.1)

                if not self.write_buffer:
                    continue

                now = time.monotonic()
                
                should_flush = False
                if len(self.write_buffer) >= self.write_window_size:
                    should_flush = True
                elif (now - self.oldest_item_timestamp) > self.max_flush_delay:
                    should_flush = True
                    logger.debug("Flushing SPDK write buffer: max delay expired.")

                if should_flush:
                    tasks_to_process = self.write_buffer
                    self.write_buffer = []

            if tasks_to_process:
                self._perform_batch_write(tasks_to_process)

    def _perform_batch_write(self, tasks: List[Tuple[CacheEngineKey, MemoryObj]]):
        if not tasks:
            return
            
        keys = [task[0] for task in tasks]
        memory_objs = [task[1] for task in tasks]
        batch_size = len(keys)
        blob_handles = []
        usage_added = 0
        io_sizes: List[int] = []

        try:
            blob_handles = [
                spdk.get_blob(timeout=self.blob_acquire_timeout_secs)
                for _ in range(batch_size)
            ]
            
            write_requests = []
            total_size = 0
            for i in range(batch_size):
                memory_obj = memory_objs[i]
                storage_size = self._get_storage_size(memory_obj)
                total_size += storage_size
                io_sizes.append(storage_size)
                write_requests.append(
                    (blob_handles[i], self._get_io_ptr(memory_obj), 0, storage_size)
                )

            with self.usage_lock:
                self.usage += total_size
                self.stats_monitor.update_local_storage_usage(self.usage)
                usage_added = total_size

            concurrent_future = spdk.write_batch_async(
                write_requests,
            )
            io_start = time.perf_counter()
            try:
                self._wait_for_io(concurrent_future)
            except Exception:
                self._log_io_trace(
                    "write_batch", io_sizes, time.perf_counter() - io_start, False
                )
                raise
            else:
                self._log_io_trace(
                    "write_batch", io_sizes, time.perf_counter() - io_start, True
                )

            for i in range(batch_size):
                self.insert_key(keys[i], memory_objs[i], blob_handles[i])
                memory_objs[i].ref_count_down()

        except Exception as e:
            logger.error(f"SPDK batch write failed: {e}", exc_info=True)
            if usage_added > 0:
                with self.usage_lock:
                    self.usage -= usage_added
                    self.stats_monitor.update_local_storage_usage(self.usage)
            for handle in blob_handles:
                spdk.release_blob(handle)
            for mem_obj in memory_objs:
                mem_obj.ref_count_down()
        finally:
            with self.put_tasks_lock:
                for key in keys:
                    if key in self.put_tasks:
                        del self.put_tasks[key]

    def __str__(self): return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.put_tasks_lock:
            if key in self.put_tasks: return True
        with self.dict_lock:
            if key not in self.dict: return False
            if pin: 
                self.dict[key].pin()
                self.keys_in_request.append(key)
            return True
            
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool: 
        return False

    def pin(self, key: CacheEngineKey) -> bool:
        with self.dict_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            return False
            
    def unpin(self, key: CacheEngineKey) -> bool:
        with self.dict_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            return False
            
    def remove(self, key: CacheEngineKey) -> None:
        with self.dict_lock:
            if key not in self.dict: return
            metadata = self.dict.pop(key)
            blob_handle = metadata.blob_handle
            size = metadata.size
        with self.usage_lock:
            self.usage -= size
            self.stats_monitor.update_local_storage_usage(self.usage)
        spdk.release_blob(blob_handle)
        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, "spdk"))
            
    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj, blob_handle: int) -> None:
        size = self._get_storage_size(memory_obj)
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
            self.dict[key] = SpdkBlobMetadata(blob_handle, size, shape, dtype, fmt, False)

        if has_stored:
            with self.usage_lock:
                self.usage -= delete_size
            logger.warning(f"Key {key} already exists in SPDK backend, overwritten.")
        
        if self.lmcache_worker is not None and not has_stored:
            self.lmcache_worker.put_msg(KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, "spdk"))

    def touch_cache(self):
        with self.dict_lock:
            for key in reversed(self.keys_in_request):
                if key in self.dict:
                    self.evictor.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def batched_submit_put_task(self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]) -> None:
        if not keys: return
        keys_to_process, objs_to_process = [], []
        with self.put_tasks_lock, self.dict_lock:
            for key, memory_obj in zip(keys, memory_objs):
                if key not in self.put_tasks and key not in self.dict:
                    keys_to_process.append(key)
                    objs_to_process.append(memory_obj)
        if not keys_to_process: return
        
        total_storage_size = sum(
            self._get_storage_size(mem_obj) for mem_obj in objs_to_process
        )
        with self.dict_lock:
            evict_keys, put_status = self.evictor.update_on_put(
                self.dict, total_storage_size
            )
        
        if put_status == PutStatus.ILLEGAL:
            logger.warning(
                f"Batch write failed: total size {total_storage_size} is too large for the cache."
            )
            return
            
        if evict_keys:
            for evict_key in evict_keys: self.remove(evict_key)
            if self.lookup_server is not None: self.lookup_server.batched_remove(evict_keys)
            
        if self.read_bios:
            with self.flush_condition:
                if not self.write_buffer:
                    self.oldest_item_timestamp = time.monotonic()
                
                for key, memory_obj in zip(keys_to_process, objs_to_process):
                    memory_obj.ref_count_up() 
                    self.write_buffer.append((key, memory_obj))
                    self.put_tasks[key] = memory_obj

                self.last_write_timestamp = time.monotonic()
                
                if len(self.write_buffer) >= self.write_window_size:
                    self.flush_condition.notify()
        else:
            tasks_to_write_now = []
            with self.put_tasks_lock:
                 for key, memory_obj in zip(keys_to_process, objs_to_process):
                    memory_obj.ref_count_up()
                    tasks_to_write_now.append((key, memory_obj))
                    self.put_tasks[key] = memory_obj
            
            self._perform_batch_write(tasks_to_write_now)
 
    def batched_get_blocking(self, keys: List[CacheEngineKey]) -> List[Optional[MemoryObj]]:
        if not self.concurrency:
            mem_objs = []
            for key in keys:
                mem_objs.append(self.get_blocking(key))
            return mem_objs
        else:
            if not keys: return []
            results: Dict[CacheEngineKey, Optional[MemoryObj]] = {key: None for key in keys}
            
            remaining_keys = list(keys)
            
            keys_after_put_tasks_check = []
            with self.put_tasks_lock:
                for key in remaining_keys:
                    if key in self.put_tasks:
                        mem_obj = self.put_tasks[key]
                        mem_obj.ref_count_up()
                        results[key] = mem_obj
                    else:
                        keys_after_put_tasks_check.append(key)

            if not keys_after_put_tasks_check:
                return [results.get(key) for key in keys]

            keys_after_prefetch_check = []
            prefetch_futures_to_wait: List[Tuple[CacheEngineKey, Future]] = []
            with self.prefetch_lock:
                for key in keys_after_put_tasks_check:
                    if key in self.prefetch_tasks:
                        prefetch_futures_to_wait.append((key, self.prefetch_tasks[key]))
                    else:
                        keys_after_prefetch_check.append(key)

            if prefetch_futures_to_wait:
                for key, future in prefetch_futures_to_wait:
                    try:
                        memory_obj = future.result(timeout=self.io_timeout_secs)
                        if memory_obj:
                            if self.local_cpu_backend.contains(key, pin=True):
                                results[key] = memory_obj
                            else:
                                logger.warning(f"Prefetched obj {key} was evicted from CPU. Re-reading.")
                                keys_after_prefetch_check.append(key)
                        else:
                            logger.warning(f"Prefetch task for {key} returned None. Will attempt SPDK read.")
                            keys_after_prefetch_check.append(key)
                    except Exception as e:
                        logger.error(f"Waiting for prefetch task for {key} failed: {e}. Will attempt SPDK read.", exc_info=True)
                        keys_after_prefetch_check.append(key)

            
            if not keys_after_prefetch_check:
                return [results.get(key) for key in keys]
            remaining_keys = keys_after_prefetch_check

            spdk_tasks_info = []
            if remaining_keys:
                with self.dict_lock:
                    for key in remaining_keys:
                        if key in self.dict:
                            self.evictor.update_on_hit(key, self.dict)
                            metadata = self.dict[key]
                            spdk_tasks_info.append({
                                "key": key,
                                "blob_handle": metadata.blob_handle,
                                "storage_size": metadata.size,
                                "dtype": metadata.dtype,
                                "shape": metadata.shape,
                                "fmt": metadata.fmt
                            })

            if spdk_tasks_info:
                start_time = time.time()
                spdk_results, total_read_bytes = self._perform_batched_spdk_read(
                    spdk_tasks_info
                )
                end_time = time.time()
                read_time = end_time - start_time
                if total_read_bytes > 0:
                    with self.stats_lock:
                        self.cumulative_read_bytes += total_read_bytes
                        self.cumulative_read_time += read_time
                        logger.info(f"current average read bandwidth: "
                                    f"{(self.cumulative_read_bytes / self.cumulative_read_time) / (1024 * 1024):.2f} MB/s")

                for task_info, result_obj in zip(spdk_tasks_info, spdk_results):
                    results[task_info["key"]] = result_obj
                    
            return [results.get(key) for key in keys]

    def batched_get_to_gpu_blocking(
        self,
        keys: List[CacheEngineKey],
        starts: List[int],
        ends: List[int],
        gpu_connector: Any,
        **kwargs,
    ) -> Optional[List[Optional[MemoryObj]]]:
        if not self.enable_chunked_gpu_overlap or len(keys) <= 1:
            return None

        if len(keys) != len(starts) or len(keys) != len(ends):
            raise ValueError(
                "SPDK batched_get_to_gpu_blocking requires keys, starts, and ends "
                "to have the same length."
            )

        results: List[Optional[MemoryObj]] = [None] * len(keys)
        remaining_indexes = list(range(len(keys)))

        indexes_after_put_tasks_check = []
        with self.put_tasks_lock:
            for idx in remaining_indexes:
                key = keys[idx]
                if key in self.put_tasks:
                    memory_obj = self.put_tasks[key]
                    memory_obj.ref_count_up()
                    results[idx] = memory_obj
                    gpu_connector.to_gpu(memory_obj, starts[idx], ends[idx], **kwargs)
                else:
                    indexes_after_put_tasks_check.append(idx)

        if not indexes_after_put_tasks_check:
            return results

        indexes_after_prefetch_check = []
        prefetch_futures_to_wait: List[Tuple[int, Future]] = []
        with self.prefetch_lock:
            for idx in indexes_after_put_tasks_check:
                key = keys[idx]
                if key in self.prefetch_tasks:
                    prefetch_futures_to_wait.append((idx, self.prefetch_tasks[key]))
                else:
                    indexes_after_prefetch_check.append(idx)

        for idx, future in prefetch_futures_to_wait:
            key = keys[idx]
            try:
                memory_obj = future.result(timeout=self.io_timeout_secs)
                if memory_obj:
                    if self.local_cpu_backend.contains(key, pin=True):
                        results[idx] = memory_obj
                        gpu_connector.to_gpu(
                            memory_obj, starts[idx], ends[idx], **kwargs
                        )
                    else:
                        logger.warning(
                            f"Prefetched obj {key} was evicted from CPU. Re-reading."
                        )
                        indexes_after_prefetch_check.append(idx)
                else:
                    logger.warning(
                        f"Prefetch task for {key} returned None. Will attempt SPDK read."
                    )
                    indexes_after_prefetch_check.append(idx)
            except Exception as e:
                logger.error(
                    f"Waiting for prefetch task for {key} failed: {e}. "
                    "Will attempt SPDK read.",
                    exc_info=True,
                )
                indexes_after_prefetch_check.append(idx)

        if not indexes_after_prefetch_check:
            return results

        spdk_tasks_info = []
        with self.dict_lock:
            for idx in indexes_after_prefetch_check:
                key = keys[idx]
                if key not in self.dict:
                    continue
                self.evictor.update_on_hit(key, self.dict)
                metadata = self.dict[key]
                spdk_tasks_info.append(
                    {
                        "index": idx,
                        "key": key,
                        "blob_handle": metadata.blob_handle,
                        "storage_size": metadata.size,
                        "dtype": metadata.dtype,
                        "shape": metadata.shape,
                        "fmt": metadata.fmt,
                        "start": starts[idx],
                        "end": ends[idx],
                    }
                )

        if not spdk_tasks_info:
            return results

        start_time = time.time()
        total_read_bytes = self._perform_batched_connector_overlap_spdk_read(
            spdk_tasks_info,
            results,
            gpu_connector,
            **kwargs,
        )
        read_time = time.time() - start_time
        if total_read_bytes > 0:
            with self.stats_lock:
                self.cumulative_read_bytes += total_read_bytes
                self.cumulative_read_time += read_time
                logger.info(
                    "current average read bandwidth: "
                    f"{(self.cumulative_read_bytes / self.cumulative_read_time) / (1024 * 1024):.2f} MB/s"
                )

        return results

    def _perform_batched_connector_overlap_spdk_read(
        self,
        spdk_tasks_info: List[Dict[str, Any]],
        results: List[Optional[MemoryObj]],
        gpu_connector: Any,
        **kwargs,
    ) -> int:
        if not spdk_tasks_info:
            return 0

        io_futures: Dict[Future, Dict[str, Any]] = {}
        io_sizes: List[int] = []
        total_request_bytes = 0
        completed_indexes = set()
        io_start = time.perf_counter()

        try:
            for submit_idx, task_info in enumerate(spdk_tasks_info):
                storage_size = task_info["storage_size"]
                memory_obj = self._allocate_host_memory_obj(
                    task_info["shape"],
                    task_info["dtype"],
                    task_info["fmt"],
                    storage_size,
                )
                if memory_obj is None:
                    logger.warning(
                        "Host allocation failed during connector-overlap SPDK read."
                    )
                    continue

                results[task_info["index"]] = memory_obj
                io_sizes.append(storage_size)
                total_request_bytes += storage_size
                future = spdk.read_async(
                    task_info["blob_handle"],
                    self._get_io_ptr(memory_obj),
                    0,
                    storage_size,
                )
                io_futures[future] = task_info
                if submit_idx + 1 < len(spdk_tasks_info):
                    time.sleep(self.sleep_time)

            if not io_futures:
                return 0

            pending_futures = set(io_futures.keys())
            while pending_futures:
                completed_futures, pending_futures = wait(
                    pending_futures,
                    timeout=self.io_timeout_secs,
                    return_when=FIRST_COMPLETED,
                )
                if not completed_futures:
                    raise TimeoutError(
                        "Timed out waiting for SPDK connector-overlap read completion."
                    )

                for future in completed_futures:
                    future.result()
                    completion = io_futures[future]
                    memory_obj = results[completion["index"]]
                    if memory_obj is None:
                        continue

                    gpu_connector.to_gpu(
                        memory_obj,
                        completion["start"],
                        completion["end"],
                        **kwargs,
                    )
                    completed_indexes.add(completion["index"])

            self._log_io_trace(
                "read_batch_connector_overlap",
                io_sizes,
                time.perf_counter() - io_start,
                True,
            )
            return total_request_bytes
        except Exception as e:
            self._log_io_trace(
                "read_batch_connector_overlap",
                io_sizes,
                time.perf_counter() - io_start,
                False,
            )
            logger.error(
                f"SPDK batched connector-overlap read operation failed or timed out: {e}",
                exc_info=True,
            )
            for task_info in spdk_tasks_info:
                idx = task_info["index"]
                if idx in completed_indexes:
                    continue
                memory_obj = results[idx]
                if memory_obj is not None:
                    memory_obj.ref_count_down()
                    results[idx] = None
            return 0

    def _perform_batched_spdk_read(
        self, spdk_tasks_info: List[Dict[str, Any]]
    ) -> Tuple[List[Optional[MemoryObj]], int]:
        if not spdk_tasks_info:
            return [], 0

        if any(
            self._should_use_chunked_gpu_overlap(task_info["storage_size"])
            for task_info in spdk_tasks_info
        ):
            if self._should_prefer_blob_gpu_overlap(spdk_tasks_info):
                return self._perform_batched_blob_overlap_spdk_read(spdk_tasks_info)
            return self._perform_batched_chunked_spdk_read(spdk_tasks_info)
        
        batch_requests = []
        allocated_memory_objs: List[Optional[MemoryObj]] = []
        total_request_bytes = 0
        io_sizes: List[int] = []
        
        for task_info in spdk_tasks_info:
            memory_obj = self._allocate_memory_obj(task_info["shape"], task_info["dtype"], task_info["fmt"])
            if memory_obj is None:
                logger.warning("Memory allocation failed during batched read, this task will be skipped.")
                allocated_memory_objs.append(None)
                continue
            storage_size = self._get_storage_size(memory_obj)
            if storage_size != task_info["storage_size"]:
                raise ValueError(
                    "Allocated memory object storage size does not match blob size: "
                    f"allocated={storage_size}, blob={task_info['storage_size']}"
                )
            allocated_memory_objs.append(memory_obj)
            request_tuple = (
                task_info["blob_handle"],
                self._get_io_ptr(memory_obj),
                0,
                storage_size,
            )
            batch_requests.append(request_tuple)
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
            logger.error(f"SPDK batch read operation failed or timed out: {e}", exc_info=True)
            for mem_obj in allocated_memory_objs:
                if mem_obj: mem_obj.ref_count_down()
            return [None] * len(spdk_tasks_info), 0

    def _perform_batched_blob_overlap_spdk_read(
        self, spdk_tasks_info: List[Dict[str, Any]]
    ) -> Tuple[List[Optional[MemoryObj]], int]:
        results: List[Optional[MemoryObj]] = [None] * len(spdk_tasks_info)
        staging_memory_objs: List[Optional[MemoryObj]] = [None] * len(spdk_tasks_info)
        io_futures: Dict[Future, Dict[str, Any]] = {}
        io_sizes: List[int] = []
        total_request_bytes = 0
        copy_stream = torch.cuda.Stream(device=self.cuda_device)
        issued_gpu_copy = False
        io_start = time.perf_counter()

        try:
            for idx, task_info in enumerate(spdk_tasks_info):
                storage_size = task_info["storage_size"]
                if self._should_use_chunked_gpu_overlap(storage_size):
                    staging_memory_obj = self._allocate_host_memory_obj(
                        task_info["shape"],
                        task_info["dtype"],
                        task_info["fmt"],
                        storage_size,
                    )
                    if staging_memory_obj is None:
                        logger.warning(
                            "Host staging allocation failed during batched blob-overlap SPDK read."
                        )
                        continue

                    output_memory_obj = self._allocate_device_memory_obj(
                        task_info["shape"],
                        task_info["dtype"],
                        task_info["fmt"],
                        storage_size,
                    )
                    if output_memory_obj is None:
                        logger.warning(
                            "GPU allocation failed during batched blob-overlap SPDK read. "
                            "Falling back to host read for this blob."
                        )
                        staging_memory_obj.ref_count_down()
                        memory_obj = self._allocate_host_memory_obj(
                            task_info["shape"],
                            task_info["dtype"],
                            task_info["fmt"],
                            storage_size,
                        )
                        if memory_obj is None:
                            logger.warning(
                                "Host allocation fallback failed during batched blob-overlap SPDK read."
                            )
                            continue

                        results[idx] = memory_obj
                        io_sizes.append(storage_size)
                        total_request_bytes += storage_size
                        future = spdk.read_async(
                            task_info["blob_handle"],
                            self._get_io_ptr(memory_obj),
                            0,
                            storage_size,
                        )
                        io_futures[future] = {"kind": "host"}
                        continue

                    staging_memory_objs[idx] = staging_memory_obj
                    results[idx] = output_memory_obj
                    staging_tensor = staging_memory_obj.tensor
                    output_tensor = output_memory_obj.tensor
                    assert staging_tensor is not None
                    assert output_tensor is not None

                    io_sizes.append(storage_size)
                    total_request_bytes += storage_size
                    future = spdk.read_async(
                        task_info["blob_handle"],
                        self._get_io_ptr(staging_memory_obj),
                        0,
                        storage_size,
                    )
                    io_futures[future] = {
                        "kind": "blob",
                        "logical_bytes": output_memory_obj.get_size(),
                        "staging_bytes": staging_tensor.view(torch.uint8).flatten(),
                        "output_bytes": output_tensor.view(torch.uint8).flatten(),
                    }
                    continue

                memory_obj = self._allocate_host_memory_obj(
                    task_info["shape"],
                    task_info["dtype"],
                    task_info["fmt"],
                    storage_size,
                )
                if memory_obj is None:
                    logger.warning(
                        "Memory allocation failed during batched blob-overlap SPDK read, this task will be skipped."
                    )
                    continue

                results[idx] = memory_obj
                io_sizes.append(storage_size)
                total_request_bytes += storage_size
                future = spdk.read_async(
                    task_info["blob_handle"],
                    self._get_io_ptr(memory_obj),
                    0,
                    storage_size,
                )
                io_futures[future] = {"kind": "host"}

            if not io_futures:
                return results, 0

            pending_futures = set(io_futures.keys())
            while pending_futures:
                completed_futures, pending_futures = wait(
                    pending_futures,
                    timeout=self.io_timeout_secs,
                    return_when=FIRST_COMPLETED,
                )
                if not completed_futures:
                    raise TimeoutError(
                        "Timed out waiting for SPDK batched blob-overlap read completion."
                    )

                for future in completed_futures:
                    future.result()
                    completion = io_futures[future]
                    if completion["kind"] != "blob":
                        continue

                    copy_size = completion["logical_bytes"]
                    if copy_size <= 0:
                        continue

                    issued_gpu_copy = True
                    with torch.cuda.stream(copy_stream):
                        completion["output_bytes"][:copy_size].copy_(
                            completion["staging_bytes"][:copy_size],
                            non_blocking=True,
                        )

            if issued_gpu_copy:
                copy_stream.synchronize()

            self._log_io_trace(
                "read_batch_blob_overlap",
                io_sizes,
                time.perf_counter() - io_start,
                True,
            )
            return results, total_request_bytes
        except Exception as e:
            try:
                if issued_gpu_copy:
                    copy_stream.synchronize()
            except Exception:
                logger.debug(
                    "Blob-overlap SPDK batch copy stream synchronize failed during error handling.",
                    exc_info=True,
                )

            self._log_io_trace(
                "read_batch_blob_overlap",
                io_sizes,
                time.perf_counter() - io_start,
                False,
            )
            logger.error(
                f"SPDK batched blob-overlap read operation failed or timed out: {e}",
                exc_info=True,
            )
            for memory_obj in results:
                if memory_obj is not None:
                    memory_obj.ref_count_down()
            return [None] * len(spdk_tasks_info), 0
        finally:
            for staging_memory_obj in staging_memory_objs:
                if staging_memory_obj is not None:
                    staging_memory_obj.ref_count_down()

    def _perform_batched_chunked_spdk_read(
        self, spdk_tasks_info: List[Dict[str, Any]]
    ) -> Tuple[List[Optional[MemoryObj]], int]:
        results: List[Optional[MemoryObj]] = [None] * len(spdk_tasks_info)
        staging_memory_objs: List[Optional[MemoryObj]] = [None] * len(spdk_tasks_info)
        io_futures: Dict[Future, Dict[str, Any]] = {}
        io_sizes: List[int] = []
        total_request_bytes = 0
        copy_stream = torch.cuda.Stream(device=self.cuda_device)
        issued_gpu_copy = False
        io_start = time.perf_counter()

        try:
            for idx, task_info in enumerate(spdk_tasks_info):
                storage_size = task_info["storage_size"]
                if self._should_use_chunked_gpu_overlap(storage_size):
                    staging_memory_obj = self._allocate_host_memory_obj(
                        task_info["shape"],
                        task_info["dtype"],
                        task_info["fmt"],
                        storage_size,
                    )
                    if staging_memory_obj is None:
                        logger.warning(
                            "Host staging allocation failed during batched chunked SPDK read."
                        )
                        continue

                    output_memory_obj = self._allocate_device_memory_obj(
                        task_info["shape"],
                        task_info["dtype"],
                        task_info["fmt"],
                        storage_size,
                    )
                    if output_memory_obj is None:
                        logger.warning(
                            "GPU allocation failed during batched chunked SPDK read. "
                            "Falling back to host read for this blob."
                        )
                        staging_memory_obj.ref_count_down()
                        memory_obj = self._allocate_host_memory_obj(
                            task_info["shape"],
                            task_info["dtype"],
                            task_info["fmt"],
                            storage_size,
                        )
                        if memory_obj is None:
                            logger.warning(
                                "Host allocation fallback failed during batched chunked SPDK read."
                            )
                            continue

                        results[idx] = memory_obj
                        io_sizes.append(storage_size)
                        total_request_bytes += storage_size
                        future = spdk.read_async(
                            task_info["blob_handle"],
                            self._get_io_ptr(memory_obj),
                            0,
                            storage_size,
                        )
                        io_futures[future] = {
                            "kind": "host",
                            "index": idx,
                        }
                        continue

                    staging_memory_objs[idx] = staging_memory_obj
                    results[idx] = output_memory_obj
                    logical_bytes = output_memory_obj.get_size()
                    staging_tensor = staging_memory_obj.tensor
                    output_tensor = output_memory_obj.tensor
                    assert staging_tensor is not None
                    assert output_tensor is not None
                    staging_bytes = staging_tensor.view(torch.uint8).flatten()
                    output_bytes = output_tensor.view(torch.uint8).flatten()

                    for offset in range(0, storage_size, self.chunked_gpu_overlap_size):
                        chunk_size = min(
                            self.chunked_gpu_overlap_size, storage_size - offset
                        )
                        io_sizes.append(chunk_size)
                        future = spdk.read_async(
                            task_info["blob_handle"],
                            self._get_io_ptr(staging_memory_obj) + offset,
                            offset,
                            chunk_size,
                        )
                        io_futures[future] = {
                            "kind": "chunk",
                            "index": idx,
                            "offset": offset,
                            "chunk_size": chunk_size,
                            "logical_bytes": logical_bytes,
                            "staging_bytes": staging_bytes,
                            "output_bytes": output_bytes,
                        }
                    total_request_bytes += storage_size
                    continue

                memory_obj = self._allocate_host_memory_obj(
                    task_info["shape"],
                    task_info["dtype"],
                    task_info["fmt"],
                    storage_size,
                )
                if memory_obj is None:
                    logger.warning(
                        "Memory allocation failed during batched SPDK read, this task will be skipped."
                    )
                    continue

                results[idx] = memory_obj
                io_sizes.append(storage_size)
                total_request_bytes += storage_size
                future = spdk.read_async(
                    task_info["blob_handle"],
                    self._get_io_ptr(memory_obj),
                    0,
                    storage_size,
                )
                io_futures[future] = {
                    "kind": "host",
                    "index": idx,
                }

            if not io_futures:
                return results, 0

            pending_futures = set(io_futures.keys())
            while pending_futures:
                completed_futures, pending_futures = wait(
                    pending_futures,
                    timeout=self.io_timeout_secs,
                    return_when=FIRST_COMPLETED,
                )
                if not completed_futures:
                    raise TimeoutError(
                        "Timed out waiting for SPDK batched chunked read completion."
                    )

                for future in completed_futures:
                    future.result()
                    completion = io_futures[future]
                    if completion["kind"] != "chunk":
                        continue

                    offset = completion["offset"]
                    chunk_size = completion["chunk_size"]
                    remaining_logical_bytes = max(
                        0, completion["logical_bytes"] - offset
                    )
                    copy_size = min(chunk_size, remaining_logical_bytes)
                    if copy_size <= 0:
                        continue

                    issued_gpu_copy = True
                    with torch.cuda.stream(copy_stream):
                        completion["output_bytes"][offset : offset + copy_size].copy_(
                            completion["staging_bytes"][offset : offset + copy_size],
                            non_blocking=True,
                        )

            if issued_gpu_copy:
                copy_stream.synchronize()

            self._log_io_trace(
                "read_batch_chunked",
                io_sizes,
                time.perf_counter() - io_start,
                True,
            )
            return results, total_request_bytes
        except Exception as e:
            try:
                if issued_gpu_copy:
                    copy_stream.synchronize()
            except Exception:
                logger.debug(
                    "Chunked SPDK batch copy stream synchronize failed during error handling.",
                    exc_info=True,
                )

            self._log_io_trace(
                "read_batch_chunked",
                io_sizes,
                time.perf_counter() - io_start,
                False,
            )
            logger.error(
                f"SPDK batched chunked read operation failed or timed out: {e}",
                exc_info=True,
            )
            for memory_obj in results:
                if memory_obj is not None:
                    memory_obj.ref_count_down()
            return [None] * len(spdk_tasks_info), 0
        finally:
            for staging_memory_obj in staging_memory_objs:
                if staging_memory_obj is not None:
                    staging_memory_obj.ref_count_down()

    def get_non_blocking(self, key: CacheEngineKey) -> Optional["Future"]:
        if self.enable_chunked_gpu_overlap:
            with self.dict_lock:
                if key not in self.dict:
                    return None

                metadata = self.dict[key]
                if self._should_use_chunked_gpu_overlap(metadata.size):
                    self.evictor.update_on_hit(key, self.dict)
                    metadata.pin()
                    future = asyncio.run_coroutine_threadsafe(
                        self.async_load_bytes_from_spdk_to_device(
                            blob_handle=metadata.blob_handle,
                            storage_size=metadata.size,
                            dtype=metadata.dtype,
                            shape=metadata.shape,
                            fmt=metadata.fmt,
                        ),
                        self.loop,
                    )
                    future.add_done_callback(
                        lambda f: self._finish_async_device_read(key, f)
                    )
                    return future
        return self.submit_prefetch_task(key)
    
    def _remove_prefetch_task(self, key: CacheEngineKey, future: Future):
        with self.prefetch_lock:
            self.prefetch_tasks.pop(key, None)
        
        try:
            future.result() 
        except Exception as e:
            logger.warning(f"Prefetch task for {key} failed: {e}. Unpinning entry.")
            with self.dict_lock:
                if key in self.dict:
                    self.dict[key].unpin()

    def _finish_async_device_read(self, key: CacheEngineKey, future: Future):
        try:
            future.result()
        except Exception as e:
            logger.warning(f"Async SPDK device read for {key} failed: {e}.")
        finally:
            with self.dict_lock:
                if key in self.dict:
                    self.dict[key].unpin()

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        with self.prefetch_lock:
            if key in self.prefetch_tasks:
                logger.debug(f"Prefetch task for {key} is already in progress.")
                return self.prefetch_tasks[key]

            with self.dict_lock:
                if key not in self.dict: return None
                self.dict[key].pin()
                metadata = self.dict[key]
                blob_info = {
                    "key": key,
                    "blob_handle": metadata.blob_handle,
                    "storage_size": metadata.size,
                    "dtype": metadata.dtype,
                    "shape": metadata.shape,
                    "fmt": metadata.fmt
                }
            assert blob_info["dtype"] is not None and blob_info["shape"] is not None
            
            coro = self.async_load_bytes_from_spdk(
                blob_info["key"],
                blob_info["blob_handle"], 
                blob_info["storage_size"],
                blob_info["dtype"], 
                blob_info["shape"], 
                blob_info["fmt"]
            )
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)

            future.add_done_callback(lambda f: self._remove_prefetch_task(key, f))
            
            self.prefetch_tasks[key] = future
            return future
    
    async def async_load_bytes_from_spdk(
        self, key: CacheEngineKey, blob_handle: int, storage_size: int, dtype, shape, fmt
    ) -> Optional[MemoryObj]:
        
        memory_obj = self._allocate_memory_obj(shape, dtype, fmt)
        if memory_obj is None:
            logger.debug("Memory allocation failed during async spdk load.")
            return None

        allocated_storage_size = self._get_storage_size(memory_obj)
        if allocated_storage_size != storage_size:
            raise ValueError(
                "Allocated memory object storage size does not match blob size: "
                f"allocated={allocated_storage_size}, blob={storage_size}"
            )
        
        concurrent_future = spdk.read_async(
            blob_handle, self._get_io_ptr(memory_obj), 0, storage_size
        )
        io_start = time.perf_counter()
        try:
            await asyncio.wait_for(
                asyncio.wrap_future(concurrent_future),
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

        self.local_cpu_backend.submit_put_task(key, memory_obj)

        return memory_obj

    async def async_load_bytes_from_spdk_to_device(
        self, blob_handle: int, storage_size: int, dtype, shape, fmt
    ) -> Optional[MemoryObj]:
        return await self._read_blob_to_device_chunked_async(
            blob_handle,
            storage_size,
            dtype,
            shape,
            fmt,
            "read_non_blocking_chunked",
        )

    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Optional[Future]:
        logger.warning("SpdkBlobBackend.submit_put_task is deprecated, use batched_submit_put_task instead.")
        self.batched_submit_put_task([key], [memory_obj])
        return None
    
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # logger.warning("SpdkBlobBackend.get_blocking is deprecated, use batched_get_blocking instead.")
        # results = self.batched_get_blocking([key])
        # return results[0] if results else None
        with self.dict_lock:
            if key not in self.dict:
                return None
            
            self.evictor.update_on_hit(key, self.dict)

            metadata = self.dict[key]
            blob_handle = metadata.blob_handle
            dtype = metadata.dtype
            shape = metadata.shape
            fmt = metadata.fmt
            start_time = time.time()
            if self._should_use_chunked_gpu_overlap(metadata.size):
                memory_obj = self._read_blob_to_device_chunked_blocking(
                    blob_handle,
                    metadata.size,
                    dtype,
                    shape,
                    fmt,
                    "read_blocking_chunked",
                )
            else:
                memory_obj = self._read_blob_to_host_blocking(
                    blob_handle,
                    metadata.size,
                    dtype,
                    shape,
                    fmt,
                    "read_blocking",
                )

            if memory_obj is None:
                return None

            end_time = time.time()
            read_time = end_time - start_time
            total_read_bytes = metadata.size
            with self.stats_lock:
                self.cumulative_read_bytes += total_read_bytes
                self.cumulative_read_time += read_time
                logger.info(f"current average read bandwidth: "
                            f"{(self.cumulative_read_bytes / self.cumulative_read_time) / (1024 * 1024):.2f} MB/s")
            return memory_obj

    def close(self) -> None:
        print("SPDK Blob Backend closing...")
        logger.info("Closing SPDK blob backend...")
        self.running.clear()
        
        logger.info("Waiting for flush thread to finish...")
        with self.flush_condition:
            self.flush_condition.notify()
        self.flush_thread.join(timeout=5)
        if self.flush_thread.is_alive():
            logger.warning(f"{self.flush_thread.name} did not exit gracefully.")

        if self.write_buffer:
            logger.info(f"Flushing {len(self.write_buffer)} remaining items from write buffer before shutdown.")
            self._perform_batch_write(self.write_buffer)
            self.write_buffer = []

        if self.lookup_server is not None:
            with self.dict_lock:
                keys_to_remove = list(self.dict.keys())
            if keys_to_remove:
                self.lookup_server.batched_remove(keys_to_remove)
                
        spdk.unload()
        logger.info("SPDK blob backend closed.")
