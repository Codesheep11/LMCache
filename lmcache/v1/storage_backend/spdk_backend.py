from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple
import threading
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, SpdkBlobMetadata
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.evictor import LRUEvictor, PutStatus
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

# Local
import spdk_controller as spdk

logger = init_logger(__name__)

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
        
        spdk_max_size = config.spdk_max_size
        self.evictor = LRUEvictor(max_cache_size=spdk_max_size)

        self.dict_lock = threading.RLock()
        self.usage_lock = threading.RLock()
        
        self.write_window_size = getattr(config, 'spdk_write_window_size', 32)
        self.idle_flush_time = getattr(config, 'spdk_idle_flush_time', 0.5)  # I/O空窗时间 (s)
        self.max_flush_delay = getattr(config, 'spdk_max_flush_delay', 1.0)  # 最大延迟 (s)
        self.read_bios = getattr(config, 'read_bios', True)

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

        try:
            blob_handles = [spdk.get_blob() for _ in range(batch_size)]
            
            write_requests = []
            total_size = 0
            for i in range(batch_size):
                memory_obj = memory_objs[i]
                size = memory_obj.get_physical_size()
                total_size += size
                write_requests.append((blob_handles[i], memory_obj.meta.address, 0, size))

            with self.usage_lock:
                self.usage += total_size
                self.stats_monitor.update_local_storage_usage(self.usage)

            concurrent_future = spdk.write_batch_async(write_requests)
            concurrent_future.result(timeout=30)

            for i in range(batch_size):
                self.insert_key(keys[i], memory_objs[i], blob_handles[i])
                memory_objs[i].ref_count_down()

        except Exception as e:
            logger.error(f"SPDK batch write failed: {e}", exc_info=True)
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
            if pin: self.dict[key].pin()
            return True
            
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool: return False

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
        size, shape, dtype, fmt = memory_obj.get_size(), memory_obj.metadata.shape, memory_obj.metadata.dtype, memory_obj.metadata.fmt
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

    def batched_submit_put_task(self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]) -> None:
        if not keys: return
        keys_to_process, objs_to_process = [], []
        with self.put_tasks_lock, self.dict_lock:
            for key, memory_obj in zip(keys, memory_objs):
                if key not in self.put_tasks and key not in self.dict:
                    keys_to_process.append(key)
                    objs_to_process.append(memory_obj)
        if not keys_to_process: return
        
        total_physical_size = sum(mem_obj.get_physical_size() for mem_obj in objs_to_process)
        with self.dict_lock:
            evict_keys, put_status = self.evictor.update_on_put(self.dict, total_physical_size)
        
        if put_status == PutStatus.ILLEGAL:
            logger.warning(f"Batch write failed: total size {total_physical_size} is too large for the cache.")
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
        remaining_keys = keys_after_put_tasks_check

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
                            "dtype": metadata.dtype,
                            "shape": metadata.shape,
                            "fmt": metadata.fmt
                        })

        if spdk_tasks_info:
            spdk_results = self._perform_batched_spdk_read(spdk_tasks_info)
            for task_info, result_obj in zip(spdk_tasks_info, spdk_results):
                results[task_info["key"]] = result_obj
                
        return [results.get(key) for key in keys]

    def _perform_batched_spdk_read(self, spdk_tasks_info: List[Dict[str, Any]]) -> List[Optional[MemoryObj]]:
        if not spdk_tasks_info: return []
        
        batch_requests = []
        allocated_memory_objs: List[Optional[MemoryObj]] = []
        
        for task_info in spdk_tasks_info:
            memory_obj = self.local_cpu_backend.allocate(task_info["shape"], task_info["dtype"], task_info["fmt"])
            if memory_obj is None:
                logger.warning("Memory allocation failed during batched read, this task will be skipped.")
                allocated_memory_objs.append(None)
                continue
            allocated_memory_objs.append(memory_obj)
            request_tuple = (task_info["blob_handle"], memory_obj.meta.address, 0, memory_obj.get_physical_size())
            batch_requests.append(request_tuple)
        
        if not batch_requests:
            return allocated_memory_objs

        future = spdk.read_batch_async(batch_requests)
        try:
            future.result(timeout=30)
            return allocated_memory_objs
        except Exception as e:
            logger.error(f"SPDK batch read operation failed or timed out: {e}", exc_info=True)
            for mem_obj in allocated_memory_objs:
                if mem_obj: mem_obj.ref_count_down()
            return [None] * len(spdk_tasks_info)

    def get_non_blocking(self, key: CacheEngineKey) -> Optional["Future"]:
        return self.submit_prefetch_task(key)
    
    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        with self.dict_lock:
            if key not in self.dict: return None
            self.evictor.update_on_hit(key, self.dict)
            metadata = self.dict[key]
            blob_info = {
                "blob_handle": metadata.blob_handle,
                "dtype": metadata.dtype,
                "shape": metadata.shape,
                "fmt": metadata.fmt
            }
        assert blob_info["dtype"] is not None and blob_info["shape"] is not None
        return asyncio.run_coroutine_threadsafe(
            self.async_load_bytes_from_spdk(blob_info["blob_handle"], blob_info["dtype"], blob_info["shape"], blob_info["fmt"]),
            self.loop
        )
    
    async def async_load_bytes_from_spdk(self, blob_handle: int, dtype, shape, fmt) -> Optional[MemoryObj]:
        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        if memory_obj is None:
            logger.debug("Memory allocation failed during async spdk load.")
            return None
        concurrent_future = spdk.read_async(blob_handle, memory_obj.meta.address, 0, memory_obj.get_physical_size())
        await asyncio.wrap_future(concurrent_future)
        return memory_obj

    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Optional[Future]:
        logger.warning("SpdkBlobBackend.submit_put_task is deprecated, use batched_submit_put_task instead.")
        self.batched_submit_put_task([key], [memory_obj])
        return None
    
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        logger.warning("SpdkBlobBackend.get_blocking is deprecated, use batched_get_blocking instead.")
        results = self.batched_get_blocking([key])
        return results[0] if results else None

    def close(self) -> None:
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