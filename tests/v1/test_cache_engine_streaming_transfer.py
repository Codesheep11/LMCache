# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace
import threading

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.storage_backend.transfer_result import (
    BackendTransferResult,
    BackendWriteSubmissionResult,
)


class FakeDirectBackend:
    def __init__(self) -> None:
        self.put_calls = []
        self.get_calls = []

    def batched_put_from_gpu_blocking(self, *args, **kwargs):
        self.put_calls.append((args, kwargs))
        return "put-result"

    def batched_get_to_gpu_blocking(self, *args, **kwargs):
        self.get_calls.append((args, kwargs))
        return "get-result"


def make_storage_manager(backends) -> StorageManager:
    manager = StorageManager.__new__(StorageManager)
    manager.storage_backends = OrderedDict(backends)
    manager._freeze = False
    manager._freeze_lock = threading.RLock()
    manager._bypassed_backends = set()
    manager._bypass_lock = threading.RLock()
    return manager


def test_storage_manager_dispatches_xds_only_streaming_store():
    backend = FakeDirectBackend()
    manager = make_storage_manager([("XDSBackend", backend)])

    result = manager.batched_put_from_gpu(
        keys=["key"],
        starts=[0],
        ends=[256],
        gpu_connector=SimpleNamespace(),
    )

    assert result == "put-result"
    assert len(backend.put_calls) == 1


def test_storage_manager_declines_streaming_store_with_multiple_backends():
    backend = FakeDirectBackend()
    manager = make_storage_manager(
        [("LocalCPUBackend", object()), ("XDSBackend", backend)]
    )

    result = manager.batched_put_from_gpu(
        keys=["key"],
        starts=[0],
        ends=[256],
        gpu_connector=SimpleNamespace(),
    )

    assert result is None
    assert backend.put_calls == []


def test_storage_manager_dispatches_requested_xds_streaming_retrieve():
    backend = FakeDirectBackend()
    manager = make_storage_manager(
        [("LocalCPUBackend", object()), ("XDSBackend", backend)]
    )

    result = manager.batched_get_to_gpu(
        keys=["key"],
        starts=[0],
        ends=[256],
        gpu_connector=SimpleNamespace(),
        location="XDSBackend",
    )

    assert result == "get-result"
    assert len(backend.get_calls) == 1


class FakeTokenDatabase:
    def __init__(self, keys) -> None:
        self.keys = keys

    def process_tokens(
        self,
        tokens=None,
        hashes=None,
        offsets=None,
        mask=None,
        request_configs=None,
    ):
        del hashes, offsets, mask, request_configs
        chunk_size = len(tokens) // len(self.keys)
        for index, key in enumerate(self.keys):
            start = index * chunk_size
            yield start, start + chunk_size, key


class FakeMemoryObj:
    def __init__(self) -> None:
        self.refs = 1

    def get_size(self) -> int:
        return 4096

    def ref_count_down(self) -> None:
        self.refs -= 1


class FakeGPUConnector:
    def __init__(self) -> None:
        self.batched_from_gpu_calls = 0
        self.batched_to_gpu_calls = 0

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs) -> None:
        del memory_objs, starts, ends, kwargs
        self.batched_from_gpu_calls += 1

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs) -> None:
        del memory_objs, starts, ends, kwargs
        self.batched_to_gpu_calls += 1


class FakeRequestStats:
    process_tokens_time = 0.0
    from_gpu_time = 0.0
    put_time = 0.0

    def profile_process_tokens(self):
        return nullcontext()

    def profile_from_gpu(self):
        return nullcontext()

    def profile_put(self):
        return nullcontext()

    def profile_to_gpu(self):
        return nullcontext()

    def time_to_store(self) -> float:
        return 1.0

    def time_to_retrieve(self) -> float:
        return 1.0


class FakeStatsMonitor:
    def __init__(self) -> None:
        self.store_finished_tokens = None
        self.retrieve_finished_tokens = None

    def on_store_request(self, num_tokens):
        del num_tokens
        return FakeRequestStats()

    def on_store_finished(self, stats, num_tokens) -> None:
        del stats
        self.store_finished_tokens = int(num_tokens)

    def on_retrieve_request(self, num_tokens):
        del num_tokens
        return FakeRequestStats()

    def on_retrieve_finished(self, stats, num_tokens) -> None:
        del stats
        self.retrieve_finished_tokens = int(num_tokens)


class FakeEngineStorageManager:
    def __init__(self, read_transfer=None, write_transfer=None, fallback=None) -> None:
        self.read_transfer = read_transfer
        self.write_transfer = write_transfer
        self.fallback = fallback or []
        self.allocate_calls = 0
        self.batched_put_calls = 0
        self.batched_get_calls = 0
        self.batched_put_from_gpu_calls = 0
        self.batched_get_to_gpu_calls = 0

    def is_frozen(self) -> bool:
        return False

    def allocate(self, shapes, dtypes, busy_loop, fmt):
        del shapes, dtypes, busy_loop, fmt
        self.allocate_calls += 1
        return FakeMemoryObj()

    def batched_put(self, keys, memory_objs, transfer_spec, location) -> None:
        del keys, transfer_spec, location
        self.batched_put_calls += 1
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()

    def batched_put_from_gpu(self, **kwargs):
        del kwargs
        self.batched_put_from_gpu_calls += 1
        return self.write_transfer

    def get_block_mapping(self, chunk_infos):
        return {"XDSBackend": chunk_infos}

    def batched_get_to_gpu(self, **kwargs):
        del kwargs
        self.batched_get_to_gpu_calls += 1
        return self.read_transfer

    def batched_get(self, keys, location):
        del keys, location
        self.batched_get_calls += 1
        return self.fallback

    def remove(self, key, locations) -> None:
        del key, locations


def make_engine(read_transfer=None, write_transfer=None, fallback=None):
    keys = [
        CacheEngineKey("model", 1, 0, 100 + index, torch.float16) for index in range(3)
    ]
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine._init_failed = False
    engine._health_monitor = None
    engine.config = SimpleNamespace(
        get_extra_config_value=lambda key, default=None: default
    )
    engine.metadata = SimpleNamespace(
        get_shapes=lambda num_tokens: torch.Size([num_tokens]),
        get_dtypes=lambda: torch.float16,
        use_mla=False,
        worker_id=0,
        is_first_rank=lambda: True,
    )
    engine.token_database = FakeTokenDatabase(keys)
    engine.storage_manager = FakeEngineStorageManager(
        read_transfer, write_transfer, fallback
    )
    engine.gpu_connector = FakeGPUConnector()
    engine.stats_monitor = FakeStatsMonitor()
    engine.save_only_first_rank = False
    engine.kvcache_check_log_enabled = False
    engine.kv_events_enabled = False
    engine.lookup_pins = {}
    engine.async_loading = False
    engine.remove_after_retrieve = False
    engine.store_location = None
    engine.retrieve_locations = None
    engine.fmt = None
    return engine


def make_transfer(completed):
    completed_bytes = sum(completed) * 4096
    return BackendTransferResult(
        completed=tuple(completed),
        requested_bytes=len(completed) * 4096,
        completed_bytes=completed_bytes,
        device="cuda:0",
        window_size=2,
        window_count=2,
        peak_staging_bytes=2 * 4096,
        storage_seconds=0.001,
        gpu_seconds=0.001,
        overlap_seconds=0.0,
        bank_wait_seconds=0.0,
        short_io_chunks=len(completed) - sum(completed),
        timeout_chunks=0,
    )


def test_cache_engine_store_uses_streaming_result_without_legacy_allocation():
    engine = make_engine(write_transfer=make_transfer([True, True, True]))

    engine.store(torch.arange(6), slot_mapping=object(), kvcaches=object())

    assert engine.storage_manager.batched_put_from_gpu_calls == 1
    assert engine.storage_manager.allocate_calls == 0
    assert engine.storage_manager.batched_put_calls == 0
    assert engine.gpu_connector.batched_from_gpu_calls == 0


def test_cache_engine_store_accepts_async_submission_without_legacy_allocation():
    submission = BackendWriteSubmissionResult(
        accepted=(True, False, True),
        accepted_bytes=8192,
        durable=False,
        staging_seconds=0.001,
        backpressure_seconds=0.0,
        queue_depth=1,
        outstanding_bytes=8192,
    )
    engine = make_engine(write_transfer=submission)

    engine.store(torch.arange(6), slot_mapping=object(), kvcaches=object())

    assert engine.storage_manager.allocate_calls == 0
    assert engine.stats_monitor.store_finished_tokens == 4


def test_cache_engine_store_preserves_legacy_path_when_fast_path_declines():
    engine = make_engine(write_transfer=None)

    engine.store(torch.arange(6), slot_mapping=object(), kvcaches=object())

    assert engine.storage_manager.batched_put_from_gpu_calls == 1
    assert engine.storage_manager.allocate_calls == 3
    assert engine.storage_manager.batched_put_calls == 1
    assert engine.gpu_connector.batched_from_gpu_calls == 1


def test_cache_engine_retrieve_streams_and_clears_failed_chunk_range():
    engine = make_engine(read_transfer=make_transfer([True, False, True]))

    ret_mask = engine.retrieve(torch.arange(6), slot_mapping=object())

    assert ret_mask.tolist() == [True, True, False, False, False, False]
    assert engine.storage_manager.batched_get_to_gpu_calls == 1
    assert engine.storage_manager.batched_get_calls == 0
    assert engine.gpu_connector.batched_to_gpu_calls == 0


def test_cache_engine_retrieve_preserves_legacy_path_when_fast_path_declines():
    fallback = [FakeMemoryObj() for _ in range(3)]
    engine = make_engine(read_transfer=None, fallback=fallback)

    ret_mask = engine.retrieve(torch.arange(6), slot_mapping=object())

    assert ret_mask.tolist() == [True] * 6
    assert engine.storage_manager.batched_get_calls == 1
    assert engine.gpu_connector.batched_to_gpu_calls == 1
    assert [obj.refs for obj in fallback] == [0, 0, 0]
