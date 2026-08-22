# LMCache v0.4.3 xDS Bounded Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make direct `XDSBackend` store and retrieve arbitrarily many KV chunks through a fixed registered GPU staging pool whose live usage is bounded by configured read/write slot counts.

**Architecture:** Add an optional direct-transfer contract between `CacheEngine`, `StorageManager`, and `XDSBackend`. The backend allocates only a bounded set of registered GPU staging objects, drives xDS requests through the existing read pump/write scheduler, scatters or gathers through the existing GPU connector, then releases each slot before admitting more work. Backends that do not implement the optional contract continue through LMCache v0.4.3's existing allocate-all path.

**Tech Stack:** Python 3.11, PyTorch 2.10, LMCache v0.4.3, pytest, xDS Python SDK, vLLM/SGLang GPU connectors.

## Global Constraints

- Baseline and target branch remain LMCache `v0.4.3` / `v0.4.3-xds`.
- `/home/yangxc/workspace/LMCache` is read-only migration reference; all production edits belong in `/home/yangxc/workspace/LMCache-v043-xds`.
- `XDSBackend` remains the SSD-to-registered-GPU direct backend; `CascadeKVBackend` remains SSD-to-CPU-to-GPU and is not routed through this fast path.
- Physical staging residency is controlled by `xds_read_slot_count` and `xds_write_slot_count`; logical window settings are accounting/telemetry fields.
- Direct streaming store is accepted only when the selected location resolves to the single `XDSBackend`; otherwise the existing v0.4.3 path runs unchanged.
- `use_layerwise=true` remains unsupported with xDS windowed I/O.
- Short I/O, timeout, unknown completion, CUDA-copy failure, and close must release all staging references and unpublished blob handles deterministically.
- All new Python APIs have type hints, public docstrings, SPDX headers, and tests against public behavior.

---

### Task 1: Transfer Events and Result Contract

**Files:**
- Modify: `lmcache/v1/gpu_connector/gpu_connectors.py`
- Modify: `lmcache/v1/storage_backend/transfer_result.py`
- Create: `tests/v1/test_gpu_connector_events.py`
- Modify: `tests/v1/storage_backend/test_transfer_result.py`

**Interfaces:**
- Produces: `GPUConnectorInterface.record_transfer_event(direction: str) -> torch.cuda.Event`.
- Produces: `BackendTransferResult` bounded-I/O telemetry fields and `BackendWriteSubmissionResult` for accepted but not-yet-durable asynchronous writes.

- [ ] **Step 1: Write failing connector-event tests**

  Add tests proving `direction="load"` records on the current stream, `direction="store"` records on `store_stream`, an unknown direction raises `ValueError`, and `SGLangGPUConnector` creates `store_stream`.

- [ ] **Step 2: Run the event tests and verify RED**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/v1/test_gpu_connector_events.py`

  Expected: failures because `record_transfer_event` and the SGLang store stream do not exist.

- [ ] **Step 3: Implement the minimal event API**

  Add the following public behavior to `GPUConnectorInterface` and initialize the missing SGLang stream:

  ```python
  def record_transfer_event(self, direction: str) -> torch.cuda.Event:
      """Record completion of the connector's current load or store work."""
      if direction == "load":
          stream = getattr(self, "load_stream", torch.cuda.current_stream())
      elif direction == "store":
          stream = getattr(self, "store_stream", torch.cuda.current_stream())
      else:
          raise ValueError(f"unknown transfer direction: {direction}")
      event = torch.cuda.Event()
      event.record(stream)
      return event
  ```

- [ ] **Step 4: Write failing transfer-result tests**

  Extend result tests to require non-negative slot counts, queue depth, in-flight I/O, wait timing, and asynchronous admission fields.

- [ ] **Step 5: Run result tests and verify RED**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/v1/storage_backend/test_transfer_result.py`

  Expected: constructor failures for fields that are not defined yet.

- [ ] **Step 6: Implement transfer result fields**

  Add defaulted telemetry to `BackendTransferResult`:

  ```python
  logical_chunk_count: int = 0
  read_slot_count: int = 0
  write_slot_count: int = 0
  io_queue_depth: int = 0
  peak_inflight_io: int = 0
  slot_wait_seconds: float = 0.0
  ```

  Add `BackendWriteSubmissionResult` with `accepted`, `accepted_bytes`, `durable`, `staging_seconds`, `backpressure_seconds`, `queue_depth`, and `outstanding_bytes`, validating all counts and durations.

- [ ] **Step 7: Run both test files and verify GREEN**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/v1/test_gpu_connector_events.py tests/v1/storage_backend/test_transfer_result.py`

  Expected: all tests pass.

### Task 2: Optional Streaming Hooks in StorageManager and CacheEngine

**Files:**
- Modify: `lmcache/v1/storage_backend/storage_manager.py`
- Modify: `lmcache/v1/cache_engine.py`
- Create: `tests/v1/test_cache_engine_streaming_transfer.py`

**Interfaces:**
- Consumes: `BackendTransferResult` and `BackendWriteSubmissionResult` from Task 1.
- Produces: `StorageManager.batched_put_from_gpu(...)` and `StorageManager.batched_get_to_gpu(...)` optional backend dispatch.
- Produces: CacheEngine store/retrieve paths that do not allocate all staging objects after a backend accepts direct streaming.

- [ ] **Step 1: Write failing StorageManager dispatch tests**

  Cover these behaviors through public methods:

  ```text
  xDS-only selected location + hook present -> dispatch to hook
  multiple/unselected backends -> return None without side effects
  backend without hook -> return None
  retrieve location XDSBackend -> dispatch direct-to-GPU hook
  ```

- [ ] **Step 2: Run dispatch tests and verify RED**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/v1/test_cache_engine_streaming_transfer.py -k storage_manager`

  Expected: `AttributeError` because the optional methods do not exist.

- [ ] **Step 3: Implement minimal StorageManager hooks**

  Add typed methods with this dispatch contract:

  ```python
  def batched_put_from_gpu(
      self, keys, starts, ends, gpu_connector, location=None, **kwargs
  ) -> Optional[Union[BackendTransferResult, BackendWriteSubmissionResult]]:
      # Resolve active backends after location/bypass filtering.
      # Dispatch only when exactly one backend remains and it exposes the hook.

  def batched_get_to_gpu(
      self, keys, starts, ends, gpu_connector, location=None, **kwargs
  ) -> Optional[BackendTransferResult]:
      # Resolve the requested backend and call its optional direct hook.
  ```

- [ ] **Step 4: Write failing CacheEngine store/retrieve tests**

  Build fake token database, connector, stats, and storage manager fixtures proving:

  - accepted store results skip `StorageManager.allocate` and legacy `batched_put`;
  - partial store results account only completed/accepted token ranges;
  - direct retrieve results skip legacy `batched_get` and clear failed ranges in `ret_mask`;
  - a `None` hook result preserves the existing v0.4.3 path.

- [ ] **Step 5: Run CacheEngine tests and verify RED**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/v1/test_cache_engine_streaming_transfer.py -k cache_engine`

  Expected: legacy allocation/get assertions fail because CacheEngine does not invoke the optional hooks.

- [ ] **Step 6: Implement CacheEngine direct transfer branches**

  In `store()`, collect `(key, start, end)` before allocating objects, invoke `batched_put_from_gpu`, consume either result type, update stats/events for accepted ranges, and return. Only a `None` result may continue to the existing allocate-all path.

  In `retrieve()`, preserve v0.4.3 prefix/block mapping, group xDS chunks by location, invoke `batched_get_to_gpu`, apply its per-chunk completion mask, and keep legacy memory-object retrieval for other locations.

- [ ] **Step 7: Run streaming tests and verify GREEN**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/v1/test_cache_engine_streaming_transfer.py`

  Expected: all streaming and fallback tests pass.

### Task 3: Bounded Direct xDS Read Pipeline

**Files:**
- Modify: `lmcache/v1/storage_backend/xds_backend.py`
- Modify: `lmcache/v1/storage_backend/xds_read_ring.py` only if a failing public test exposes a drain/cleanup defect.
- Modify: `tests/v1/storage_backend/test_xds_backend.py`
- Modify: `tests/v1/storage_backend/test_xds_read_ring.py`

**Interfaces:**
- Consumes: `GPUConnectorInterface.to_gpu`, `record_transfer_event("load")`, and Task 2's direct retrieval hook.
- Produces: `XDSBackend.batched_get_to_gpu_blocking(keys, starts, ends, gpu_connector, **kwargs) -> Optional[BackendTransferResult]`.

- [ ] **Step 1: Write failing bounded-read tests**

  Use five stored fake blobs with `xds_read_slot_count=2` and assert:

  ```python
  assert result.completed == (True, True, True, True, True)
  assert allocator_peak_active <= 2
  assert result.peak_staging_bytes <= 2 * blob_size
  assert result.peak_inflight_io <= config.xds_io_queue_depth
  assert all(fake_event.synchronized for fake_event in connector.events)
  ```

  Add separate failure tests for short I/O, unknown completion, scatter failure, and allocation failure; every case must return/release every live slot and unpin metadata.

- [ ] **Step 2: Run bounded-read tests and verify RED**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/v1/storage_backend/test_xds_backend.py -k 'bounded_read or windowed_read'`

  Expected: active staging reaches all requested chunks or the direct-to-GPU method is missing.

- [ ] **Step 3: Implement the read ring refill loop**

  Implement a live-entry map capped by `xds_read_slot_count`:

  ```text
  fill free slots -> enqueue descriptors -> receive one completion
  -> validate exact bytes -> scatter to final KV -> synchronize event
  -> ref_count_down/unpin -> fill the freed slot -> repeat
  ```

  Protect xDS client completion ownership with the backend I/O lock, retain original logical indexes, and return one `BackendTransferResult` for the whole request.

- [ ] **Step 4: Run read tests and verify GREEN**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/v1/storage_backend/test_xds_read_ring.py tests/v1/storage_backend/test_xds_backend.py -k 'read or ring'`

  Expected: all selected tests pass and peak staging remains within configured slots.

### Task 4: Bounded Direct xDS Write Pipeline

**Files:**
- Modify: `lmcache/v1/storage_backend/xds_backend.py`
- Modify: `lmcache/v1/storage_backend/xds_write_scheduler.py` only when failing lease tests require reader priority, cancellation, or multi-object cleanup.
- Modify: `tests/v1/storage_backend/test_xds_backend.py`
- Modify: `tests/v1/storage_backend/test_xds_write_scheduler.py`

**Interfaces:**
- Consumes: `GPUConnectorInterface.batched_from_gpu`, `record_transfer_event("store")`, and Task 2's direct store hook.
- Produces: `XDSBackend.batched_put_from_gpu_blocking(keys, starts, ends, gpu_connector, **kwargs) -> Optional[Union[BackendTransferResult, BackendWriteSubmissionResult]]`.

- [ ] **Step 1: Write failing synchronous bounded-write tests**

  Store five chunks with `xds_write_slot_count=2`, then assert all keys are published, gather batches are `[2, 2, 1]`, peak staging is at most two blobs, and every object is released. Add short-I/O and gather-failure rollback tests.

- [ ] **Step 2: Run synchronous tests and verify RED**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/v1/storage_backend/test_xds_backend.py -k 'bounded_write and not async'`

  Expected: method missing or legacy allocation exceeds the two-slot budget.

- [ ] **Step 3: Implement synchronous windowed gather/write/release**

  Collect shapes from `metadata.get_shapes(end - start)`, allocate at most `xds_write_slot_count` objects, gather those ranges, synchronize their store event, submit exact descriptors with queue depth bounded by `xds_io_queue_depth`, publish only complete writes, release the window, and continue.

- [ ] **Step 4: Write failing asynchronous lease tests**

  Require large requests to be admitted in leases of at most `xds_write_slot_count`, never exceed `write_slot_count * blob_size` outstanding staging, apply backpressure before gathering another lease, make pending objects readable, and drain all leases during `close()`.

- [ ] **Step 5: Run asynchronous tests and verify RED**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/v1/storage_backend/test_xds_backend.py tests/v1/storage_backend/test_xds_write_scheduler.py -k async`

  Expected: current per-object scheduler path cannot accept directly from the GPU without prior allocate-all staging.

- [ ] **Step 6: Implement direct asynchronous admission**

  Acquire credits before gathering each bounded lease, retain its memory objects until xDS completion, publish successful metadata through the scheduler commit callback, roll back handles/entries/references once on error, and return `BackendWriteSubmissionResult(durable=False, ...)` after accepted leases are safely owned by the scheduler.

- [ ] **Step 7: Run write tests and verify GREEN**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/v1/storage_backend/test_xds_backend.py tests/v1/storage_backend/test_xds_write_scheduler.py -k 'write or async'`

  Expected: all selected tests pass with bounded staging.

### Task 5: Regression, Formatting, and Source Integration Verification

**Files:**
- Modify: `docs/source/kv_cache/storage_backends/index.rst` or the existing xDS runbook only if runtime configuration behavior changed.
- Verify all files changed in Tasks 1-4.

**Interfaces:**
- Consumes: all prior task deliverables.
- Produces: a source-installable LMCache v0.4.3 direct xDS implementation with regression evidence.

- [ ] **Step 1: Run focused xDS and CacheEngine tests**

  Run:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
    tests/v1/storage_backend/test_transfer_result.py \
    tests/v1/storage_backend/test_xds_read_ring.py \
    tests/v1/storage_backend/test_xds_write_scheduler.py \
    tests/v1/storage_backend/test_xds_backend.py \
    tests/v1/test_gpu_connector_events.py \
    tests/v1/test_cache_engine_streaming_transfer.py
  ```

  Expected: zero failures.

- [ ] **Step 2: Run adjacent v0.4.3 regressions**

  Run:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
    tests/v1/test_config.py \
    tests/v1/storage_backend/test_storage_manager.py \
    tests/v1/test_cache_engine.py
  ```

  Expected: zero failures; if environment-owned vLLM/Transformers imports fail before collection, report that separately from implementation failures.

- [ ] **Step 3: Run formatting and static checks on changed Python files**

  Run `ruff check` and `ruff format --check` on every modified Python source and test file. Expected: exit code 0.

- [ ] **Step 4: Verify source import and version in `xds043`**

  Run an import smoke test that prints `lmcache.__file__`, LMCache version, `torch.__version__`, `xds.__file__`, and confirms `XDSBackend` exposes both direct streaming methods. Expected: imports resolve to `/home/yangxc/workspace/LMCache-v043-xds` with LMCache 0.4.3 and torch 2.10.

- [ ] **Step 5: Inspect final repository state**

  Run `git status --short`, `git diff --check`, and `git diff --stat`. Confirm no unrelated files changed and no whitespace errors exist.

- [ ] **Step 6: Hardware prompt verification when the xDS service is available**

  Start the existing vLLM 0.18.0 + LMCache source configuration with `local_cpu: false`, `xds_windowed_io: true`, and a registered pool sized only for configured slots. Send the same prompt twice and require the second request to retrieve every expected external-cache chunk. Record `[xDS Window]` logs showing total logical chunks greater than the physical slot count without allocation failure.

