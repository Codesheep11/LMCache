# LMCache v0.4.3 CascadeKV Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the pre-migration CascadeKV write aggregation and add a bounded SSD-to-CPU-to-GPU read pipeline while keeping the external xDS service architecture.

**Architecture:** CascadeKV asks the xDS service to allocate one SPDK hugepage-backed CPU buffer, maps that buffer into LMCache, and registers the mapped address with CUDA without copying it. The buffer is partitioned into independent read and write slot banks. Writes gather GPU KV into write slots and hand multi-object leases to a bounded background scheduler; reads use a completion-driven xDS ring and start CPU-to-GPU transfer as each SSD read completes. A slot is recycled only after its xDS request and CUDA event finish.

**Tech Stack:** Python 3.11, PyTorch 2.10, LMCache 0.4.3, xDS service SDK, pytest, vLLM 0.18.0 GPU connector.

## Constraints

- Production edits stay in `/home/yangxc/workspace/LMCache-v043-xds`; `/home/yangxc/workspace/LMCache` is a read-only behavior reference.
- `CascadeKVBackend` remains `SSD -> service hugepage CPU staging -> GPU`; `XDSBackend` remains the separate SSD-to-GPU direct backend.
- CascadeKV continues to call `XDSClient.alloc_buffer()`; ordinary Torch pinned allocations cannot replace the service buffer because xDS I/O descriptors require its `buffer_ref`.
- CUDA host registration must apply to the original `torch.frombuffer()` address. Do not use `.pin_memory()` when it could create a second allocation.
- The first implementation overlaps complete LMCache chunks and does not restore the old experimental sub-blob chunking modes.
- `use_layerwise: false` is the performance path. Layerwise and single-object APIs remain functionally compatible.
- Failures must release blob handles, memory-object references, scheduler credits, metadata pins, CUDA registration, and the service buffer exactly once.

---

### Task 1: CascadeKV Configuration and Dispatch Contract

**Files:**
- Modify: `lmcache/v1/config.py`
- Modify: `lmcache/v1/storage_backend/storage_manager.py`
- Modify: `tests/v1/test_config.py`
- Modify: `tests/v1/test_cache_engine_streaming_transfer.py`

- [ ] Add failing tests for `cascadekv_overlap`, `cascadekv_async_write`, read/write slot counts, queue depth, write window size, positive-value validation, and the legacy `cascadekv_slot_count` fallback.
- [ ] Add failing StorageManager tests proving the optional whole-batch store/retrieve hooks dispatch to a sole or explicitly selected `CascadeKVBackend`, while mixed unselected backends still fall back.
- [ ] Run the focused tests and verify RED:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
    tests/v1/test_config.py -k cascadekv \
    tests/v1/test_cache_engine_streaming_transfer.py -k storage_manager
  ```

- [ ] Implement the configuration definitions and validation. Keep `cascadekv_slot_count` as compatibility input only when the new counts are not explicitly supplied.
- [ ] Generalize StorageManager's hook allow-list from only `XDSBackend` to `XDSBackend` and `CascadeKVBackend`.
- [ ] Re-run the focused tests and verify GREEN.

### Task 2: Service Hugepage Pinning and Independent Slot Banks

**Files:**
- Modify: `lmcache/v1/storage_backend/cascadekv_backend.py`
- Modify: `tests/v1/storage_backend/test_cascadekv_backend.py`

- [ ] Add failing public-behavior tests proving one service allocation covers read plus write banks, objects from each bank remain inside the original mapped address range, and no copied Tensor replaces the service mapping.
- [ ] Add CUDA lifecycle tests with a fake cudart proving registration uses `staging_tensor.data_ptr()` and `staging_size`, registration failure frees the service buffer, close unregisters before `free_buffer`, and repeated close does not repeat cleanup.
- [ ] Run CascadeKV tests and verify RED.
- [ ] Allocate `(read_slot_count + write_slot_count) * blob_size` through `XDSClient.alloc_buffer()`, create the zero-copy Tensor view, call `cudaHostRegister`, then partition it into separately aligned `TensorMemoryAllocator` read/write banks.
- [ ] Route public allocation/store staging to the write bank and all SSD retrieval allocation to the read bank. Report a bounded chunk budget without requiring slots for every logical prompt chunk.
- [ ] Implement cleanup ordering: drain workers/events, close allocators, `cudaHostUnregister`, release blobs, `free_buffer`, close client.
- [ ] Re-run CascadeKV tests and verify GREEN.

### Task 3: Aggregated Asynchronous CascadeKV Writes

**Files:**
- Modify: `lmcache/v1/storage_backend/cascadekv_backend.py`
- Reuse: `lmcache/v1/storage_backend/xds_write_scheduler.py`
- Reuse: `lmcache/v1/storage_backend/xds_read_ring.py`
- Modify: `tests/v1/storage_backend/test_cascadekv_backend.py`

- [ ] Add failing tests showing a write window submits multiple `submit_write` calls before the first completion wait, peak in-flight I/O respects `cascadekv_io_queue_depth`, and metadata appears only after exact completions.
- [ ] Add failing tests for async admission, pending-key reads, write-slot backpressure, short I/O rollback, callback/future completion, and close-time drain.
- [ ] Run the write-focused tests and verify RED.
- [ ] Build multi-object `StagingLease` windows capped by `cascadekv_write_window_size` and the available write slots. Use `XDSAsyncWriteScheduler` for bounded ownership and an `XDSReadRingPump` as the write completion pump.
- [ ] For whole-batch vLLM store, gather GPU ranges into write slots, record one store event per lease, and let the background worker wait for that event before xDS submission. Return `BackendWriteSubmissionResult` after ownership is transferred safely.
- [ ] Keep pending objects readable until durable commit. Commit publishes `XDSBlobMetadata`; rollback releases all acquired handles and references exactly once.
- [ ] Preserve regular `batched_submit_put_task` by grouping its already-staged objects into the same lease machinery.
- [ ] Re-run write-focused tests and verify GREEN.

### Task 4: Completion-Driven CascadeKV Read Overlap

**Files:**
- Modify: `lmcache/v1/storage_backend/cascadekv_backend.py`
- Reuse: `lmcache/v1/storage_backend/xds_read_ring.py`
- Modify: `tests/v1/storage_backend/test_cascadekv_backend.py`

- [ ] Add failing tests with more logical chunks than read slots. Assert multiple reads are submitted before the first wait, at most `cascadekv_read_slot_count` memory objects are live, and every completed SSD read immediately calls `gpu_connector.to_gpu` while the pump thread continues.
- [ ] Add fake CUDA-event tests proving a read slot cannot be reused before its load event synchronizes. Cover pending-write hits, out-of-order completion, short read, connector failure, unknown completion, and metadata unpin cleanup.
- [ ] Run read-focused tests and verify RED.
- [ ] Implement `batched_get_to_gpu_blocking()` with a bounded live-entry map and `XDSReadRingPump`: fill read slots, consume any completion, validate exact bytes, launch CPU-to-GPU, wait for its event before freeing that slot, refill, and preserve original logical indexes.
- [ ] Return `BackendTransferResult` containing completed mask, byte counts, read/write slot counts, queue depth, peak staging, and peak in-flight I/O.
- [ ] Upgrade `batched_get_blocking()` to concurrent xDS reads through the read bank while preserving the memory-object return contract.
- [ ] Re-run read-focused tests and verify GREEN.

### Task 5: Benchmark Configuration and Verification

**Files:**
- Modify: `benchmarks/xds_e2e/cascadekv_40k.yaml`
- Verify all files modified above.

- [ ] Set the CascadeKV performance configuration to `use_layerwise: false`, `chunk_size: 256`, overlap/async write enabled, 8 read slots, 64 write slots, queue depth 4, and write window 64.
- [ ] Run focused tests:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
    tests/v1/test_config.py \
    tests/v1/test_cache_engine_streaming_transfer.py \
    tests/v1/storage_backend/test_xds_read_ring.py \
    tests/v1/storage_backend/test_xds_write_scheduler.py \
    tests/v1/storage_backend/test_cascadekv_backend.py \
    tests/v1/storage_backend/test_xds_backend_factory.py
  ```

- [ ] Run adjacent storage/cache-engine regressions and targeted Ruff checks. Record environmental collection failures separately from implementation failures.
- [ ] Run an editable-source import smoke test in Conda environment `xds043` and verify LMCache resolves to this repository with Torch 2.10.
- [ ] When GPUs and the xDS service are idle, run the existing 40K vLLM prompt pair. Require the second request to hit the external CascadeKV cache with logical chunks greater than read slots and no staging allocation failure.
- [ ] Compare elapsed time, TTFT/cache-hit logs, peak staging, and peak in-flight I/O against the saved pre-optimization CascadeKV result and Local Disk result.

