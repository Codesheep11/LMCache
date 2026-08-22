# LMCache v0.4.3 CascadeKV Legacy-Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bounded-slot LMCache v0.4.3 CascadeKV implementation with the legacy feature and concurrency behavior while using the external xDS service request-ID API without Python Futures.

**Architecture:** Expose the existing native xDS `poll_burst` operation to Python, then give CascadeKV one completion pump that routes request IDs to condition-backed request groups. A service-owned hugepage buffer supplies one large LMCache CPU allocator, while the legacy CascadeKV backend remains the source for write aggregation, overlap reads, prefetch, LRU, worker/lookup integration, statistics, and shutdown behavior.

**Tech Stack:** Python 3.12, CPython C API, C11 xDS client, PyTorch 2.10/CUDA, LMCache 0.4.3, pytest, Ruff.

## Global Constraints

- LMCache baseline and branch remain `v0.4.3` and `v0.4.3-xds`.
- xDS source remains `/home/yangxc/workspace/xDS`; do not commit unrelated pre-existing changes in its dirty worktree.
- Legacy behavior source is `/home/yangxc/workspace/LMCache/lmcache/v1/storage_backend/cascadekv_backend.py`.
- `xds_backend` and `cascadekv_backend` remain independent and mutually exclusive.
- CascadeKV data flow remains `SSD -> xDS hugepage CPU memory -> GPU`.
- xDS service I/O is represented by request IDs and completion records, never `concurrent.futures.Future`.
- Do not restore fixed CascadeKV staging slots, queue-depth admission, full-I/O `_io_lock`, poison, quarantine, or active-operation close gates.
- Do not modify the xDS service protocol, completion-ring ABI, SPDK service workers, or normal I/O hot path.
- Preserve the legacy write buffer/flush thread, read overlap, prefetch, LRU, pinning, worker messages, lookup, stats, trace, timeout, and close semantics.
- Prefer the `xds043` Conda environment with PyTorch `2.10.0` for LMCache verification.

---

### Task 1: Expose non-blocking xDS completion bursts to Python

**Files:**
- Modify: `/home/yangxc/workspace/xDS/tests/python/test_client_wrapper.py`
- Modify: `/home/yangxc/workspace/xDS/sdk/csrc/py_wrapper.c`
- Modify: `/home/yangxc/workspace/xDS/sdk/src/xds/__init__.py`

**Interfaces:**
- Consumes: existing C function `int xds_client_poll_burst(xds_client_t *, struct xds_client_result *, uint32_t)` and `XDS_CLIENT_MAX_POLL_BURST == 64`.
- Produces: `_xds.client_poll_burst(max_count: int) -> list[tuple[int, int, int, int, int]]` and `XDSClient.poll_burst(max_count: int = 64) -> list[XDSCompletion]`.

- [ ] **Step 1: Add failing facade tests**

Add a fake extension method and two public API tests:

```python
def client_poll_burst(self, max_count: int) -> list[tuple[int, int, int, int, int]]:
    assert max_count == 8
    return [
        (101, 0, 0, 4096, 0),
        (102, -5, 5, 2048, 0),
    ]


def test_poll_burst_returns_completion_records(facade) -> None:
    xds, _ = facade
    client = xds.connect("cache")

    assert client.poll_burst(8) == [
        xds.XDSCompletion(101, 0, 0, 4096, 0),
        xds.XDSCompletion(102, -5, 5, 2048, 0),
    ]


@pytest.mark.parametrize("max_count", [0, 65])
def test_poll_burst_rejects_invalid_count(facade, max_count: int) -> None:
    xds, _ = facade
    client = xds.connect("cache")

    with pytest.raises(ValueError, match="between 1 and 64"):
        client.poll_burst(max_count)
```

- [ ] **Step 2: Run the facade test and confirm the missing API failure**

Run:

```bash
conda run -n xds043 pytest -q tests/python/test_client_wrapper.py
```

Expected: the new tests fail because `XDSClient` has no `poll_burst` method.

- [ ] **Step 3: Export the existing native burst poll**

Add a CPython wrapper that validates `1..64`, calls the existing C client function, and builds a Python list of five-field tuples:

```c
static PyObject *py_c_api_client_poll_burst(PyObject *self, PyObject *args)
{
    xds_client_t *client;
    unsigned int max_count;
    struct xds_client_result results[XDS_CLIENT_MAX_POLL_BURST] = {0};
    PyObject *items;
    int count;

    if (!PyArg_ParseTuple(args, "I:client_poll_burst", &max_count))
        return NULL;
    if (max_count == 0 || max_count > XDS_CLIENT_MAX_POLL_BURST) {
        PyErr_SetString(PyExc_ValueError, "max_count must be between 1 and 64");
        return NULL;
    }
    if (require_xds_client(&client) != 0)
        return NULL;
    count = xds_client_poll_burst(client, results, max_count);
    if (count < 0)
        return retcode_to_py(count);
    items = PyList_New(count);
    if (!items)
        return NULL;
    for (int i = 0; i < count; i++) {
        PyObject *item = Py_BuildValue(
            "(KiIKK)",
            (unsigned long long)results[i].request_id,
            results[i].status,
            results[i].error_code,
            (unsigned long long)results[i].completed_bytes,
            (unsigned long long)results[i].result);
        if (!item) {
            Py_DECREF(items);
            return NULL;
        }
        PyList_SET_ITEM(items, i, item);
    }
    return items;
}
```

Register it in `SpdkMethods` as `client_poll_burst`. Add the public Python method:

```python
def poll_burst(self, max_count: int = 64) -> list[XDSCompletion]:
    """Return up to ``max_count`` currently available completions."""
    self._require_connected()
    if max_count < 1 or max_count > 64:
        raise ValueError("max_count must be between 1 and 64")
    raw_results = self._require_client_api("client_poll_burst")(max_count)
    return [XDSCompletion(*raw) for raw in raw_results]
```

- [ ] **Step 4: Build the editable SDK and run focused tests**

Run:

```bash
conda run -n xds043 python -m pip install -e /home/yangxc/workspace/xDS/sdk
conda run -n xds043 pytest -q tests/python/test_client_wrapper.py
```

Expected: all facade tests pass and the editable import resolves to `/home/yangxc/workspace/xDS/sdk/src/xds`.

- [ ] **Step 5: Commit only the three xDS files**

Because the xDS worktree contains unrelated staged changes, use an explicit path-limited commit:

```bash
git commit --only tests/python/test_client_wrapper.py sdk/csrc/py_wrapper.c sdk/src/xds/__init__.py -m "feat: expose service completion burst polling"
```

Expected: the commit contains only the three named files; all other xDS status entries remain untouched.

---

### Task 2: Add request-ID completion groups for CascadeKV

**Files:**
- Create: `lmcache/v1/storage_backend/cascadekv_completion.py`
- Create: `tests/v1/storage_backend/test_cascadekv_completion.py`

**Interfaces:**
- Consumes: `XDSClient.poll_burst(max_count: int = 64) -> list[XDSCompletion]`.
- Produces: `CascadeKVRequestGroup.add(request_id, context)`, `wait_next(timeout)`, `wait_all(timeout)`, and `CascadeKVCompletionPump.submit(group, context, submitter)`, `close()`.

- [ ] **Step 1: Write request-group behavior tests**

Use an integer context and a fake client whose completion list is explicitly controlled. Tests must prove that two groups receive only their own out-of-order completions, `wait_next` follows device completion order, `wait_all` returns input registration order, and no `Future` appears in the module:

```python
def test_completion_pump_routes_out_of_order_ids_to_groups() -> None:
    client = FakePollClient()
    pump = CascadeKVCompletionPump(client)
    first = pump.create_group()
    second = pump.create_group()

    pump.submit(first, "read-a", lambda: 11)
    pump.submit(second, "write-b", lambda: 22)
    client.publish(XDSCompletion(22, 0, 0, 4096, 0))
    client.publish(XDSCompletion(11, 0, 0, 4096, 0))

    assert second.wait_next(1.0).context == "write-b"
    assert first.wait_next(1.0).context == "read-a"
    pump.close()


def test_module_does_not_import_concurrent_futures() -> None:
    source = inspect.getsource(cascadekv_completion)
    assert "concurrent.futures" not in source
    assert "Future" not in source
```

- [ ] **Step 2: Verify the tests fail before the module exists**

Run:

```bash
conda run -n xds043 pytest -q tests/v1/storage_backend/test_cascadekv_completion.py
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement condition-backed request groups and one poller**

Define immutable completion records and generic contexts:

```python
@dataclass(frozen=True)
class CascadeKVCompletedRequest(Generic[ContextT]):
    request_id: int
    context: ContextT
    completion: Any


class CascadeKVRequestGroup(Generic[ContextT]):
    def add(self, request_id: int, context: ContextT) -> None:
        with self._condition:
            self._contexts[request_id] = context
            self._registration_order.append(request_id)

    def complete(self, request_id: int, completion: Any) -> None:
        with self._condition:
            context = self._contexts[request_id]
            result = CascadeKVCompletedRequest(request_id, context, completion)
            self._results[request_id] = result
            self._completion_order.append(request_id)
            self._condition.notify_all()

    def wait_next(self, timeout: float) -> CascadeKVCompletedRequest[ContextT]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._completion_order:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for xDS completion")
                self._condition.wait(remaining)
            request_id = self._completion_order.popleft()
            return self._results[request_id]

    def wait_all(self, timeout: float) -> list[CascadeKVCompletedRequest[ContextT]]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self._results) != len(self._registration_order):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for xDS completion group")
                self._condition.wait(remaining)
            return [self._results[item] for item in self._registration_order]
```

`wait_next` waits on a `threading.Condition`; `wait_all` repeatedly consumes completed records and returns them in request-registration order. Timeout raises `TimeoutError` without cancelling the xDS request, matching the old backend's timed `Future.result()` behavior.

Implement one daemon poller with a public atomic submit operation:

```python
def submit(
    self,
    group: CascadeKVRequestGroup[ContextT],
    context: ContextT,
    submitter: Callable[[], int],
) -> int:
    with self._client_lock:
        request_id = submitter()
        group.add(request_id, context)
        self._owners[request_id] = group
    self._work_available.set()
    return request_id
```

The poll loop calls `client.poll_burst(64)` under `_client_lock`, releases the lock before dispatch, and waits briefly on `_work_available` when no request is outstanding. It must not hold `_client_lock` while a request group waits.

- [ ] **Step 4: Run and lint the completion module**

Run:

```bash
conda run -n xds043 pytest -q tests/v1/storage_backend/test_cascadekv_completion.py
conda run -n xds043 ruff check lmcache/v1/storage_backend/cascadekv_completion.py tests/v1/storage_backend/test_cascadekv_completion.py
```

Expected: tests and Ruff pass.

- [ ] **Step 5: Commit the completion router**

```bash
git add lmcache/v1/storage_backend/cascadekv_completion.py tests/v1/storage_backend/test_cascadekv_completion.py
git commit -m "feat: route CascadeKV request completions"
```

---

### Task 3: Restore one large xDS service DMA allocator

**Files:**
- Modify: `lmcache/v1/memory_management.py`
- Modify: `tests/v1/test_memory_management.py`

**Interfaces:**
- Consumes: `XDSClient.alloc_buffer(size) -> tuple[memoryview, int]`, `free_buffer(buffer_ref)`, and service `io_unit_size`.
- Produces: `XDSServiceMemoryAllocator(client, size, align_bytes)`, public `buffer_ref`, `base_ptr`, `size`, `get_buffer_offset(memory_obj)`, and `close()`.

- [ ] **Step 1: Add failing allocator ownership tests**

Create tests with a bytearray-backed fake service allocation and fake CUDA runtime. Assert one allocation covers `max_local_cpu_size`, tensor slices share the original service address, offsets are I/O-unit aligned, and close order is unregister then free:

```python
allocator = XDSServiceMemoryAllocator(client, size=16384, align_bytes=4096)
obj = allocator.allocate(torch.Size([4096]), torch.uint8)

assert obj is not None
assert allocator.get_buffer_offset(obj) == 0
assert obj.raw_data.data_ptr() == allocator.base_ptr
obj.ref_count_down()
allocator.close()
assert lifecycle[-2:] == ["cudaHostUnregister", "free_buffer"]
```

- [ ] **Step 2: Run the focused allocator test and confirm the missing class**

Run:

```bash
conda run -n xds043 pytest -q tests/v1/test_memory_management.py -k XDSService
```

Expected: failure because `XDSServiceMemoryAllocator` is not defined.

- [ ] **Step 3: Implement the allocator as a TensorMemoryAllocator specialization**

The constructor performs these exact operations:

```python
self._view, self.buffer_ref = client.alloc_buffer(size)
tensor = torch.frombuffer(self._view, dtype=torch.uint8).flatten()
self.base_ptr = tensor.data_ptr()
status = torch.cuda.cudart().cudaHostRegister(self.base_ptr, size, 0)
if status != 0:
    client.free_buffer(self.buffer_ref)
    raise RuntimeError(f"cudaHostRegister failed with status {status}")
super().__init__(tensor, align_bytes=align_bytes)
```

`get_buffer_offset` subtracts `base_ptr` from `memory_obj.raw_data.data_ptr()`, validates the object lies entirely inside the registered range, and returns the aligned offset. `close()` synchronizes CUDA, unregisters the original address, then frees the service buffer exactly once.

- [ ] **Step 4: Run allocator tests and Ruff**

Run:

```bash
conda run -n xds043 pytest -q tests/v1/test_memory_management.py -k XDSService
conda run -n xds043 ruff check lmcache/v1/memory_management.py tests/v1/test_memory_management.py
```

Expected: focused tests and Ruff pass.

- [ ] **Step 5: Commit the allocator**

```bash
git add lmcache/v1/memory_management.py tests/v1/test_memory_management.py
git commit -m "feat: add xDS service DMA allocator"
```

---

### Task 4: Restore legacy CascadeKV configuration and factory wiring

**Files:**
- Modify: `lmcache/v1/config.py`
- Modify: `lmcache/v1/storage_backend/__init__.py`
- Modify: `tests/v1/test_config.py`
- Modify: `tests/v1/storage_backend/test_xds_backend_factory.py`

**Interfaces:**
- Consumes: `XDSServiceMemoryAllocator` and existing `connect_xds_client(config, metadata)`.
- Produces: legacy config fields and a `CascadeKVBackend` constructed with the service-DMA `LocalCPUBackend`.

- [ ] **Step 1: Write failing configuration tests**

Add parsing/default assertions:

```python
config = LMCacheEngineConfig.from_dict(
    {
        "cascadekv_backend": True,
        "xds_service_name": "xds0",
        "xds_max_size": 500,
        "xds_enable_chunked_overlap": True,
        "xds_chunked_size_kb": 4096,
        "xds_write_window_size": 64,
        "xds_idle_flush_time": 0.5,
        "xds_max_flush_delay": 1.0,
    }
)
assert config.xds_max_size == 500
assert config.xds_enable_chunked_overlap is True
assert config.xds_max_flush_delay == 1.0
```

Also assert deprecated `cascadekv_*slot*`, queue-depth, async-write, and overlap fields parse but do not participate in CascadeKV validation.

- [ ] **Step 2: Run configuration tests and verify missing fields fail**

Run:

```bash
conda run -n xds043 pytest -q tests/v1/test_config.py -k cascadekv
```

Expected: legacy field assertions fail before schema changes.

- [ ] **Step 3: Add exact legacy schema entries and validation**

Add to `_CONFIG_DEFINITIONS`:

```python
"xds_max_size": {"type": float, "default": 10.0, "env_converter": float},
"xds_enable_chunked_overlap": {
    "type": bool,
    "default": False,
    "env_converter": _to_bool,
},
"xds_chunked_size_kb": {"type": int, "default": 4096, "env_converter": int},
"xds_idle_flush_time": {"type": float, "default": 0.5, "env_converter": float},
"xds_max_flush_delay": {"type": float, "default": 1.0, "env_converter": float},
```

Reuse existing `xds_write_window_size`. Validate positive max size, chunk size, write window and flush delays when CascadeKV is enabled. Remove slot-count and queue-depth coupling from CascadeKV validation while leaving those schema keys accepted as deprecated ignored inputs.

- [ ] **Step 4: Change the factory to supply the service DMA local CPU backend**

When `cascadekv_backend` is selected, connect with `require_direct=False`, allocate `int(config.max_local_cpu_size * 1024**3)` bytes aligned to `client.metadata.io_unit_size`, and construct:

```python
allocator = XDSServiceMemoryAllocator(
    xds_client,
    int(config.max_local_cpu_size * 1024**3),
    int(xds_client.metadata.io_unit_size),
)
local_cpu_backend = LocalCPUBackend(
    config,
    metadata,
    dst_device,
    lmcache_worker,
    memory_allocator=allocator,
)
cascade_backend = CascadeKVBackend(
    config,
    metadata,
    loop,
    local_cpu_backend,
    xds_client,
    dst_device,
    lmcache_worker,
)
```

Keep `LocalCPUBackend` before `CascadeKVBackend` in the ordered backend mapping, matching the old storage hierarchy. The direct `XDSBackend` branch remains unchanged.

- [ ] **Step 5: Run config/factory tests and commit**

Run:

```bash
conda run -n xds043 pytest -q tests/v1/test_config.py -k 'xds or cascadekv'
conda run -n xds043 pytest -q tests/v1/storage_backend/test_xds_backend_factory.py
```

Expected: both focused suites pass.

Commit:

```bash
git add lmcache/v1/config.py lmcache/v1/storage_backend/__init__.py tests/v1/test_config.py tests/v1/storage_backend/test_xds_backend_factory.py
git commit -m "feat: restore legacy CascadeKV configuration"
```

---

### Task 5: Port the legacy CascadeKV backend onto request groups

**Files:**
- Replace: `lmcache/v1/storage_backend/cascadekv_backend.py`
- Rewrite: `tests/v1/storage_backend/test_cascadekv_backend.py`

**Interfaces:**
- Consumes: legacy source backend, `CascadeKVCompletionPump`, `XDSServiceMemoryAllocator`, LMCache v0.4.3 backend interfaces, and `XDSClient` blob/submit APIs.
- Produces: `CascadeKVBackend` with legacy behavior and LMCache v0.4.3 signatures.

- [ ] **Step 1: Replace slot tests with legacy characterization tests**

Delete assertions about read/write slot counts, queue depth, quarantine, poison, lifecycle gates, and bounded rings. Add public behavior tests for:

```python
def test_cross_call_write_buffer_flushes_at_legacy_window(legacy_backend):
    backend, client, keys, objects = legacy_backend
    backend.batched_submit_put_task(keys[:2], objects[:2])
    assert client.submitted_writes == []
    backend.batched_submit_put_task(keys[2:4], objects[2:4])
    wait_until(lambda: len(client.submitted_writes) == 4)
    assert [item.key for item in client.submitted_writes] == keys


def test_blob_overlap_submits_all_reads_before_first_h2d(legacy_gpu_backend):
    backend, connector, keys, starts, ends, trace = legacy_gpu_backend
    results = backend.batched_get_to_gpu_blocking(keys, connector, starts, ends)
    assert trace.index("submit-read-last") < trace.index("h2d-first")
    assert results.completed_count == len(keys)


def test_pending_write_is_readable_before_ssd_completion(legacy_backend):
    backend, _client, keys, objects = legacy_backend
    key, obj = keys[0], objects[0]
    backend.batched_submit_put_task([key], [obj])
    loaded = backend.get_blocking(key)
    assert loaded is obj
```

Add separately named tests `test_delayed_flush_uses_legacy_max_delay`,
`test_close_flushes_remaining_legacy_write_buffer`,
`test_batch_read_returns_input_order`,
`test_single_blob_read_overlaps_chunks`,
`test_get_non_blocking_deduplicates_prefetch`,
`test_lru_evicts_oldest_unpinned_key`,
`test_pin_prevents_legacy_eviction`,
`test_controller_messages_follow_admit_and_evict`,
`test_trace_accumulates_completed_bytes`, and
`test_batch_larger_than_64_has_no_slot_limit` using the same fake client and
event trace.

- [ ] **Step 2: Run characterization tests against the current backend**

Run:

```bash
conda run -n xds043 pytest -q tests/v1/storage_backend/test_cascadekv_backend.py
```

Expected: legacy aggregation, prefetch, and unbounded-batch tests fail against the bounded-slot implementation.

- [ ] **Step 3: Use the legacy file as the implementation body**

Start from the complete source file:

```text
/home/yangxc/workspace/LMCache/lmcache/v1/storage_backend/cascadekv_backend.py
```

Preserve its method order, locks, dictionaries, flush loop, overlap branches, reference counting, LRU operations, prefetch flow, worker/lookup notifications, stats, trace, and close flow. Apply only these explicit substitutions:

```text
StorageBackendInterface -> AllocatorBackendInterface-compatible signatures
xds.get_blob             -> self.client.get_blob
xds.release_blob         -> self.client.release_blob
xds.read/write *_async   -> completion-pump submit plus request-group wait
raw pointer descriptors  -> allocator.buffer_ref + allocator.get_buffer_offset(obj)
xds.get_io_unit_size     -> self.client.metadata.io_unit_size
xds.get_blob_size_in_bytes -> self.client.metadata.blob_size_bytes
xds.init_blob_pool       -> remove; external service already owns the blob pool
xds.unload               -> completion pump close; client lifetime remains backend-owned
```

For old `FIRST_COMPLETED` loops, replace only the completion primitive:

```python
group = self.completion_pump.create_group()
for task in tasks:
    self.completion_pump.submit(
        group,
        task,
        lambda task=task: self.client.submit_read(
            task.blob_handle,
            allocator.buffer_ref,
            allocator.get_buffer_offset(task.memory_obj),
            task.blob_offset,
            task.size,
        ),
    )
while group.pending_count:
    completed = group.wait_next(self.io_timeout_secs)
    self._copy_completed_read(completed.context, copy_stream)
```

For old batch waits, submit every descriptor first and then call `group.wait_all`. Do not wait between submissions.

- [ ] **Step 4: Add the LMCache v0.4.3 interface shims**

Implement `initialize_allocator`, `get_allocator_backend`, `get_memory_allocator`, `allocate`, `batched_allocate`, and `calculate_chunk_budget` by delegating to the injected DMA `LocalCPUBackend`. Accept `transfer_spec` and `on_complete_callback` in put methods; invoke the callback after the corresponding key reaches the same point at which old `insert_key` publishes it. Implement async batch methods as thin event-loop wrappers around the same legacy state machine.

LMCache may still return a Future from the public prefetch API because that is an LMCache method contract; no Future may represent an xDS service request or completion.

- [ ] **Step 5: Run backend, streaming, and storage-manager tests**

Run:

```bash
conda run -n xds043 pytest -q tests/v1/storage_backend/test_cascadekv_backend.py
conda run -n xds043 pytest -q tests/v1/test_cache_engine_streaming_transfer.py
conda run -n xds043 pytest -q tests/v1/storage_backend/test_xds_backend_factory.py
```

Expected: all focused suites pass; tests prove more than 64 chunks do not require 64 fixed slots.

- [ ] **Step 6: Run Ruff and commit the backend port**

Run:

```bash
conda run -n xds043 ruff check lmcache/v1/storage_backend/cascadekv_backend.py tests/v1/storage_backend/test_cascadekv_backend.py
```

Expected: Ruff passes.

Commit:

```bash
git add lmcache/v1/storage_backend/cascadekv_backend.py tests/v1/storage_backend/test_cascadekv_backend.py
git commit -m "feat: restore legacy CascadeKV pipelines"
```

---

### Task 6: Restore worker, lookup, stats, and StorageManager parity

**Files:**
- Modify: `lmcache/v1/storage_backend/storage_manager.py`
- Modify: `lmcache/v1/storage_backend/cascadekv_backend.py`
- Modify: `tests/v1/storage_backend/test_cascadekv_backend.py`
- Modify: `tests/v1/storage_backend/test_storage_manager.py`

**Interfaces:**
- Consumes: LMCache v0.4.3 worker message sender, lookup interfaces, transfer-result types, and the ported backend.
- Produces: legacy-visible admit/evict, lookup removal, prefix behavior, and direct transfer routing.

- [ ] **Step 1: Add failing integration-contract tests**

Assert that successful writes emit one admit per key, eviction/remove emits one evict and lookup removal, `touch_cache` updates legacy LRU order, and StorageManager routes batched GPU retrieval to `CascadeKVBackend` without slot-specific result handling.

- [ ] **Step 2: Run focused tests and capture API mismatches**

Run:

```bash
conda run -n xds043 pytest -q tests/v1/storage_backend/test_cascadekv_backend.py -k 'worker or lookup or touch'
conda run -n xds043 pytest -q tests/v1/storage_backend/test_storage_manager.py -k cascadekv
```

Expected: failures identify only LMCache v0.4.3 API name/signature differences.

- [ ] **Step 3: Implement only the v0.4.3 adapters**

Construct `BatchedMessageSender(metadata, config, str(self), lmcache_worker)` when
a worker exists. At the old `insert_key` transition call
`add_kv_op(OpType.ADMIT, key.chunk_hash)`; at successful remove/eviction call
`add_kv_op(OpType.EVICT, key.chunk_hash)`. LMCache v0.4.3 lookup servers query the
engine dynamically rather than receiving backend insert/remove callbacks, so dict
publication/removal is the lookup adaptation and no private lookup-server state is
accessed. Keep input-order prefix truncation and per-key completion callbacks.
Remove StorageManager assumptions about `BackendTransferResult` that exist solely
for the fixed-slot CascadeKV path; retain them for direct `XDSBackend` where
required.

- [ ] **Step 4: Run focused tests, Ruff, and commit**

Run:

```bash
conda run -n xds043 pytest -q tests/v1/storage_backend/test_cascadekv_backend.py tests/v1/storage_backend/test_storage_manager.py
conda run -n xds043 ruff check lmcache/v1/storage_backend/storage_manager.py lmcache/v1/storage_backend/cascadekv_backend.py
```

Expected: tests and Ruff pass.

Commit:

```bash
git add lmcache/v1/storage_backend/storage_manager.py lmcache/v1/storage_backend/cascadekv_backend.py tests/v1/storage_backend/test_cascadekv_backend.py tests/v1/storage_backend/test_storage_manager.py
git commit -m "fix: preserve CascadeKV storage integration semantics"
```

---

### Task 7: Run full verification and one vLLM prompt pair

**Files:**
- Modify only if failures prove an interface mismatch: `lmcache/integration/vllm/vllm_v1_adapter.py`
- Test only if the adapter changes: `tests/v1/test_vllm_integration.py`
- Reuse: `/home/yangxc/workspace/test` CascadeKV service and prompt scripts
- Record: `/home/yangxc/workspace/test/results/` timestamped logs and xDS statistics

**Interfaces:**
- Consumes: source-installed xDS SDK, LMCache v0.4.3, and vLLM 0.18.0 in `xds043`.
- Produces: unit/lint evidence plus an end-to-end cache-hit trace.

- [ ] **Step 1: Verify source imports and versions**

Run:

```bash
conda run -n xds043 python -c 'import torch,xds,lmcache,vllm; print(torch.__version__); print(xds.__file__); print(lmcache.__file__); print(vllm.__version__)'
```

Expected: Torch version starts with `2.10.0+`; xDS and LMCache paths point to the two source worktrees; vLLM reports `0.18.0`.

- [ ] **Step 2: Run the complete focused xDS and LMCache suites**

Run:

```bash
conda run -n xds043 pytest -q /home/yangxc/workspace/xDS/tests/python/test_client_wrapper.py
conda run -n xds043 pytest -q tests/v1/test_config.py tests/v1/test_memory_management.py tests/v1/storage_backend/test_cascadekv_completion.py tests/v1/storage_backend/test_cascadekv_backend.py tests/v1/storage_backend/test_xds_backend_factory.py tests/v1/test_cache_engine_streaming_transfer.py
conda run -n xds043 ruff check lmcache/v1/config.py lmcache/v1/memory_management.py lmcache/v1/storage_backend/cascadekv_completion.py lmcache/v1/storage_backend/cascadekv_backend.py lmcache/v1/storage_backend/__init__.py lmcache/v1/storage_backend/storage_manager.py
```

Expected: every command exits zero.

- [ ] **Step 3: Run regression tests for both xDS backends**

Run the existing direct-xDS and CascadeKV test selection:

```bash
conda run -n xds043 pytest -q tests/v1/storage_backend -k 'xds or cascadekv'
```

Expected: both independently selectable backends pass and no test imports the unused backend eagerly.

- [ ] **Step 4: Run the existing 40K-token vLLM CascadeKV scenario twice**

Use the same SSD service configuration and model setup already present under `/home/yangxc/workspace/test`, with `chunk_size: 256`, `cascadekv_backend: true`, `xds_enable_chunked_overlap: true`, and the restored write-window settings. Send the identical 40K-token prompt twice.

Expected evidence:

```text
first request: CascadeKV stores chunks through SSD -> CPU hugepage
second request: positive external-cache hit count
read trace: multiple xDS requests submitted before first H2D completion
write trace: multiple calls aggregated into the configured legacy window
no failure mentioning staging/read/write slot exhaustion
```

- [ ] **Step 5: Inspect both worktrees and commit only required final adjustments**

Run:

```bash
git status --short
git log --oneline -8
```

Expected in LMCache: only intentional uncommitted result/config adjustments, or a clean tree. In xDS: pre-existing unrelated dirty entries remain, while the `poll_burst` commit contains only its three intended files.

If Task 7 proves an LMCache connector correction is required, add a regression in
`tests/v1/test_vllm_integration.py`, modify only
`lmcache/integration/vllm/vllm_v1_adapter.py`, rerun that test, and commit exactly
those files:

```bash
git commit --only lmcache/integration/vllm/vllm_v1_adapter.py tests/v1/test_vllm_integration.py -m "fix: adapt vLLM CascadeKV transfer contract"
```

If no connector correction was required, do not create an empty final commit.
