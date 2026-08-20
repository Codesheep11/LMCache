# LMCache v0.4.3 xDS and CascadeKV Backend Design

**Status:** Approved in conversation on 2026-08-20

## Goal

Forward-port the local xDS work onto upstream LMCache v0.4.3 while keeping the
repository, Python distribution, and import package named `lmcache`. The result
provides two mutually exclusive storage backends that connect to the same
rank-local xDS service:

- `XDSBackend`: SSD to GPU through a service-registered GPU buffer.
- `CascadeKVBackend`: SSD to service-owned CPU staging memory, then CPU to GPU
  through LMCache's serving-engine GPU connector.

Both paths must work with source-installed vLLM 0.18.0. SGLang remains a
source-installed consumer and is validated after the LMCache backend port is
working.

## Version and Compatibility Constraints

- LMCache baseline: upstream tag `v0.4.3`.
- LMCache working branch: `v0.4.3-xds`.
- vLLM target: upstream tag `v0.18.0`.
- PyTorch target: `2.10.0`, matching both source projects' build metadata.
- The existing `/home/yangxc/workspace/LMCache` checkout remains unchanged and
  serves only as the source of the local implementation and tests.
- The xDS Python SDK is source-installed from
  `/home/yangxc/workspace/xDS/sdk` and supplies `XDSClient`.
- Backend selection and resource ownership are per LMCache worker/rank.

## Selected Architecture

### Shared service client

LMCache creates one connected `XDSClient` for each worker rank whenever either
backend is enabled. The connection uses `xds_service_name`, the LMCache worker
rank, world size, and a backend-specific `require_direct` value. The client
owns service connection state and blob handles; the selected backend owns all
buffer registrations or allocations made through that client.

The shared connection helper lives in a new, focused
`lmcache/v1/xds_utils.py`. Common blob metadata also lives there so the port
does not add xDS-only types to LMCache's general-purpose `lmcache/utils.py`.

### Direct xDS backend

`XDSBackend` requires a CUDA destination and `xds_require_direct: true`. It
registers the LMCache GPU allocation once with
`XDSClient.register_gpu_buffer()`. Each I/O descriptor addresses a range by
`buffer_ref`, buffer offset, blob offset, and size. Reads therefore follow:

```text
SSD -> xDS service -> registered LMCache GPU buffer
```

Writes follow the reverse path. Windowed I/O, queue-depth limits, staged write
leases, completion validation, timeout handling, and deterministic buffer
unregistration are forward-ported from the local implementation. A failed or
short I/O never publishes blob metadata as durable.

### CascadeKV backend

`CascadeKVBackend` uses the same service protocol without requiring the direct
GPU capability. It allocates service-owned CPU DMA buffers with
`XDSClient.alloc_buffer()`. The returned `memoryview` and `buffer_ref` remain
paired for their full lifetime. Reads follow:

```text
SSD -> xDS service -> service-owned CPU staging buffer -> GPU
```

Writes follow:

```text
GPU -> service-owned CPU staging buffer -> xDS service -> SSD
```

The CPU view is exposed to LMCache as a memory object without an intermediate
CPU-to-CPU copy. LMCache's existing GPU connector performs CPU/GPU transfers.
The backend maintains a bounded staging-buffer pool sized in whole aligned blob
slots. Slots cannot be reused until both xDS I/O and the corresponding GPU
copy have completed. On shutdown, the backend drains I/O, releases blob
handles that were not committed, frees every CPU buffer with
`XDSClient.free_buffer()`, and only then permits the client to close.

### Backend factory and storage manager

`CreateStorageBackends` constructs exactly one of `XDSBackend` and
`CascadeKVBackend`. The direct backend is the allocator backend because its
objects reside in the registered GPU allocation. CascadeKV uses LMCache's
normal local CPU allocation and storage flow, with CascadeKV acting as the
persistent storage backend. Generic LMCache backends retain their v0.4.3
behavior.

## Configuration Contract

The public YAML keys are:

```yaml
xds_service_name: xds0

# Direct SSD -> GPU path
xds_backend: false
xds_require_direct: false
xds_windowed_io: true
xds_read_window_size: 64
xds_write_window_size: 64
xds_io_queue_depth: 1
xds_read_slot_count: 2
xds_write_slot_count: 8
xds_async_write: false

# Staged SSD -> CPU -> GPU path
cascadekv_backend: false
cascadekv_slot_count: 8
```

Validation rules are:

1. `xds_backend` and `cascadekv_backend` cannot both be true.
2. Enabling either backend requires a non-empty `xds_service_name`.
3. `xds_backend` requires `xds_require_direct: true`.
4. `cascadekv_backend` connects with `require_direct: false`; setting
   `xds_require_direct: true` with CascadeKV is rejected to keep the selected
   path unambiguous.
5. Window sizes, queue depth, and slot counts must be positive.
6. Direct-backend queue depth cannot exceed its read or write slot count.
7. `cascadekv_slot_count` determines the bounded CPU staging capacity and must
   be positive.

Legacy in-process keys such as `bdev_name`, `rpc_addr`, and `xds_max_size` are
not used by either new backend. Storage device ownership and capacity belong to
the external xDS service.

## Code Boundaries

The implementation is limited to the following responsibilities:

- `lmcache/v1/storage_backend/xds_backend.py`: direct GPU backend and its
  direct-I/O scheduling.
- `lmcache/v1/storage_backend/cascadekv_backend.py`: CPU-staged backend,
  staging-slot pool, and CPU memory-object adaptation.
- `lmcache/v1/xds_utils.py`: client creation, alignment helpers, and common xDS
  blob metadata.
- `lmcache/v1/storage_backend/xds_read_ring.py`: bounded direct-read slot
  scheduling used only by `XDSBackend`.
- `lmcache/v1/storage_backend/xds_write_scheduler.py`: direct asynchronous
  write leases used only by `XDSBackend`.
- `lmcache/v1/storage_backend/transfer_result.py`: explicit per-chunk transfer
  outcomes when the v0.4.3 generic interface cannot express partial success.
- `lmcache/v1/config.py`: the configuration keys and validation contract above.
- `lmcache/v1/storage_backend/__init__.py`: lazy imports and mutually exclusive
  backend construction.
- `lmcache/v1/storage_backend/storage_manager.py`: minimal allocator-backend
  selection and transfer-result consumption.
- `lmcache/v1/cache_engine.py`: only the xDS client/allocator lifecycle hooks
  required for the direct backend.

The upstream v0.4.3 vLLM connector is retained by default. Connector changes
are allowed only when a failing integration test proves that an xDS buffer
lifecycle or shutdown hook cannot be expressed through the upstream connector.
No project-wide refactor or package rename is in scope.

## Error Handling and Resource Ownership

- Service connection failure aborts backend construction with the service name
  and worker rank in the error.
- Alignment, offset, and registered-range checks occur before I/O submission.
- Timeouts and short completions are failures; failed chunks remain cache
  misses and their newly acquired blob handles are released.
- CascadeKV staging slots use an explicit state transition:
  `free -> xds_io -> gpu_copy -> free`. Exceptions transition through cleanup
  before returning the slot.
- Direct write leases use `reserved -> submitted -> committed` or
  `reserved/submitted -> rolled_back`.
- `close()` is idempotent for both backends. New operations are rejected after
  shutdown begins.
- The backend closes its buffers and workers before the rank-local client is
  closed. `XDSClient.close()` must observe no outstanding CPU buffer refs.

## Test Strategy

Testing proceeds test-first and does not require live SSD hardware for unit
coverage.

1. Configuration tests prove mutual exclusion, required service name, direct
   capability rules, positive sizing, YAML parsing, and environment overrides.
2. Backend-factory tests prove the selected backend is constructed lazily and
   the unused backend's optional dependencies are not imported.
3. Mock-client unit tests for `XDSBackend` cover GPU registration, descriptor
   bounds, successful read/write, short completion rollback, timeout cleanup,
   async write draining, and idempotent close.
4. Mock-client unit tests for `CascadeKVBackend` cover `alloc_buffer`, zero
   intermediate CPU copies, SSD-to-CPU-to-GPU ordering, GPU-to-CPU-to-SSD
   ordering, slot backpressure, timeout cleanup, `free_buffer`, and close
   ordering.
5. Existing LMCache v0.4.3 storage and configuration tests remain green.
6. Source-install smoke tests import `lmcache`, `xds`, vLLM, and SGLang from
   their expected checkout paths and report the pinned versions.
7. The hardware validation starts the xDS service and sends the same long
   prompt twice through vLLM 0.18.0 for each backend. The second request must
   report an external-cache hit and logs must identify the selected path.
8. SGLang validation repeats one prompt pair after its source integration is
   adapted, without changing the LMCache backend contract.

## Acceptance Criteria

- The repository and import package remain named `lmcache`.
- `xds_backend: true` selects and exercises SSD-to-GPU direct I/O.
- `cascadekv_backend: true` selects and exercises SSD-to-CPU-to-GPU I/O through
  the external xDS service.
- Invalid dual-backend configurations fail before allocating resources.
- Unit tests demonstrate deterministic cleanup for success, timeout, short I/O,
  and shutdown.
- LMCache, vLLM, and SGLang are editable source installations in a new Conda
  environment; the existing validated `xds` environment remains recoverable.
- One repeated-prompt validation for vLLM and one for SGLang show an xDS-backed
  cache hit rather than only a local GPU prefix-cache hit.

## Rejected Approaches

- Merging v0.4.3 directly into the old dirty branch: it combines 836 upstream
  commits with old interfaces and obscures whether failures come from the port
  or from merge resolution.
- Retaining the in-process `xds.init_blob_pool()` CascadeKV path: it violates
  the requirement that both backends use the external xDS service.
- Copying service CPU data into a second LMCache CPU buffer: it adds an
  avoidable CPU-to-CPU copy and doubles staging-memory pressure.
- Renaming the Python distribution or imports to `cascadekv`: it breaks serving
  engine integrations without helping backend selection.
