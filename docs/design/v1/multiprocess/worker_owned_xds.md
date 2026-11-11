# Worker-owned xDS transfers

## Scope and implementation status

The target is an MP control plane with SSD-to-final-HBM transfers executed by
the inference worker. The initial implementation provides `XDSCatalog` and
the `modules.xds_metadata` RPC extension with MP heartbeat invalidation and
scheduler-to-worker read-pin handoff (lookup, claim, cancellation).
LMCache-Ascend provides `RegisteredNPUKVPool` to retain and register owned,
aligned NHD planes once per worker, including alias deduplication and cleanup.
The experimental `xds_worker` mode now connects the vLLM scheduler's token
lookup to the metadata extension and selects LMCache-Ascend's
`XDSWorkerTransferContext`. Support includes dense NHD attention and MLA
latent/RoPE plane pairs with one engine KV group. MLA uses one writer and explicit physical reader epochs for
each logical KV rank. Hybrid/multi-group layouts are rejected. A chunk
must fit one xDS blob. Four-card Qwen short-context restoration and four-card
DeepSeek-V2-Lite 30K-token MLA restoration have passed NaN poisoning and exact
KV-byte checks. General multi-request stress, APC-enabled inference and broader
model coverage remain to be validated; this mode is not yet production-ready.

## Why the worker owns I/O

Ascend driver 25.5.1 rejects P2P page export for IPC-imported allocations in
`devmm_is_support_p2p_get_pages`. Fixing xDS's HAL large-page table expansion
allows registration of locally owned PyTorch allocations, but does not remove
that driver restriction. The server therefore must not import worker HBM to
register it for DMA.

xDS also scopes SSD blob leases and registered buffer references to the client
queue. An independent worker can read a blob only after importing its owner-issued
read-only lease token. HBM registration remains local to the reading worker. Each worker must own both its fixed HBM registrations and
its xDS blob leases. The server allocates object extents inside the capacity
announced by that worker; it never submits I/O itself.

```text
worker                              LMCache MP server
  connect xDS, lease blobs
  register fixed HBM pools once
  announce SSD extents ------------> catalog.register
  reserve content key -------------> catalog.reserve_write
  <------------------------------- write ticket + SSD extent
  wait for producer NPU operations
  writev(HBM pages -> SSD)
  wait for successful I/O completion
  publish(ticket) -----------------> entry becomes visible

scheduler/worker ------------------> acquire_reads(all required TP shards)
  <------------------------------- all leases, or miss without any pins
  readv(SSD -> final HBM pages)
  wait for DMA before consuming KV
  release_read(ticket) ------------> entry can be evicted after last reader
```

No KV bytes or HBM addresses appear in the metadata protocol. CPU payload
copies used to seed or inspect hardware tests are outside the transfer path.

## Catalog contract

* Worker IDs identify connection epochs, not rank numbers or reusable PIDs.
  A reconnect requires a new epoch and fresh xDS leases. Cached data from the
  previous connection is invalidated; sharing it across replicas is not
  supported by this protocol.
* `ObjectKey` preserves the existing model, chunk hash, KV rank, object group,
  and cache salt identity. Token hashing is not reimplemented here.
* Capacity consists of nonoverlapping `(blob, byte_offset, byte_size)` extents.
  The worker validates blob ownership and xDS I/O alignment before announcing
  capacity. Exact-size allocation supports different KV group sizes without
  mixing their physical representations.
* Writes are invisible until `publish`. Concurrent writes to an existing key
  receive no reservation. A pending write is never an eviction candidate.
* Read acquisition is atomic across its object list. A missing shard or
  unfinished write returns a miss without leaving partial pins. Every reader
  has its own opaque ticket. All tickets must be released before eviction.
* LRU eviction skips pending writes and pinned objects. Catalog entries and
  outstanding leases are protected by a single lock; no I/O runs under it.
* Completion tickets cannot be reused for replacement objects. Stale or
  foreign completion tickets fail rather than publishing unrelated bytes.

## Failure and shutdown ordering

RPC timeout does **not** establish DMA completion. The worker must drain xDS
operations before calling `abort_write` or `release_read`. A partial write
cannot be published. Graceful unregister rejects outstanding leases; shutdown
must drain operations, return leases, unregister the catalog epoch, release
the xDS blobs, and finally unregister HBM before the allocation is freed.

For dead workers, `forget_dead_worker` invalidates catalog metadata without
recycling their capacity. xDS connection cleanup owns the hardware drain and
blob release. The module implements `InstanceLivenessTarget`, so the existing
MP management module automatically delivers PINGs and reaps stale workers.
Traffic refreshes liveness without shortening the initial registration grace;
scheduler lookups do not prove worker liveness. The RPC extension does not
expose forced hardware reclamation.

The vLLM 0.26 default `--shutdown-timeout 0` aborts workers immediately. Use
`--shutdown-timeout 30` and `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=30` in the
validation environment so normal shutdown can drain transfers and release
registrations. This does not replace dead-worker cleanup for forced exits.

## RPC extension

The existing MP `server_module_call` envelope carries MessagePack-encoded
dataclasses from `xds_protocol.py`. Methods use the `xds.v1` namespace:

| Method | Request | Response |
| --- | --- | --- |
| register | Register(worker, extents) | nil |
| ensure_registered | Register(worker, extents), after local DMA drain | nil |
| reserve_write | ReserveWrite(worker, key, size) | XDSLease or nil |
| publish / abort_write | Complete(worker, ticket) | nil |
| publish_many / release_many | CompleteMany(worker, tickets) | nil |
| acquire_reads | AcquireReads(worker/key pairs) | tuple of XDSLease or nil |
| lookup_prefix | LookupPrefix(request epoch, token-ordered shard sets) | complete prefix chunk count |
| claim_reads | ClaimReads(request epoch, worker, ordered keys) | tuple of XDSLease |
| cancel_lookup | request epoch string | nil |
| release_read | Complete(worker, ticket) | nil |
| unregister | worker epoch integer | nil |
| bind_worker | BindWorker(epoch, model, world size, worker rank, chunk bytes) | nil |
| lookup_tokens | scheduler IPCCacheServerKey (low-level validation) | complete prefix chunk count |
| lookup_owned_tokens | SchedulerLookup(scheduler epoch, key) | complete prefix chunk count |
| reserve_tokens | TokenTransfer(epoch, worker key) | optional write lease per chunk |
| claim_range | ClaimRange(epoch, request ID, model, world size, rank, start, end) | ordered read leases |
| claim_tokens | TokenTransfer(epoch, worker key) | ordered read leases |
| release_tokens | scheduler key for unused range | nil |
| clear_idle | nil | evicted idle object count |

The standard extension router returns validation/lifecycle errors through its
`success=False` response envelope. RPCs are not automatically retried: an
ambiguous completion must not release capacity early.

Load the metadata extension for development with:

```text
--server-module '{"module_path":"lmcache.v1.multiprocess.modules.xds_metadata"}'
```

## Read-pin handoff

`lookup_prefix` takes token-ordered chunks, each listing all required rank/group
objects. It pins only the consecutive complete prefix. A missing shard cannot
leave a partially pinned chunk. Duplicate object keys and duplicate request
epochs are rejected; request IDs must be scoped to a scheduler session.

`claim_reads` atomically transfers the exact requested pins to one worker and
preserves their existing tickets. No unlock/relock occurs. A missing or already
claimed key fails the entire claim. Once claimed, only worker completion can
release the pin: `cancel_lookup` releases unclaimed pins and is idempotent.
A dead worker invalidates affected lookups, releasing other ranks' unclaimed
pins while preserving their already claimed DMA leases. Each scheduler request
must cancel its lookup at completion, including misses and fully claimed hits.
The scheduler adapter uses `lookup_owned_tokens` with a negative process epoch
and sends that epoch through the existing MP heartbeat. The metadata reaper
cancels this scheduler's unclaimed lookups after its heartbeat expires. Claimed
worker DMA leases remain pinned. Registration grace applies until the first
PING; a healthy scheduler can keep a queued request alive without a per-request
TTL. Normal request cancellation removes the ownership record. Nonnegative
worker epochs occupy a separate namespace.

The real ZMQ hardware harness exercises lookup, claim, cancellation before DMA,
and SSD-to-HBM restoration at 1, 3, 257 and 300 noncontiguous slices. Each case
checks the full destination allocation, including untouched guard bytes. It
also verifies that cancellation cannot make a claimed extent available to a
replacement write. This is a protocol/hardware test, not inference validation.

## Worker integration and remaining validation

Select `lmcache.mp.mp_transfer_mode: xds_worker` in the vLLM connector extra
configuration (or `LMCACHE_MP_TRANSFER_MODE=xds_worker` for both scheduler and
workers). The MP server still loads the extension shown above. It resolves
token ranges through its existing `TokenHasher` and `ipc_key_to_object_keys`;
there is no second hashing implementation. Model/world-size/rank bindings
identify the live worker epochs. Duplicate active rank bindings are rejected.

The worker owns fixed registrations and blob capacity, uses plane-major
writev and matching indexed readv, and fences NPU execution. The worker API is synchronous; each transfer pipelines up to
`LMCACHE_XDS_QUEUE_DEPTH` outstanding requests (default 4) and completes the
metadata tickets in one batch after draining all I/O. Blob count and
service are configured through `LMCACHE_XDS_BLOB_COUNT` and
`LMCACHE_XDS_SERVICE`; startup validates that each chunk fits the service's
reported blob size. No intermediate payload allocation is made.

The adapter calls `recover_registration()` on the existing worker context.
A transfer lock serializes recovery, registration, I/O and close. Recovery
rejects a closed context or uncertain DMA completion. `ensure_registered`
preserves an identical live catalog, including pins, after transient transport
loss; after server restart or reaping it recreates empty capacity. Exact rank
bindings are idempotent; conflicting bindings or changed live capacity fail.
The same xDS connection, blob leases and HBM registrations remain owned by the
worker. Unregister prevents new transfers before sending its metadata RPC.

Four-card Qwen cold/warm inference and poisoned-HBM byte verification passed
for 30K cached + 3K fresh tokens. Performance measurements still require an
isolated run without correctness instrumentation. Remaining work includes
scheduler-crash integration verification, actual server-restart
integration verification, and MiniMax eight-card correctness/performance.
An uncertain DMA timeout remains fail-closed rather than automatically retrying.

Validation must compare the same final-HBM endpoint and context lengths as the
previous version, including cache poisoning or equivalent byte checks to prove
that inference actually uses restored KV. Protocol unit tests alone do not
prove this integration complete.


## Shared prefix inside a chunk

vLLM aligns the load range down to a cache chunk and passes the APC-hit prefix
as `skip_first_n_tokens`. A block-aligned skip preserves those shared pages.
Whole chunks use indexed readv; a partially skipped chunk uses one contiguous
SSD range per K/V plane and scatters it into that plane's remaining blocks.
The on-disk plane stride is the full chunk size, including skipped blocks.
The same QD limit applies to these requests, and all read leases remain pinned
until every transfer finishes. Fully skipped chunks submit no I/O but their
claimed leases are still released. No payload staging allocation is introduced.

Sub-block skips fail before claiming metadata: a byte range narrower than a
physical block may violate NVMe alignment, and rounding down would overwrite
pages shared by other requests. This contract still needs actual APC-enabled
inference validation; simulated byte-level tests prove offsets and guard
preservation, not hardware scheduling behavior.


## Indexed range claims

Token lookup retains the canonical keys for its matched prefix, indexed by
worker rank and token offset. `claim_range` uses this request-scoped index;
workers transmit no token list and the server performs no prefix hashing or
object-key reconstruction during claim. Model, world size, connection epoch
and chunk-aligned hit bounds are checked before the catalog atomically hands
off existing pins. Cache salt is part of the original lookup identity.
Disjoint subranges can be claimed separately; duplicate/overlapping claims
fail atomically. This applies to arbitrary chunk-aligned lengths and starts.

Cancellation, scheduler expiry, worker invalidation and module shutdown remove
the index. Claimed DMA leases retain their existing completion lifetime.
After metadata loss, an old request cannot claim from a new worker epoch.
The token-based claim RPC remains available for existing clients.


## MLA replicas and unequal planes

`BindWorker` distinguishes logical `worker_id` from physical `reader_id`, declares
`num_readers`, and identifies the single writer. Read-only registrations have
empty capacity. Lookups require a complete reader cohort and exactly one writer.
The catalog pins one SSD entry with independent tickets for every physical reader;
claim indices use the connection epoch, so replicas with the same logical rank
cannot consume each other's tickets. Owner unregister rejects any outstanding
reader, including other workers. Dead owners invalidate metadata without reusing
capacity; already-claimed reader completion retains its entry reference.

`XDSExtent.read_token` carries the owner's xDS read-only export token. Each
non-owner worker imports distinct blobs in a restore batch, executes I/O using
its own registered HBM, drains I/O, unimports the blobs, then releases metadata
tickets. A timeout keeps imports and HBM registration pinned. The xDS service
prevents owner blob release/reuse while any import remains, including owner
process death. Reader disconnect drains its queue before removing imports.

Plane-major SSD packing uses each plane's actual block-byte stride. Equal-width
planes keep the indexed readv path; unequal MLA latent/RoPE planes use general
readv. A skipped prefix advances each plane's source offset independently and
preserves skipped destination pages. Only descriptors are constructed on CPU;
no KV staging allocation or payload copy is added.

MLA stores one unique SSD copy, but this direct-read implementation performs one
SSD read per physical HBM replica. It does not implement NVMe multicast. Report
unique stored KV bytes, actual SSD bytes read, and aggregate HBM bytes separately
when comparing against Disk's CPU fanout.


## MLA leader reads and final-page broadcast

The shared worker implementation now lives in `lmcache.v1.multiprocess.xds_transfer`.
Its CUDA adapter supports contiguous block-major MLA `[blocks, tokens, width]`
and NHD K/V planes; LMCache-Ascend subclasses its registration and fencing hooks
for Ascend's separate, padded latent/RoPE planes. CUDA peer-memory registration
still depends on the deployed xDS/NVIDIA driver accepting the allocation type;
unsupported VMM allocations fail rather than falling back to a staging buffer.

For replicated MLA KV, `LMCACHE_XDS_MLA_READ_MODE=leader` (default) selects one
writer/reader per logical shard. The existing vLLM TP device group provides
NCCL on CUDA or HCCL on Ascend; its Gloo group exchanges only control metadata.
The group's membership must exactly match the shard's reader cohort. Ordinary
nonreplicated K/V does not use collectives. DCP/multiple-server subsets are not
automatically constructed: use explicit matching groups or `direct` mode.

Every replica still claims its independent metadata pin. The leader reads each
SSD extent into its own final pages; broadcasts write directly into each other
replica's final pages. Different local page IDs are allowed. A planner finds
logical intervals contiguous on all replicas and creates narrow tensor views,
never a packed KV buffer. Fragmented page maps may produce more collectives.
Application-level payload staging is absent; communication libraries may retain
their own internal workspaces.

`LMCACHE_XDS_MLA_BATCH_BYTES` defaults to 67108864 (64 MiB of unique KV). A chunk
is indivisible, so a larger chunk sets the minimum batch size. This byte-based
policy is independent of context length, layer count and TP count. The leader
reads the next disjoint batch while the preceding broadcast is outstanding.
Only one broadcast batch is retained at a time. The final device fence precedes
lease release and attention consumption. The source pages remain retained for
the full transfer. `direct` mode keeps the previous all-rank SSD read path for
comparison or unsupported replica-group layouts.

Before data collectives, ranks compare request identity, range, plane geometry,
skipped prefix and writer count. Page maps must not alias. Validation/claim
failures are shared by a readiness handshake; leader I/O failures are propagated
before starting the next broadcast. An uncertain DMA/communication completion
retains pins and HBM until the worker exits. No timeout path frees active pages.
The engine owns process-group creation, timeout and failure teardown; this
transport does not destroy groups shared with model computation.

Validation status: CPU collective simulation covers CUDA/NCCL and Ascend/HCCL
routing, unequal widths, arbitrary page maps, skipped prefixes, byte-based
batch sizes and failure propagation. CUDA registration has mock contract tests.
NVIDIA hardware validation remains required in a separate GPU environment.
Ascend DeepSeek-V2-Lite TP4 passed three 30K-token restores with NaN poisoning
and matching byte digests on all four cards (10.678711 GiB total checked).
Only one worker read SSD data: 0.889893 GiB per restore. The following 3K-token
prefill produced the same output as the cold request. These shared-machine
correctness runs do not establish performance. Raw results are outside the
repository.


### Coalescing independently of logical page order

Within each completed SSD batch, the planner sorts selected logical block
indices by rank zero's final-page address. It extends a run only when the
corresponding page advances by one on every replica. This merges shared reverse
or arbitrary permutations without moving payloads. Each descriptor carries the
per-rank starting block and a common block count; each rank broadcasts a narrow
view of its own final allocation. The same plan is reused for every plane.

This is optimal for contiguous, increasing-address tensor views within the
selected batch: any possible merged run must also be consecutive in rank zero's
sorted address order. Incompatible permutations between ranks still require
separate runs. Protected prefix pages and pages outside the completed batch are
excluded before sorting. The planner does not assume a particular token count,
TP count, or page numbering convention.


### Bounded SSD read-ahead

A single host reader thread executes the leader's reads while the caller submits
NCCL/HCCL broadcasts. Two batches may be admitted ahead of the consumer; the
reader uses the existing per-batch QD and final registered pages, with no payload
staging. Batches are consumed in order, and no view is broadcast before its read
future succeeds and all ranks agree. This removes the dependency of the next
read on broadcast submission; it does not guarantee a nonempty hardware queue
when read-ahead is exhausted or at per-batch drains. The byte-based batch policy
is unchanged and is independent of context length.

Device collectives and fences remain on the caller thread. The read callback
uses already registered SDK handles and must not require caller thread-local
device state. Only the reader accesses the xDS client during this phase. On
exit, queued work is cancelled and active reads are joined before the transfer
lock is released. Any failed transfer keeps its existing fail-closed pin/HBM
retention, including read-ahead failures. CUDA and Ascend share this policy.
