xDS Worker-Owned MLA Transfers
==============================

This experimental transfer mode keeps KV payloads in the inference worker.
The MP service manages metadata and leases; xDS performs SSD I/O directly
against the worker's registered final KV pages.

Platform support
----------------

* LMCache provides the common transfer implementation and CUDA registration.
  NVIDIA replicas use NCCL.
* LMCache-Ascend provides the NPU registration and fencing adapter. Ascend
  replicas use HCCL.
* Both paths require an installed xDS SDK, a compatible running xDS service,
  peer-memory driver support, and the existing MP ``xds_worker`` setup.
  This mode does not install or configure those dependencies automatically.

For replicated MLA, the default is to read each SSD chunk once on its writer
and broadcast directly into the other workers' final KV pages. Local block IDs
may differ. No application KV packing/staging buffer is allocated. Device
communication libraries can still use internal workspaces.

Configuration
-------------

Set these variables in every inference worker before starting vLLM:

.. code-block:: bash

   export LMCACHE_MP_TRANSFER_MODE=xds_worker
   export LMCACHE_XDS_MLA_READ_MODE=leader
   export LMCACHE_XDS_MLA_BATCH_BYTES=67108864

``LMCACHE_XDS_MLA_READ_MODE``
   ``leader`` (default) performs one SSD read per logical MLA shard and
   distributes its data via the TP process group. ``direct`` retains the
   previous path where every replica reads the SSD. Nonreplicated K/V continues
   to use direct I/O without collectives in either mode.

``LMCACHE_XDS_MLA_BATCH_BYTES``
   Maximum target bytes of unique KV per pipeline batch, default 64 MiB.
   Whole chunks are indivisible; a chunk larger than this value forms its own
   batch. The leader can read the next batch while the previous broadcast is
   outstanding. This setting does not allocate a payload buffer.

Requirements and limits
-----------------------

The existing TP device and Gloo groups must contain exactly one logical KV
shard's readers, including one writer. DCP and multiple-server subsets require
explicit matching groups and are not automatically derived; use ``direct`` for
those layouts until the correct replica groups are supplied.

CUDA supports contiguous block-major MLA tensors and NHD K/V planes. Ascend
supports separate block-major latent/RoPE or K/V planes with owned padding for
HAL registration. Hybrid/multiple KV groups remain unsupported. A driver's
rejection of CUDA VMM allocations is an error, not a CPU staging fallback.

Collective ordering and failure handling
----------------------------------------

All replicas must restore the same logical range and protected prefix in the
same order. CPU control messages validate this contract and exchange page maps.
Within each completed SSD batch, descriptors are sorted by one rank's page
addresses. A run is merged only when corresponding addresses are consecutive
on every rank. Shared reverse/permuted allocations can therefore be merged
without packing data. The resulting plan is reused across all planes.
Incompatible layouts between ranks still increase the collective count.
Protected prefixes and unread batches are never included in a run. Data is
fenced before attention uses the pages.

An I/O or collective failure retains HBM and leases when completion is uncertain.
Restart the failed inference workers; do not release their memory while DMA may
still be active. The engine owns process-group timeouts and teardown.

Validation
----------

Unit tests cover both platform routes with CPU collective simulation. NVIDIA
hardware validation must be performed on the target GPU/driver deployment.
Ascend DeepSeek-V2-Lite TP4 validation passed three restores of a 30K-token
MLA prefix followed by 3K fresh input: all four final HBM replicas matched
pre-store byte digests after NaN poisoning. Only the leader submitted SSD reads
(955514880 bytes per restore). Detailed results remain in external deployment
artifacts. Do not infer an end-to-end speedup from unit tests or tests on a shared
machine; compare complete SSD-to-final-HBM windows under controlled conditions.

Bounded MLA read-ahead
---------------------

The leader uses one host reader thread to prefetch up to two batches ahead
of the broadcast consumer, directly into final registered KV pages. This
allows SSD reads to progress during NCCL/HCCL submission without allocating
KV staging buffers. Batch size and QD retain their existing limits. Broadcasts
consume completed batches in order. Shutdown or transfer failure cancels
queued reads and joins the reader before leaving the transfer; uncertain
DMA still retains its leases and registered memory. Per-batch drains and
backpressure can still leave the SSD temporarily idle.
