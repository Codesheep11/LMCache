# SPDX-License-Identifier: Apache-2.0
"""Public planner contract: exact bytes, supported layouts and safe fallback."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.xds_readv import plan_paged_readv
from lmcache.v1.memory_management import MemoryFormat


@pytest.mark.parametrize("layout", ["2NB", "NB2"])
def test_readv_matches_kv2ltd_mapping(layout: str) -> None:
    shape = (2, 8, 16, 2, 128) if layout == "2NB" else (8, 2, 16, 2, 128)
    tensors = [torch.zeros(shape, dtype=torch.float16, device="cuda") for _ in range(3)]
    slots = torch.tensor(list(range(64, 80)) + list(range(16, 32)))
    source_shape = torch.Size([2, 3, 32, 256])
    plan = plan_paged_readv(
        tensors,
        [source_shape],
        [torch.float16],
        MemoryFormat.KV_2LTD,
        slots,
        0,
        32,
        4096,
        layout,
    )
    assert plan is not None
    source = (
        torch.arange(source_shape.numel(), device="cuda", dtype=torch.int32)
        .remainder(997)
        .to(torch.float16)
        .reshape(source_shape)
    )
    raw = source.view(torch.uint8).flatten()
    cursor = 0
    for index, offset, size in plan.slices:
        tensors[index].view(torch.uint8).flatten()[offset : offset + size].copy_(
            raw[cursor : cursor + size]
        )
        cursor += size
    assert cursor == plan.size == source.numel() * source.element_size()
    for layer, tensor in enumerate(tensors):
        for kv in range(2):
            expected = torch.zeros((8, 16, 2, 128), device="cuda", dtype=torch.float16)
            expected[4] = source[kv, layer, :16].reshape(16, 2, 128)
            expected[1] = source[kv, layer, 16:].reshape(16, 2, 128)
            assert torch.equal(
                tensor[kv] if layout == "2NB" else tensor[:, kv], expected
            )


@pytest.mark.parametrize(
    "case",
    [
        "partial",
        "misaligned",
        "negative",
        "duplicate",
        "range",
        "hnd",
        "alias",
        "dtype",
        "small_block",
    ],
)
def test_readv_unsupported_returns_none(case: str) -> None:
    tensors = [
        torch.empty((2, 8, 16, 2, 128), dtype=torch.float16, device="cuda")
        for _ in range(2)
    ]
    slots = torch.arange(32)
    shape = torch.Size([2, 2, 32, 256])
    dtype = torch.float16
    layout = "2NB"
    if case == "partial":
        slots = slots[:31]
        shape = torch.Size([2, 2, 31, 256])
    if case == "misaligned":
        slots += 1
    if case == "negative":
        slots[0] = -1
    if case == "duplicate":
        slots[16:] = slots[:16]
    if case == "range":
        slots += 128
    if case == "hnd":
        layout = "HND"
    if case == "alias":
        tensors[1] = tensors[0]
    if case == "dtype":
        dtype = torch.float32
    if case == "small_block":
        tensors = [
            torch.empty((2, 8, 16, 1, 8), dtype=torch.float16, device="cuda")
            for _ in range(2)
        ]
        shape = torch.Size([2, 2, 32, 8])
    assert (
        plan_paged_readv(
            tensors,
            [shape],
            [dtype],
            MemoryFormat.KV_2LTD,
            slots,
            0,
            slots.numel(),
            4096,
            layout,
        )
        is None
    )


@pytest.mark.parametrize("layout", ["2NB", "NB2"])
def test_batch_plans_bind_source_order_and_refresh_slots(layout: str) -> None:
    from lmcache.v1.gpu_connector.xds_readv import plan_paged_readv_batch

    shape = (2, 8, 16, 2, 128) if layout == "2NB" else (8, 2, 16, 2, 128)
    tensors = [torch.empty(shape, dtype=torch.float16, device="cuda") for _ in range(2)]
    chunks = [
        (
            [torch.Size([2, 2, 32, 256])],
            [torch.float16],
            MemoryFormat.KV_2LTD,
            a,
            a + 32,
        )
        for a in [0, 32]
    ]
    refs = [101, 202]
    for ids in [[4, 1, 7, 2], [3, 6, 0, 5]]:
        slots = torch.tensor([b * 16 + t for b in ids for t in range(16)])
        plans = plan_paged_readv_batch(tensors, chunks, slots, 4096, layout)
        assert plans is not None and len(plans) == 2
        for chunk, plan in enumerate(plans):
            expected = [
                (
                    ref,
                    ((kv * 8 + block) if layout == "2NB" else (block * 2 + kv)) * 8192,
                    8192,
                )
                for kv in range(2)
                for ref in refs
                for block in ids[chunk * 2 : chunk * 2 + 2]
            ]
            assert plan.bind_slices(refs) == expected
            assert [(refs[i], a, n) for i, a, n in plan.slices] == expected
            assert plan.size == sum(x[2] for x in expected)
            with pytest.raises(ValueError, match="reference count"):
                plan.bind_slices(refs[:1])


@pytest.mark.parametrize("failure", ["overlap", "last_partial", "last_dtype"])
def test_batch_rejects_invalid_late_chunk(failure: str) -> None:
    from lmcache.v1.gpu_connector.xds_readv import plan_paged_readv_batch

    tensors = [torch.empty((2, 8, 16, 2, 128), dtype=torch.float16, device="cuda")]
    chunks = [
        (
            [torch.Size([2, 1, 32, 256])],
            [torch.float16],
            MemoryFormat.KV_2LTD,
            a,
            a + 32,
        )
        for a in [0, 32]
    ]
    slots = torch.arange(64)
    if failure == "overlap":
        slots[32:] = slots[:32]
    elif failure == "last_partial":
        slots[-1] += 1
    else:
        chunks[-1][1][0] = torch.float32
    assert plan_paged_readv_batch(tensors, chunks, slots, 4096, "2NB") is None


@pytest.mark.parametrize("layout", ["2NB", "NB2"])
def test_layerwise_plans_preserve_per_layer_kv_major_order(layout: str) -> None:
    from lmcache.v1.gpu_connector.xds_readv import plan_layerwise_readv_batch

    shape = (2, 8, 16, 2, 128) if layout == "2NB" else (8, 2, 16, 2, 128)
    tensors = [torch.empty(shape, dtype=torch.float16, device="cuda") for _ in range(2)]
    chunks = [([torch.Size([2, 32, 256])], [torch.float16], MemoryFormat.KV_2TD, 0, 32)]
    slots = torch.tensor(list(range(64, 80)) + list(range(16, 32)))
    plans = plan_layerwise_readv_batch(tensors, chunks, slots, 4096, layout)
    assert plans is not None and len(plans) == 2
    for layer, chunk_plans in enumerate(plans):
        assert chunk_plans[0].tensors[0] is tensors[layer]
        expected = [
            (
                101 + layer,
                ((kv * 8 + block) if layout == "2NB" else block * 2 + kv) * 8192,
                8192,
            )
            for kv in range(2)
            for block in [4, 1]
        ]
        assert chunk_plans[0].bind_slices([101 + layer]) == expected
        assert chunk_plans[0].size == 32768
    legacy = [([torch.Size([32, 2, 256])], [torch.float16], MemoryFormat.KV_T2D, 0, 32)]
    assert plan_layerwise_readv_batch(tensors, legacy, slots, 4096, layout) is None


@pytest.mark.parametrize("legacy", [False, True])
def test_layerwise_connector_store_fallback_and_completion_events(legacy: bool) -> None:
    from types import SimpleNamespace
    from lmcache.v1.gpu_connector.xds_layerwise import XDSReadvLayerwiseGPUConnector

    connector = XDSReadvLayerwiseGPUConnector(
        256,
        2,
        use_gpu=False,
        chunk_size=32,
        dtype=torch.float16,
        device=torch.device("cuda"),
        layout_hints={"kv_layout": "NHD"},
    )
    targets = [
        torch.randn((2, 8, 16, 2, 128), dtype=torch.float16, device="cuda")
        for _ in range(2)
    ]
    originals = [t.clone() for t in targets]
    objects = [
        [
            SimpleNamespace(
                tensor=torch.empty((2, 32, 256), device="cuda", dtype=torch.float16),
                metadata=SimpleNamespace(fmt=MemoryFormat.KV_2TD),
            )
        ]
        for _ in range(2)
    ]
    slots = torch.tensor(list(range(64, 80)) + list(range(16, 32)), device="cuda")
    kwargs = dict(kvcaches=targets, slot_mapping=slots, sync=True)
    assert len(list(connector.batched_from_gpu(objects, [0], [32], **kwargs))) == 3
    for layer, source in zip(objects, originals, strict=True):
        expected = torch.cat([source[:, 4], source[:, 1]], dim=1).reshape(2, 32, 256)
        assert torch.equal(layer[0].tensor, expected)
        if legacy:
            layer[0].tensor = layer[0].tensor.permute(1, 0, 2).contiguous()
            layer[0].metadata.fmt = MemoryFormat.KV_T2D
    for t in targets:
        t.zero_()
    torch.cuda.synchronize()
    events = []
    consumer = connector.batched_to_gpu(
        [0], [32], **kwargs, h2d_copy_completion_events=events
    )
    next(consumer)
    for layer in objects:
        consumer.send(layer)
    next(consumer)
    consumer.close()
    assert len(events) == 2
    for event in events:
        event.synchronize()
    for t, source in zip(targets, originals, strict=True):
        expected = torch.zeros_like(t)
        expected[:, 4] = source[:, 4]
        expected[:, 1] = source[:, 1]
        assert torch.equal(t, expected)


@pytest.mark.parametrize("layout", ["2NB", "NB2"])
@pytest.mark.parametrize("layerwise", [False, True])
def test_native_plan_submission_preserves_byte_mapping(
    layout: str, layerwise: bool
) -> None:
    # Real SDK planner, fake DMA endpoint; no privileged service required.
    import xds
    from lmcache.v1.gpu_connector.xds_readv import plan_layerwise_readv_batch

    shape = (2, 8, 16, 2, 128) if layout == "2NB" else (8, 2, 16, 2, 128)
    tensors = [torch.empty(shape, dtype=torch.float16, device="cuda") for _ in range(3)]
    slots = torch.tensor(list(range(32, 64)) + list(range(0, 16)))
    if layerwise:
        layers = plan_layerwise_readv_batch(
            tensors,
            [([torch.Size([2, 48, 256])], [torch.float16], MemoryFormat.KV_2TD, 0, 48)],
            slots,
            4096,
            layout,
        )
        plans = [p for layer in layers for p in layer]
    else:
        plans = [
            plan_paged_readv(
                tensors,
                [torch.Size([2, 3, 48, 256])],
                [torch.float16],
                MemoryFormat.KV_2LTD,
                slots,
                0,
                48,
                4096,
                layout,
            )
        ]

    class NativeClient:
        def prepare_indexed_readv(self, ids, groups, size):
            return xds.prepare_indexed_readv(ids, groups, size, 4096)

        def submit_readv_plan(self, handle, offset, plan):
            assert (handle, offset) == (17, 4096)
            self.slices = plan.to_slices()
            return 19

    for plan in plans:
        refs = list(range(1, len(plan.tensors) + 1))
        client = NativeClient()
        assert plan.submit(client, 17, 4096, refs) == 19

        def pages(slices):
            return [(r, o + i) for r, o, n in slices for i in range(0, n, 4096)]

        assert pages(client.slices) == pages(plan.bind_slices(refs))

        def submit_batch(requests, plan=plan, refs=refs):
            assert len(requests) == 2
            for handle, offset, native in requests:
                assert (handle, offset) == (17, 4096)
                assert pages(native.to_slices()) == pages(plan.bind_slices(refs))
            return [29, 30]

        client.submit_readv_plans = submit_batch
        assert plan.submit_batch(client, [(plan, 17, 4096, refs)] * 2) == [29, 30]
        # A later preparation error must prevent submission of the entire batch.
        failures = plan.submit_batch(
            client, [(plan, 17, 4096, refs), (plan, 17, 4096, refs + [99])]
        )
        assert all(isinstance(error, ValueError) for error in failures)
        # A new facade with an older native extension must still use readv.
        client.supports_native_readv = False
        legacy_calls = []

        def legacy_submit(handle, offset, slices, calls=legacy_calls):
            calls.append((handle, offset, slices))
            return 23

        client.submit_readv = legacy_submit
        assert plan.submit(client, 17, 4096, refs) == 23
        assert legacy_calls == [(17, 4096, plan.bind_slices(refs))]
        with pytest.raises(ValueError, match="reference count"):
            plan.submit(client, 17, 4096, refs + [99])


@pytest.mark.parametrize(
    "mapping",
    [
        {"block_ids": ([],)},
        {"block_ids": (4,)},
        {"block_ids": (1, 1), "offsets": (0, 2), "sizes": (16384,)},
        {"offsets": [0, 1]},
        {"offsets": (0, 0)},
        {"sizes": (8191,)},
    ],
)
def test_compact_batch_rejects_invalid_owned_mapping(mapping: dict) -> None:
    """Malformed immutable mappings must fail before any SDK submission."""
    from lmcache.v1.gpu_connector.xds_readv import (
        XDSPagedKVGeometry,
        XDSPagedReadvBatch,
    )

    geometry = XDSPagedKVGeometry(
        (torch.empty(1, device="cpu"),),
        "2NB",
        "cuda",
        torch.float16,
        1,
        4,
        16,
        128,
        4096,
        512,
        4096,
    )
    fields = {"block_ids": (1,), "offsets": (0, 1), "sizes": (8192,)}
    fields.update(mapping)
    with pytest.raises(ValueError, match="invalid compact readv"):
        XDSPagedReadvBatch(geometry, **fields)


@pytest.mark.parametrize("layout", ["2NB", "NB2"])
@pytest.mark.parametrize("compact", [False, True])
def test_cuda_fixed_pool_reuses_registration_and_refreshes_slots(
    layout: str, compact: bool
) -> None:
    """Fixed storage keeps refs, but each request gets its own block mapping."""
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2

    shape = (2, 8, 16, 2, 128) if layout == "2NB" else (8, 2, 16, 2, 128)
    tensors = [torch.empty(shape, dtype=torch.float16, device="cuda")]
    connector = VLLMPagedMemGPUConnectorV2(hidden_dim_size=256, num_layers=1)
    pool = connector.initialize_xds_readv_pool(tensors)
    assert connector.initialize_xds_readv_pool(tensors) is pool
    pool.buffer_refs = (123,)
    chunk = (
        [torch.Size([2, 1, 16, 256])],
        [torch.float16],
        MemoryFormat.KV_2LTD,
        0,
        16,
    )
    for block in (0, 3):
        plans = connector.plan_xds_readv_batch(
            [chunk],
            4096,
            kvcaches=tensors,
            slot_mapping_cpu=torch.arange(block * 16, (block + 1) * 16),
            xds_compact_batch=compact,
        )
        assert plans is not None
        assert plans[0].buffer_refs == (123,)
        assert plans[0].block_ids == (block,)
    with pytest.raises(RuntimeError, match="replace"):
        connector.initialize_xds_readv_pool([torch.empty_like(tensors[0])])
    pool.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        connector.initialize_xds_readv_pool(tensors)


def test_cuda_dma_fence_waits_for_other_streams() -> None:
    """External DMA must not race a producer running outside the current stream."""
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2

    connector = VLLMPagedMemGPUConnectorV2(hidden_dim_size=256, num_layers=1)
    tensor = torch.empty(1024 * 1024, device="cuda")
    stream = torch.cuda.Stream()
    done = torch.cuda.Event()
    with torch.cuda.stream(stream):
        tensor.fill_(7)
        done.record()
    connector.prepare_xds_readv_dma(tensor.device)
    assert done.query()
    connector.finish_xds_readv_dma(tensor.device)
    assert torch.all(tensor == 7)


def test_cuda_registered_geometry_rejects_in_place_metadata_changes() -> None:
    """A retained tensor object must not bypass registration validation."""
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2

    tensor = torch.empty((2, 8, 16, 2, 128), dtype=torch.float16, device="cuda")
    connector = VLLMPagedMemGPUConnectorV2(hidden_dim_size=256, num_layers=1)
    pool = connector.initialize_xds_readv_pool([tensor])
    assert pool.geometry is not None
    tensor.transpose_(3, 4)
    with pytest.raises(RuntimeError, match="replace"):
        pool.validate_storage([tensor])


def test_cuda_registered_geometry_is_reused_with_fresh_slot_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated restores reuse fixed geometry but still reject aliased slots."""
    from lmcache.v1.gpu_connector import gpu_connectors
    from lmcache.v1.gpu_connector import xds_readv

    tensors = [torch.empty((2, 8, 16, 2, 128), dtype=torch.float16, device="cuda")]
    connector = gpu_connectors.VLLMPagedMemGPUConnectorV2(256, 1)
    pool = connector.initialize_xds_readv_pool(tensors)
    pool.buffer_refs = (123,)

    def unexpected(*args, **kwargs):
        raise AssertionError("fixed geometry must not be rediscovered per request")

    monkeypatch.setattr(gpu_connectors, "discover_gpu_kv_format", unexpected)
    monkeypatch.setattr(xds_readv, "validate_paged_readv_pool", unexpected)
    chunk = (
        [torch.Size([2, 1, 32, 256])],
        [torch.float16],
        MemoryFormat.KV_2LTD,
        0,
        32,
    )
    plans = connector.plan_xds_readv_batch(
        [chunk],
        4096,
        kvcaches=tensors,
        slot_mapping_cpu=torch.arange(32),
        xds_compact_batch=True,
    )
    assert plans is not None and plans.geometry is pool.geometry
    assert (
        connector.plan_xds_readv_batch(
            [chunk],
            4096,
            kvcaches=tensors,
            slot_mapping_cpu=torch.arange(16).repeat(2),
        )
        is None
    )
