# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.xds_utils import (
    XDSBlobMetadata,
    align_to_io_unit,
    connect_xds_client,
)


def _chunk_size_bytes(metadata) -> int:
    return sum(
        shape.numel() * dtype.itemsize
        for shape, dtype in zip(
            metadata.get_shapes(metadata.chunk_size),
            metadata.get_dtypes(),
            strict=True,
        )
    )


def test_align_to_io_unit_rejects_non_positive_values():
    assert align_to_io_unit(4097, 4096) == 8192
    with pytest.raises(ValueError, match="size must be positive"):
        align_to_io_unit(0, 4096)
    with pytest.raises(ValueError, match="io_unit_size must be positive"):
        align_to_io_unit(4096, 0)


def test_blob_metadata_pin_count_is_balanced():
    metadata = XDSBlobMetadata(
        blob_handle=11,
        size=4096,
        shapes=[torch.Size([2, 4])],
        dtypes=[torch.float16],
        fmt=MemoryFormat.KV_2LTD,
    )

    assert metadata.is_pinned is False
    assert metadata.pin() is True
    assert metadata.pin() is True
    assert metadata.is_pinned is True
    assert metadata.unpin() is True
    assert metadata.unpin() is True
    assert metadata.unpin() is False
    assert metadata.is_pinned is False


def test_connect_xds_client_binds_rank_and_service(lmcache_engine_metadata):
    calls = []
    config = LMCacheEngineConfig.from_defaults()
    config.xds_service_name = "xds0"
    config.cascadekv_backend = True
    config.use_layerwise = False
    logical_size = _chunk_size_bytes(lmcache_engine_metadata)
    blob_size = align_to_io_unit(logical_size, 4096)

    def connect(service_name, rank_id, world_size, require_direct):
        calls.append((service_name, rank_id, world_size, require_direct))
        return SimpleNamespace(
            metadata=SimpleNamespace(
                queue_id=rank_id,
                worker_id=rank_id,
                io_unit_size=4096,
                blob_size_bytes=blob_size,
                capacity_bytes=blob_size * 32,
                gpu_buffer_size_bytes=blob_size * 8,
            )
        )

    client = connect_xds_client(
        config,
        lmcache_engine_metadata,
        connect_fn=connect,
    )

    assert client.metadata.queue_id == lmcache_engine_metadata.worker_id
    assert calls == [
        (
            "xds0",
            lmcache_engine_metadata.worker_id,
            lmcache_engine_metadata.world_size,
            False,
        )
    ]


def test_non_layerwise_cascadekv_validates_one_full_block(
    lmcache_engine_metadata,
):
    """Non-layerwise metadata already includes every layer exactly once."""
    config = LMCacheEngineConfig.from_defaults()
    config.xds_service_name = "xds0"
    config.cascadekv_backend = True
    config.use_layerwise = False
    blob_size = align_to_io_unit(_chunk_size_bytes(lmcache_engine_metadata), 4096)

    def connect(*_args):
        return SimpleNamespace(
            metadata=SimpleNamespace(
                queue_id=0,
                worker_id=0,
                io_unit_size=4096,
                blob_size_bytes=blob_size,
                capacity_bytes=blob_size * 32,
                gpu_buffer_size_bytes=0,
            )
        )

    client = connect_xds_client(config, lmcache_engine_metadata, connect_fn=connect)
    assert client.metadata.blob_size_bytes == blob_size


def test_connect_xds_client_rejects_undersized_blob(lmcache_engine_metadata):
    config = LMCacheEngineConfig.from_defaults()
    config.xds_service_name = "xds0"
    config.xds_backend = True
    config.xds_require_direct = True
    logical_size = _chunk_size_bytes(lmcache_engine_metadata)

    def connect(*_args):
        return SimpleNamespace(
            metadata=SimpleNamespace(
                queue_id=0,
                worker_id=0,
                io_unit_size=4096,
                blob_size_bytes=logical_size - 4096,
                capacity_bytes=logical_size * 2,
                gpu_buffer_size_bytes=logical_size * 2,
            )
        )

    with pytest.raises(ValueError, match="smaller than one LMCache chunk"):
        connect_xds_client(config, lmcache_engine_metadata, connect_fn=connect)


def test_layerwise_xds_validates_one_layer_per_blob():
    metadata = LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 32, 8, 128),
        chunk_size=32,
    )
    config = LMCacheEngineConfig.from_defaults()
    config.xds_service_name = "xds0"
    config.xds_backend = True
    config.xds_require_direct = True
    config.use_layerwise = True

    def connect(*_args):
        return SimpleNamespace(
            metadata=SimpleNamespace(
                queue_id=0,
                worker_id=0,
                io_unit_size=4096,
                blob_size_bytes=1024 * 1024,
                capacity_bytes=64 * 1024 * 1024,
                gpu_buffer_size_bytes=64 * 1024 * 1024,
            )
        )

    client = connect_xds_client(config, metadata, connect_fn=connect)
    assert client.metadata.blob_size_bytes == 1024 * 1024
