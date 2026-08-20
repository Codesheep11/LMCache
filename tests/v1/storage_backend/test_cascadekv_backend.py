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
from lmcache.v1.storage_backend.cascadekv_backend import CascadeKVBackend


class FakeXDSClient:
    def __init__(self):
        self.metadata = SimpleNamespace(
            io_unit_size=4096,
            blob_size_bytes=4096,
        )
        self.alloc_sizes = []
        self.free_refs = []
        self._buffers = {}

    def alloc_buffer(self, size):
        self.alloc_sizes.append(size)
        buffer_ref = 101
        view = memoryview(bytearray(size))
        self._buffers[buffer_ref] = view
        return view, buffer_ref

    def free_buffer(self, buffer_ref):
        self.free_refs.append(buffer_ref)
        del self._buffers[buffer_ref]


@pytest.fixture
def cascade_config():
    config = LMCacheEngineConfig.from_defaults()
    config.xds_service_name = "xds0"
    config.cascadekv_backend = True
    config.cascadekv_slot_count = 2
    return config


@pytest.fixture
def cascade_metadata():
    return LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(1, 2, 4, 1, 8),
        chunk_size=4,
    )


def test_cascadekv_allocator_wraps_service_cpu_buffer_without_copy(
    cascade_config,
    cascade_metadata,
):
    fake_xds_client = FakeXDSClient()
    backend = CascadeKVBackend(
        cascade_config,
        cascade_metadata,
        fake_xds_client,
        dst_device="cpu",
    )

    obj = backend.allocate(
        cascade_metadata.get_shapes(),
        cascade_metadata.get_dtypes(),
        MemoryFormat.KV_2LTD,
    )

    assert obj is not None
    assert obj.raw_data.data_ptr() >= backend.staging_base_ptr
    assert obj.raw_data.data_ptr() < backend.staging_base_ptr + backend.staging_size
    assert fake_xds_client.alloc_sizes == [backend.staging_size]

    obj.ref_count_down()
    backend.close()
    assert fake_xds_client.free_refs == [backend.staging_buffer_ref]


def test_cascadekv_close_rejects_live_staging_objects(
    cascade_config,
    cascade_metadata,
):
    fake_xds_client = FakeXDSClient()
    backend = CascadeKVBackend(
        cascade_config,
        cascade_metadata,
        fake_xds_client,
        dst_device="cpu",
    )
    obj = backend.allocate(
        cascade_metadata.get_shapes(),
        cascade_metadata.get_dtypes(),
    )
    assert obj is not None

    with pytest.raises(RuntimeError, match="active staging allocations"):
        backend.close()
    assert fake_xds_client.free_refs == []

    obj.ref_count_down()
    backend.close()
    backend.close()
    assert fake_xds_client.free_refs == [backend.staging_buffer_ref]
