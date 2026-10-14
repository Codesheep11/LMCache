# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector


class _TestConnector(RemoteConnector):
    async def exists(self, key):
        raise NotImplementedError

    def exists_sync(self, key):
        raise NotImplementedError

    async def get(self, key):
        raise NotImplementedError

    async def put(self, key, memory_obj):
        raise NotImplementedError

    async def list(self):
        raise NotImplementedError

    async def close(self):
        raise NotImplementedError


def test_reshape_partial_chunk_updates_mla_group_sizes() -> None:
    full_shape = torch.Size([1, 27, 256, 576])
    full_bytes = full_shape.numel() * torch.bfloat16.itemsize
    partial_tokens = 160
    partial_bytes = full_bytes * partial_tokens // 256

    metadata = MemoryObjMetadata(
        shape=full_shape,
        dtype=torch.bfloat16,
        address=0,
        phy_size=full_bytes,
        ref_count=1,
        fmt=MemoryFormat.KV_MLA_FMT,
        shapes=[full_shape],
        dtypes=[torch.bfloat16],
    )
    memory_obj = TensorMemoryObj(
        raw_data=torch.empty(full_bytes, dtype=torch.uint8),
        metadata=metadata,
        parent_allocator=None,
    )

    connector = object.__new__(_TestConnector)
    connector.full_chunk_size_bytes = full_bytes
    connector.single_token_size = full_bytes // 256

    result = connector.reshape_partial_chunk(memory_obj, partial_bytes)

    assert result.meta.shape == torch.Size([1, 27, partial_tokens, 576])
    assert result.meta.shapes == [torch.Size([1, 27, partial_tokens, 576])]
    assert result.get_size() == partial_bytes
    assert result.raw_tensor is not None
    assert result.raw_tensor.numel() == partial_bytes
    assert result.tensor is not None
    assert result.tensor.shape == result.meta.shape
