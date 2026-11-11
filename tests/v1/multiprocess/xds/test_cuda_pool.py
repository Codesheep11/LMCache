# SPDX-License-Identifier: Apache-2.0
"""CUDA registration contracts without requiring a GPU or peer-memory driver."""

# Standard
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.xds_cuda_pool import RegisteredCUDAKVPool


def plane(pointer: int = 65536, *, contiguous: bool = True) -> Mock:
    """Create a CUDA view with an explicit owned range and block stride."""
    tensor = Mock(device=torch.device("cuda:0"), ndim=3, shape=(8, 16, 64))
    tensor.is_contiguous.return_value = contiguous
    tensor.numel.return_value = 8192
    tensor.element_size.return_value = 2
    tensor.stride.return_value = 1024
    tensor.data_ptr.return_value = pointer
    return tensor


def test_aliases_register_once_and_close_is_idempotent() -> None:
    client = Mock()
    client.register_gpu_buffer.side_effect = [11, 12]
    tensor = plane()
    pool = RegisteredCUDAKVPool(client, (tensor, tensor, plane(131072)))
    assert client.register_gpu_buffer.call_count == 2
    assert [p.buffer_ref for p in pool.planes] == [11, 11, 12]
    assert [p.block_bytes for p in pool.planes] == [2048] * 3
    pool.close()
    pool.close()
    assert client.unregister_buffer.call_count == 2


def test_registration_failure_drains_already_registered_ranges() -> None:
    client = Mock()
    client.register_gpu_buffer.side_effect = [11, OSError("driver rejected VMM")]
    with pytest.raises(OSError, match="VMM"):
        RegisteredCUDAKVPool(client, (plane(), plane(131072)))
    client.unregister_buffer.assert_called_once_with(11)


def test_close_failure_retains_registration_for_retry() -> None:
    client = Mock()
    client.register_gpu_buffer.return_value = 11
    client.unregister_buffer.side_effect = [OSError("busy"), None]
    pool = RegisteredCUDAKVPool(client, (plane(),))
    with pytest.raises(OSError, match="busy"):
        pool.close()
    pool.close()
    assert client.unregister_buffer.call_count == 2


@pytest.mark.parametrize("invalid", ["empty", "noncontiguous", "cpu", "mixed"])
def test_invalid_pool_performs_no_registration(invalid: str) -> None:
    client = Mock()
    tensors = [plane()]
    if invalid == "empty":
        tensors = []
    elif invalid == "noncontiguous":
        tensors = [plane(contiguous=False)]
    elif invalid == "cpu":
        tensors[0].device = torch.device("cpu")
    else:
        other = plane()
        other.device = torch.device("cuda:1")
        tensors.append(other)
    with pytest.raises(ValueError):
        RegisteredCUDAKVPool(client, tuple(tensors))
    client.register_gpu_buffer.assert_not_called()
