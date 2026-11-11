# SPDX-License-Identifier: Apache-2.0
"""Fixed CUDA final-page registrations for worker-owned xDS I/O."""

# Standard
from dataclasses import dataclass
from typing import Protocol

# Third Party
import torch


class RegistrationClient(Protocol):
    """Connection-local registration API implemented by the xDS SDK."""

    def register_gpu_buffer(self, ptr: int, size: int) -> int:
        """Pin an owned CUDA range and return its buffer handle."""
        ...

    def unregister_buffer(self, buffer_ref: int) -> None:
        """Unpin a range after all external DMA and device users have drained."""
        ...


@dataclass(frozen=True)
class RegisteredKVPlane:
    """Block-major plane used to generate SSD scatter descriptors."""

    buffer_ref: int
    block_bytes: int
    num_blocks: int


class KVPool(Protocol):
    """Platform-owned registrations retained until transfer completion."""

    planes: tuple[RegisteredKVPlane, ...]

    def close(self) -> None:
        """Unregister drained ranges, retaining tensors on cleanup failure."""
        ...


class RegisteredCUDAKVPool:
    """Register contiguous CUDA planes without allocating payload buffers.

    Args:
        client: Connected xDS client in the allocation-owning worker.
        tensors: Final KV views, sharing one device and a block-major layout.

    Raises:
        ValueError: Empty, non-CUDA, noncontiguous, or mixed-device planes.
        OSError: The driver's peer-memory registration rejects an allocation.
            CUDA VMM support depends on the deployed driver; no staging fallback
            or allocator replacement is performed here.
    """

    def __init__(
        self, client: RegistrationClient, tensors: tuple[torch.Tensor, ...]
    ) -> None:
        if not tensors or len({t.device for t in tensors}) != 1:
            raise ValueError("CUDA KV pool requires nonempty single-device planes")
        if any(
            t.device.type != "cuda"
            or t.ndim < 2
            or not t.is_contiguous()
            or t.numel() == 0
            for t in tensors
        ):
            raise ValueError("xDS needs contiguous owned CUDA block-major KV planes")
        self._client = client
        self._tensors = tensors
        self._registrations: dict[tuple[int, int], int] = {}
        planes = []
        try:
            for tensor in tensors:
                region = (tensor.data_ptr(), tensor.numel() * tensor.element_size())
                if region not in self._registrations:
                    self._registrations[region] = client.register_gpu_buffer(*region)
                planes.append(
                    RegisteredKVPlane(
                        self._registrations[region],
                        tensor.stride(0) * tensor.element_size(),
                        tensor.shape[0],
                    )
                )
        except (RuntimeError, OSError):
            self.close()
            raise
        self.planes = tuple(planes)

    def close(self) -> None:
        """Unregister a drained pool before dropping its owning tensor references."""
        for region, handle in tuple(self._registrations.items()):
            self._client.unregister_buffer(handle)
            del self._registrations[region]
        self._tensors = ()
