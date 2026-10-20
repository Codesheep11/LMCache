# SPDX-License-Identifier: Apache-2.0
"""Allocation guard for xDS I/O directly against the serving KV pool."""

# Standard
from typing import Any, Optional

# First Party
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj


class XDSDirectKVAllocator(MemoryAllocatorInterface):
    """Own no payload memory and reject staging/fallback allocation requests.

    Only direct backend transfer APIs are valid with this allocator. Metadata
    and I/O descriptors still use ordinary host memory.
    """

    num_active_allocations = 0
    total_allocated_size = 0

    def allocate(self, *args: Any, **kwargs: Any) -> Optional[MemoryObj]:
        """Reject a payload allocation instead of silently staging KV bytes."""
        raise RuntimeError("xDS direct KV mode forbids staging allocations")

    def batched_allocate(self, *args: Any, **kwargs: Any) -> list[MemoryObj]:
        """Reject batched payload allocations."""
        raise RuntimeError("xDS direct KV mode forbids staging allocations")

    def free(self, *args: Any, **kwargs: Any) -> None:
        """Reject objects that cannot have originated from this allocator."""
        raise RuntimeError("xDS direct KV allocator owns no payload objects")

    def batched_free(self, *args: Any, **kwargs: Any) -> None:
        """Reject objects that cannot have originated from this allocator."""
        raise RuntimeError("xDS direct KV allocator owns no payload objects")

    def memcheck(self) -> bool:
        """Return True because this allocator never owns payload memory."""
        return True
