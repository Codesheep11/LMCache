# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for LMCache storage backends backed by an xDS service."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

# Third Party
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata


@dataclass
class XDSBlobMetadata:
    """LMCache metadata associated with one blob owned by the xDS service."""

    blob_handle: int
    size: int
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]
    fmt: MemoryFormat
    pin_count: int = 0

    def pin(self) -> bool:
        self.pin_count += 1
        return True

    def unpin(self) -> bool:
        if self.pin_count == 0:
            return False
        self.pin_count -= 1
        return True

    @property
    def is_pinned(self) -> bool:
        return self.pin_count > 0


def align_to_io_unit(size: int, io_unit_size: int) -> int:
    """Round ``size`` up to the xDS service I/O unit."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if io_unit_size <= 0:
        raise ValueError(f"io_unit_size must be positive, got {io_unit_size}")
    return ((size + io_unit_size - 1) // io_unit_size) * io_unit_size


def get_chunk_size_bytes(
    metadata: LMCacheMetadata,
    use_layerwise: bool = False,
) -> int:
    """Return the largest logical object submitted for one cache chunk."""
    shapes = metadata.get_shapes(metadata.chunk_size)
    dtypes = metadata.get_dtypes()
    if len(shapes) != len(dtypes):
        raise ValueError(
            "LMCache metadata returned different shape and dtype counts: "
            f"{len(shapes)} != {len(dtypes)}"
        )
    sizes = [
        shape.numel() * torch.empty((), dtype=dtype).element_size()
        for shape, dtype in zip(shapes, dtypes, strict=True)
    ]
    if not use_layerwise:
        return sum(sizes)

    layer_groups = metadata.kv_layer_groups_manager.kv_layer_groups
    if layer_groups:
        return max(
            size // group.num_layers
            for size, group in zip(sizes, layer_groups, strict=True)
        )

    num_layers = metadata.kv_shape[0]
    if num_layers <= 0 or sizes[0] % num_layers:
        raise ValueError(
            "LMCache layerwise metadata must have a positive, evenly sized "
            f"layer dimension, got num_layers={num_layers}, size={sizes[0]}"
        )
    return sizes[0] // num_layers


def _validate_service_metadata(
    client: Any,
    metadata: LMCacheMetadata,
    require_direct: bool,
    use_layerwise: bool = False,
) -> None:
    service_metadata = getattr(client, "metadata", None)
    if service_metadata is None:
        raise RuntimeError("xDS service did not return metadata")

    io_unit_size = int(service_metadata.io_unit_size)
    blob_size = int(service_metadata.blob_size_bytes)
    capacity = int(service_metadata.capacity_bytes)
    queue_id = int(service_metadata.queue_id)
    worker_id = int(service_metadata.worker_id)

    if io_unit_size <= 0:
        raise ValueError(f"xDS io_unit_size must be positive, got {io_unit_size}")
    if blob_size <= 0 or blob_size % io_unit_size != 0:
        raise ValueError(
            "xDS blob_size_bytes must be positive and I/O-unit aligned, "
            f"got blob_size={blob_size}, io_unit_size={io_unit_size}"
        )
    if capacity < blob_size:
        raise ValueError(
            "xDS capacity_bytes must hold at least one blob, "
            f"got capacity={capacity}, blob_size={blob_size}"
        )
    if queue_id < 0 or worker_id < 0:
        raise ValueError(
            "xDS service returned an invalid queue/worker binding: "
            f"queue_id={queue_id}, worker_id={worker_id}"
        )

    logical_chunk_size = get_chunk_size_bytes(metadata, use_layerwise=use_layerwise)
    required_blob_size = align_to_io_unit(logical_chunk_size, io_unit_size)
    if blob_size < required_blob_size:
        raise ValueError(
            "xDS blob is smaller than one LMCache chunk: "
            f"blob_size={blob_size}, required={required_blob_size}"
        )

    if require_direct:
        gpu_buffer_size = int(service_metadata.gpu_buffer_size_bytes)
        if gpu_buffer_size < blob_size or gpu_buffer_size % io_unit_size != 0:
            raise ValueError(
                "xDS direct GPU buffer must hold at least one aligned blob, "
                f"got gpu_buffer_size={gpu_buffer_size}, blob_size={blob_size}"
            )


def connect_xds_client(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    connect_fn: Optional[Callable[..., Any]] = None,
) -> Any:
    """Connect this LMCache rank to the configured external xDS service."""
    config.validate()
    service_name = config.xds_service_name
    if not service_name:
        raise ValueError("xds_service_name must be set")

    require_direct = bool(config.xds_backend)
    if connect_fn is None:
        # Import lazily so normal LMCache users do not need the xDS SDK.
        import xds

        connect_fn = xds.connect

    try:
        client = connect_fn(
            service_name,
            metadata.worker_id,
            metadata.world_size,
            require_direct,
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to connect to xDS service {service_name!r} "
            f"for rank {metadata.worker_id}"
        ) from exc

    try:
        _validate_service_metadata(
            client,
            metadata,
            require_direct,
            use_layerwise=config.use_layerwise,
        )
    except Exception:
        close = getattr(client, "close", None)
        if callable(close):
            close()
        raise
    return client
