# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional
import asyncio

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import MemoryAllocatorInterface
from lmcache.v1.spdk_utils import init_spdk_if_needed
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.gds_backend import GdsBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.weka_gds_backend import WekaGdsBackend
from lmcache.v1.storage_backend.spdk_backend import SpdkBlobBackend
from lmcache.v1.storage_backend.xds_backend import XDSBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


def _spdk_enable_direct_p2p(config: LMCacheEngineConfig) -> bool:
    return config.xds_enable_direct

def CreateStorageBackends(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    loop: asyncio.AbstractEventLoop,
    memory_allocator: MemoryAllocatorInterface,
    dst_device: str = "cuda",
    lmcache_worker: Optional["LMCacheWorker"] = None,
    lookup_server: Optional[LookupServerInterface] = None,
) -> OrderedDict[str, StorageBackendInterface]:
    # Replace 'cuda' with 'cuda:<device id>'
    if dst_device == "cuda":
        dst_device = f"cuda:{torch.cuda.current_device()}"

    storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()
    local_cpu_backend: Optional[LocalCPUBackend] = None

    if config.enable_nixl:
        if config.enable_xpyd:
            # First Party
            from lmcache.v1.storage_backend.nixl_backend_v3 import NixlBackend

            storage_backends["NixlBackend"] = NixlBackend.CreateNixlBackend(
                config, metadata, memory_allocator
            )
        else:
            # First Party
            from lmcache.v1.storage_backend.nixl_backend import NixlBackend

            storage_backends["NixlBackend"] = NixlBackend.CreateNixlBackend(
                config, metadata
            )
            assert config.nixl_buffer_device is not None

    # TODO(Jiayi): The hierarchy is fixed for now
    # NOTE(Jiayi): The local_cpu backend is always created because
    # other backends might need it as a buffer.
    if (config.enable_nixl and not config.local_cpu) or (
        _spdk_enable_direct_p2p(config) and not config.local_cpu
    ):
        pass
    else:
        local_cpu_backend = LocalCPUBackend(
            config,
            memory_allocator,
            lookup_server,
            lmcache_worker,
        )
        backend_name = str(local_cpu_backend)
        storage_backends[backend_name] = local_cpu_backend

    if config.local_disk and config.max_local_disk_size > 0:
        assert local_cpu_backend is not None
        local_disk_backend = LocalDiskBackend(
            config,
            loop,
            local_cpu_backend,
            dst_device,
            lmcache_worker,
            lookup_server,
        )

        backend_name = str(local_disk_backend)
        storage_backends[backend_name] = local_disk_backend

    if (
        config.bdev_name is not None
        and config.xds_max_size is not None
        and config.xds_max_size > 0
    ):
        init_spdk_if_needed(config)
        if _spdk_enable_direct_p2p(config):
            spdk_backend = XDSBackend(
                config,
                loop,
                memory_allocator,
                dst_device,
                lmcache_worker,
                lookup_server,
            )
        else:
            assert local_cpu_backend is not None
            spdk_backend = SpdkBlobBackend(
                config,
                loop,
                local_cpu_backend,
                dst_device,
                lmcache_worker,
                lookup_server,
            )
        backend_name = str(spdk_backend)
        storage_backends[backend_name] = spdk_backend

    if config.weka_path is not None:
        weka_backend = WekaGdsBackend(config, loop, memory_allocator, dst_device)
        # TODO(Serapheim): there's a chance we don't want the local
        # CPU cache in front of ours. Let's experiment and potentially
        # change that in the future.
        storage_backends[str(weka_backend)] = weka_backend
    if config.gds_path is not None:
        gds_backend = GdsBackend(config, loop, memory_allocator, dst_device)
        storage_backends[str(gds_backend)] = gds_backend
    if config.remote_url is not None:
        assert local_cpu_backend is not None
        remote_backend = RemoteBackend(
            config, metadata, loop, local_cpu_backend, dst_device, lookup_server
        )
        backend_name = str(remote_backend)
        storage_backends[backend_name] = remote_backend

    return storage_backends
