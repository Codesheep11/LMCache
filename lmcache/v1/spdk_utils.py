import spdk_controller as spdk

from lmcache.v1.config import LMCacheEngineConfig


def init_spdk_if_needed(config: LMCacheEngineConfig) -> None:
    if config.bdev_name is None:
        return

    if getattr(spdk.engine, "initialized", False):
        return

    spdk.init(
        config.bdev_name,
        config.json_config_file,
        config.reactor_mask,
        config.main_core,
        config.rpc_addr,
        peer_bdf=config.xds_peer_bdfs if config.xds_enable_direct else None,
        cluster_size_bytes=(
            1024 * 1024
            if config.xds_cluster_size_kb is None
            else config.xds_cluster_size_kb * 1024
        ),
    )


def get_spdk_io_unit_size() -> int:
    io_unit_size = spdk.get_io_unit_size()
    if io_unit_size <= 0:
        raise RuntimeError(
            "SPDK io_unit_size is unavailable. Make sure SPDK is initialized first."
        )
    return io_unit_size


def align_size_to_io_unit(size_in_bytes: int) -> int:
    if size_in_bytes <= 0:
        raise ValueError(
            f"size_in_bytes must be a positive integer, but got {size_in_bytes}"
        )

    io_unit_size = get_spdk_io_unit_size()
    return ((size_in_bytes + io_unit_size - 1) // io_unit_size) * io_unit_size
