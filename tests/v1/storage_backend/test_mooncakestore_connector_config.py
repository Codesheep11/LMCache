# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# First Party
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
)


def test_mooncake_store_config_enables_rank_separated_ssd_offload() -> None:
    config = SimpleNamespace(
        extra_config={
            "local_hostname": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8005/metadata",
            "master_server_address": "127.0.0.1:50051",
            "mooncake_enable_ssd_offload": True,
            "mooncake_ssd_offload_path": "/mnt/cache/mooncake",
            "mooncake_separate_ssd_path_by_rank": True,
            "mooncake_tenant_id": "offline-test",
            "benchmark_transfer_metrics": True,
        }
    )

    result = MooncakeStoreConfig.load_from_lmcache_config(config)

    assert result.enable_ssd_offload is True
    assert result.ssd_offload_path == "/mnt/cache/mooncake"
    assert result.separate_ssd_path_by_rank is True
    assert result.tenant_id == "offline-test"
    assert result.benchmark_transfer_metrics is True


def test_mooncake_store_config_keeps_ssd_offload_disabled_by_default() -> None:
    config = SimpleNamespace(
        extra_config={
            "local_hostname": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8005/metadata",
            "master_server_address": "127.0.0.1:50051",
        }
    )

    result = MooncakeStoreConfig.load_from_lmcache_config(config)

    assert result.enable_ssd_offload is False
    assert result.ssd_offload_path == ""
    assert result.separate_ssd_path_by_rank is True
