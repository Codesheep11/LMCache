# SPDX-License-Identifier: Apache-2.0
"""Direct layer readiness preserves the engine's public generator protocol."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig


@pytest.mark.parametrize("cancel", [False, True])
def test_direct_layerwise_engine_yields_ready_layers_and_closes(cancel: bool) -> None:
    config = LMCacheEngineConfig.from_defaults()
    config.xds_backend = True
    config.use_layerwise = True
    config.extra_config = {
        "xds_layerwise_block_blob": True,
        "xds_layerwise_readv": True,
    }
    engine = MagicMock()
    engine.config = config
    engine.num_layers = 2
    engine.is_healthy.return_value = True
    engine.retrieve_locations = ["XDSBackend"]
    key = CacheEngineKey("test", 1, 0, 1, torch.float16)
    engine.token_database.process_tokens.return_value = [(0, 32, key)]
    engine.storage_manager.contains.return_value = "XDSBackend"
    events = []

    def layers():
        try:
            for layer in range(2):
                events.append(layer)
                yield SimpleNamespace(completed=(True,), completed_bytes=32768)
        finally:
            events.append("closed")

    engine.storage_manager.prepare_layerwise_readv.return_value = layers()
    iterator = LMCacheEngine.retrieve_layer(engine, list(range(32)))
    assert next(iterator).item() == 32
    assert next(iterator) is None and events == [0]
    if cancel:
        iterator.close()
        assert events == [0, "closed"]
    else:
        assert next(iterator) is None and events == [0, 1]
        mask = next(iterator)
        assert mask.dtype == torch.bool and mask.all()
        assert events == [0, 1, "closed"]
        with pytest.raises(StopIteration):
            next(iterator)
        engine.stats_monitor.on_retrieve_finished.assert_called_once()
    engine.storage_manager.blockwise_batched_get.assert_not_called()
    engine.gpu_connector.batched_to_gpu.assert_not_called()
