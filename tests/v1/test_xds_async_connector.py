# SPDX-License-Identifier: Apache-2.0
"""Scheduler contract for reserved asynchronous xDS destinations."""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata, LoadSpec
from lmcache.integration.vllm.xds_async_connector import XDSAsyncConnector
from lmcache.v1.config import LMCacheEngineConfig


class FakeAdapter:
    def __init__(self) -> None:
        self.config = LMCacheEngineConfig.from_defaults()
        self.config.xds_backend = True
        self.load_specs: dict[str, LoadSpec] = {}
        self.hit: int | None = 32

    def get_num_new_matched_tokens(self, request: Any, computed: int) -> int | None:
        self.load_specs[request.request_id] = LoadSpec(computed, self.hit or 0, False)
        return None if self.hit is None else max(0, self.hit - computed)

    def update_state_after_alloc(self, request: Any, count: int) -> None:
        self.load_specs[request.request_id].can_load = count > 0

    def build_connector_meta(self, output: Any) -> LMCacheConnectorMetadata:
        return LMCacheConnectorMetadata()


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch) -> tuple[XDSAsyncConnector, FakeAdapter]:
    adapter = FakeAdapter()

    # Replace service startup only; exercise the connector's public scheduler API.
    def init(self: Any, config: Any, role: Any, cache: Any = None) -> None:
        self._lmcache_engine = adapter

    monkeypatch.setattr(LMCacheConnectorV1Dynamic, "__init__", init)
    connector = XDSAsyncConnector(
        SimpleNamespace(cache_config=SimpleNamespace(block_size=16)), None
    )
    return connector, adapter


def request() -> Any:
    return SimpleNamespace(
        request_id="r", all_token_ids=list(range(48)), sampling_params=None
    )


@pytest.mark.parametrize(
    "hit,expected", [(None, (None, False)), (0, (0, False)), (32, (32, True))]
)
def test_async_flag_only_for_positive_hits(
    scheduler: Any, hit: Any, expected: Any
) -> None:
    connector, adapter = scheduler
    adapter.hit = hit
    assert connector.get_num_new_matched_tokens(request(), 0) == expected


def test_reserved_blocks_survive_abort_and_are_dispatched_once(scheduler: Any) -> None:
    connector, adapter = scheduler
    req = request()
    assert connector.get_num_new_matched_tokens(req, 0) == (32, True)
    blocks = SimpleNamespace(get_block_ids=lambda: ([4, 1],))
    connector.update_state_after_alloc(req, blocks, 32)
    # Even an abort must dispatch/drain the load: vLLM holds these blocks until
    # finished_recving arrives, so silently dropping metadata would leak them.
    output = SimpleNamespace(finished_req_ids={"r"})
    meta = connector.build_connector_meta(output)
    assert len(meta.loads) == 1
    load = meta.loads[0]
    assert load.slot_mapping.tolist() == list(range(64, 80)) + list(range(16, 32))
    assert load.token_ids == list(range(32))
    assert load.load_spec.can_load
    assert not adapter.load_specs["r"].can_load
    assert connector.build_connector_meta(output).loads == []
    # Resume allocates remaining prompt blocks, but must not repeat the SSD read.
    connector.update_state_after_alloc(req, blocks, 0)
    assert connector.build_connector_meta(output).loads == []


@pytest.mark.parametrize("groups", [([1],), ([1, 2], [3, 4])])
def test_rejects_incomplete_or_multiple_target_groups(
    scheduler: Any, groups: Any
) -> None:
    connector, _ = scheduler
    req = request()
    connector.get_num_new_matched_tokens(req, 0)
    with pytest.raises(ValueError):
        connector.update_state_after_alloc(
            req, SimpleNamespace(get_block_ids=lambda: groups), 32
        )


def test_multimodal_request_fails_before_lookup(scheduler: Any) -> None:
    connector, adapter = scheduler
    req = request()
    req.mm_features = [object()]
    with pytest.raises(ValueError, match="text-only"):
        connector.get_num_new_matched_tokens(req, 0)
    assert adapter.load_specs == {}


@pytest.mark.parametrize(
    "option", ["use_layerwise", "enable_async_loading", "enable_blending"]
)
def test_incompatible_modes_rejected_before_serving(
    scheduler: Any, option: str
) -> None:
    _, adapter = scheduler
    setattr(adapter.config, option, True)
    with pytest.raises(ValueError, match="direct xDS"):
        XDSAsyncConnector(
            SimpleNamespace(cache_config=SimpleNamespace(block_size=16)), None
        )
