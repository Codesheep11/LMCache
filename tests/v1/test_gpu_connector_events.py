# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.gpu_connector.gpu_connectors import SGLangGPUConnector


class DummyConnector(GPUConnectorInterface):
    def __init__(self) -> None:
        self.store_stream = object()

    def to_gpu(self, memory_obj, start: int, end: int, **kwargs):
        return None

    def from_gpu(self, memory_obj, start: int, end: int, **kwargs):
        return None

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        return None

    def batched_to_gpu(self, memory_objs=None, starts=None, ends=None, **kwargs):
        return None

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens])


class FakeEvent:
    def __init__(self) -> None:
        self.recorded_stream = None

    def record(self, stream) -> None:
        self.recorded_stream = stream


@pytest.mark.parametrize("direction", ["load", "store"])
def test_record_transfer_event_uses_direction_stream(monkeypatch, direction):
    connector = DummyConnector()
    load_stream = object()
    event = FakeEvent()
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: load_stream)

    result = connector.record_transfer_event(direction)

    assert result is event
    expected_stream = load_stream if direction == "load" else connector.store_stream
    assert event.recorded_stream is expected_stream


def test_record_transfer_event_rejects_unknown_direction():
    connector = DummyConnector()

    with pytest.raises(ValueError, match="unknown transfer direction"):
        connector.record_transfer_event("copy")


def test_sglang_connector_initializes_store_stream(monkeypatch):
    store_stream = object()
    monkeypatch.setattr(torch.cuda, "Stream", lambda: store_stream)

    connector = SGLangGPUConnector(hidden_dim_size=4, num_layers=2)

    assert connector.store_stream is store_stream
