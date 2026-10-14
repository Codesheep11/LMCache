# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.gpu_connector.gpu_connectors import SGLangGPUConnector


class DummyConnector(GPUConnectorInterface):
    def __init__(self) -> None:
        self.load_stream = object()
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
    event = FakeEvent()
    event_kwargs = {}

    def make_event(**kwargs):
        event_kwargs.update(kwargs)
        return event

    monkeypatch.setattr(torch.cuda, "Event", make_event)

    result = connector.record_transfer_event(direction)

    assert result is event
    expected_stream = (
        connector.load_stream if direction == "load" else connector.store_stream
    )
    assert event.recorded_stream is expected_stream
    assert event_kwargs == {"enable_timing": False}


def test_record_transfer_event_can_enable_timing(monkeypatch):
    connector = DummyConnector()
    event = FakeEvent()
    event_kwargs = {}

    def make_event(**kwargs):
        event_kwargs.update(kwargs)
        return event

    monkeypatch.setattr(torch.cuda, "Event", make_event)

    assert connector.record_transfer_event("load", enable_timing=True) is event
    assert event_kwargs == {"enable_timing": True}


def test_record_transfer_event_rejects_unknown_direction():
    connector = DummyConnector()

    with pytest.raises(ValueError, match="unknown transfer direction"):
        connector.record_transfer_event("copy")


def test_set_transfer_stream_priority_reuses_matching_stream(monkeypatch):
    connector = DummyConnector()
    created_streams = []

    def make_stream(*, priority):
        stream = ("stream", priority)
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(torch.cuda, "Stream", make_stream)

    connector.set_transfer_stream_priority("load", -1)
    first_stream = connector.load_stream
    connector.set_transfer_stream_priority("load", -1)

    assert first_stream == ("stream", -1)
    assert connector.load_stream is first_stream
    assert created_streams == [first_stream]


def test_sglang_connector_initializes_store_stream(monkeypatch):
    store_stream = object()
    monkeypatch.setattr(torch.cuda, "Stream", lambda: store_stream)

    connector = SGLangGPUConnector(hidden_dim_size=4, num_layers=2)

    assert connector.store_stream is store_stream
