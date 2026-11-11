# SPDX-License-Identifier: Apache-2.0
"""CUDA transport routing and final-view layout contracts without GPU hardware."""

# Standard
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess import xds_transfer
from lmcache.v1.multiprocess.transfer_context import worker_transfer


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> xds_transfer.XDSWorkerTransferContext:
    """Construct the real CUDA worker transport through its public factory."""
    monkeypatch.setattr(xds_transfer.xds, "XDSClient", Mock(return_value=Mock()))
    monkeypatch.setattr(
        worker_transfer, "get_device", lambda value: torch.device("cuda:0")
    )
    result = worker_transfer.create_transfer_context(
        {"layer": torch.empty((8, 16, 64), device="cpu")},
        instance_id=1,
        req_client=Mock(),
        mode="xds_worker",
    )
    assert type(result) is xds_transfer.XDSWorkerTransferContext
    return result


def test_mla_fused_plane_is_retained_without_copy(
    context: xds_transfer.XDSWorkerTransferContext,
) -> None:
    tensor = torch.empty((8, 16, 576), device="cpu")
    planes = context.prepare_planes({"layer": tensor})
    assert len(planes) == 1
    assert planes[0] is tensor
    context.close()


def test_dense_planes_are_storage_views(
    context: xds_transfer.XDSWorkerTransferContext,
) -> None:
    tensor = torch.empty((2, 8, 16, 4, 64), device="cpu")
    planes = context.prepare_planes({"layer": tensor})
    assert len(planes) == 2
    assert all(p.is_contiguous() for p in planes)
    assert all(
        p.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for p in planes
    )
    context.close()


def test_unsupported_layout_fails_without_repacking(
    context: xds_transfer.XDSWorkerTransferContext,
) -> None:
    with pytest.raises(ValueError, match="block-major"):
        context.prepare_planes({"layer": torch.empty((8, 16), device="cpu")})
    context.close()
