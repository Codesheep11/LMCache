# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.transfer_result import BackendTransferResult


def test_backend_transfer_result_counts_failed_chunks():
    result = BackendTransferResult(
        completed=(True, False),
        requested_bytes=8192,
        completed_bytes=4096,
        device="cuda:0",
        window_size=2,
        window_count=1,
        peak_staging_bytes=8192,
        storage_seconds=0.1,
        gpu_seconds=0.0,
        overlap_seconds=0.0,
        bank_wait_seconds=0.0,
        short_io_chunks=1,
        timeout_chunks=0,
    )
    assert result.failed_chunks == 1


def test_backend_transfer_result_rejects_impossible_byte_count():
    with pytest.raises(ValueError, match="completed bytes exceed"):
        BackendTransferResult(
            completed=(True,),
            requested_bytes=4096,
            completed_bytes=8192,
            device="cuda:0",
            window_size=1,
            window_count=1,
            peak_staging_bytes=4096,
            storage_seconds=0.1,
            gpu_seconds=0.0,
            overlap_seconds=0.0,
            bank_wait_seconds=0.0,
            short_io_chunks=0,
            timeout_chunks=0,
        )
