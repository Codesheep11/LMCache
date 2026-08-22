# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.transfer_result import (
    BackendTransferResult,
    BackendWriteSubmissionResult,
)


def _valid_result_values():
    return {
        "completed": (True, False),
        "requested_bytes": 8192,
        "completed_bytes": 4096,
        "device": "cuda:0",
        "window_size": 2,
        "window_count": 1,
        "peak_staging_bytes": 8192,
        "storage_seconds": 0.1,
        "gpu_seconds": 0.0,
        "overlap_seconds": 0.0,
        "bank_wait_seconds": 0.0,
        "short_io_chunks": 1,
        "timeout_chunks": 0,
    }


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


def test_backend_transfer_result_accepts_bounded_io_metrics():
    values = _valid_result_values()
    values.update(
        logical_chunk_count=5,
        read_slot_count=2,
        write_slot_count=8,
        io_queue_depth=1,
        peak_inflight_io=1,
        slot_wait_seconds=0.25,
    )

    result = BackendTransferResult(**values)

    assert result.logical_chunk_count == 5
    assert result.read_slot_count == 2
    assert result.write_slot_count == 8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logical_chunk_count", -1),
        ("read_slot_count", -1),
        ("write_slot_count", -1),
        ("io_queue_depth", -1),
        ("peak_inflight_io", -1),
        ("slot_wait_seconds", -0.1),
    ],
)
def test_backend_transfer_result_rejects_negative_bounded_io_metrics(field, value):
    values = _valid_result_values()
    values[field] = value

    with pytest.raises(ValueError, match="bounded I/O metrics"):
        BackendTransferResult(**values)


def test_backend_write_submission_result_validates_admission_metrics():
    result = BackendWriteSubmissionResult(
        accepted=(True, False),
        accepted_bytes=4096,
        durable=False,
        staging_seconds=0.1,
        backpressure_seconds=0.2,
        queue_depth=1,
        outstanding_bytes=4096,
    )

    assert result.accepted == (True, False)
    assert result.accepted_bytes == 4096
    assert result.durable is False

    with pytest.raises(ValueError, match="write submission metrics"):
        BackendWriteSubmissionResult(
            accepted=(True,),
            accepted_bytes=-1,
            durable=False,
            staging_seconds=0.0,
            backpressure_seconds=0.0,
            queue_depth=0,
            outstanding_bytes=0,
        )
