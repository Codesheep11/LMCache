import json

from benchmarks.xds_e2e.analyze_cascadekv_io_profile import (
    IO_MARKER,
    build_report,
    marker_records,
)


def test_marker_records_ignores_logger_suffix(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(f"prefix {IO_MARKER}{json.dumps({'io_count': 1})} suffix\n")

    assert marker_records(log, IO_MARKER) == [{"io_count": 1}]


def test_build_report_combines_io_layer_and_service_stages():
    summary = {
        "count": 2,
        "mean": 10.0,
        "p50": 9.0,
        "p95": 15.0,
        "p99": 15.0,
        "max": 15.0,
    }
    io_record = {
        "operation": "read",
        "io_count": 2,
        "bytes": 8192,
        "task_ms": 1.0,
        "engine_queue_us": summary,
        "native_submit_us": summary,
        "submit_to_python_us": summary,
        "submit_to_service_publish_us": summary,
        "completion_ring_us": summary,
        "dequeue_to_python_us": summary,
        "layer_callback_us": summary,
    }
    timeline = {
        "layers": [
            {
                "ssd_ready_ns": 1_000,
                "future_wait_start_ns": 1_500,
                "future_wait_end_ns": 2_000,
                "h2d_submit_start_ns": 3_000,
                "h2d_submit_end_ns": 4_000,
                "h2d_sync_start_ns": 4_000,
                "h2d_complete_ns": 6_000,
                "h2d_cuda_ms": 0.001,
            }
        ]
    }
    service = {
        "service_prepare_count": 2,
        "service_prepare_avg_ns": 1_000.0,
        "prepare_p95_ns": 2_000,
        "prepare_p99_ns": 3_000,
        "prepare_max_ns": 4_000,
    }
    direct = {
        "blocks": 2,
        "cuda_sample_shift": 4,
        "transfer_ms": 1.5,
        "final_drain_us": 20.0,
        **{
            name: summary
            for name in (
                "ready_get_us",
                "ssd_completion_gap_us",
                "ready_to_h2d_us",
                "storage_release_us",
                "h2d_host_submit_us",
                "h2d_host_gap_us",
                "event_record_us",
                "event_query_us",
                "event_sync_us",
                "event_timing_read_us",
                "h2d_cuda_us",
                "memory_release_us",
            )
        },
    }

    result, markdown = build_report([io_record], [timeline], service, [direct])

    assert (
        result["operations"]["read"]["metrics"]["engine_queue_us"]["weighted_mean_us"]
        == 10.0
    )
    assert result["layer_pipeline"]["ssd_ready_to_h2d_submit_us"]["mean_us"] == 2.0
    assert result["direct_pipeline"]["h2d_cuda_us"]["weighted_mean_us"] == 10.0
    assert result["direct_pipeline"]["transfer_ms"]["mean_us"] == 1500.0
    assert "requests=1, blocks=2, CUDA sample shift=4" in markdown
    assert "| service prepare | 2 | 1.00 | 2.00 | 3.00 | 4.00 |" in markdown
