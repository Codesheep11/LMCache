# SPDX-License-Identifier: Apache-2.0

from analyze_bandwidth_compare import marker_records


def test_marker_records_accepts_colored_lmcache_log(tmp_path):
    log_path = tmp_path / "backend.log"
    log_path.write_text(
        "LMCache INFO: [LMCACHE_SSD_GPU_METRICS] "
        '\x1b[0m{"worker_id":2,"bytes":4096,"seconds":0.5}'
        " \x1b[3m(cache_engine.py:1)\x1b[0m\n"
    )

    assert marker_records(log_path, "[LMCACHE_SSD_GPU_METRICS]") == [
        {"worker_id": 2, "bytes": 4096, "seconds": 0.5}
    ]
