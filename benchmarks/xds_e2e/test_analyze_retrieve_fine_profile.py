# SPDX-License-Identifier: Apache-2.0
"""Tests for request/client timestamp correlation."""

from analyze_retrieve_fine_profile import correlate_requests


def test_correlate_reuse_request_uses_tp_critical_path():
    clients = {
        (7, 1): {
            "start_ns": 1_000_000_000,
            "first_token_ns": 1_500_000_000,
            "end_ns": 1_800_000_000,
        }
    }
    ranks = [
        {
            "req_id": "cmpl-stress-s007-t01-a",
            "request_start_ns": 1_100_000_000,
            "request_end_ns": 1_300_000_000,
        },
        {
            "req_id": "cmpl-stress-s007-t01-b",
            "request_start_ns": 1_120_000_000,
            "request_end_ns": 1_350_000_000,
        },
    ]

    result = correlate_requests(clients, ranks, "request_start_ns")

    assert result["pre_lmcache"]["mean_ms"] == 100
    assert result["lmcache"]["mean_ms"] == 250
    assert result["post_lmcache"]["mean_ms"] == 150
    assert result["decode"]["mean_ms"] == 300
