# SPDX-License-Identifier: Apache-2.0
"""Metadata transports must not initialize unrelated native device engines."""

# Standard
import subprocess
import sys


def test_metadata_client_does_not_import_mooncake_native_engine() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lmcache.v1.multiprocess.xds_client; import sys; "
            "assert 'mooncake.engine' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
