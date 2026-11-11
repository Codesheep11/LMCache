# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallResponse
from lmcache.v1.multiprocess.xds_client import xds_rpc


def test_rpc_polling_and_failure_propagation() -> None:
    client = MagicMock()
    source = MessagingFuture[ServerModuleCallResponse]()
    client.server_module_call.return_value = source
    future = xds_rpc(client, "lookup_tokens", None, int)
    assert not future.query()
    assert not future.wait(0)
    source.set_result(
        ServerModuleCallResponse(
            success=True, payload=msgspec.msgpack.encode(7), error=""
        )
    )
    assert future.query()
    assert future.result(0) == 7
    failed = MessagingFuture[ServerModuleCallResponse]()
    client.server_module_call.return_value = failed
    future = xds_rpc(client, "claim_tokens", None, int)
    failed.set_result(
        ServerModuleCallResponse(success=False, error="stale epoch", payload=b"")
    )
    with pytest.raises(RuntimeError, match="stale epoch"):
        future.result(0)
