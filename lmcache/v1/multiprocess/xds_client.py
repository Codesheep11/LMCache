# SPDX-License-Identifier: Apache-2.0
"""Typed, nonblocking decoding of the xDS metadata extension envelope."""

# Standard
from typing import Generic, TypeVar

# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)
from lmcache.v1.multiprocess.transport.base import RequestClient

T = TypeVar("T")


def xds_rpc(
    client: RequestClient, method: str, payload: object, response_type: type[T]
) -> "XDSMetadataFuture[T]":
    """Submit metadata without waiting; decode its typed response on result().

    Args:
        client: Existing MP transport, owned by the caller.
        method: xds.v1 method suffix.
        payload: MessagePack-compatible metadata request.
        response_type: Expected response schema accepted by msgspec.

    Returns:
        A polling future which propagates transport and server errors.
    """
    future = client.server_module_call(
        ServerModuleCallRequest(
            method="xds.v1." + method, payload=msgspec.msgpack.encode(payload)
        )
    )
    return XDSMetadataFuture(future, response_type)


class XDSMetadataFuture(MessagingFuture[T], Generic[T]):
    """Map a metadata response without a background waiter or device event."""

    def __init__(
        self, source: MessagingFuture[ServerModuleCallResponse], response_type: type[T]
    ) -> None:
        super().__init__()
        self._source = source
        self._response_type = response_type

    def query(self) -> bool:
        """Return whether the underlying transport response is available."""
        return self._source.query()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait up to timeout seconds for the metadata response."""
        return self._source.wait(timeout)

    def result(self, timeout: float | None = None) -> T:
        """Decode the response, raising RuntimeError on a server rejection."""
        response = self._source.result(timeout)
        if not response.success:
            raise RuntimeError(response.error)
        return msgspec.msgpack.decode(response.payload, type=self._response_type)
