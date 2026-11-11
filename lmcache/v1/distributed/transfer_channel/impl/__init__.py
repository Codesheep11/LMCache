# SPDX-License-Identifier: Apache-2.0
"""Transfer channel implementations.

Importing this package imports each implementation module, which in turn
self-registers its factory via ``register_transfer_channel_factory``.
"""

# Standard
from importlib.util import find_spec

# First Party
from lmcache.v1.distributed.transfer_channel.impl import nixl_impl  # noqa: F401

# Discover the optional package without initializing its native transport.
if find_spec("mooncake") is not None:
    # First Party
    from lmcache.v1.distributed.transfer_channel.impl import (  # noqa: F401
        mooncake_te_impl,
    )
