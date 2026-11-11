# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.ext_server_module import build_server_module_router
from lmcache.v1.multiprocess.modules import xds_metadata
from lmcache.v1.multiprocess.modules.xds_metadata import XDSMetadataModule
from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest
from lmcache.v1.multiprocess.xds_catalog import XDSExtent, XDSLease
from lmcache.v1.multiprocess.xds_protocol import (
    AcquireReads,
    BindWorker,
    ClaimReads,
    Complete,
    LookupPrefix,
    Register,
    ReserveWrite,
)


def test_metadata_wire_contract_and_failure_responses() -> None:
    ctx = MagicMock()
    router = build_server_module_router(ctx, [XDSMetadataModule(ctx)])
    assert router is not None

    def call(method: str, payload: object) -> bytes:
        response = router.server_module_call(
            ServerModuleCallRequest(
                method="xds.v1." + method, payload=msgspec.msgpack.encode(payload)
            )
        )
        assert response.success, response.error
        return response.payload

    key = ObjectKey(b"hash", "model", 1)
    call("register", Register(1, (XDSExtent(100, 0, 4096),)))
    write = msgspec.msgpack.decode(
        call("reserve_write", ReserveWrite(1, key, 4096)), type=XDSLease
    )
    assert (
        msgspec.msgpack.decode(call("acquire_reads", AcquireReads(((1, key),)))) is None
    )
    call("publish", Complete(1, write.ticket))
    leases = msgspec.msgpack.decode(
        call("acquire_reads", AcquireReads(((1, key),))), type=tuple[XDSLease, ...]
    )
    assert leases[0].extent == write.extent
    failure = router.server_module_call(
        ServerModuleCallRequest(
            method="xds.v1.unregister", payload=msgspec.msgpack.encode(1)
        )
    )
    assert not failure.success and "outstanding" in failure.error
    call("release_read", Complete(1, leases[0].ticket))
    call("unregister", 1)
    malformed = router.server_module_call(
        ServerModuleCallRequest(
            method="xds.v1.register", payload=msgspec.msgpack.encode({})
        )
    )
    assert not malformed.success


def test_heartbeat_reaps_metadata_without_reusing_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(xds_metadata, "monotonic", lambda: now[0])
    module = XDSMetadataModule(MagicMock())
    module.register(msgspec.msgpack.encode(Register(1, (XDSExtent(100, 0, 4096),))))
    key = ObjectKey(b"hash", "model", 1)
    write = msgspec.msgpack.decode(
        module.reserve_write(msgspec.msgpack.encode(ReserveWrite(1, key, 4096))),
        type=XDSLease,
    )
    now[0] = 11.0
    assert module.reap_stale_instances(10, 100) == []
    module.touch_instance(1)
    now[0] = 20.0
    assert module.reap_stale_instances(10, 100) == []
    now[0] = 21.0
    assert module.reap_stale_instances(10, 100) == [1]
    assert module.tracked_instance_count() == 0
    with pytest.raises(KeyError):
        module.publish(msgspec.msgpack.encode(Complete(1, write.ticket)))
    assert (
        msgspec.msgpack.decode(
            module.acquire_reads(msgspec.msgpack.encode(AcquireReads(((1, key),))))
        )
        is None
    )
    module.touch_instance(1)
    assert module.tracked_instance_count() == 0


def test_worker_traffic_preserves_registration_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(xds_metadata, "monotonic", lambda: now[0])
    module = XDSMetadataModule(MagicMock())
    module.register(msgspec.msgpack.encode(Register(1, (XDSExtent(100, 0, 4096),))))
    now[0] = 90.0
    module.reserve_write(
        msgspec.msgpack.encode(ReserveWrite(1, ObjectKey(b"hash", "model", 1), 4096))
    )
    now[0] = 110.0
    assert module.reap_stale_instances(10, 100) == []
    module.close()
    assert module.tracked_instance_count() == 0


def test_lookup_handoff_wire_contract() -> None:
    ctx = MagicMock()
    module = XDSMetadataModule(ctx)
    router = build_server_module_router(ctx, [module])
    assert router is not None

    def call(method: str, payload: object) -> bytes:
        response = router.server_module_call(
            ServerModuleCallRequest(
                method="xds.v1." + method, payload=msgspec.msgpack.encode(payload)
            )
        )
        assert response.success, response.error
        return response.payload

    key = ObjectKey(b"wire-handoff", "model", 1)
    call("register", Register(1, (XDSExtent(100, 0, 4096),)))
    write = msgspec.msgpack.decode(
        call("reserve_write", ReserveWrite(1, key, 4096)), type=XDSLease
    )
    call("publish", Complete(1, write.ticket))
    assert (
        msgspec.msgpack.decode(
            call("lookup_prefix", LookupPrefix("request-epoch", (((1, key),),)))
        )
        == 1
    )
    reads = msgspec.msgpack.decode(
        call("claim_reads", ClaimReads("request-epoch", 1, (key,))),
        type=tuple[XDSLease, ...],
    )
    call("cancel_lookup", "request-epoch")
    assert (
        msgspec.msgpack.decode(
            call(
                "reserve_write", ReserveWrite(1, ObjectKey(b"other", "model", 1), 4096)
            )
        )
        is None
    )
    call("release_read", Complete(1, reads[0].ticket))
    call("unregister", 1)


def test_recovery_preserves_live_entries_and_pins() -> None:
    module = XDSMetadataModule(MagicMock())
    registration = Register(1, (XDSExtent(100, 0, 4096),))
    payload = msgspec.msgpack.encode(registration)
    module.ensure_registered(payload)
    key = ObjectKey(b"hash", "model", 1)
    write = msgspec.msgpack.decode(
        module.reserve_write(msgspec.msgpack.encode(ReserveWrite(1, key, 4096))),
        type=XDSLease,
    )
    module.publish(msgspec.msgpack.encode(Complete(1, write.ticket)))
    readers = msgspec.msgpack.decode(
        module.acquire_reads(msgspec.msgpack.encode(AcquireReads(((1, key),)))),
        type=tuple[XDSLease, ...],
    )
    module.ensure_registered(payload)
    with pytest.raises(RuntimeError, match="outstanding"):
        module.unregister(msgspec.msgpack.encode(1))
    module.release_read(msgspec.msgpack.encode(Complete(1, readers[0].ticket)))
    with pytest.raises(ValueError, match="capacity differs"):
        module.ensure_registered(
            msgspec.msgpack.encode(Register(1, (XDSExtent(200, 0, 4096),)))
        )
    module.unregister(msgspec.msgpack.encode(1))
    module.ensure_registered(payload)
    assert (
        msgspec.msgpack.decode(
            module.acquire_reads(msgspec.msgpack.encode(AcquireReads(((1, key),))))
        )
        is None
    )


def test_recovery_binding_is_idempotent_but_rejects_conflicts() -> None:
    module = XDSMetadataModule(MagicMock())
    registration = msgspec.msgpack.encode(Register(1, (XDSExtent(100, 0, 4096),)))
    binding = msgspec.msgpack.encode(BindWorker(1, "model", 1, 0, 4096))
    module.ensure_registered(registration)
    module.bind_worker(binding)
    module.ensure_registered(registration)
    module.bind_worker(binding)
    with pytest.raises(ValueError, match="already bound"):
        module.bind_worker(msgspec.msgpack.encode(BindWorker(1, "model", 1, 0, 8192)))
    restarted = XDSMetadataModule(MagicMock())
    restarted.ensure_registered(registration)
    restarted.bind_worker(binding)
    assert restarted.tracked_instance_count() == 1
