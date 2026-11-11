# SPDX-License-Identifier: Apache-2.0
"""Public replica contract tests with deterministic CPU collective simulation."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from threading import Condition, Event, local
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess import xds_replica as replica

pytestmark = pytest.mark.no_shared_allocator


class CollectiveSimulator:
    """Model ordered asynchronous collectives, retaining each passed final view."""

    def __init__(self, ranks: int, backend: str) -> None:
        self.ranks = ranks
        self.backend = backend
        self.thread = local()
        self.condition = Condition()
        self.operations: dict = {}
        self.broadcasts = 0

    def enter(self, rank: int) -> None:
        self.thread.rank = rank
        self.thread.sequence = 0

    def gather(self, output: list, value: object, group: object) -> None:
        with self.condition:
            sequence = self.thread.sequence
            self.thread.sequence += 1
            state = self.operations.setdefault(sequence, {})
            state[self.thread.rank] = value
            self.condition.notify_all()
            if not self.condition.wait_for(
                lambda: len(state) == self.ranks, timeout=10
            ):
                raise TimeoutError("Simulated control collective did not match")
            output[:] = [state[r] for r in range(self.ranks)]

    def broadcast(
        self, view: torch.Tensor, src: int, group: object, async_op: bool
    ) -> object:
        with self.condition:
            sequence = self.thread.sequence
            self.thread.sequence += 1
            state = self.operations.setdefault(sequence, {})
            state[self.thread.rank] = view
            if len(state) == self.ranks:
                for rank, tensor in state.items():
                    if rank != src:
                        tensor.copy_(state[src])
                self.broadcasts += 1
            self.condition.notify_all()

        def wait() -> bool:
            with self.condition:
                if not self.condition.wait_for(
                    lambda: len(state) == self.ranks, timeout=10
                ):
                    raise TimeoutError("Simulated data collective did not match")
            return True

        return SimpleNamespace(wait=wait)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(replica.dist, "get_world_size", lambda group: self.ranks)
        monkeypatch.setattr(replica.dist, "get_rank", lambda group: self.thread.rank)
        monkeypatch.setattr(replica.dist, "get_global_rank", lambda group, rank: rank)
        monkeypatch.setattr(
            replica.dist,
            "get_backend",
            lambda group: self.backend if group == "data" else "gloo",
        )
        monkeypatch.setattr(replica.dist, "all_gather_object", self.gather)
        monkeypatch.setattr(replica.dist, "broadcast", self.broadcast)


class FinalPlane:
    """CPU backing with explicit platform identity; narrow returns a real view."""

    def __init__(self, width: int, device: str) -> None:
        self.tensor = torch.full((32, 2, width), -1, dtype=torch.float32, device="cpu")
        self.device = SimpleNamespace(type=device)
        self.shape = self.tensor.shape
        self.dtype = self.tensor.dtype

    def stride(self, dim: int) -> int:
        return self.tensor.stride(dim)

    def element_size(self) -> int:
        return self.tensor.element_size()

    def narrow(self, dim: int, start: int, length: int) -> torch.Tensor:
        view = self.tensor.narrow(dim, start, length)
        assert (
            view.untyped_storage().data_ptr()
            == self.tensor.untyped_storage().data_ptr()
        )
        return view


@pytest.mark.parametrize("device,backend", [("cuda", "nccl"), ("npu", "hccl")])
@pytest.mark.parametrize("skip", [0, 1, 3, 6])
@pytest.mark.parametrize("batch_bytes", [1, 192, 100000])
@pytest.mark.parametrize("layout", ["fragmented", "reverse", "permuted"])
def test_only_leader_reads_and_every_replica_preserves_guards(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    backend: str,
    skip: int,
    batch_bytes: int,
    layout: str,
) -> None:
    sim = CollectiveSimulator(4, backend)
    sim.install(monkeypatch)
    monkeypatch.setenv("LMCACHE_XDS_MLA_BATCH_BYTES", str(batch_bytes))
    maps = [
        [1, 2, 6, 7, 9, 12],
        [3, 4, 8, 12, 13, 19],
        [8, 7, 6, 3, 2, 1],
        [21, 22, 23, 24, 25, 26],
    ]
    if layout != "fragmented":
        order = [5, 4, 3, 2, 1, 0] if layout == "reverse" else [3, 1, 4, 0, 5, 2]
        maps = [[base + i for i in order] for base in (1, 8, 16, 24)]
    planes = [(FinalPlane(4, device), FinalPlane(1, device)) for _ in maps]
    reads = []

    def worker(rank: int) -> None:
        sim.enter(rank)
        transfer = replica.ReplicaBroadcast(
            replica.ReplicaGroups("data", "cpu"),
            planes[rank],
            rank == 0,
            4,
            lambda: None,
        )
        chunks = [maps[rank][i : i + 2] for i in range(0, 6, 2)]

        def read(first: int, last: int, skips: list[int]) -> None:
            assert rank == 0
            for chunk_index, protected in zip(range(first, last), skips, strict=True):
                for offset in range(protected, 2):
                    logical = chunk_index * 2 + offset
                    reads.append(logical)
                    for plane_index, plane in enumerate(planes[rank]):
                        plane.tensor[maps[rank][logical]].fill_(
                            100 * plane_index + logical
                        )

        transfer.restore(("request", 0, 768), chunks, skip, read)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, rank) for rank in range(4)]
        for future in futures:
            future.result(timeout=20)
    assert sorted(reads) == list(range(skip, 6))
    if layout != "fragmented" and batch_bytes == 100000 and skip == 0:
        assert sim.broadcasts == 2  # One collective per final plane, no packing.

    for rank, pair in enumerate(planes):
        for plane_index, plane in enumerate(pair):
            for block in range(32):
                logical = maps[rank].index(block) if block in maps[rank] else -1
                value = 100 * plane_index + logical if logical >= skip else -1
                assert bool(torch.all(plane.tensor[block] == value))


@pytest.mark.parametrize("failure", ["read", "identity", "alias", "writers"])
def test_collective_failures_reach_every_replica_before_new_broadcast(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    sim = CollectiveSimulator(2, "nccl")
    sim.install(monkeypatch)

    def worker(rank: int) -> None:
        sim.enter(rank)
        transfer = replica.ReplicaBroadcast(
            replica.ReplicaGroups("data", "cpu"),
            (FinalPlane(4, "cuda"),),
            rank == 0 or failure == "writers",
            2,
            lambda: None,
        )

        def read(first: int, last: int, skips: list[int]) -> None:
            raise TimeoutError("SSD timeout")

        identity = (rank,) if failure == "identity" else ("same",)
        blocks = [1, 1] if failure == "alias" and rank == 1 else [1, 2]
        with pytest.raises((RuntimeError, ValueError)):
            transfer.restore(identity, [blocks], 0, read)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, rank) for rank in range(2)]
        for future in futures:
            future.result(timeout=20)
    assert sim.broadcasts == 0


@pytest.mark.parametrize("permutation", [[0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]])
def test_shared_permutation_coalesces_without_payload_reordering(
    permutation: list[int],
) -> None:
    maps = [[base + i for i in permutation] for base in (10, 30, 60, 90)]
    assert replica.plan_broadcast_runs(maps, 0, 4) == [
        replica.BroadcastRun((10, 30, 60, 90), 4)
    ]


def test_incompatible_rank_order_cannot_be_merged() -> None:
    maps = [[1, 2, 3, 4], [9, 8, 7, 6]]
    runs = replica.plan_broadcast_runs(maps, 0, 4)
    assert len(runs) == 4
    assert all(run.blocks == 1 for run in runs)


def test_runs_never_include_skipped_or_outside_batch_pages() -> None:
    maps = [[3, 1, 4, 2, 5], [13, 11, 14, 12, 15]]
    assert replica.plan_broadcast_runs(maps, 1, 4) == [
        replica.BroadcastRun((1, 11), 2),
        replica.BroadcastRun((4, 14), 1),
    ]
    assert replica.plan_broadcast_runs(maps, 2, 2) == []
    with pytest.raises(ValueError):
        replica.plan_broadcast_runs([[1], [1, 2]], 0, 1)


@pytest.mark.parametrize("ranks", [2, 3, 4, 8])
def test_random_plans_preserve_every_logical_correspondence(ranks: int) -> None:
    # Standard
    import random

    rng = random.Random(42)
    for length in (1, 7, 16, 31):
        for _ in range(10):
            maps = [rng.sample(range(length * 3), length) for rank in range(ranks)]
            begin = rng.randrange(length)
            end = rng.randrange(begin + 1, length + 1)
            runs = replica.plan_broadcast_runs(maps, begin, end)
            expected = {tuple(m[i] for m in maps) for i in range(begin, end)}
            actual = [
                tuple(start + offset for start in run.starts)
                for run in runs
                for offset in range(run.blocks)
            ]
            assert len(actual) == end - begin
            assert set(actual) == expected
            assert all(
                a.starts[0] + a.blocks <= b.starts[0]
                for a, b in zip(runs, runs[1:], strict=False)
            )


@pytest.mark.parametrize("failure", ["none", "read", "collective"])
def test_read_ahead_progresses_during_broadcast_and_joins_on_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Reads progress without broadcast progress, stay bounded, and are joined."""
    sim = CollectiveSimulator(2, "nccl")
    sim.install(monkeypatch)
    monkeypatch.setenv("LMCACHE_XDS_MLA_BATCH_BYTES", "1")
    second_started = Event()
    release_second = Event()
    second_finished = Event()
    reads = []
    original_broadcast = sim.broadcast
    planes = [(FinalPlane(4, "cuda"),) for _ in range(2)]

    def broadcast(
        view: torch.Tensor, src: int, group: object, async_op: bool
    ) -> object:
        assert second_started.wait(5), "Reader stalled behind broadcast"
        # The consumer has only admitted the current batch plus two ahead.
        if not release_second.is_set():
            assert len(reads) <= 3
        release_second.set()
        if failure == "collective":
            raise RuntimeError("Injected collective failure")
        return original_broadcast(view, src, group, async_op)

    monkeypatch.setattr(replica.dist, "broadcast", broadcast)

    def worker(rank: int) -> None:
        sim.enter(rank)
        transfer = replica.ReplicaBroadcast(
            replica.ReplicaGroups("data", "cpu"),
            planes[rank],
            rank == 0,
            2,
            lambda: None,
        )

        def read(first: int, last: int, skips: list[int]) -> None:
            reads.append(first)
            if first == 1:
                second_started.set()
                assert release_second.wait(5)
                second_finished.set()
                if failure == "read":
                    raise OSError("Injected read failure")
            planes[rank][0].tensor[first + 1].fill_(first + 10)

        if failure == "none":
            transfer.restore(("same",), [[i] for i in range(1, 7)], 0, read)
        else:
            with pytest.raises(RuntimeError):
                transfer.restore(("same",), [[i] for i in range(1, 7)], 0, read)
        if rank == 0:
            assert second_finished.is_set(), "Reader outlived restore"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, rank) for rank in range(2)]
        for future in futures:
            future.result(timeout=20)
    if failure == "none":
        assert reads == list(range(6))
        assert torch.equal(planes[0][0].tensor, planes[1][0].tensor)
