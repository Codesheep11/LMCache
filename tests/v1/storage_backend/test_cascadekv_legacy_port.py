# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the LMCache 0.4.3 CascadeKV legacy port."""

# Standard
from pathlib import Path
from collections import OrderedDict
import ast

# First Party
from lmcache.v1.storage_backend.storage_manager import StorageManager


BACKEND_PATH = (
    Path(__file__).parents[3]
    / "lmcache"
    / "v1"
    / "storage_backend"
    / "cascadekv_backend.py"
)


def test_cascadekv_xds_io_uses_service_completion_groups() -> None:
    source = BACKEND_PATH.read_text()
    tree = ast.parse(source)

    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "xds" not in imported_names
    assert "xds.read_async" not in source
    assert "xds.unload" not in source
    assert "_wait_for_io" not in source
    assert "_get_io_ptr" not in source


def test_cascadekv_owns_lru_without_lmcache_evictor() -> None:
    source = BACKEND_PATH.read_text()

    assert "update_on_hit" not in source
    assert "update_on_get" not in source
    assert "evict_oldest" in source


def test_cascadekv_exposes_lmcache_043_async_batch_api() -> None:
    source = BACKEND_PATH.read_text()
    tree = ast.parse(source)
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CascadeKVBackend"
    )
    method_names = {
        node.name
        for node in backend.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "async_batched_submit_put_task" in method_names
    assert "batched_async_contains" in method_names
    assert "batched_get_non_blocking" in method_names


def test_storage_manager_closes_dma_owner_after_dependent_backends() -> None:
    manager = object.__new__(StorageManager)
    manager.storage_backends = OrderedDict(
        [
            ("LocalCPUBackend", object()),
            ("CascadeKVBackend", object()),
            ("LocalDiskBackend", object()),
        ]
    )

    assert [name for name, _backend in manager._get_backend_close_order()] == [
        "CascadeKVBackend",
        "LocalDiskBackend",
        "LocalCPUBackend",
    ]
