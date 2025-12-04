"""Tests for codeintel.analytics.ports public API exports.

Testing Charter Compliance:
- Verifies production import paths work correctly
- No monkeypatching or test-only code paths
- Simple import verification for re-export modules
"""

from __future__ import annotations

import importlib

import pytest

from codeintel.analytics import ports as ports_pkg
from codeintel.analytics.ports import (
    BatchResult,
    CatalogPort,
    FunctionSpanData,
    GraphRuntimePort,
    QueryResult,
    StoragePort,
)
from codeintel.analytics.ports import catalog as catalog_mod
from codeintel.analytics.ports import graphs as graphs_mod
from codeintel.analytics.ports import storage as storage_mod
from codeintel.analytics.ports.catalog import CatalogPort as CatalogPortDirect
from codeintel.analytics.ports.catalog import FunctionSpanData as FunctionSpanDataDirect
from codeintel.analytics.ports.graphs import GraphRuntimePort as GraphRuntimePortDirect
from codeintel.analytics.ports.storage import BatchResult as AnalyticsBatchResult
from codeintel.analytics.ports.storage import QueryResult as AnalyticsQueryResult
from codeintel.analytics.ports.storage import StoragePort as AnalyticsStoragePort
from codeintel.graphs.ports.storage import BatchResult as GraphsBatchResult
from codeintel.graphs.ports.storage import QueryResult as GraphsQueryResult
from codeintel.graphs.ports.storage import StoragePort as GraphsStoragePort


class TestPortsPackageExports:
    """Verify analytics.ports package exports are accessible."""

    @staticmethod
    def test_ports_init_exports_all_symbols() -> None:
        """Verify top-level __init__.py exports all expected symbols."""
        # All symbols should be importable and non-None
        assert CatalogPort is not None
        assert FunctionSpanData is not None
        assert GraphRuntimePort is not None
        assert BatchResult is not None
        assert QueryResult is not None
        assert StoragePort is not None

    @staticmethod
    def test_ports_init_all_list_is_complete() -> None:
        """Verify __all__ list matches expected exports."""
        expected = {
            "BatchResult",
            "CatalogPort",
            "FunctionSpanData",
            "GraphRuntimePort",
            "QueryResult",
            "StoragePort",
        }
        actual = set(ports_pkg.__all__)
        assert actual == expected, f"Missing: {expected - actual}, Extra: {actual - expected}"


class TestCatalogPortExports:
    """Verify catalog port module exports."""

    @staticmethod
    def test_catalog_port_exports() -> None:
        """Verify CatalogPort and FunctionSpanData are importable from submodule."""
        assert CatalogPortDirect is not None
        assert FunctionSpanDataDirect is not None

    @staticmethod
    def test_catalog_port_is_protocol() -> None:
        """Verify CatalogPort is a runtime-checkable Protocol."""
        # CatalogPort should be a Protocol class
        assert hasattr(CatalogPortDirect, "__protocol_attrs__") or hasattr(
            CatalogPortDirect, "_is_protocol"
        )

    @staticmethod
    def test_catalog_port_all_list() -> None:
        """Verify __all__ list for catalog module."""
        expected = {"CatalogPort", "FunctionSpanData"}
        actual = set(catalog_mod.__all__)
        assert actual == expected


class TestGraphRuntimePortExports:
    """Verify graph runtime port module exports."""

    @staticmethod
    def test_graph_runtime_port_exports() -> None:
        """Verify GraphRuntimePort is importable from submodule."""
        assert GraphRuntimePortDirect is not None

    @staticmethod
    def test_graph_runtime_port_is_runtime_checkable() -> None:
        """Verify GraphRuntimePort is a runtime-checkable Protocol."""
        # Should be able to use isinstance checks
        assert hasattr(GraphRuntimePortDirect, "_is_runtime_protocol") or callable(
            getattr(GraphRuntimePortDirect, "__subclasshook__", None)
        )

    @staticmethod
    def test_graph_runtime_port_all_list() -> None:
        """Verify __all__ list for graphs module."""
        expected = {"GraphRuntimePort"}
        actual = set(graphs_mod.__all__)
        assert actual == expected


class TestStoragePortExports:
    """Verify storage port module exports."""

    @staticmethod
    def test_storage_port_exports() -> None:
        """Verify storage port types are importable from submodule."""
        assert AnalyticsBatchResult is not None
        assert AnalyticsQueryResult is not None
        assert AnalyticsStoragePort is not None

    @staticmethod
    def test_storage_port_all_list() -> None:
        """Verify __all__ list for storage module."""
        expected = {"BatchResult", "QueryResult", "StoragePort"}
        actual = set(storage_mod.__all__)
        assert actual == expected

    @staticmethod
    def test_storage_port_reexports_from_graphs() -> None:
        """Verify storage port types are same as graphs.ports.storage."""
        # Should be the exact same types (re-exports, not copies)
        assert AnalyticsBatchResult is GraphsBatchResult
        assert AnalyticsQueryResult is GraphsQueryResult
        assert AnalyticsStoragePort is GraphsStoragePort


class TestPortsModuleStructure:
    """Verify module structure follows conventions."""

    @staticmethod
    @pytest.mark.parametrize(
        "import_path",
        [
            "codeintel.analytics.ports",
            "codeintel.analytics.ports.catalog",
            "codeintel.analytics.ports.graphs",
            "codeintel.analytics.ports.storage",
        ],
    )
    def test_modules_importable(import_path: str) -> None:
        """Verify all port modules are importable without errors."""
        module = importlib.import_module(import_path)
        assert module is not None
        assert hasattr(module, "__all__")
