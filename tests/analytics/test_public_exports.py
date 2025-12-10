"""Public API export checks for analytics packages."""

from __future__ import annotations

import importlib

from codeintel.analytics import functions as functions_mod
from codeintel.analytics import graphs as analytics_graphs_mod
from codeintel.analytics import history as history_mod
from codeintel.analytics import ports as ports_pkg
from codeintel.analytics.ports import catalog as catalog_mod
from codeintel.analytics.ports import graphs as graphs_mod
from codeintel.analytics.ports import storage as storage_mod
from tests._helpers import assert_frozen
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_true,
)
from tests._helpers.rows import list_public_exports


def _assert_exports(module: object, expected: set[str]) -> None:
    actual = set(list_public_exports(module))
    expect_equal(actual, expected)
    assert_frozen(tuple(actual), "__setitem__", expected)


def test_ports_exports() -> None:
    """Top-level ports package should export stable symbols."""
    expected = {
        "BatchResult",
        "CatalogPort",
        "FunctionSpanData",
        "GraphRuntimePort",
        "QueryResult",
        "StoragePort",
    }
    _assert_exports(ports_pkg, expected)


def test_ports_submodules_exports() -> None:
    """Submodule __all__ lists should match expected exports."""
    _assert_exports(catalog_mod, {"CatalogPort", "FunctionSpanData"})
    _assert_exports(graphs_mod, {"GraphRuntimePort"})
    _assert_exports(storage_mod, {"BatchResult", "QueryResult", "StoragePort"})


def test_ports_reexports_alignment() -> None:
    """Storage port types should match graphs storage definitions."""
    storage = importlib.import_module("codeintel.analytics.ports.storage")
    graphs_storage = importlib.import_module("codeintel.graphs.ports.storage")
    expect_true(storage.BatchResult is graphs_storage.BatchResult)
    expect_true(storage.QueryResult is graphs_storage.QueryResult)
    expect_true(storage.StoragePort is graphs_storage.StoragePort)


def test_functions_module_exports() -> None:
    """Functions API should retain expected callables and options."""
    expected = {
        "FunctionAnalyticsOptions",
        "compute_function_contracts",
        "compute_function_effects",
        "compute_function_history",
        "compute_function_metrics_and_types",
    }
    exports = set(list_public_exports(functions_mod))
    for name in expected:
        expect_in(name, exports)
        expect_true(callable(getattr(functions_mod, name)))
    assert_frozen(tuple(sorted(exports)), "__len__", len(exports))


def test_graphs_module_exports() -> None:
    """Graphs API should surface expected compute helpers."""
    expected = {
        "build_subsystems",
        "compute_cfg_metrics",
        "compute_config_data_flow",
        "compute_config_graph_metrics",
        "compute_dfg_metrics",
        "compute_graph_metrics",
        "compute_graph_metrics_functions_ext",
        "compute_graph_metrics_modules_ext",
        "compute_graph_stats",
        "compute_subsystem_agreement",
        "compute_subsystem_graph_metrics",
        "compute_symbol_graph_metrics_functions",
        "compute_symbol_graph_metrics_modules",
    }
    exports = set(list_public_exports(analytics_graphs_mod))
    for name in expected:
        expect_in(name, exports)
        expect_true(callable(getattr(analytics_graphs_mod, name)))
    assert_frozen(tuple(sorted(exports)), "__len__", len(exports))


def test_history_module_exports() -> None:
    """History API should expose timeseries helpers."""
    expected = {
        "compute_history_timeseries",
        "compute_history_timeseries_gateways",
    }
    exports = set(list_public_exports(history_mod))
    for name in expected:
        expect_in(name, exports)
        expect_true(callable(getattr(history_mod, name)))
    assert_frozen(tuple(sorted(exports)), "__len__", len(exports))
