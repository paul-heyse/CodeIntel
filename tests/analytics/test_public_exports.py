"""Public API export checks for analytics packages."""

from __future__ import annotations

from codeintel.analytics import functions as functions_mod
from codeintel.analytics import graphs as analytics_graphs_mod
from codeintel.analytics import history as history_mod
from tests._helpers import assert_frozen
from tests._helpers.assertions.expectation_assertions import (
    expect_in,
    expect_true,
)
from tests._helpers.rows import list_public_exports


def test_functions_module_exports() -> None:
    """Functions API should retain expected callables and options."""
    expected = {
        "FunctionAnalyticsOptions",
        "build_function_history_rows",
        "compute_function_contracts",
        "compute_function_effects",
        "compute_function_metrics_and_types",
    }
    exports = set(list_public_exports(functions_mod))
    for name in expected:
        expect_in(name, exports)
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
        "FileCommitDelta",
        "build_history_timeseries_rows",
        "iter_file_history",
    }
    exports = set(list_public_exports(history_mod))
    for name in expected:
        expect_in(name, exports)
    assert_frozen(tuple(sorted(exports)), "__len__", len(exports))
