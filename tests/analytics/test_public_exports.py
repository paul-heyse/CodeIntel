"""Public API export checks for analytics packages."""

from __future__ import annotations

from codeintel.analytics import functions as functions_mod
from codeintel.analytics import graphs as analytics_graphs_mod
from codeintel.analytics import history as history_mod
from tests._helpers import assert_frozen
from tests._helpers.assertions.expectation_assertions import (
    expect_in,
)
from tests._helpers.rows import list_public_exports


def test_functions_module_exports() -> None:
    """Functions API should retain expected callables and options."""
    expected = {
        "FunctionAnalyticsOptions",
        "FunctionAnalyticsResult",
    }
    exports = set(list_public_exports(functions_mod))
    for name in expected:
        expect_in(name, exports)
    assert_frozen(tuple(sorted(exports)), "__len__", len(exports))


def test_graphs_module_exports() -> None:
    """Graphs API should surface expected compute helpers."""
    expected = {
        "CONFIG_DATA_FLOW_COLS",
        "CONFIG_GRAPH_METRICS_KEYS_COLS",
        "CONFIG_GRAPH_METRICS_MODULES_COLS",
        "CONFIG_PROJECTION_KEY_EDGES_COLS",
        "CONFIG_PROJECTION_MODULE_EDGES_COLS",
        "ConfigDataFlowResult",
        "ConfigGraphMetricsResult",
    }
    exports = set(list_public_exports(analytics_graphs_mod))
    for name in expected:
        expect_in(name, exports)
    assert_frozen(tuple(sorted(exports)), "__len__", len(exports))


def test_history_module_exports() -> None:
    """History API should expose timeseries helpers."""
    expected = {
        "FileCommitDelta",
    }
    exports = set(list_public_exports(history_mod))
    for name in expected:
        expect_in(name, exports)
    assert_frozen(tuple(sorted(exports)), "__len__", len(exports))
