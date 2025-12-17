"""Ensure analytics domain packages expose stable public APIs."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

EXPECTED_EXPORTS: Mapping[str, set[str]] = {
    "codeintel.analytics.functions": {
        "FUNCTION_HISTORY_COLS",
        "FunctionAnalyticsOptions",
        "build_function_history_rows",
        "compute_function_contracts",
        "compute_function_effects",
        "compute_function_metrics_and_types",
    },
    "codeintel.analytics.graphs": {
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
    },
    "codeintel.analytics.history": {
        "FileCommitDelta",
        "HISTORY_TIMESERIES_COLS",
        "build_history_timeseries_rows",
        "iter_file_history",
    },
    "codeintel.analytics.parsing": {
        "BaseValidationReporter",
        "FunctionParserRegistry",
        "FunctionValidationReporter",
        "GraphValidationReporter",
        "ParsedFunction",
        "ParsedModule",
        "SourceSpan",
        "SpanResolutionError",
        "SpanResolutionResult",
        "ValidationResult",
        "build_span_index",
        "get_parser",
        "get_validation_rows",
        "parse_python_module",
        "register_parser",
        "resolve_span",
    },
}


@pytest.mark.parametrize(("module_path", "expected"), EXPECTED_EXPORTS.items())
def test_domain_exports(module_path: str, expected: set[str]) -> None:
    """Verify each analytics domain package exports the agreed public API."""
    module = importlib.import_module(module_path)
    actual = set(cast("Iterable[str]", getattr(module, "__all__", ())))
    missing = expected - actual
    extras = actual - expected

    if missing or extras:
        message = f"{module_path} exports drift: missing={sorted(missing)} extras={sorted(extras)}"
        pytest.fail(message)
