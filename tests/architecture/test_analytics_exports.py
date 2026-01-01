"""Ensure analytics domain packages expose stable public APIs."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

EXPECTED_EXPORTS: Mapping[str, set[str]] = {
    "codeintel.build.analytics.functions": {
        "FunctionAnalyticsOptions",
        "FunctionAnalyticsResult",
    },
    "codeintel.build.analytics.graphs": {
        "ConfigDataFlowResult",
        "ConfigGraphMetricsResult",
        "CONFIG_DATA_FLOW_COLS",
        "CONFIG_GRAPH_METRICS_KEYS_COLS",
        "CONFIG_GRAPH_METRICS_MODULES_COLS",
        "CONFIG_PROJECTION_KEY_EDGES_COLS",
        "CONFIG_PROJECTION_MODULE_EDGES_COLS",
    },
    "codeintel.build.analytics.parsing": {
        "BaseValidationReporter",
        "FunctionParserRegistry",
        "FunctionValidationReporter",
        "GraphValidationReporter",
        "ParsedFunction",
        "ParsedModule",
        "SourceSpan",
        "SpanResolutionError",
        "SpanResolutionResult",
        "ValidationRows",
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
