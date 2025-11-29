"""MCP scope parsing and plan response shape coverage.

Raises
------
AssertionError
    When parsing or metadata expectations are not met.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from codeintel.serving.mcp.models import (
    GraphPlanPluginMetadata,
    GraphPlanResponse,
    GraphScopePayload,
    parse_graph_scope,
)


def test_parse_graph_scope_coerces_time_window_strings() -> None:
    """GraphScopePayload should accept ISO strings and coerce to datetimes."""
    start = datetime.now(tz=UTC)
    end = start + timedelta(hours=2)
    payload = GraphScopePayload.model_validate(
        {
            "paths": ("src/foo.py",),
            "modules": ("pkg.mod",),
            "time_window": (start.isoformat(), end.isoformat()),
        }
    )
    scope = parse_graph_scope(payload)
    if scope is None or scope.paths != ("src/foo.py",) or scope.modules != ("pkg.mod",):
        message = "Scope payload should be parsed with paths/modules"
        pytest.fail(message)
    if scope.time_window != (start, end):
        message = "Scope time window should be coerced from ISO strings"
        pytest.fail(message)


def test_graph_plan_response_accepts_plugin_metadata() -> None:
    """GraphPlanResponse should validate isolation/scope metadata in plugin_metadata."""
    plugin_meta = GraphPlanPluginMetadata(
        stage="core",
        severity="fatal",
        requires_isolation=False,
        isolation_kind=None,
        scope_aware=True,
        supported_scopes=("paths", "modules"),
        description="desc",
        enabled_by_default=True,
        depends_on=("core_graph_metrics",),
        provides=("graph_metrics_ext",),
        requires=("docs.graph_metrics",),
        resource_hints={"max_runtime_ms": 1000, "memory_mb_hint": None},
        options_model="MyOptions",
        options_default={"sample": True},
        version_hash="abc123",
        contract_checkers=2,
        config_schema_ref="schemas/graph",
        row_count_tables=("analytics.graph_metrics_functions",),
    )
    response = GraphPlanResponse(
        plan_id="pid",
        ordered_plugins=("p1",),
        skipped_plugins=(),
        dep_graph={"p1": ()},
        plugin_metadata={"p1": plugin_meta},
    )
    meta = response.plugin_metadata["p1"]
    if meta.scope_aware is not True:
        message = "Plan metadata should persist scope_aware flag"
        pytest.fail(message)
    if meta.depends_on != ("core_graph_metrics",):
        message = "Plan metadata should roundtrip depends_on"
        pytest.fail(message)
    if meta.resource_hints != {"max_runtime_ms": 1000, "memory_mb_hint": None}:
        message = "Plan metadata should include resource_hints map"
        pytest.fail(message)
