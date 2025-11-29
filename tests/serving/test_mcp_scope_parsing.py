"""MCP scope payload parsing helpers."""

from __future__ import annotations

from datetime import datetime

import pytest

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.mcp.models import GraphScopePayload, parse_graph_scope


def test_parse_graph_scope_roundtrip() -> None:
    """GraphScopePayload should coerce ISO strings into datetimes."""
    start = "2024-01-01T00:00:00+00:00"
    end = "2024-02-01T00:00:00+00:00"
    payload = GraphScopePayload.model_validate(
        {"paths": ("a.py",), "modules": ("m",), "time_window": (start, end)}
    )
    scope = parse_graph_scope(payload)
    if not isinstance(scope, GraphRunScope):
        pytest.fail("Expected GraphRunScope result")
    if scope.paths != ("a.py",):
        pytest.fail("Scope paths should roundtrip from payload")
    if scope.modules != ("m",):
        pytest.fail("Scope modules should roundtrip from payload")
    if scope.time_window is None:
        pytest.fail("Scope time_window should parse when provided")
    if scope.time_window[0] != datetime.fromisoformat(start):
        pytest.fail("Scope start time should parse ISO string")
    if scope.time_window[1] != datetime.fromisoformat(end):
        pytest.fail("Scope end time should parse ISO string")
