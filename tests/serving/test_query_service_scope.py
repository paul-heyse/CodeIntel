"""Scope propagation for query services."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.config.graph_helpers import GraphRunScope
from codeintel.serving.mcp.models import GraphScopePayload
from tests._helpers.fakes.serving import build_serving_scope_pack
from tests._helpers.gateway import build_scope_parsing_service


def test_local_query_service_parses_graph_scope() -> None:
    """LocalQueryService should parse GraphScopePayload into GraphRunScope."""
    pack = build_serving_scope_pack()
    service = build_scope_parsing_service(delegate=pack.query.get_function_summary)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 2, 1, tzinfo=UTC)
    payload = GraphScopePayload(paths=("a.py",), modules=("m",), time_window=(start, end))

    _ = service.get_function_summary(scope=payload)

    if len(pack.query.scopes) != 1:
        pytest.fail("Scope should be captured once")
    parsed = pack.query.scopes[0]
    if not isinstance(parsed, GraphRunScope):
        pytest.fail("Scope should be parsed into GraphRunScope")
    if parsed.paths != ("a.py",):
        pytest.fail("Paths should propagate through LocalQueryService")
    if parsed.modules != ("m",):
        pytest.fail("Modules should propagate through LocalQueryService")
    if parsed.time_window != (start, end):
        pytest.fail("Time window should propagate through LocalQueryService")
