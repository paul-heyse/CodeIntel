"""Scope propagation for query services."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.mcp.models import GraphScopePayload
from codeintel.serving.services.query_service import LocalQueryService, ResponseMeta


class _FakeQuery:
    """Minimal stub that records scopes passed from services."""

    def __init__(self) -> None:
        self.scopes: list[GraphRunScope | None] = []

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphRunScope | None = None,
    ) -> object:
        _ = (urn, goid_h128, rel_path, qualname)
        self.scopes.append(scope)
        return {
            "found": True,
            "summary": None,
            "meta": ResponseMeta().model_dump(),
        }


def test_local_query_service_parses_graph_scope() -> None:
    """LocalQueryService should parse GraphScopePayload into GraphRunScope."""
    query = _FakeQuery()
    service = LocalQueryService(query=query)  # type: ignore[arg-type]
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 2, 1, tzinfo=UTC)
    payload = GraphScopePayload(paths=("a.py",), modules=("m",), time_window=(start, end))

    _ = service.get_function_summary(scope=payload)

    if len(query.scopes) != 1:
        pytest.fail("Scope should be captured once")
    parsed = query.scopes[0]
    if not isinstance(parsed, GraphRunScope):
        pytest.fail("Scope should be parsed into GraphRunScope")
    if parsed.paths != ("a.py",):
        pytest.fail("Paths should propagate through LocalQueryService")
    if parsed.modules != ("m",):
        pytest.fail("Modules should propagate through LocalQueryService")
    if parsed.time_window != (start, end):
        pytest.fail("Time window should propagate through LocalQueryService")
