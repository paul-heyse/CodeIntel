"""Smoke tests for helper fixtures used across adapter/service tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.mcp.models import GraphScopePayload
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import build_duckdb_backend, build_scope_parsing_service


def test_build_duckdb_backend_defaults_to_gateway_config(
    fresh_gateway: StorageGateway,
) -> None:
    """Backend helper should inherit repo/commit from the gateway config."""
    backend = build_duckdb_backend(fresh_gateway)
    if backend.repo != fresh_gateway.config.repo:
        pytest.fail("Backend should inherit repo from gateway config")
    if backend.commit != fresh_gateway.config.commit:
        pytest.fail("Backend should inherit commit from gateway config")
    backend.get_function_summary(urn="urn:foo")


def test_scope_parsing_service_round_trips_graph_scope() -> None:
    """Scope parsing helper should produce GraphRunScope from payloads."""
    captured: dict[str, object | None] = {}

    def _delegate(**kwargs: object) -> dict[str, object]:
        captured["scope"] = kwargs.get("scope")
        return {
            "found": True,
            "summary": None,
            "meta": {"generated_at": datetime.now(tz=UTC)},
        }

    service = build_scope_parsing_service(delegate=_delegate)
    payload = GraphScopePayload(paths=("a.py",), modules=("m",), time_window=None)
    _ = service.get_function_summary(scope=payload)
    if not isinstance(captured["scope"], GraphRunScope):
        pytest.fail("Expected GraphRunScope from parsing helper")
