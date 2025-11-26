"""Adapter-level tests for MCP backend validation and delegation."""

from __future__ import annotations

import pytest

from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.models import FunctionSummaryResponse
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway


def test_require_identifier_validation(fresh_gateway: StorageGateway) -> None:
    """Backend raises invalid_argument when identifiers are missing."""
    backend = DuckDBBackend(
        gateway=fresh_gateway,  # type: ignore[arg-type]
        repo="r",
        commit="c",
        service_override=None,
    )
    with pytest.raises(errors.McpError):
        backend.get_function_summary()
    # should succeed with identifier
    resp = backend.get_function_summary(urn="urn:foo")
    if not isinstance(resp, FunctionSummaryResponse):
        pytest.fail("Expected FunctionSummaryResponse")


def test_validate_direction_accepts_expected(fresh_gateway: StorageGateway) -> None:
    """Backend validates direction before delegation."""
    backend = DuckDBBackend(
        gateway=fresh_gateway,  # type: ignore[arg-type]
        repo="r",
        commit="c",
        service_override=None,
    )
    with pytest.raises(errors.McpError):
        backend.get_callgraph_neighbors(goid_h128=1, direction="bad")
    backend.get_callgraph_neighbors(goid_h128=1, direction="both")


def test_backend_delegates_after_validation(fresh_gateway: StorageGateway) -> None:
    """Backend methods delegate to the service after validation."""
    backend = DuckDBBackend(
        gateway=fresh_gateway,  # type: ignore[arg-type]
        repo="r",
        commit="c",
        service_override=None,
    )
    backend.get_function_summary(urn="urn:foo")
    if not isinstance(backend.service, LocalQueryService):
        pytest.fail("Expected LocalQueryService")
