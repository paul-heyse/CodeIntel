"""Adapter-level tests for MCP backend validation and delegation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from codeintel.mcp import errors
from codeintel.mcp.backend import DuckDBBackend
from codeintel.mcp.models import CallGraphNeighborsResponse, FunctionSummaryResponse, ResponseMeta
from codeintel.services.query_service import LocalQueryService


class StubService(LocalQueryService):
    """Stub LocalQueryService that records method calls."""

    def __init__(self) -> None:
        super().__init__(query=None, dataset_tables={})  # type: ignore[arg-type]
        self.calls: list[str] = []

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        """
        Record function summary requests.

        Returns
        -------
        FunctionSummaryResponse
            Minimal stub response.
        """
        _ = (urn, goid_h128, rel_path, qualname)
        self.calls.append("get_function_summary")
        return FunctionSummaryResponse(found=True, summary=None)

    def get_callgraph_neighbors(  # noqa: PLR6301
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
    ) -> CallGraphNeighborsResponse:  # pragma: no cover
        """
        Stub callgraph neighbors response.

        Returns
        -------
        CallGraphNeighborsResponse
            Empty neighbor sets for tests.
        """
        _ = (goid_h128, direction, limit)
        return CallGraphNeighborsResponse(outgoing=[], incoming=[], meta=ResponseMeta())


def _fake_gateway() -> object:
    """
    Minimal gateway stub to satisfy DuckDBBackend signature.

    Returns
    -------
    object
        Stub object with gateway attributes.
    """
    return SimpleNamespace(
        con=None,
        config=SimpleNamespace(repo="r", commit="c", read_only=True),
        datasets=SimpleNamespace(mapping={}),
        close=lambda: None,
    )


def test_require_identifier_validation() -> None:
    """Backend raises invalid_argument when identifiers are missing."""
    backend = DuckDBBackend(
        gateway=_fake_gateway(),  # type: ignore[arg-type]
        repo="r",
        commit="c",
        service_override=StubService(),
    )
    with pytest.raises(errors.McpError):
        backend.get_function_summary()
    # should succeed with identifier
    resp = backend.get_function_summary(urn="urn:foo")
    if not resp.found:
        pytest.fail("Expected function summary to be found")


def test_validate_direction_accepts_expected() -> None:
    """Backend validates direction before delegation."""
    backend = DuckDBBackend(
        gateway=_fake_gateway(),  # type: ignore[arg-type]
        repo="r",
        commit="c",
        service_override=StubService(),
    )
    with pytest.raises(errors.McpError):
        backend.get_callgraph_neighbors(goid_h128=1, direction="bad")
    backend.get_callgraph_neighbors(goid_h128=1, direction="incoming")


def test_backend_delegates_after_validation() -> None:
    """Backend methods delegate to the service after validation."""
    backend = DuckDBBackend(
        gateway=_fake_gateway(),  # type: ignore[arg-type]
        repo="r",
        commit="c",
        service_override=StubService(),
    )
    backend.get_function_summary(urn="urn:foo")
    if not isinstance(backend.service, LocalQueryService):
        pytest.fail("Expected LocalQueryService")
    if backend.service.calls != ["get_function_summary"]:
        pytest.fail(f"Unexpected call log: {backend.service.calls}")
