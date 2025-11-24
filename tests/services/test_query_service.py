"""Unit tests for LocalQueryService and HttpQueryService."""

from __future__ import annotations

import importlib
from http import HTTPStatus
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from codeintel.mcp.backend import QueryBackend
from codeintel.mcp.config import McpServerConfig
from codeintel.mcp.models import (
    DatasetRowsResponse,
    FunctionSummaryResponse,
    ResponseMeta,
    ViewRow,
)
from codeintel.mcp.query_service import BackendLimits, DuckDBQueryService
from codeintel.server.fastapi import ApiAppConfig, BackendResource, create_app
from codeintel.services.query_service import HttpQueryService, LocalQueryService


class StubDuckQuery:
    """Minimal DuckDBQueryService stand-in for LocalQueryService tests."""

    def __init__(self) -> None:
        self.limits = BackendLimits(default_limit=3, max_rows_per_call=5)
        self.last_call: tuple[str, int | None, int] | None = None
        self.summary_calls = 0

    def read_dataset_rows(
        self, *, dataset_name: str, limit: int | None, offset: int
    ) -> DatasetRowsResponse:
        """
        Record the dataset read request and return a stubbed response.

        Returns
        -------
        DatasetRowsResponse
            Response containing a single row for test assertions.

        Raises
        ------
        ValueError
            When the dataset is not recognized.
        """
        self.last_call = (dataset_name, limit, offset)
        if dataset_name == "missing":
            message = "Unknown dataset"
            raise ValueError(message)
        return DatasetRowsResponse(
            dataset=dataset_name,
            limit=self.limits.max_rows_per_call,
            offset=offset,
            rows=[ViewRow.model_validate({"issue": "ok"})],
            meta=ResponseMeta(),
        )

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        """
        Record function summary requests and return a canned response.

        Returns
        -------
        FunctionSummaryResponse
            Response containing the requested qualname when provided.
        """
        _ = (goid_h128, rel_path)
        self.summary_calls += 1
        return FunctionSummaryResponse(
            found=True,
            summary=ViewRow.model_validate({"qualname": qualname or "fn", "urn": urn}),
            meta=ResponseMeta(),
        )


def test_local_read_dataset_rows_defaults_limit() -> None:
    """Local service uses backend default limit when none is provided."""
    query = StubDuckQuery()
    service = LocalQueryService(query=cast("DuckDBQueryService", query), dataset_tables={})
    _resp = service.read_dataset_rows(dataset_name="docs", limit=None, offset=0)
    if query.last_call != ("docs", query.limits.default_limit, 0):
        message = f"Unexpected call args: {query.last_call}"
        pytest.fail(message)


def test_local_read_dataset_rows_propagates_errors() -> None:
    """Local service does not swallow errors from the query layer."""
    service = LocalQueryService(
        query=cast("DuckDBQueryService", StubDuckQuery()), dataset_tables={}
    )
    with pytest.raises(ValueError, match="Unknown dataset"):
        service.read_dataset_rows(dataset_name="missing", limit=1, offset=0)


def test_http_read_dataset_rows_clamps_limit() -> None:
    """HTTP service clamps large limits and forwards applied values."""
    calls: list[tuple[str, dict[str, object]]] = []

    def request_json(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((path, params))
        return {
            "dataset": "functions",
            "limit": params["limit"],
            "offset": params["offset"],
            "rows": [{"issue": "ok"}],
        }

    limits = BackendLimits(default_limit=10, max_rows_per_call=20)
    service = HttpQueryService(request_json=request_json, limits=limits)
    resp = service.read_dataset_rows(dataset_name="functions", limit=1000, offset=0)
    expected_call = [("/datasets/functions", {"limit": 20, "offset": 0})]
    if calls != expected_call:
        message = f"Unexpected HTTP calls: {calls}"
        pytest.fail(message)
    codes = {msg.code for msg in resp.meta.messages}
    if "limit_clamped" not in codes:
        message = f"Missing clamp message: {codes}"
        pytest.fail(message)


def test_http_read_dataset_rows_invalid_offset() -> None:
    """HTTP service returns empty rows and messages when offset is invalid."""
    limits = BackendLimits(default_limit=5, max_rows_per_call=10)
    service = HttpQueryService(request_json=lambda _p, _q: {}, limits=limits)
    resp = service.read_dataset_rows(dataset_name="functions", limit=2, offset=-1)
    if resp.rows:
        pytest.fail("Expected no rows when offset is invalid")
    codes = {msg.code for msg in resp.meta.messages}
    if "offset_invalid" not in codes:
        message = f"Missing offset_invalid message: {codes}"
        pytest.fail(message)


def test_http_high_risk_clamps_using_limits() -> None:
    """High-risk listing uses clamped limits before issuing HTTP calls."""
    limits = BackendLimits(default_limit=5, max_rows_per_call=5)
    requested: dict[str, object] = {}

    def capture(path: str, params: dict[str, object]) -> dict[str, object]:
        requested["path"] = path
        requested["params"] = params
        return {"functions": [], "truncated": False, "meta": {}}

    service = HttpQueryService(request_json=capture, limits=limits)
    _resp = service.list_high_risk_functions(limit=20)
    expected_params = {"min_risk": 0.7, "limit": 5, "tested_only": False}
    if requested.get("path") != "/functions/high-risk":
        pytest.fail(f"Unexpected path: {requested}")
    if requested.get("params") != expected_params:
        pytest.fail(f"Unexpected params: {requested}")


def test_fastapi_delegates_to_query_service(tmp_path: Path) -> None:
    """FastAPI routes delegate to the injected QueryService instance."""
    calls: list[str] = []

    class RecordingService(LocalQueryService):
        """Service that records method calls."""

        def __init__(self) -> None:
            super().__init__(query=cast("DuckDBQueryService", StubDuckQuery()), dataset_tables={})

        def get_function_summary(
            self,
            *,
            urn: str | None = None,
            goid_h128: int | None = None,
            rel_path: str | None = None,
            qualname: str | None = None,
        ) -> FunctionSummaryResponse:
            _ = (goid_h128, rel_path)
            calls.append("get_function_summary")
            return super().get_function_summary(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
            )

    service = RecordingService()
    backend_obj = type("StubBackend", (), {"service": service})()
    backend = cast("QueryBackend", backend_obj)

    def load_config() -> ApiAppConfig:
        server_cfg = McpServerConfig(
            mode="remote_api",
            repo_root=tmp_path,
            repo="r",
            commit="c",
            api_base_url="http://test",
        )
        return ApiAppConfig(server=server_cfg, read_only=True)

    def backend_factory(_: ApiAppConfig, *, _gateway: object | None = None) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)
    with TestClient(app) as client:
        response = client.get("/function/summary", params={"urn": "urn:foo"})
        if response.status_code != HTTPStatus.OK:
            message = f"Unexpected status: {response.status_code}"
            pytest.fail(message)
    if "get_function_summary" not in calls:
        pytest.fail("Service was not invoked by route")


def test_mcp_tool_delegation() -> None:
    """MCP tool registration delegates to the provided service."""
    fastmcp_mod = pytest.importorskip("mcp.server.fastmcp")
    registry = importlib.import_module("codeintel.mcp.registry")

    calls: list[str] = []

    class DummyService(LocalQueryService):
        """Service that records method invocations."""

        def __init__(self) -> None:
            super().__init__(query=cast("DuckDBQueryService", StubDuckQuery()), dataset_tables={})

        def get_function_summary(
            self,
            *,
            urn: str | None = None,
            goid_h128: int | None = None,
            rel_path: str | None = None,
            qualname: str | None = None,
        ) -> FunctionSummaryResponse:
            _ = (goid_h128, rel_path)
            calls.append("get_function_summary")
            return super().get_function_summary(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
            )

    mcp = fastmcp_mod.FastMCP("test")
    if not hasattr(mcp, "tools"):
        pytest.skip("FastMCP tools registry not available")
    service = DummyService()
    registry.register_tools(mcp, service)
    tool = mcp.tools["get_function_summary"]
    _result = tool({"urn": "urn:foo"})
    if "get_function_summary" not in calls:
        pytest.fail("Service was not invoked via MCP tool")
