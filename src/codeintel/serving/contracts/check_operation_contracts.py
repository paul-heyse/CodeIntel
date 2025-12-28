"""Validate that the serving-layer public surfaces remain coherent.

This module is executed by `tools.quality_report` to ensure the semantic HTTP and MCP
surfaces expose the expected operations and JSON schemas.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from starlette.requests import Request

from codeintel.serving.contracts.http_routes import SERVING_HTTP_ROUTES
from codeintel.serving.errors import (
    ERROR_CODE_CATALOG,
    AuthForbiddenError,
    CodeIntelDomainError,
    ErrorContext,
    ExportTooLargeError,
    SemanticViewNotFoundError,
    exception_to_error_response,
)
from codeintel.serving.http.errors import problem_from_domain_error
from codeintel.serving.http.routes import router as http_router
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.meta.models import ServingKernelMetaResponse
from codeintel.serving.search.models import SearchQueryResponse
from codeintel.serving.semantic.models import (
    SemanticCatalogResponse,
    SemanticCatalogView,
    SemanticExplainResponse,
    SemanticQueryResponse,
    SemanticViewDefaults,
    SemanticViewDescriptionResponse,
)
from codeintel.serving.settings import ServingSettings
from codeintel.serving.snapshot.models import ServingSnapshotIdentity

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from codeintel.serving.db.manager import ServingDBManager
    from codeintel.serving.operations.cancellation import CancelCheck
    from codeintel.serving.search.models import SearchQueryRequest
    from codeintel.serving.semantic.models import SemanticExportRequest, SemanticQueryRequest


EXPECTED_MCP_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "code_search",
        "semantic_catalog",
        "semantic_describe",
        "semantic_explain",
        "semantic_export",
        "semantic_query",
        "serving_meta",
    }
)


class OperationContractsError(RuntimeError):
    """Raised when serving operation contracts are invalid."""

    def __init__(self, *, issues: Iterable[str]) -> None:
        lines = "\n".join(f"- {issue}" for issue in issues)
        super().__init__(f"Serving operation contracts check failed:\n{lines}")


@dataclass(frozen=True, slots=True)
class _DummyKernel:
    @property
    def db(self) -> ServingDBManager:
        msg = "_DummyKernel has no ServingDBManager"
        raise RuntimeError(msg)

    @staticmethod
    def catalog() -> SemanticCatalogResponse:
        return SemanticCatalogResponse(
            version="v1",
            snapshot=ServingSnapshotIdentity(repo="demo/repo", commit="deadbeef", run_id="run-1"),
            views=[
                SemanticCatalogView(
                    id="demo.view",
                    table_key="docs.v_demo",
                    entity="demo",
                    grain="demo",
                    description=None,
                    column_count=0,
                ),
            ],
        )

    @staticmethod
    def describe(view_id: str) -> SemanticViewDescriptionResponse:
        return SemanticViewDescriptionResponse(
            id=view_id,
            table_key="docs.v_demo",
            kind="view",
            entity="demo",
            grain="demo",
            description=None,
            primary_key=[],
            columns=[],
            column_types={},
            joins=[],
            defaults=SemanticViewDefaults(),
            snapshot=ServingSnapshotIdentity(repo="demo/repo", commit="deadbeef", run_id="run-1"),
        )

    @staticmethod
    def query(
        request: SemanticQueryRequest, *, cancel_check: CancelCheck | None = None
    ) -> SemanticQueryResponse:
        _ = cancel_check
        return SemanticQueryResponse(
            view_id=request.view_id,
            columns=[],
            rows=[],
            truncated=False,
            snapshot=ServingSnapshotIdentity(repo="demo/repo", commit="deadbeef", run_id="run-1"),
        )

    @staticmethod
    def query_ipc_stream(
        request: SemanticQueryRequest, *, cancel_check: CancelCheck | None = None
    ) -> Iterator[bytes]:
        _ = cancel_check
        _ = request
        return iter(())

    @staticmethod
    def explain(request: SemanticQueryRequest) -> SemanticExplainResponse:
        return SemanticExplainResponse(
            view_id=request.view_id,
            sql="SELECT 1",
            plan="demo plan",
            snapshot=ServingSnapshotIdentity(repo="demo/repo", commit="deadbeef", run_id="run-1"),
        )

    @staticmethod
    def compile_query_sql(request: SemanticQueryRequest) -> str:
        _ = request
        return "SELECT 1"

    @staticmethod
    def search(request: SearchQueryRequest) -> SearchQueryResponse:
        return SearchQueryResponse(
            query=request.query,
            results=[],
            truncated=False,
            snapshot=ServingSnapshotIdentity(repo="demo/repo", commit="deadbeef", run_id="run-1"),
            engine="pandas",
        )

    @staticmethod
    def meta() -> ServingKernelMetaResponse:
        return ServingKernelMetaResponse(
            repo="demo/repo",
            commit="deadbeef",
            run_id="run-1",
            published_at=datetime.now(UTC),
            semantic_layer_version="demo",
            buildspec_hash="deadbeef",
            buildspec_version=1,
            duckdb={"version": "0.0.0"},
            environment={},
            semantic_views=[],
            datasets=[],
            targets=[],
            schema_inventory={},
        )

    @staticmethod
    def export_rows(
        request: SemanticExportRequest, *, cancel_check: CancelCheck | None = None
    ) -> Iterator[dict[str, object]]:
        _ = cancel_check
        _ = request
        return iter(())

    @staticmethod
    def export_sql(request: SemanticExportRequest) -> str:
        _ = request
        return "SELECT 1"

    @staticmethod
    def export_fingerprint(request: SemanticExportRequest) -> tuple[str, str | None]:
        _ = request
        return ("q_dummy", None)

    @staticmethod
    def export_to_parquet(
        request: SemanticExportRequest,
        *,
        output_path: Path,
        cancel_check: CancelCheck | None = None,
    ) -> int:
        _ = cancel_check
        _ = request
        _ = output_path
        msg = "_DummyKernel does not support parquet exports"
        raise RuntimeError(msg)

    @staticmethod
    def export_to_arrow_ipc(
        request: SemanticExportRequest,
        *,
        output_path: Path,
        cancel_check: CancelCheck | None = None,
    ) -> int:
        _ = cancel_check
        _ = request
        _ = output_path
        msg = "_DummyKernel does not support arrow exports"
        raise RuntimeError(msg)


def _check_semantic_http_routes() -> list[str]:
    issues: list[str] = []
    observed: set[tuple[str, str]] = set()
    expected_routes = {(spec.method, spec.path) for spec in SERVING_HTTP_ROUTES}
    for route in http_router.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not isinstance(path, str) or not isinstance(methods, set):
            continue
        for method in methods:
            if isinstance(method, str):
                observed.add((method, path))

    prefixes = ("/v1/semantic", "/v1/search", "/v1/export")
    semantic_paths = {item for item in observed if item[1].startswith(prefixes)}

    missing = expected_routes - semantic_paths
    if missing:
        issues.append(f"Missing serving HTTP routes: {sorted(missing)}")

    extra = semantic_paths - expected_routes
    if extra:
        issues.append(f"Unexpected serving HTTP routes: {sorted(extra)}")

    return issues


def _request_for_contracts(*, path: str) -> Request:
    scope: dict[str, object] = {
        "type": "http",
        "asgi": {"spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
        "root_path": "",
    }
    request = Request(scope)
    request.state.correlation_id = "contract-check"
    return request


def _check_error_parity() -> list[str]:
    issues: list[str] = []
    ctx = ErrorContext(operation="contract-check", request_id="contract-check")
    request = _request_for_contracts(path="/contract-check")

    samples: list[tuple[str, Exception]] = [
        ("CODEINTEL_SEMANTIC_VIEW_NOT_FOUND", SemanticViewNotFoundError("demo.view")),
        (
            "CODEINTEL_SEMANTIC_INVALID_QUERY",
            CodeIntelDomainError(code="CODEINTEL_SEMANTIC_INVALID_QUERY"),
        ),
        ("CODEINTEL_AUTH_FORBIDDEN", AuthForbiddenError(reason="missing api key")),
        ("CODEINTEL_EXPORT_TOO_LARGE", ExportTooLargeError(row_count=10_000_000)),
        (
            "CODEINTEL_EXPORT_INVALID_REQUEST",
            CodeIntelDomainError(code="CODEINTEL_EXPORT_INVALID_REQUEST"),
        ),
    ]

    for expected_code, exc in samples:
        error_response = exception_to_error_response(exc, context=ctx)
        if error_response.error.code != expected_code:
            issues.append(
                "MCP error mapping mismatch: "
                f"expected={expected_code} got={error_response.error.code} exc={type(exc).__name__}"
            )

        if not isinstance(exc, CodeIntelDomainError):
            msg = f"Contract test expects domain errors only, got {type(exc).__name__}"
            raise TypeError(msg)
        problem = problem_from_domain_error(request, exc)
        problem_code = problem.extensions.get("code")
        if problem_code != expected_code:
            issues.append(
                "HTTP problem code mismatch: "
                f"expected={expected_code} got={problem_code} exc={type(exc).__name__}"
            )
        tmpl = ERROR_CODE_CATALOG.get(expected_code)
        if tmpl is None:
            issues.append(f"Missing error code entry for {expected_code} in canonical catalog")
        elif problem.status != tmpl.error_code.status:
            issues.append(
                "HTTP problem status mismatch: "
                f"code={expected_code} expected={tmpl.error_code.status} got={problem.status}"
            )

    return issues


def _get_required_fields(schema: Mapping[str, object]) -> frozenset[str]:
    raw = schema.get("required", [])
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(item for item in raw if isinstance(item, str))


def _get_tool_input_schema(tool: object) -> Mapping[str, object] | None:
    schema = getattr(tool, "inputSchema", None)
    if isinstance(schema, dict):
        return schema

    schema = getattr(tool, "parameters", None)
    if isinstance(schema, dict):
        return schema

    return None


def _resolve_schema_ref(
    schema: Mapping[str, object],
    *,
    root: Mapping[str, object],
) -> Mapping[str, object] | None:
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return schema
    if not ref.startswith("#/$defs/"):
        return None
    defs = root.get("$defs")
    if not isinstance(defs, Mapping):
        return None
    target = defs.get(ref.removeprefix("#/$defs/"))
    if isinstance(target, Mapping):
        return target
    return None


def _request_schema(schema: Mapping[str, object]) -> Mapping[str, object] | None:
    props = _tool_properties(schema)
    if props is None:
        return None
    request_schema = props.get("request")
    if not isinstance(request_schema, Mapping):
        return None
    resolved = _resolve_schema_ref(request_schema, root=schema)
    return resolved if resolved is not None else request_schema


_MCP_TOOLS_REQUIRE_VIEW_ID: frozenset[str] = frozenset(
    {"semantic_describe", "semantic_explain", "semantic_export", "semantic_query"}
)
_MCP_TOOLS_NO_REQUIRED_ARGS: frozenset[str] = frozenset({"semantic_catalog", "serving_meta"})
_MCP_TOOLS_REQUIRED_ARG: dict[str, str] = {"code_search": "query"}
_MCP_TOOL_EXPECTED_PROPERTIES: dict[str, frozenset[str]] = {
    "semantic_explain": frozenset({"view_id", "filters", "select", "order_by", "pagination"}),
    "semantic_query": frozenset({"view_id", "filters", "select", "order_by", "pagination"}),
    "semantic_export": frozenset({"view_id", "filters", "export_format", "limit"}),
}


def _tool_properties(schema: Mapping[str, object]) -> Mapping[str, object] | None:
    props = schema.get("properties", {})
    if isinstance(props, dict):
        return props
    return None


def _check_mcp_tool_properties(
    schema: Mapping[str, object], *, name: str, expected: frozenset[str]
) -> list[str]:
    issues: list[str] = []
    props = _tool_properties(schema)
    if props is None:
        issues.append(f"MCP tool {name} inputSchema.properties must be a dict")
        return issues
    missing_props = expected - set(props)
    if missing_props:
        issues.append(f"MCP tool {name} missing properties: {sorted(missing_props)}")
    return issues


def _check_mcp_tool_schema(tool: object, *, name: str) -> list[str]:
    issues: list[str] = []
    schema = _get_tool_input_schema(tool)
    if schema is None:
        issues.append(f"MCP tool {name} has no JSON schema for parameters")
        return issues

    schema_to_check = _request_schema(schema) or schema
    required = _get_required_fields(schema_to_check)
    if name in _MCP_TOOLS_REQUIRE_VIEW_ID and "view_id" not in required:
        issues.append(f"MCP tool {name} must require view_id")
    if name in _MCP_TOOLS_NO_REQUIRED_ARGS and required:
        issues.append(f"MCP tool {name} must not require arguments, got {sorted(required)}")
    required_arg = _MCP_TOOLS_REQUIRED_ARG.get(name)
    if required_arg is not None and required_arg not in required:
        issues.append(f"MCP tool {name} must require {required_arg}")

    expected_props = _MCP_TOOL_EXPECTED_PROPERTIES.get(name)
    if expected_props is not None:
        issues.extend(
            _check_mcp_tool_properties(schema_to_check, name=name, expected=expected_props)
        )

    return issues


async def _check_mcp_tools() -> list[str]:
    issues: list[str] = []
    settings = ServingSettings(serve_dir=Path.cwd() / "build" / "operation-contracts")
    mcp = build_mcp_app(
        kernel=_DummyKernel(),
        settings=settings,
    )
    tool_map = await mcp.get_tools()
    tools = list(tool_map.values())
    tool_names = set(tool_map)

    missing = EXPECTED_MCP_TOOL_NAMES - tool_names
    if missing:
        issues.append(f"Missing MCP tools: {sorted(missing)}")

    extra = tool_names - EXPECTED_MCP_TOOL_NAMES
    if extra:
        issues.append(f"Unexpected MCP tools: {sorted(extra)}")

    for tool in tools:
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name in EXPECTED_MCP_TOOL_NAMES:
            issues.extend(_check_mcp_tool_schema(tool, name=name))

    return issues


def main() -> int:
    """Run serving operation-contract checks.

    Returns
    -------
    int
        Process exit code (0 = success, 1 = failure).
    """
    issues: list[str] = []
    issues.extend(_check_semantic_http_routes())
    issues.extend(_check_error_parity())
    issues.extend(asyncio.run(_check_mcp_tools()))
    if issues:
        error = OperationContractsError(issues=issues)
        sys.stderr.write(f"{error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
