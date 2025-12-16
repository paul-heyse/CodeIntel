"""Validate that the serving-layer public surfaces remain coherent.

This module is executed by `tools.quality_report` to ensure the semantic HTTP and MCP
surfaces expose the expected operations and JSON schemas.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving.http.routes import semantic
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.search.models import SearchQueryResponse
from codeintel.serving.semantic.models import SemanticExplainResponse, SemanticQueryResponse

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.serving.search.models import SearchQueryRequest
    from codeintel.serving.semantic.models import SemanticQueryRequest


EXPECTED_MCP_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "code_search",
        "semantic_catalog",
        "semantic_describe",
        "semantic_explain",
        "semantic_query",
        "serving_meta",
    }
)

EXPECTED_SEMANTIC_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/semantic/views"),
        ("GET", "/semantic/views/{view_id}"),
        ("POST", "/semantic/explain"),
        ("POST", "/semantic/query"),
    }
)


class OperationContractsError(RuntimeError):
    """Raised when serving operation contracts are invalid."""

    def __init__(self, *, issues: Iterable[str]) -> None:
        lines = "\n".join(f"- {issue}" for issue in issues)
        super().__init__(f"Serving operation contracts check failed:\n{lines}")


@dataclass(frozen=True, slots=True)
class _DummyKernel:
    @staticmethod
    def catalog() -> dict[str, object]:
        return {"version": "v1", "snapshot": {}, "views": []}

    @staticmethod
    def describe(view_id: str) -> dict[str, object]:
        return {"id": view_id, "table_key": "docs.v_demo"}

    @staticmethod
    def query(request: SemanticQueryRequest) -> SemanticQueryResponse:
        return SemanticQueryResponse(
            view_id=request.view_id,
            columns=[],
            rows=[],
            truncated=False,
            snapshot={"repo": "demo/repo", "commit": "deadbeef", "run_id": "run-1"},
        )

    @staticmethod
    def explain(request: SemanticQueryRequest) -> SemanticExplainResponse:
        return SemanticExplainResponse(
            view_id=request.view_id,
            sql="SELECT 1",
            plan="demo plan",
            snapshot={"repo": "demo/repo", "commit": "deadbeef", "run_id": "run-1"},
        )

    @staticmethod
    def search(request: SearchQueryRequest) -> SearchQueryResponse:
        return SearchQueryResponse(
            query=request.query,
            results=[],
            truncated=False,
            snapshot={"repo": "demo/repo", "commit": "deadbeef", "run_id": "run-1"},
            engine="pandas",
        )

    @staticmethod
    def meta() -> dict[str, object]:
        return {"repo": "demo/repo", "commit": "deadbeef", "run_id": "run-1"}


def _check_semantic_http_routes() -> list[str]:
    issues: list[str] = []
    observed: set[tuple[str, str]] = set()
    for route in semantic.router.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not isinstance(path, str) or not isinstance(methods, set):
            continue
        for method in methods:
            if isinstance(method, str):
                observed.add((method, path))

    missing = EXPECTED_SEMANTIC_ROUTES - observed
    if missing:
        issues.append(f"Missing semantic HTTP routes: {sorted(missing)}")

    extra = observed - EXPECTED_SEMANTIC_ROUTES
    if extra:
        issues.append(f"Unexpected semantic HTTP routes: {sorted(extra)}")

    return issues


def _get_required_fields(schema: Mapping[str, object]) -> frozenset[str]:
    raw = schema.get("required", [])
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(item for item in raw if isinstance(item, str))


def _check_mcp_tool_schema(tool: object, *, name: str) -> list[str]:
    issues: list[str] = []
    schema = getattr(tool, "inputSchema", None)
    if not isinstance(schema, dict):
        issues.append(f"MCP tool {name} has no inputSchema dict")
        return issues

    required = _get_required_fields(schema)
    if (
        name in {"semantic_describe", "semantic_explain", "semantic_query"}
        and "view_id" not in required
    ):
        issues.append(f"MCP tool {name} must require view_id")
    if name in {"semantic_catalog", "serving_meta"} and required:
        issues.append(f"MCP tool {name} must not require arguments, got {sorted(required)}")
    if name == "code_search" and "query" not in required:
        issues.append("MCP tool code_search must require query")

    if name in {"semantic_explain", "semantic_query"}:
        props = schema.get("properties", {})
        if not isinstance(props, dict):
            issues.append(f"MCP tool {name} inputSchema.properties must be a dict")
            return issues
        expected_props = {"view_id", "filters", "select", "order_by", "pagination"}
        missing_props = expected_props - set(props)
        if missing_props:
            issues.append(f"MCP tool {name} missing properties: {sorted(missing_props)}")

    return issues


async def _check_mcp_tools() -> list[str]:
    issues: list[str] = []
    mcp = build_mcp_app(kernel=_DummyKernel())
    tools = await mcp.list_tools()
    tool_names = {tool.name for tool in tools}

    missing = EXPECTED_MCP_TOOL_NAMES - tool_names
    if missing:
        issues.append(f"Missing MCP tools: {sorted(missing)}")

    extra = tool_names - EXPECTED_MCP_TOOL_NAMES
    if extra:
        issues.append(f"Unexpected MCP tools: {sorted(extra)}")

    for tool in tools:
        if tool.name in EXPECTED_MCP_TOOL_NAMES:
            issues.extend(_check_mcp_tool_schema(tool, name=tool.name))

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
    issues.extend(asyncio.run(_check_mcp_tools()))
    if issues:
        error = OperationContractsError(issues=issues)
        sys.stderr.write(f"{error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
