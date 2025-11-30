"""Profile-oriented MCP tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import OperationSpec, get_operation_spec


def _require_spec(op_id: str, expected_tool: str) -> OperationSpec:
    spec = get_operation_spec(op_id)
    if spec is None or spec.tool_name != expected_tool:
        message = f"OperationSpec {op_id} has mismatched tool name"
        raise ValueError(message)
    return spec


def register_profile_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register profile-oriented MCP tools."""
    _ = _require_spec("profiles.function", "get_function_profile")
    _ = _require_spec("profiles.file", "get_file_profile")
    _ = _require_spec("profiles.module", "get_module_profile")

    @mcp.tool()
    @_wrap
    def get_function_profile(goid_h128: int) -> dict[str, object] | dict[str, ProblemDetail]:
        resp = backend.get_function_profile(goid_h128=goid_h128)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_file_profile(rel_path: str) -> dict[str, object] | dict[str, ProblemDetail]:
        resp = backend.get_file_profile(rel_path=rel_path)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_module_profile(module: str) -> dict[str, object] | dict[str, ProblemDetail]:
        resp = backend.get_module_profile(module=module)
        return resp.model_dump()


__all__ = ["register_profile_tools"]
