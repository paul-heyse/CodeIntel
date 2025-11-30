"""Profile-oriented MCP tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap


def register_profile_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register profile-oriented MCP tools."""

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
