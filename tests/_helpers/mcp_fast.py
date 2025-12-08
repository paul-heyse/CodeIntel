"""FastMCP registrar wrapper for tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.tools_base import as_registrar


class FastMcpRegistrar:
    """Adapter exposing McpToolRegistrar over a FastMCP instance."""

    def __init__(self, mcp: FastMCP) -> None:
        self._mcp = mcp
        self.name = mcp.name
        self._registrar = as_registrar(mcp)

    def tool(
        self,
        name: str | None = None,
        **options: object,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        return self._registrar.tool(name=name, **options)

    def list_tools(self) -> list[Any]:
        return self._mcp.list_tools()

    def run(self, *args: object, **kwargs: object) -> object:
        """Forward run to the underlying FastMCP instance."""
        return self._mcp.run(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        """Delegate unknown attributes to the underlying FastMCP."""
        return getattr(self._mcp, item)


def wrap_fastmcp(name: str) -> FastMcpRegistrar:
    """Construct a FastMCP and wrap it as a registrar."""
    return FastMcpRegistrar(FastMCP(name, json_response=True))


__all__ = ["FastMcpRegistrar", "wrap_fastmcp"]
