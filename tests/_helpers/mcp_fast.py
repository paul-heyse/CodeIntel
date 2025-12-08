"""FastMCP registrar wrapper for tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal

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
        """Return registered tools; awaits FastMCP coroutine if needed.

        Returns
        -------
        list[Any]
            Tool entries from the underlying FastMCP.
        """
        result = self._mcp.list_tools()
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        return result

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        mount_path: str | None = None,
    ) -> object:
        """Forward run to the underlying FastMCP instance.

        Parameters
        ----------
        transport
            Transport type for FastMCP.
        mount_path
            Optional mount path for HTTP transports.

        Returns
        -------
        object
            Result of FastMCP.run.
        """
        return self._mcp.run(transport=transport, mount_path=mount_path)

    def __getattr__(self, item: str) -> object:
        """Delegate unknown attributes to the underlying FastMCP.

        Returns
        -------
        object
            Attribute from the wrapped FastMCP.
        """
        return getattr(self._mcp, item)


def wrap_fastmcp(name: str) -> FastMcpRegistrar:
    """Construct a FastMCP and wrap it as a registrar.

    Returns
    -------
    FastMcpRegistrar
        Registrar adapter wrapping a FastMCP instance.
    """
    return FastMcpRegistrar(FastMCP(name, json_response=True))


__all__ = ["FastMcpRegistrar", "wrap_fastmcp"]
