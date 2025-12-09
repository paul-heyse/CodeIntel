"""Unified MCP registrars for tests.

Provides a single recording registrar that supports sync decorators, async
`list_tools`, and FastMCP compatibility. All tool registrations are captured
for assertions in tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from mcp.server.fastmcp import FastMCP


@dataclass(frozen=True)
class ToolRegistration:
    """Recorded MCP tool registration."""

    name: str
    options: dict[str, object]


@dataclass(frozen=True)
class ToolDescriptor:
    """Typed tool descriptor payload returned by registrars."""

    name: str
    options: dict[str, object]


@runtime_checkable
class ToolRegistrar(Protocol):
    """Protocol for MCP registrars used in tests."""

    def tool(
        self,
        name: str | None = None,
        **options: object,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]: ...

    def list_tools(self) -> list[ToolDescriptor]: ...


class McpRegistrationRecorder:
    """Recorder that tracks tool registrations with simple counters."""

    def __init__(self) -> None:
        self.calls: list[ToolRegistration] = []

    def increment(self, name: str, options: dict[str, object]) -> None:
        """Record a tool registration."""
        self.calls.append(ToolRegistration(name=name, options=options))

    def count(self, name: str | None = None) -> int:
        """Count registrations, optionally filtered by name.

        Returns
        -------
        int
            Number of registrations matching the filter.
        """
        if name is None:
            return len(self.calls)
        return sum(1 for call in self.calls if call.name == name)


class RecordingMcpRegistrar(ToolRegistrar):
    """Recording registrar providing sync decorator API."""

    def __init__(self, app_name: str = "recorder") -> None:
        self.app_name = app_name
        self.registry: dict[str, Callable[..., object]] = {}
        self._registrations: list[ToolRegistration] = []
        self.registrations = McpRegistrationRecorder()

    def tool(
        self,
        name: str | None = None,
        **options: object,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Register a tool and record metadata.

        Returns
        -------
        Callable[[Callable[..., object]], Callable[..., object]]
            Decorator that registers the function.
        """

        def _decorator(func: Callable[..., object]) -> Callable[..., object]:
            tool_name = name or func.__name__
            opts = dict(options)
            self.registrations.increment(tool_name, opts)
            self._registrations.append(ToolRegistration(name=tool_name, options=opts))
            self.registry[tool_name] = func
            return func

        return _decorator

    def list_tools(self) -> list[ToolDescriptor]:
        """Return registered tools in FastMCP-compatible shape.

        Returns
        -------
        list[ToolDescriptor]
            Serialized tool metadata.
        """
        return [
            ToolDescriptor(name=reg.name, options=dict(reg.options)) for reg in self._registrations
        ]


class AsyncRecordingMcpRegistrar(RecordingMcpRegistrar):
    """Async-compatible registrar exposing async list_tools."""

    def list_tools(self) -> list[ToolDescriptor]:
        """Return registered tools as sync API for compatibility.

        Returns
        -------
        list[ToolDescriptor]
            Tool descriptors recorded on the registrar.
        """
        return super().list_tools()


class FastMcpAdapter:
    """Adapter exposing RecordingMcpRegistrar over a FastMCP instance."""

    def __init__(self, mcp: FastMCP) -> None:
        self._mcp = mcp
        self._registrar = RecordingMcpRegistrar(app_name=mcp.name)
        self.name = mcp.name

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
        return list(result)

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        mount_path: str | None = None,
    ) -> object:
        """Forward run to the underlying FastMCP instance.

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
            Attribute from the wrapped FastMCP instance.
        """
        return getattr(self._mcp, item)


def wrap_fastmcp(name: str) -> FastMcpAdapter:
    """Construct a FastMCP and wrap it as a registrar.

    Returns
    -------
    FastMcpAdapter
        Adapter exposing the recording registrar API.
    """
    return FastMcpAdapter(FastMCP(name, json_response=True))


__all__ = [
    "AsyncRecordingMcpRegistrar",
    "FastMcpAdapter",
    "McpRegistrationRecorder",
    "RecordingMcpRegistrar",
    "ToolRegistration",
    "wrap_fastmcp",
]
