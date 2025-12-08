"""Async-capable MCP registrar stub for tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tests._helpers.records import CallRecorder


@dataclass(frozen=True)
class _ToolRegistration:
    name: str
    options: dict[str, object]


class AsyncRecordingMcp:
    """Lightweight MCP stand-in that records tool registrations and supports async list_tools."""

    def __init__(self, app_name: str = "async-recorder") -> None:
        self.app_name = app_name
        self.registry: dict[str, Callable[..., object]] = {}
        self.registrations: CallRecorder[_ToolRegistration] = CallRecorder()

    def tool(
        self,
        name: str | None = None,
        **options: object,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Decorate a tool and record its registration metadata.

        Returns
        -------
        Callable[[Callable[..., object]], Callable[..., object]]
            Decorator that registers the provided function.
        """

        def _decorator(func: Callable[..., object]) -> Callable[..., object]:
            tool_name = name or func.__name__
            self.registrations.record(_ToolRegistration(name=tool_name, options=dict(options)))
            self.registry[tool_name] = func
            return func

        return _decorator

    async def list_tools(self) -> list[object]:
        """Return registered tools in the FastMCP-compatible shape.

        Returns
        -------
        list[object]
            Serialized tool entries with names.
        """
        return [{"name": name} for name in self.registry]


__all__ = ["AsyncRecordingMcp"]
