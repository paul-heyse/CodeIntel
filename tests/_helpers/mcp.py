"""Reusable MCP recording helpers for tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tests._helpers.records import CallRecorder


class RecordingMcp:
    """Lightweight MCP stand-in that records tool registrations and handlers."""

    def __init__(self, app_name: str = "recorder") -> None:
        self.app_name = app_name
        self.registry: dict[str, Callable[..., object]] = {}
        self.registrations: CallRecorder[McpRegistration] = CallRecorder()

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
            self.registrations.record(McpRegistration(name=tool_name, options=dict(options)))
            self.registry[tool_name] = func
            return func

        return _decorator


@dataclass(frozen=True)
class McpRegistration:
    """Record of a registered MCP tool."""

    name: str
    options: dict[str, object]


__all__ = ["McpRegistration", "RecordingMcp"]
