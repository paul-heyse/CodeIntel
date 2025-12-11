"""HTTPX client stubs for async MCP backend tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Mapping


class RecordingAsyncClient:
    """Minimal AsyncClient stand-in recording requests and closures."""

    def __init__(self) -> None:
        self.get_calls: list[tuple[str, Mapping[str, str] | None]] = []
        self.closed_count = 0

    async def get(self, path: str, params: Mapping[str, str] | None = None) -> httpx.Response:
        self.get_calls.append((path, params))
        return httpx.Response(200, json={"ok": True, "path": path, "params": dict(params or {})})

    async def aclose(self) -> None:
        self.closed_count += 1


__all__ = ["RecordingAsyncClient"]
