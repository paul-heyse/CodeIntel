"""Tests for function MCP tool wrappers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp import models
from codeintel.serving.mcp.function_tools import register_function_tools
from codeintel.serving.operations.catalog import Operation


@dataclass
class _DomainModel:
    value: str

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}

    @classmethod
    def from_domain(cls, payload: object) -> _DomainModel:
        return cls(value=str(payload))


class _Backend:
    def __init__(self) -> None:
        self.gateway = SimpleNamespace()

    def get_fn(self, *, payload: object = "x") -> _DomainModel:
        return _DomainModel.from_domain(payload)


class _RecordingMcp(FastMCP):
    def __init__(self) -> None:
        super().__init__("recorder")
        self.registered: list[str] = []

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:  # type: ignore[override]
        def _decorator(func: Callable[..., object]) -> Callable[..., object]:
            _ = description
            self.registered.append(name or func.__name__)
            return func

        return _decorator


def test_register_function_tools_registers_and_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Function tool registration wraps backend calls with context and serialization."""
    spec = Operation(
        id="functions.summary",
        category="functions",
        summary="fn",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="functions_summary",
        output_model_name="_DomainModel",
        backend_method="get_fn",
        data_source="docs",
        source_name=None,
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )
    monkeypatch.setattr(models, "_DomainModel", _DomainModel, raising=False)
    monkeypatch.setattr(
        "codeintel.serving.mcp.function_tools.iter_operations",
        lambda: (spec,),
    )
    backend = _Backend()
    mcp = _RecordingMcp()
    register_function_tools(mcp, backend, config=SimpleNamespace(repo="r", commit="c"))
    assert mcp.registered == ["functions_summary"]
    tool_func = mcp.tool()(backend.get_fn)
    result = tool_func(payload="hello")
    assert result == {"value": "hello"}
