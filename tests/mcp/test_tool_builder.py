"""Coverage-focused tests for MCP tool builder helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from codeintel.serving.context import current_request_context
from codeintel.serving.mcp import models
from codeintel.serving.mcp.tool_builder import (
    build_tool_from_operation,
    register_tools_for_category,
)
from codeintel.serving.operations.catalog import Operation


@dataclass
class _ModelFromDomain:
    value: str

    @classmethod
    def from_domain(cls, payload: object) -> _ModelFromDomain:
        return cls(value=str(payload))

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}


class _Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.repo = "demo/repo"
        self.commit = "deadbeef"
        self.gateway = SimpleNamespace()  # marker for auto-pipeline branch

    def do_echo(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("do_echo", dict(kwargs)))
        return {"ok": True, **kwargs}

    def do_model(self, **kwargs: object) -> _ModelFromDomain:
        self.calls.append(("do_model", dict(kwargs)))
        return _ModelFromDomain.from_domain(kwargs.get("payload", "p"))


def _make_operation(
    op_id: str,
    backend_method: str,
    *,
    output_model_name: str = "",
) -> Operation:
    return Operation(
        id=op_id,
        category="functions",
        summary="test op",
        description=None,
        http_method=None,
        http_path=None,
        tool_name=f"tool_{op_id}",
        output_model_name=output_model_name,
        backend_method=backend_method,
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


def test_build_tool_from_operation_model_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool builder should handle from_domain/model_dump paths."""
    backend = _Backend()
    spec = _make_operation("echo", "do_model", output_model_name="_ModelFromDomain")
    monkeypatch.setattr(models, "_ModelFromDomain", _ModelFromDomain, raising=True)

    tool = build_tool_from_operation(spec, backend, config=None)
    response = tool(payload="hello", extra=1)
    assert response == {"value": "hello"}
    # Request context set/reset around call
    ctx = current_request_context.get()
    assert ctx is None


def test_build_tool_from_operation_missing_backend_method() -> None:
    """Missing backend method should raise TypeError."""
    backend = _Backend()
    spec = _make_operation("missing", "nope")
    with pytest.raises(TypeError):
        build_tool_from_operation(spec, backend, config=None)


class _RecordingMcp(FastMCP):
    def __init__(self) -> None:
        super().__init__("recorder")
        self.registered: list[tuple[str, str]] = []

    def tool(self, name: str | None = None, description: str | None = None) -> Callable[[Callable[..., object]], Callable[..., object]]:  # type: ignore[override]
        def _decorator(func: Callable[..., object]) -> Callable[..., object]:
            self.registered.append((name or func.__name__, description or ""))
            return func

        return _decorator


def test_register_tools_for_category_registers_expected_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only matching categories should be registered."""
    backend = _Backend()
    mcp = _RecordingMcp()
    specs = [
        _make_operation("fn.one", "do_echo"),
        _make_operation("datasets.list", "do_echo"),
    ]
    monkeypatch.setattr(
        "codeintel.serving.mcp.tool_builder.iter_operations",
        lambda: specs,
    )
    register_tools_for_category(mcp, backend, categories={"functions"})
    names = {name for name, _ in mcp.registered}
    assert names == {"tool_fn.one"}
    # Tool executes and serializes dict payloads
    registered_callable = mcp.tool()(backend.do_echo)
    result = registered_callable()
    assert isinstance(result, dict)


def test_build_tool_auto_pipeline_invokes_prereqs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-pipeline branch should trigger ensure_prereqs_for_mcp when enabled."""
    backend = _Backend()
    spec = _make_operation("echo", "do_echo")

    calls: list[tuple[str, str]] = []

    def _ensure_prereqs(**kwargs: object) -> None:
        calls.append((kwargs.get("op_id", ""), kwargs.get("backend").repo))  # type: ignore[index]

    monkeypatch.setenv("CODEINTEL_AUTO_PIPELINE", "1")
    monkeypatch.setattr(
        "codeintel.serving.auto_pipeline.ensure_prereqs_for_mcp",
        _ensure_prereqs,
    )
    tool = build_tool_from_operation(spec, backend, config=SimpleNamespace(repo="demo", commit="c"))
    _ = tool()
    assert ("echo", "demo/repo") in calls
    # Reset env
    monkeypatch.delenv("CODEINTEL_AUTO_PIPELINE", raising=False)
