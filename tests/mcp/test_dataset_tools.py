"""Tests for dataset MCP tool wrappers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp import models
from codeintel.serving.mcp.dataset_tools import _serialize_payload, register_dataset_tools
from codeintel.serving.operations.catalog import Operation


@dataclass
class _Dumpable:
    value: str

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}


@dataclass
class _FromDomain:
    value: str

    @classmethod
    def from_domain(cls, payload: object) -> _FromDomain:
        return cls(value=str(payload))

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}


def test_serialize_payload_prefers_model_dump_and_from_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serialization handles model_dump and from_domain fallbacks."""
    dumpable = _Dumpable("x")
    assert _serialize_payload(dumpable, None) == {"value": "x"}

    monkeypatch.setattr(models, "_FromDomain", _FromDomain, raising=True)
    payload = _serialize_payload("y", models._FromDomain)
    assert payload == {"value": "y"}


class _Backend:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.gateway = SimpleNamespace()

    def list_dataset(self, **_: object) -> list[_Dumpable]:
        self.calls.append("list_dataset")
        return [_Dumpable("one"), _Dumpable("two")]


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
            self.registered.append(name or func.__name__)
            return func

        return _decorator


def test_register_dataset_tools_registers_and_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dataset tool registration should wrap backend methods and serialize lists."""
    spec = Operation(
        id="datasets.list",
        category="datasets",
        summary="list datasets",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="datasets_list",
        output_model_name="_Dumpable",
        backend_method="list_dataset",
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
    monkeypatch.setattr(models, "_Dumpable", _Dumpable, raising=True)
    monkeypatch.setattr(
        "codeintel.serving.mcp.dataset_tools.iter_operations",
        lambda: (spec,),
    )
    backend = _Backend()
    mcp = _RecordingMcp()
    register_dataset_tools(mcp, backend, config=SimpleNamespace(repo="r", commit="c"))
    assert mcp.registered == ["datasets_list"]
    # Execute registered tool
    tool_func = mcp.tool()(backend.list_dataset)
    result = tool_func()
    assert isinstance(result, list)
    assert result[0] == {"value": "one"}
