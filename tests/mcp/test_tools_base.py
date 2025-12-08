"""Tests for MCP tool registration orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp import tool_builder, tools_base
from tests._helpers.assertions import expect_equal

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class _RecorderMcp:
    def __init__(self) -> None:
        self.recorded_tools: list[object] = []
        self.tools: list[object] | None = None

    def list_tools(self) -> list[object]:
        return self.recorded_tools


def test_register_tools_delegates_to_category_registrars(monkeypatch: pytest.MonkeyPatch) -> None:
    """register_tools should delegate to each category and expose tools list."""
    calls: list[tuple[str, object | None]] = []

    def _recorder(
        name: str,
    ) -> Callable[[_RecorderMcp, object, ServingConfig | None], None]:
        def _stub(mcp: _RecorderMcp, backend: object, config: ServingConfig | None = None) -> None:
            _ = mcp
            _ = backend
            calls.append((name, getattr(config, "mode", None)))

        return _stub

    monkeypatch.setattr(tools_base, "register_function_tools", _recorder("functions"))
    monkeypatch.setattr(tools_base, "register_profile_tools", _recorder("profiles"))
    monkeypatch.setattr(tools_base, "register_architecture_tools", _recorder("architecture"))
    monkeypatch.setattr(tools_base, "register_dataset_tools", _recorder("datasets"))
    monkeypatch.setattr(tools_base, "register_meta_tools", _recorder("meta"))
    mcp = _RecorderMcp()
    backend = SimpleNamespace()
    config = ServingConfig(mode="remote_api", api_base_url="https://example.invalid")

    tools_base.register_tools(
        cast("FastMCP", mcp), cast("tools_base.QueryBackendOrService", backend), config
    )

    expect_equal(
        calls,
        [
            ("functions", "remote_api"),
            ("profiles", "remote_api"),
            ("architecture", "remote_api"),
            ("datasets", "remote_api"),
            ("meta", None),
        ],
    )
    expect_equal(mcp.tools, mcp.recorded_tools)


def test_register_tools_for_category_type_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """TypeError from build phase should bubble up for unknown backend method."""
    backend = SimpleNamespace()
    mcp = _RecorderMcp()
    spec = SimpleNamespace(
        id="unknown.op",
        category="unknown",
        tool_name="bad_tool",
        backend_method="does_not_exist",
        output_model_name="",
        summary="bad",
        description=None,
    )
    monkeypatch.setattr(
        "codeintel.serving.mcp.tool_builder.iter_operations",
        lambda: (spec,),
    )
    with pytest.raises(TypeError):
        tool_builder.register_tools_for_category(
            cast("FastMCP", mcp),
            cast("tools_base.QueryBackendOrService", backend),
            categories={"unknown"},
            config=None,
        )
