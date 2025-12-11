"""Tests for MCP model resolver fallbacks and tool name requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from codeintel.serving.mcp import tool_builder
from codeintel.serving.operations.catalog import DataSourceType, Operation
from tests._helpers.assertions import expect_equal, expect_true

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class _Backend:
    calls: int = 0

    def missing(self, *, value: int) -> dict[str, object]:
        self.calls += 1
        return {"value": value}


def test_build_tool_raises_when_model_resolver_missing_output_model() -> None:
    """Model resolver returning None should fall back to raw payload."""
    spec = Operation(
        id="test.missing_model",
        category="test",
        summary="missing model",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="missing_model",
        output_model_name="NonexistentModel",
        backend_method="missing",
        data_source=DataSourceType.COMPUTED,
        source_name=None,
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )
    backend = _Backend()
    tool = tool_builder.build_tool_from_operation(
        spec,
        cast("tool_builder.QueryBackendOrService", backend),
        model_resolver=lambda _name: None,
    )

    result = tool(value=1)
    expect_true(isinstance(result, dict))
    expect_equal(result["value"], 1)


def test_build_tool_requires_tool_name_for_registration() -> None:
    """Operations without tool_name should not be registered by register_all_tools."""
    spec = Operation(
        id="test.no_tool",
        category="test",
        summary="no tool",
        description=None,
        http_method=None,
        http_path=None,
        tool_name=None,
        output_model_name="NonexistentModel",
        backend_method="missing",
        data_source=DataSourceType.COMPUTED,
        source_name=None,
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )
    backend = _Backend()
    recorded: list[str] = []

    class _Mcp:
        def __init__(self, sink: list[str]) -> None:
            self._sink = sink

        def tool(
            self, name: str | None = None, **_: object
        ) -> Callable[[Callable[..., object]], Callable[..., object]]:
            def _decorator(func: Callable[..., object]) -> Callable[..., object]:
                self._sink.append(name or func.__name__)
                return func

            return _decorator

    mcp = _Mcp(recorded)
    tool_builder.register_all_tools(
        mcp,
        cast("tool_builder.QueryBackendOrService", backend),
        options=tool_builder.ToolRegistrationOptions(operations=[spec]),
    )

    expect_true(not recorded)
