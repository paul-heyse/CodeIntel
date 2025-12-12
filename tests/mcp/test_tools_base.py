"""Tests for MCP tool registration orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.serving.mcp.tool_builder import ToolRegistrationOptions, register_tools_for_category
from codeintel.serving.operations.catalog import DataSourceType, Operation
from tests._helpers.assertions import expect_equal, expect_in
from tests._helpers.mcp_tools import make_mcp_context

if TYPE_CHECKING:
    from codeintel.serving.mcp.tool_utils import QueryBackendOrService


@dataclass
class _StubBackend:
    """Backend with generic call recording."""

    calls: list[str]

    def functions(self) -> str:
        self.calls.append("functions")
        return "ok"

    def profiles(self) -> str:
        self.calls.append("profiles")
        return "ok"

    def architecture(self) -> str:
        self.calls.append("architecture")
        return "ok"

    def datasets(self) -> str:
        self.calls.append("datasets")
        return "ok"


def _operations() -> list[Operation]:
    """Minimal operations across categories.

    Returns
    -------
    list[Operation]
        Operations spanning multiple categories for MCP registration tests.
    """
    return [
        Operation(
            id="fn.op",
            category="functions",
            summary="fn",
            description=None,
            http_method=None,
            http_path=None,
            tool_name="fn_tool",
            output_model_name="",
            backend_method="functions",
            data_source=DataSourceType.VIEW,
            source_name=None,
            repository_method=None,
            required_datasets=(),
            required_graphs=(),
            exposed_datasets=(),
            supports_pagination=False,
            default_limit=None,
            max_limit=None,
        ),
        Operation(
            id="prof.op",
            category="profiles",
            summary="prof",
            description=None,
            http_method=None,
            http_path=None,
            tool_name="prof_tool",
            output_model_name="",
            backend_method="profiles",
            data_source=DataSourceType.VIEW,
            source_name=None,
            repository_method=None,
            required_datasets=(),
            required_graphs=(),
            exposed_datasets=(),
            supports_pagination=False,
            default_limit=None,
            max_limit=None,
        ),
        Operation(
            id="arch.op",
            category="architecture",
            summary="arch",
            description=None,
            http_method=None,
            http_path=None,
            tool_name="arch_tool",
            output_model_name="",
            backend_method="architecture",
            data_source=DataSourceType.VIEW,
            source_name=None,
            repository_method=None,
            required_datasets=(),
            required_graphs=(),
            exposed_datasets=(),
            supports_pagination=False,
            default_limit=None,
            max_limit=None,
        ),
        Operation(
            id="data.op",
            category="datasets",
            summary="data",
            description=None,
            http_method=None,
            http_path=None,
            tool_name="data_tool",
            output_model_name="",
            backend_method="datasets",
            data_source=DataSourceType.VIEW,
            source_name=None,
            repository_method=None,
            required_datasets=(),
            required_graphs=(),
            exposed_datasets=(),
            supports_pagination=False,
            default_limit=None,
            max_limit=None,
        ),
    ]


class _MissingBackend:
    """Backend without the requested method for negative testing."""


def test_register_tools_registers_all_categories() -> None:
    """Tools are registered for all categories with injected operations."""
    backend = _StubBackend(calls=[])
    ctx = make_mcp_context(
        backend=cast("QueryBackendOrService", backend),
        operations=_operations(),
    )

    ctx.register({"functions", "profiles", "architecture", "datasets"})

    expect_equal(set(ctx.mcp.registry.keys()), {"fn_tool", "prof_tool", "arch_tool", "data_tool"})

    ctx.mcp.registry["fn_tool"]()
    ctx.mcp.registry["prof_tool"]()
    ctx.mcp.registry["arch_tool"]()
    ctx.mcp.registry["data_tool"]()
    expect_equal(backend.calls, ["functions", "profiles", "architecture", "datasets"])


def test_register_tools_for_category_type_error_propagates() -> None:
    """TypeError bubbles when backend method is missing."""
    backend = _MissingBackend()
    mcp_ctx = make_mcp_context(
        backend=cast("QueryBackendOrService", backend),
        operations=[
            Operation(
                id="unknown.op",
                category="unknown",
                summary="bad",
                description=None,
                http_method=None,
                http_path=None,
                tool_name="bad_tool",
                output_model_name="",
                backend_method="does_not_exist",
                data_source=DataSourceType.VIEW,
                source_name=None,
                repository_method=None,
                required_datasets=(),
                required_graphs=(),
                exposed_datasets=(),
                supports_pagination=False,
                default_limit=None,
                max_limit=None,
            )
        ],
    )
    with pytest.raises(TypeError) as excinfo:
        register_tools_for_category(
            mcp_ctx.mcp,
            mcp_ctx.backend,
            categories={"unknown"},
            config=None,
            options=ToolRegistrationOptions(operations=mcp_ctx.operations),
        )
    expect_in("does_not_exist", str(excinfo.value))
