"""Tests for architecture MCP tools registration and execution."""

from __future__ import annotations

from collections.abc import Callable, ItemsView, Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.serving.context import get_current_request_context
from codeintel.serving.mcp import architecture_tools, errors
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from tests._helpers.assertions import expect_equal, expect_in, expect_true
from tests._helpers.mcp import RecordingMcp

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


@dataclass
class _Model:
    value: str

    @classmethod
    def from_domain(cls, payload: object) -> _Model:
        return cls(value=str(payload))

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}


class _GraphPlanResponse:
    def __init__(
        self,
        *,
        plan_id: str,
        ordered_plugins: tuple[str, ...],
        skipped_plugins: tuple[object, ...],
        dep_graph: dict[str, tuple[str, ...]],
        plugin_metadata: dict[str, object],
    ) -> None:
        self.plan_id = plan_id
        self.ordered_plugins = ordered_plugins
        self.skipped_plugins = skipped_plugins
        self.dep_graph = dep_graph
        self.plugin_metadata = plugin_metadata

    def model_dump(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "ordered_plugins": self.ordered_plugins,
            "skipped_plugins": self.skipped_plugins,
            "dep_graph": self.dep_graph,
            "plugin_metadata": self.plugin_metadata,
        }


class _GraphPlanPluginMetadata:
    def __init__(self, **kwargs: object) -> None:
        self.payload: dict[str, object] = dict(kwargs)

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def items(self) -> ItemsView[str, object]:
        return self.payload.items()


@dataclass
class _GraphPlanSkipped:
    name: str
    reason: str


class _Backend:
    def __init__(self) -> None:
        self.repo = "demo/repo"
        self.commit = "abc123"
        self.calls: list[str] = []

    def get_function_architecture(self, **_: object) -> _Model:
        self.calls.append("function")
        return _Model.from_domain("fn")

    def get_module_architecture(self, **_: object) -> _Model:
        self.calls.append("module")
        return _Model.from_domain("mod")


class _ExplodingBackend(_Backend):
    """Backend that raises a backend failure for function architecture."""

    def get_function_architecture(self, **_: object) -> _Model:
        self.calls.append("function_error")
        error_message = "bad"
        raise errors.backend_failure(error_message)


class _BadSignatureBackend(_Backend):
    """Backend that raises TypeError to simulate signature mismatch."""

    def get_module_architecture(self, **kwargs: object) -> _Model:
        module_name = str(kwargs.get("module", "unknown"))
        self.calls.append(f"bad_module:{module_name}")
        error_message = "Invalid module architecture request"
        raise TypeError(error_message)

    def list_subsystems(self, **_: object) -> _Model:
        self.calls.append("list_subsystems")
        return _Model.from_domain("subs")

    def get_module_subsystems(self, **_: object) -> _Model:
        self.calls.append("module_subsystems")
        return _Model.from_domain("mods")

    def get_file_hints(self, **_: object) -> _Model:
        self.calls.append("hints")
        return _Model.from_domain("hint")

    def get_subsystem_modules(self, **_: object) -> _Model:
        self.calls.append("subsystem_modules")
        return _Model.from_domain("submods")

    def search_subsystems(self, **_: object) -> _Model:
        self.calls.append("search")
        return _Model.from_domain("search")

    def summarize_subsystem(self, **_: object) -> _Model:
        self.calls.append("summarize")
        return _Model.from_domain("summary")


def _patch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(architecture_tools, "FunctionArchitectureResponse", _Model, raising=False)
    monkeypatch.setattr(architecture_tools, "ModuleArchitectureResponse", _Model, raising=False)
    monkeypatch.setattr(architecture_tools, "SubsystemSummaryResponse", _Model, raising=False)
    monkeypatch.setattr(architecture_tools, "ModuleSubsystemResponse", _Model, raising=False)
    monkeypatch.setattr(architecture_tools, "FileHintsResponse", _Model, raising=False)
    monkeypatch.setattr(architecture_tools, "SubsystemModulesResponse", _Model, raising=False)
    monkeypatch.setattr(architecture_tools, "SubsystemSearchResponse", _Model, raising=False)
    monkeypatch.setattr(architecture_tools, "SubsystemSummaryResponse", _Model, raising=False)
    monkeypatch.setattr(architecture_tools, "GraphPlanResponse", _GraphPlanResponse, raising=False)
    monkeypatch.setattr(
        architecture_tools, "GraphPlanPluginMetadata", _GraphPlanPluginMetadata, raising=False
    )
    monkeypatch.setattr(architecture_tools, "GraphPlanSkipped", _GraphPlanSkipped, raising=False)


def _stub_plan_response() -> SimpleNamespace:
    return SimpleNamespace(
        plan_id="plan-1",
        ordered_names=("p1",),
        skipped_plugins=(SimpleNamespace(name="skip", reason="r"),),
        dep_graph={"p1": ("dep",)},
        plugins=(
            SimpleNamespace(
                metadata=SimpleNamespace(
                    name="p1",
                    stage="stage",
                    severity="high",
                    isolation_kind="none",
                    supports_incremental=False,
                    enabled_by_default=True,
                    depends_on=(),
                    provides=(),
                    requires=(),
                    resource_hints=None,
                    options_model=None,
                    options_default=None,
                    version_hash="v1",
                    config_schema_ref=None,
                    row_count_tables=(),
                    cache_populates=(),
                    cache_consumes=(),
                    description="desc",
                )
            ),
        ),
    )


def test_register_architecture_tools_registers_and_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Architecture tools should register expected handlers and serialize payloads."""
    _patch_models(monkeypatch)
    monkeypatch.setattr(
        architecture_tools,
        "plan_graph_plugins",
        lambda **_: _stub_plan_response(),
    )
    backend = _Backend()
    mcp = RecordingMcp()

    architecture_tools.register_architecture_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend), config=None
    )

    expected = {
        "graph_plugin_plan",
        "get_function_architecture",
        "get_module_architecture",
        "list_subsystems",
        "get_module_subsystems",
        "get_file_hints",
        "get_subsystem_modules",
        "search_subsystems",
        "summarize_subsystem",
    }
    expect_true(expected.issubset(set(mcp.registry)))

    # Execute a couple of tools to ensure serialization and context reset
    result_fn = cast("Callable[..., dict[str, object]]", mcp.registry["get_function_architecture"])(
        goid_h128="1"
    )
    result_plan = cast("Callable[[], dict[str, object]]", mcp.registry["graph_plugin_plan"])()
    expect_equal(result_fn, {"value": "fn"})
    expect_in("plan_id", result_plan)
    expect_true(get_current_request_context() is None)


def test_architecture_tools_wrap_mcp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """McpError should be converted to ProblemDetail payload and reset context."""
    _patch_models(monkeypatch)
    monkeypatch.setattr(
        architecture_tools,
        "plan_graph_plugins",
        lambda **_: _stub_plan_response(),
    )
    backend = _ExplodingBackend()
    mcp = RecordingMcp()

    architecture_tools.register_architecture_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend), config=None
    )
    result = cast("Callable[..., dict[str, object]]", mcp.registry["get_function_architecture"])(
        goid_h128="1"
    )
    expect_in("error", result)
    expect_true(get_current_request_context() is None)


def test_architecture_tools_type_error_matches_backend_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend signature mismatches should surface as TypeError without leaking context."""
    _patch_models(monkeypatch)
    backend = _BadSignatureBackend()
    monkeypatch.setattr(
        architecture_tools,
        "plan_graph_plugins",
        lambda **_: _stub_plan_response(),
    )
    mcp = RecordingMcp()
    architecture_tools.register_architecture_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend), config=None
    )
    with pytest.raises(TypeError):
        cast("Callable[..., dict[str, object]]", mcp.registry["get_module_architecture"])(
            module="pkg.mod"
        )
    expect_true(get_current_request_context() is None)
