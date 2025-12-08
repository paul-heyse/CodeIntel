"""Tests for architecture MCP tools registration and execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pytest

from codeintel.serving import domain_models as dm
from codeintel.serving.context import get_current_request_context
from codeintel.serving.mcp import architecture_tools, errors
from codeintel.serving.mcp.serialization import ResponseFactory
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.operations.catalog import DataSourceType, Operation
from tests._helpers.assertions import expect_equal, expect_in, expect_true
from tests._helpers.mcp import RecordingMcp


@dataclass
class _Model:
    value: str

    @classmethod
    def from_domain(cls, payload: object) -> _Model:
        return cls(value=str(payload))

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}


@dataclass
class _GraphPlanSkipped:
    name: str
    reason: str


@dataclass
class _PlanPluginMetadata:
    name: str
    stage: str
    severity: str
    isolation_kind: str
    supports_incremental: bool
    enabled_by_default: bool
    depends_on: tuple[str, ...]
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    resource_hints: object | None
    options_model: object | None
    options_default: object | None
    version_hash: str
    config_schema_ref: str | None
    row_count_tables: tuple[str, ...]
    cache_populates: tuple[str, ...]
    cache_consumes: tuple[str, ...]
    description: str


@dataclass
class _PlanPlugin:
    metadata: _PlanPluginMetadata


@dataclass
class _GraphPlanResponse:
    plan_id: str
    ordered_plugins: tuple[str, ...]
    skipped_plugins: tuple[_GraphPlanSkipped, ...]
    dep_graph: dict[str, tuple[str, ...]]
    plugin_metadata: dict[str, object]

    @classmethod
    def from_domain(cls, plan: dm.GraphPlan) -> _GraphPlanResponse:
        return cls(
            plan_id=plan.plan_id,
            ordered_plugins=plan.ordered_plugins,
            skipped_plugins=tuple(
                _GraphPlanSkipped(name=str(entry["name"]), reason=str(entry["reason"]))
                for entry in plan.skipped_plugins
            ),
            dep_graph=plan.dep_graph,
            plugin_metadata=cast("dict[str, object]", plan.plugin_metadata),
        )

    def model_dump(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "ordered_plugins": self.ordered_plugins,
            "skipped_plugins": self.skipped_plugins,
            "dep_graph": self.dep_graph,
            "plugin_metadata": self.plugin_metadata,
        }


def _model_resolver(name: str) -> ResponseFactory | None:
    models: dict[str, ResponseFactory] = {
        "FunctionArchitectureResponse": _Model,
        "ModuleArchitectureResponse": _Model,
        "SubsystemSummaryResponse": _Model,
        "ModuleSubsystemResponse": _Model,
        "FileHintsResponse": _Model,
        "SubsystemModulesResponse": _Model,
        "SubsystemSearchResponse": _Model,
        "GraphPlanResponse": _GraphPlanResponse,
    }
    return models.get(name)


def _make_operation(
    op_id: str,
    tool_name: str,
    backend_method: str,
    output_model_name: str,
) -> Operation:
    return Operation(
        id=op_id,
        category="architecture",
        summary=tool_name,
        description=None,
        http_method=None,
        http_path=None,
        tool_name=tool_name,
        output_model_name=output_model_name,
        backend_method=backend_method,
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


@dataclass
class _GraphPlanStub:
    plan_id: str
    ordered_names: tuple[str, ...]
    skipped_plugins: tuple[_GraphPlanSkipped, ...]
    dep_graph: dict[str, tuple[str, ...]]
    plugins: tuple[_PlanPlugin, ...]


@dataclass
class _PlanCall:
    names: tuple[str, ...] | None
    enable: tuple[str, ...] | None
    disable: tuple[str, ...] | None
    options: architecture_tools.PlanningOptions


_plan_calls: list[_PlanCall] = []


def _stub_plan_response() -> _GraphPlanStub:
    metadata = _PlanPluginMetadata(
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
    return _GraphPlanStub(
        plan_id="plan-1",
        ordered_names=("p1",),
        skipped_plugins=(_GraphPlanSkipped(name="skip", reason="r"),),
        dep_graph={"p1": ("dep",)},
        plugins=(_PlanPlugin(metadata=metadata),),
    )


def _reset_plan_calls() -> None:
    _plan_calls.clear()


def _plan_builder_stub(
    names: tuple[str, ...] | None,
    enable: tuple[str, ...] | None,
    disable: tuple[str, ...] | None,
    plan_options: architecture_tools.PlanningOptions,
) -> _GraphPlanStub:
    _plan_calls.append(
        _PlanCall(
            names=names,
            enable=enable,
            disable=disable,
            options=plan_options,
        )
    )
    return _stub_plan_response()


def test_register_architecture_tools_registers_and_executes() -> None:
    """Architecture tools should register expected handlers and serialize payloads."""
    _reset_plan_calls()
    operations = [
        _make_operation(
            "graph.plugins.plan",
            "graph_plugin_plan",
            "graph_plugin_plan",
            "GraphPlanResponse",
        ),
        _make_operation(
            "architecture.function",
            "get_function_architecture",
            "get_function_architecture",
            "FunctionArchitectureResponse",
        ),
        _make_operation(
            "architecture.module",
            "get_module_architecture",
            "get_module_architecture",
            "ModuleArchitectureResponse",
        ),
        _make_operation(
            "subsystems.list",
            "list_subsystems",
            "list_subsystems",
            "SubsystemSummaryResponse",
        ),
        _make_operation(
            "subsystems.module_memberships",
            "get_module_subsystems",
            "get_module_subsystems",
            "ModuleSubsystemResponse",
        ),
        _make_operation(
            "ide.hints",
            "get_file_hints",
            "get_file_hints",
            "FileHintsResponse",
        ),
        _make_operation(
            "subsystems.detail",
            "get_subsystem_modules",
            "get_subsystem_modules",
            "SubsystemModulesResponse",
        ),
        _make_operation(
            "subsystems.search",
            "search_subsystems",
            "search_subsystems",
            "SubsystemSearchResponse",
        ),
        _make_operation(
            "subsystems.summarize",
            "summarize_subsystem",
            "summarize_subsystem",
            "SubsystemModulesResponse",
        ),
    ]
    options = architecture_tools.ArchitectureToolOptions(
        operations=operations,
        model_resolver=_model_resolver,
        plan_builder=_plan_builder_stub,
    )
    backend = _Backend()
    mcp = RecordingMcp()

    architecture_tools.register_architecture_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=options,
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
    last_call = _plan_calls[-1]
    expect_true(last_call.names is None)
    expect_true(last_call.enable is None)
    expect_true(last_call.disable is None)
    expect_true(last_call.options.allow_missing_dependencies)
    expect_equal(str(last_call.options.dependency_policy), "DependencyPolicy.SKIP")
    expect_equal(str(last_call.options.selection_policy), "SelectionPolicy.LENIENT")
    expect_true(get_current_request_context() is None)


def test_architecture_tools_wrap_mcp_error() -> None:
    """McpError should be converted to ProblemDetail payload and reset context."""
    _reset_plan_calls()
    operations = [
        _make_operation(
            "graph.plugins.plan",
            "graph_plugin_plan",
            "graph_plugin_plan",
            "GraphPlanResponse",
        ),
        _make_operation(
            "architecture.function",
            "get_function_architecture",
            "get_function_architecture",
            "FunctionArchitectureResponse",
        ),
    ]
    options = architecture_tools.ArchitectureToolOptions(
        operations=operations,
        model_resolver=_model_resolver,
        plan_builder=_plan_builder_stub,
    )
    backend = _ExplodingBackend()
    mcp = RecordingMcp()

    architecture_tools.register_architecture_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=options,
    )
    result = cast("Callable[..., dict[str, object]]", mcp.registry["get_function_architecture"])(
        goid_h128="1"
    )
    expect_in("error", result)
    expect_true(get_current_request_context() is None)


def test_architecture_tools_type_error_matches_backend_signature() -> None:
    """Backend signature mismatches should surface as TypeError without leaking context."""
    _reset_plan_calls()
    operations = [
        _make_operation(
            "graph.plugins.plan",
            "graph_plugin_plan",
            "graph_plugin_plan",
            "GraphPlanResponse",
        ),
        _make_operation(
            "architecture.module",
            "get_module_architecture",
            "get_module_architecture",
            "ModuleArchitectureResponse",
        ),
        _make_operation(
            "subsystems.list",
            "list_subsystems",
            "list_subsystems",
            "SubsystemSummaryResponse",
        ),
        _make_operation(
            "subsystems.module_memberships",
            "get_module_subsystems",
            "get_module_subsystems",
            "ModuleSubsystemResponse",
        ),
        _make_operation(
            "ide.hints",
            "get_file_hints",
            "get_file_hints",
            "FileHintsResponse",
        ),
        _make_operation(
            "subsystems.detail",
            "get_subsystem_modules",
            "get_subsystem_modules",
            "SubsystemModulesResponse",
        ),
        _make_operation(
            "subsystems.search",
            "search_subsystems",
            "search_subsystems",
            "SubsystemSearchResponse",
        ),
        _make_operation(
            "subsystems.summarize",
            "summarize_subsystem",
            "summarize_subsystem",
            "SubsystemModulesResponse",
        ),
    ]
    options = architecture_tools.ArchitectureToolOptions(
        operations=operations,
        model_resolver=_model_resolver,
        plan_builder=_plan_builder_stub,
    )
    backend = _BadSignatureBackend()
    mcp = RecordingMcp()
    architecture_tools.register_architecture_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=options,
    )
    with pytest.raises(TypeError):
        cast("Callable[..., dict[str, object]]", mcp.registry["get_module_architecture"])(
            module="pkg.mod"
        )
    expect_true(get_current_request_context() is None)
