"""Generic analytics plugin definitions and registry."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntime
from codeintel.analytics.graphs.contracts import ContractChecker
from codeintel.analytics.runtime_manifest import AnalyticsSkippedStep
from codeintel.config.steps_analytics import (
    BehavioralCoverageStepConfig,
    CoverageAnalyticsStepConfig,
    DataModelsStepConfig,
    DataModelUsageStepConfig,
    EntryPointsStepConfig,
    FunctionAnalyticsStepConfig,
    FunctionContractsStepConfig,
    FunctionEffectsStepConfig,
    FunctionHistoryStepConfig,
    HistoryTimeseriesStepConfig,
    HotspotsStepConfig,
    ProfilesAnalyticsStepConfig,
    SemanticRolesStepConfig,
    SubsystemsStepConfig,
    TestCoverageStepConfig,
    TestProfileStepConfig,
)
from codeintel.config.steps_graphs import (
    ConfigDataFlowStepConfig,
    ExternalDependenciesStepConfig,
    GraphMetricsStepConfig,
    GraphRunScope,
)
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.graphs.function_catalog_service import FunctionCatalogProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceHints:
    """
    Runtime resource hints for schedulers / harness.

    This is intentionally generic so both graph and non-graph analytics can use it.
    """

    max_runtime_ms: int | None = None
    max_memory_mb: int | None = None
    requires_gpu: bool = False
    priority: int = 0


@dataclass
class AnalyticsExecutionContext:
    """
    Shared execution context for generic analytics plugins.

    Graph plugins can adapt this into GraphMetricExecutionContext using a
    context_factory; non-graph plugins typically consume this directly.
    """

    gateway: StorageGateway
    analytics_context: AnalyticsContext | None
    repo: str
    commit: str

    graph_runtime: GraphRuntime | None = None
    catalog_provider: FunctionCatalogProvider | None = None

    function_cfg: FunctionAnalyticsStepConfig | None = None
    function_effects_cfg: FunctionEffectsStepConfig | None = None
    function_contracts_cfg: FunctionContractsStepConfig | None = None
    function_history_cfg: FunctionHistoryStepConfig | None = None
    test_profile_cfg: TestProfileStepConfig | None = None
    behavioral_cfg: BehavioralCoverageStepConfig | None = None
    hotspots_cfg: HotspotsStepConfig | None = None
    subsystems_cfg: SubsystemsStepConfig | None = None
    semantic_roles_cfg: SemanticRolesStepConfig | None = None
    data_models_cfg: DataModelsStepConfig | None = None
    data_model_usage_cfg: DataModelUsageStepConfig | None = None
    entrypoints_cfg: EntryPointsStepConfig | None = None
    profiles_cfg: ProfilesAnalyticsStepConfig | None = None
    history_cfg: HistoryTimeseriesStepConfig | None = None
    config_data_flow_cfg: ConfigDataFlowStepConfig | None = None
    coverage_functions_cfg: CoverageAnalyticsStepConfig | None = None
    test_coverage_cfg: TestCoverageStepConfig | None = None
    external_deps_cfg: ExternalDependenciesStepConfig | None = None
    graph_cfg: GraphMetricsStepConfig | None = None

    options: object | None = None
    plugin_name: str | None = None
    scope: GraphRunScope = field(default_factory=GraphRunScope)
    run_id: str | None = None
    scratch: object | None = None

    extra: dict[str, Any] = field(default_factory=dict)


Severity = Literal["fatal", "soft_fail", "skip_on_error"]
Stage = Literal[
    "graph",
    "function",
    "function_history",
    "test",
    "coverage",
    "subsystem",
    "data_model",
    "data_model_usage",
    "entrypoints",
    "profiles",
    "history",
    "other",
]

@dataclass(frozen=True)
class AnalyticsPlugin:
    """
    Declarative description of any analytics task (graph or non-graph).

    `run` is where you call your existing "big step" functions.
    """

    name: str
    description: str
    stage: Stage
    enabled_by_default: bool
    run: Callable[[AnalyticsExecutionContext], object | None]

    severity: Severity = "fatal"
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    options_model: type[BaseModel] | None = None
    options_default: object | None = None
    resource_hints: ResourceHints | None = None
    version_hash: str | None = None
    row_count_tables: tuple[str, ...] = ()
    contract_checkers: tuple[ContractChecker, ...] = ()
    requires_isolation: bool = False
    isolation_kind: Literal["process", "thread"] | None = None

    context_factory: Callable[[AnalyticsExecutionContext], AnalyticsExecutionContext] | None = None


@dataclass(frozen=True)
class AnalyticsPluginPlan:
    """Resolved execution plan for a set of analytics plugins."""

    plugins: tuple[AnalyticsPlugin, ...]
    plan_id: str
    skipped_plugins: tuple[AnalyticsSkippedStep, ...]
    dep_graph: dict[str, tuple[str, ...]]

    @property
    def ordered_names(self) -> tuple[str, ...]:
        """Plugin names in execution order."""
        return tuple(plugin.name for plugin in self.plugins)


_ANALYTICS_PLUGINS: dict[str, AnalyticsPlugin] = {}


def register_analytics_plugin(plugin: AnalyticsPlugin) -> None:
    """
    Register an analytics plugin.

    Intended for module-level registration in analytics subpackages.

    Raises
    ------
    ValueError
        If the plugin name is already registered.
    """
    if plugin.name in _ANALYTICS_PLUGINS:
        message = f"Duplicate analytics plugin name: {plugin.name}"
        raise ValueError(message)
    _ANALYTICS_PLUGINS[plugin.name] = plugin
    log.debug("Registered analytics plugin %s (stage=%s)", plugin.name, plugin.stage)


def get_analytics_plugin(name: str) -> AnalyticsPlugin:
    """
    Return a registered analytics plugin by name.

    Returns
    -------
    AnalyticsPlugin
        Registered plugin instance.

    Raises
    ------
    KeyError
        If no analytics or graph plugin exists with the requested name.
    """
    plugin = _ANALYTICS_PLUGINS.get(name)
    if plugin is not None:
        return plugin

    _autoregister_graph_plugin(name)
    try:
        return _ANALYTICS_PLUGINS[name]
    except KeyError as exc:
        message = f"Unknown analytics plugin: {name}"
        raise KeyError(message) from exc


def list_analytics_plugins() -> tuple[AnalyticsPlugin, ...]:
    """
    Return all registered analytics plugins.

    Returns
    -------
    tuple[AnalyticsPlugin, ...]
        Registered plugins in registration order.
    """
    return tuple(_ANALYTICS_PLUGINS.values())


def _resolve_requested_plugins(
    *,
    plugin_names: Sequence[str] | None,
    enabled: Sequence[str] | None,
    disabled: Sequence[str] | None,
    defaults: Sequence[str],
) -> tuple[tuple[str, ...], tuple[AnalyticsSkippedStep, ...]]:
    """
    Resolve requested plugin names and capture disabled entries.

    Returns
    -------
    tuple[tuple[str, ...], tuple[AnalyticsSkippedStep, ...]]
        Selected plugin names and skipped records.
    """
    if enabled:
        selected = tuple(enabled)
    elif plugin_names:
        selected = tuple(plugin_names)
    else:
        selected = tuple(defaults)
    disabled_set = set(disabled or ())
    resolved: list[str] = []
    skipped: list[AnalyticsSkippedStep] = []
    for name in selected:
        if name in disabled_set:
            skipped.append(
                AnalyticsSkippedStep(
                    name=name,
                    reason="disabled",
                    kind="analytics_plugin",
                )
            )
            continue
        resolved.append(name)
    return tuple(resolved), tuple(skipped)


def _validate_plugin_deps(requested: dict[str, AnalyticsPlugin]) -> dict[str, set[str]]:
    """
    Validate that dependencies exist within the requested set.

    Returns
    -------
    dict[str, set[str]]
        Dependency mapping for each requested plugin.
    """
    dependencies: dict[str, set[str]] = {name: set() for name in requested}
    for plugin in requested.values():
        for dep in plugin.depends_on:
            if dep not in requested:
                log.debug(
                    "Skipping unmet dependency %s for analytics plugin %s",
                    dep,
                    plugin.name,
                )
                continue
            dependencies[plugin.name].add(dep)
    return dependencies


def plan_analytics_plugins(
    plugin_names: Sequence[str] | None = None,
    *,
    enabled: Sequence[str] | None = None,
    disabled: Sequence[str] | None = None,
    defaults: Sequence[str] | None = None,
) -> AnalyticsPluginPlan:
    """
    Build an execution plan with dependency validation and topological ordering.

    Raises
    ------
    ValueError
        If duplicate plugins are requested or a dependency cycle is detected.

    Returns
    -------
    AnalyticsPluginPlan
        Ordered analytics plugin plan with dependency graph metadata.
    """
    selection, skipped = _resolve_requested_plugins(
        plugin_names=plugin_names,
        enabled=enabled,
        disabled=disabled,
        defaults=defaults or tuple(_ANALYTICS_PLUGINS.keys()),
    )
    requested: dict[str, AnalyticsPlugin] = {}
    for name in selection:
        if name in requested:
            message = f"Analytics plugin '{name}' listed more than once"
            raise ValueError(message)
        requested[name] = get_analytics_plugin(name)

    dependencies = _validate_plugin_deps(requested)

    ordered: list[AnalyticsPlugin] = []
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(name: str) -> None:
        if name in permanent:
            return
        if name in temporary:
            message = f"Detected dependency cycle involving analytics plugin '{name}'"
            raise ValueError(message)
        temporary.add(name)
        plugin = requested[name]
        for dep in dependencies[plugin.name]:
            visit(dep)
        temporary.remove(name)
        permanent.add(name)
        ordered.append(plugin)

    for name in selection:
        visit(name)

    dep_graph = {name: tuple(sorted(dependencies[name])) for name in selection}
    return AnalyticsPluginPlan(
        plugins=tuple(ordered),
        plan_id=uuid4().hex,
        skipped_plugins=skipped,
        dep_graph=dep_graph,
    )


def _autoregister_graph_plugin(name: str) -> None:
    """Attempt to mirror a graph plugin into the analytics registry when missing."""
    if name in _ANALYTICS_PLUGINS:
        return
    try:
        graphs_plugins = import_module("codeintel.analytics.graphs.plugins")
        graph_plugin = graphs_plugins.get_graph_metric_plugin(name)
        register_analytics_plugin(graphs_plugins.graph_metric_plugin_to_analytics(graph_plugin))
    except (ImportError, AttributeError, KeyError):
        return


__all__ = [
    "AnalyticsExecutionContext",
    "AnalyticsPlugin",
    "AnalyticsPluginPlan",
    "ResourceHints",
    "Stage",
    "get_analytics_plugin",
    "list_analytics_plugins",
    "plan_analytics_plugins",
    "register_analytics_plugin",
]
