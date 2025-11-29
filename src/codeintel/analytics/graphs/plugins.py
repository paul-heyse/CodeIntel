"""Plugin registry for graph metric computations."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.analytics.graphs.contracts import ContractChecker
from codeintel.config import GraphMetricsStepConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphRunScope
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass
class GraphRuntimeScratch:
    """Ephemeral scratch/cache store shared across plugin executions in a run."""

    _store: dict[str, object] = field(default_factory=dict)
    _cleanup: list[Callable[[], None]] = field(default_factory=list)

    def declare(self, key: str, value: object) -> None:
        """Record a value for later consumption."""
        self._store[key] = value

    def consume(self, key: str, default: object | None = None) -> object | None:
        """
        Retrieve a value populated by another plugin.

        Returns
        -------
        object | None
            Cached value or provided default.
        """
        return self._store.get(key, default)

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback executed after the run completes."""
        self._cleanup.append(callback)

    def cleanup(self) -> None:
        """Execute cleanup callbacks and clear stored values."""
        for callback in reversed(self._cleanup):
            try:
                callback()
            except Exception:
                log.exception("scratch.cleanup_failed")
        self._store.clear()
        self._cleanup.clear()

    def __len__(self) -> int:
        """
        Return the number of declared cache entries.

        Returns
        -------
        int
            Count of cached entries.
        """
        return len(self._store)

    def keys(self) -> tuple[str, ...]:
        """
        Return declared cache keys.

        Returns
        -------
        tuple[str, ...]
            Cache key names.
        """
        return tuple(self._store.keys())


@dataclass(frozen=True)
class GraphMetricExecutionContext:
    """
    Shared execution context for graph metric plugins.

    Plugins receive everything they need to resolve graphs for a given
    repo/commit and write results to analytics.* tables.
    """

    gateway: StorageGateway
    runtime: GraphRuntime
    repo: str
    commit: str
    config: GraphMetricsStepConfig | None
    analytics_context: AnalyticsContext | None
    catalog_provider: FunctionCatalogProvider | None
    options: object | None = None
    plugin_name: str | None = None
    run_id: str | None = None
    scope: GraphRunScope = field(default_factory=GraphRunScope)
    scratch: GraphRuntimeScratch = field(default_factory=GraphRuntimeScratch)


@dataclass(frozen=True)
class GraphPluginResult:
    """Optional structured result returned by graph metric plugins."""

    row_counts: dict[str, int] | None = None
    input_hash: str | None = None
    options_hash: str | None = None


def _row_counts_for_tables(
    ctx: GraphMetricExecutionContext, tables: Sequence[str]
) -> dict[str, int]:
    """
    Compute row counts for a set of tables scoped by repo/commit.

    Parameters
    ----------
    ctx:
        Plugin execution context containing gateway/repo/commit.
    tables:
        Iterable of table names to count.

    Returns
    -------
    dict[str, int]
        Mapping of table name to row count for the requested repo/commit.
    """
    counts: dict[str, int] = {}
    connection = getattr(ctx.gateway, "con", None)
    if connection is None:
        return counts
    for table in tables:
        try:
            escaped_repo = ctx.repo.replace("'", "''")
            escaped_commit = ctx.commit.replace("'", "''")
            relation = connection.table(table).filter(
                f"repo = '{escaped_repo}' AND commit = '{escaped_commit}'"
            )
            row_count = relation.count().fetchone()[0]
            counts[table] = int(row_count)
        except Exception:  # noqa: BLE001 - defensive, counting must not break plugins
            log.debug("row_count.failed table=%s repo=%s commit=%s", table, ctx.repo, ctx.commit)
    return counts


@dataclass(frozen=True)
class GraphMetricResourceHints:
    """Optional resource hints used for planning/observability."""

    max_runtime_ms: int | None = None
    memory_mb_hint: int | None = None


@dataclass(frozen=True)
class GraphMetricPlugin:
    """
    Declarative description of a graph metric task.

    Attributes
    ----------
    name:
        Stable identifier used in config and logs.
    description:
        Human-readable description of what the plugin computes.
    stage:
        Rough grouping for reporting / ordering. Examples:
        "core", "cfg", "dfg", "test", "symbol", "subsystem", "config", "stats".
    enabled_by_default:
        Whether this plugin runs when no explicit plugin list is provided.
    run:
        Callable that performs the metric computation given the shared context.
    """

    name: str
    description: str
    stage: Literal[
        "core",
        "cfg",
        "dfg",
        "test",
        "symbol",
        "subsystem",
        "config",
        "stats",
    ]
    enabled_by_default: bool
    run: Callable[[GraphMetricExecutionContext], GraphPluginResult | None]
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    resource_hints: GraphMetricResourceHints | None = None
    severity: Literal["fatal", "soft_fail", "skip_on_error"] = "fatal"
    options_model: type[BaseModel] | None = None
    options_default: object | None = None
    version_hash: str | None = None
    contract_checkers: tuple[ContractChecker, ...] = ()
    scope_aware: bool = False
    supported_scopes: tuple[Literal["paths", "modules", "time_window"], ...] = ()
    requires_isolation: bool = False
    isolation_kind: Literal["process", "thread"] | None = None
    config_schema_ref: str | None = None
    row_count_tables: tuple[str, ...] = ()
    description_ref: str | None = None
    stage_rank: int | None = None
    cache_populates: tuple[str, ...] = ()
    cache_consumes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphMetricPluginMetadata:
    """Snapshot of plugin metadata for planning or documentation."""

    name: str
    description: str
    stage: Literal[
        "core",
        "cfg",
        "dfg",
        "test",
        "symbol",
        "subsystem",
        "config",
        "stats",
    ]
    severity: Literal["fatal", "soft_fail", "skip_on_error"]
    enabled_by_default: bool
    depends_on: tuple[str, ...]
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    resource_hints: GraphMetricResourceHints | None
    options_model: type[BaseModel] | None
    options_default: object | None
    version_hash: str | None
    contract_checkers: tuple[ContractChecker, ...]
    scope_aware: bool
    supported_scopes: tuple[Literal["paths", "modules", "time_window"], ...]
    requires_isolation: bool
    isolation_kind: Literal["process", "thread"] | None
    config_schema_ref: str | None
    row_count_tables: tuple[str, ...] = ()
    cache_populates: tuple[str, ...] = ()
    cache_consumes: tuple[str, ...] = ()


def graph_metric_plugin_metadata(plugin: GraphMetricPlugin) -> GraphMetricPluginMetadata:
    """
    Return immutable metadata describing a plugin.

    Parameters
    ----------
    plugin:
        Plugin instance to summarize.

    Returns
    -------
    GraphMetricPluginMetadata
        Snapshot of the plugin's declarative metadata.
    """
    return GraphMetricPluginMetadata(
        name=plugin.name,
        description=plugin.description,
        stage=plugin.stage,
        severity=plugin.severity,
        enabled_by_default=plugin.enabled_by_default,
        depends_on=plugin.depends_on,
        provides=plugin.provides,
        requires=plugin.requires,
        resource_hints=plugin.resource_hints,
        options_model=plugin.options_model,
        options_default=plugin.options_default,
        version_hash=plugin.version_hash,
        contract_checkers=plugin.contract_checkers,
        scope_aware=plugin.scope_aware,
        supported_scopes=plugin.supported_scopes,
        requires_isolation=plugin.requires_isolation,
        isolation_kind=plugin.isolation_kind,
        config_schema_ref=plugin.config_schema_ref,
        row_count_tables=plugin.row_count_tables,
        cache_populates=plugin.cache_populates,
        cache_consumes=plugin.cache_consumes,
    )


@dataclass(frozen=True)
class GraphMetricPluginPlan:
    """Resolved execution plan for a set of graph metric plugins."""

    plugins: tuple[GraphMetricPlugin, ...]
    plan_id: str = field(default_factory=lambda: uuid4().hex)
    skipped_plugins: tuple[GraphMetricPluginSkip, ...] = ()
    dep_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ordered_names(self) -> tuple[str, ...]:
        """Plugin names in execution order."""
        return tuple(plugin.name for plugin in self.plugins)


_PLUGINS: dict[str, GraphMetricPlugin] = {}
_ENTRYPOINTS_LOADED_FLAG = [False]


def _normalize_options_payload(options: object | None) -> dict[str, object]:
    """
    Normalize plugin options into a mapping.

    Returns
    -------
    dict[str, object]
        Normalized mapping of option keys to values.

    Raises
    ------
    TypeError
        If the options are not a mapping or BaseModel instance.
    """
    if options is None:
        return {}
    if isinstance(options, BaseModel):
        return options.model_dump()
    if isinstance(options, dict):
        return options
    message = "Plugin options must be a mapping or BaseModel instance"
    raise TypeError(message)


def _validate_plugin_default_options(plugin: GraphMetricPlugin) -> None:
    if plugin.options_default is None and plugin.options_model is None:
        return
    payload = _normalize_options_payload(plugin.options_default)
    if plugin.options_model is not None:
        plugin.options_model.model_validate(payload)


def register_graph_metric_plugin(plugin: GraphMetricPlugin) -> None:
    """
    Register a graph metric plugin at import time.

    Raises
    ------
    ValueError
        When a plugin with the same name has already been registered.
    """
    if plugin.name in _PLUGINS:
        message = f"Duplicate graph metric plugin name: {plugin.name}"
        raise ValueError(message)
    _validate_plugin_default_options(plugin)
    _PLUGINS[plugin.name] = plugin
    log.debug("Registered graph metric plugin %s (stage=%s)", plugin.name, plugin.stage)


def unregister_graph_metric_plugin(name: str) -> None:
    """Remove a registered graph metric plugin (primarily for tests)."""
    _PLUGINS.pop(name, None)


def get_graph_metric_plugin(name: str) -> GraphMetricPlugin:
    """
    Return a plugin by name or raise KeyError.

    Returns
    -------
    GraphMetricPlugin
        Registered plugin matching the provided name.

    Raises
    ------
    KeyError
        If no plugin is registered under the provided name.
    """
    _ensure_entrypoint_plugins_loaded()
    plugin = _PLUGINS.get(name)
    if plugin is None:
        raise KeyError(name)
    return plugin


def list_graph_metric_plugins() -> tuple[GraphMetricPlugin, ...]:
    """
    Return all registered plugins.

    Returns
    -------
    tuple[GraphMetricPlugin, ...]
        Registered plugins in insertion order.
    """
    _ensure_entrypoint_plugins_loaded()
    return tuple(_PLUGINS.values())


@dataclass(frozen=True)
class GraphMetricPluginSkip:
    """Skip metadata for planned plugins that will not execute."""

    name: str
    reason: Literal["disabled"]


def load_graph_metric_plugins_from_entrypoints(
    *,
    group: str = "codeintel.graph_metric_plugins",
    force: bool = False,
) -> tuple[GraphMetricPlugin, ...]:
    """
    Discover and register graph metric plugins from entrypoints.

    Parameters
    ----------
    group :
        Entrypoint group to load plugins from.
    force :
        When True, load entrypoints even if discovery already ran.

    Returns
    -------
    tuple[GraphMetricPlugin, ...]
        Plugins loaded and registered from entrypoints.

    Raises
    ------
    TypeError
        If an entrypoint does not yield a GraphMetricPlugin.
    """
    if _ENTRYPOINTS_LOADED_FLAG[0] and not force:
        return ()
    discovered: list[GraphMetricPlugin] = []
    for entry_point in importlib.metadata.entry_points().select(group=group):
        plugin = entry_point.load()
        if not isinstance(plugin, GraphMetricPlugin):
            message = f"Entrypoint {entry_point.name} did not return GraphMetricPlugin"
            raise TypeError(message)
        register_graph_metric_plugin(plugin)
        discovered.append(plugin)
        log.info(
            "Discovered graph metric plugin from entrypoint name=%s group=%s",
            plugin.name,
            group,
        )
    _ENTRYPOINTS_LOADED_FLAG[0] = True
    return tuple(discovered)


def list_graph_metric_plugin_metadata() -> tuple[GraphMetricPluginMetadata, ...]:
    """
    Return metadata for all registered plugins.

    Returns
    -------
    tuple[GraphMetricPluginMetadata, ...]
        Ordered metadata matching the current registry contents.
    """
    return tuple(graph_metric_plugin_metadata(plugin) for plugin in list_graph_metric_plugins())


def _ensure_entrypoint_plugins_loaded() -> None:
    if _ENTRYPOINTS_LOADED_FLAG[0]:
        return
    load_graph_metric_plugins_from_entrypoints()


def _validate_plugin_deps(
    selected: dict[str, GraphMetricPlugin],
) -> dict[str, set[str]]:
    dependencies: dict[str, set[str]] = {}
    for plugin in selected.values():
        deps = set(plugin.depends_on)
        for dep in deps:
            if dep not in selected:
                message = (
                    f"Graph metric plugin '{plugin.name}' depends on '{dep}', "
                    "which is not in the selected plugin set"
                )
                raise ValueError(message)
        dependencies[plugin.name] = deps
    return dependencies


def _build_provider_index(selected: dict[str, GraphMetricPlugin]) -> dict[str, set[str]]:
    provider_index: dict[str, set[str]] = {}
    for plugin in selected.values():
        for capability in plugin.provides:
            provider_index.setdefault(capability, set()).add(plugin.name)
    return provider_index


def _attach_required_providers(
    selected: dict[str, GraphMetricPlugin], dependencies: dict[str, set[str]]
) -> None:
    provider_index = _build_provider_index(selected)
    for plugin in selected.values():
        plugin_dependencies = dependencies[plugin.name]
        for requirement in plugin.requires:
            providers = provider_index.get(requirement, set())
            if not providers:
                message = (
                    f"Graph metric plugin '{plugin.name}' requires capability '{requirement}', "
                    "but no provider plugin is selected"
                )
                raise ValueError(message)
            if plugin.name in providers:
                continue
            explicit_provider = providers.intersection(plugin_dependencies)
            if explicit_provider:
                continue
            if len(providers) > 1:
                provider_list = ", ".join(sorted(providers))
                message = (
                    f"Graph metric plugin '{plugin.name}' requires capability '{requirement}', "
                    f"but multiple providers are available ({provider_list}). "
                    "Add an explicit depends_on entry to disambiguate."
                )
                raise ValueError(message)
            plugin_dependencies.add(next(iter(providers)))


def _resolve_requested_plugins(
    *,
    plugin_names: Sequence[str] | None,
    enabled: Sequence[str] | None,
    disabled: Sequence[str] | None,
    defaults: Sequence[str],
) -> tuple[tuple[str, ...], tuple[GraphMetricPluginSkip, ...]]:
    if enabled:
        selected = tuple(enabled)
    elif plugin_names:
        selected = tuple(plugin_names)
    else:
        selected = tuple(defaults)
    disabled_set = set(disabled or ())
    resolved: list[str] = []
    skipped: list[GraphMetricPluginSkip] = []
    for name in selected:
        if name in disabled_set:
            skipped.append(GraphMetricPluginSkip(name=name, reason="disabled"))
            continue
        resolved.append(name)
    return tuple(resolved), tuple(skipped)


def plan_graph_metric_plugins(
    plugin_names: Sequence[str] | None = None,
    *,
    enabled: Sequence[str] | None = None,
    disabled: Sequence[str] | None = None,
    defaults: Sequence[str] | None = None,
) -> GraphMetricPluginPlan:
    """
    Build an execution plan with dependency validation and topological order.

    Parameters
    ----------
    plugin_names:
        Explicit plugin names requested for execution (used when `enabled` is not set).
    enabled:
        When provided and non-empty, this ordered list is used verbatim.
    disabled:
        Plugins to drop from the selected set (recorded as skipped).
    defaults:
        Default plugin list to use when neither `enabled` nor `plugin_names` is provided.

    Returns
    -------
    GraphMetricPluginPlan
        Ordered plugins ready for execution plus skip/graph metadata.

    Raises
    ------
    ValueError
        If a plugin name is unknown, duplicated, or a dependency is missing/cyclic.
    """
    _ensure_entrypoint_plugins_loaded()
    selection, skipped = _resolve_requested_plugins(
        plugin_names=plugin_names,
        enabled=enabled,
        disabled=disabled,
        defaults=defaults or DEFAULT_GRAPH_METRIC_PLUGINS,
    )
    requested: dict[str, GraphMetricPlugin] = {}
    for name in selection:
        if name in requested:
            message = f"Graph metric plugin '{name}' listed more than once"
            raise ValueError(message)
        requested[name] = get_graph_metric_plugin(name)

    dependencies = _validate_plugin_deps(requested)
    _attach_required_providers(requested, dependencies)

    ordered: list[GraphMetricPlugin] = []
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(name: str) -> None:
        if name in permanent:
            return
        if name in temporary:
            message = f"Detected dependency cycle involving graph metric plugin '{name}'"
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
    return GraphMetricPluginPlan(tuple(ordered), skipped_plugins=skipped, dep_graph=dep_graph)


def validate_plugin_options(
    plugin: GraphMetricPlugin, options: dict[str, object] | None
) -> object | None:
    """
    Validate and normalize plugin options against the declared schema.

    Returns
    -------
    object | None
        Parsed options object (BaseModel instance or raw mapping) or None.
    """
    payload = _normalize_options_payload(options)
    if plugin.options_model is None:
        return payload or None
    return plugin.options_model.model_validate(payload)


def resolve_plugin_options(
    plugin: GraphMetricPlugin,
    config_options: dict[str, object] | None,
    runtime_options: dict[str, object] | None,
) -> object | None:
    """
    Merge default, config, and runtime options and validate against plugin schema.

    Parameters
    ----------
    plugin:
        Plugin whose options are being resolved.
    config_options:
        Options supplied via GraphMetricsStepConfig.
    runtime_options:
        Options supplied via GraphPluginRunOptions.

    Returns
    -------
    object | None
        Validated options object (BaseModel instance or raw mapping) or None.
    """
    merged: dict[str, object] = {}
    merged.update(_normalize_options_payload(plugin.options_default))
    merged.update(_normalize_options_payload(config_options))
    merged.update(_normalize_options_payload(runtime_options))
    return validate_plugin_options(plugin, merged)


def _ensure_graph_metrics_cfg(ctx: GraphMetricExecutionContext) -> GraphMetricsStepConfig:
    """
    Resolve a GraphMetricsStepConfig from context.config or derive from runtime.

    This ensures that plugins which rely on GraphMetricsStepConfig can always
    obtain one, even when called from outside the pipeline.

    Returns
    -------
    GraphMetricsStepConfig
        Graph metrics configuration resolved for the current context.
    """
    if ctx.config is not None:
        return ctx.config

    options: GraphRuntimeOptions = ctx.runtime.options
    snapshot = options.snapshot or SnapshotRef(
        repo=ctx.repo,
        commit=ctx.commit,
        repo_root=Path(),
    )
    return GraphMetricsStepConfig(snapshot=snapshot)


def _plugin_core_graph_metrics(ctx: GraphMetricExecutionContext) -> GraphPluginResult | None:
    module = importlib.import_module("codeintel.analytics.graphs.graph_metrics")

    cfg = _ensure_graph_metrics_cfg(ctx)
    deps = module.GraphMetricsDeps(
        catalog_provider=ctx.catalog_provider,
        runtime=ctx.runtime,
        analytics_context=ctx.analytics_context,
        filters=None,
    )
    module.compute_graph_metrics(ctx.gateway, cfg, deps=deps)
    return GraphPluginResult(
        row_counts=_row_counts_for_tables(
            ctx,
            (
                "analytics.graph_metrics_functions",
                "analytics.graph_metrics_modules",
            ),
        )
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="core_graph_metrics",
        description="Core function/module graph metrics (centrality, neighbors, components).",
        stage="core",
        enabled_by_default=True,
        run=_plugin_core_graph_metrics,
        row_count_tables=(
            "analytics.graph_metrics_functions",
            "analytics.graph_metrics_modules",
        ),
    )
)


def _plugin_graph_metrics_functions_ext(
    ctx: GraphMetricExecutionContext,
) -> GraphPluginResult | None:
    module = importlib.import_module("codeintel.analytics.graphs.graph_metrics_ext")

    cfg = _ensure_graph_metrics_cfg(ctx)
    module.compute_graph_metrics_functions_ext(
        ctx.gateway,
        repo=cfg.repo,
        commit=cfg.commit,
        runtime=ctx.runtime,
        filters=None,
    )
    return GraphPluginResult(
        row_counts=_row_counts_for_tables(
            ctx,
            ("analytics.graph_metrics_functions_ext",),
        )
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="graph_metrics_functions_ext",
        description="Extended call graph metrics for functions.",
        stage="core",
        enabled_by_default=True,
        run=_plugin_graph_metrics_functions_ext,
        row_count_tables=("analytics.graph_metrics_functions_ext",),
    )
)


def _plugin_graph_metrics_modules_ext(
    ctx: GraphMetricExecutionContext,
) -> GraphPluginResult | None:
    module = importlib.import_module("codeintel.analytics.graphs.module_graph_metrics_ext")

    cfg = _ensure_graph_metrics_cfg(ctx)
    module.compute_graph_metrics_modules_ext(
        ctx.gateway,
        repo=cfg.repo,
        commit=cfg.commit,
        runtime=ctx.runtime,
        filters=None,
    )
    return GraphPluginResult(
        row_counts=_row_counts_for_tables(
            ctx,
            ("analytics.graph_metrics_modules_ext",),
        )
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="graph_metrics_modules_ext",
        description="Extended import graph metrics for modules.",
        stage="core",
        enabled_by_default=True,
        run=_plugin_graph_metrics_modules_ext,
        row_count_tables=("analytics.graph_metrics_modules_ext",),
    )
)


def _plugin_cfg_metrics(ctx: GraphMetricExecutionContext) -> None:
    module = importlib.import_module("codeintel.analytics.cfg_dfg")

    module.compute_cfg_metrics(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        context=ctx.analytics_context,
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="cfg_metrics",
        description="Control-flow graph metrics for functions and blocks.",
        stage="cfg",
        enabled_by_default=True,
        run=_plugin_cfg_metrics,
    )
)


def _plugin_dfg_metrics(ctx: GraphMetricExecutionContext) -> None:
    module = importlib.import_module("codeintel.analytics.cfg_dfg")

    module.compute_dfg_metrics(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        context=ctx.analytics_context,
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="dfg_metrics",
        description="Data-flow graph metrics for functions and blocks.",
        stage="dfg",
        enabled_by_default=True,
        run=_plugin_dfg_metrics,
    )
)


def _plugin_test_graph_metrics(ctx: GraphMetricExecutionContext) -> GraphPluginResult | None:
    module = importlib.import_module("codeintel.analytics.tests.graph_metrics")

    module.compute_test_graph_metrics(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
    )
    return GraphPluginResult(
        row_counts=_row_counts_for_tables(
            ctx,
            (
                "analytics.test_graph_metrics_tests",
                "analytics.test_graph_metrics_functions",
            ),
        )
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="test_graph_metrics",
        description="Metrics over the test <-> function bipartite graph.",
        stage="test",
        enabled_by_default=True,
        run=_plugin_test_graph_metrics,
        row_count_tables=(
            "analytics.test_graph_metrics_tests",
            "analytics.test_graph_metrics_functions",
        ),
    )
)


def _plugin_symbol_graph_metrics_modules(
    ctx: GraphMetricExecutionContext,
) -> GraphPluginResult | None:
    module = importlib.import_module("codeintel.analytics.graphs.symbol_graph_metrics")

    module.compute_symbol_graph_metrics_modules(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
    )
    return GraphPluginResult(
        row_counts=_row_counts_for_tables(
            ctx,
            ("analytics.symbol_graph_metrics_modules",),
        )
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="symbol_graph_metrics_modules",
        description="Symbol graph metrics at the module level.",
        stage="symbol",
        enabled_by_default=True,
        run=_plugin_symbol_graph_metrics_modules,
        row_count_tables=("analytics.symbol_graph_metrics_modules",),
    )
)


def _plugin_symbol_graph_metrics_functions(
    ctx: GraphMetricExecutionContext,
) -> GraphPluginResult | None:
    module = importlib.import_module("codeintel.analytics.graphs.symbol_graph_metrics")

    module.compute_symbol_graph_metrics_functions(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
    )
    return GraphPluginResult(
        row_counts=_row_counts_for_tables(
            ctx,
            ("analytics.symbol_graph_metrics_functions",),
        )
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="symbol_graph_metrics_functions",
        description="Symbol graph metrics at the function level.",
        stage="symbol",
        enabled_by_default=True,
        run=_plugin_symbol_graph_metrics_functions,
        row_count_tables=("analytics.symbol_graph_metrics_functions",),
    )
)


def _plugin_subsystem_graph_metrics(
    ctx: GraphMetricExecutionContext,
) -> GraphPluginResult | None:
    module = importlib.import_module("codeintel.analytics.graphs.subsystem_graph_metrics")

    module.compute_subsystem_graph_metrics(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
        filters=None,
    )
    return GraphPluginResult(
        row_counts=_row_counts_for_tables(ctx, ("analytics.subsystem_graph_metrics",))
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="subsystem_graph_metrics",
        description="Subsystem-level condensed import graph metrics.",
        stage="subsystem",
        enabled_by_default=True,
        run=_plugin_subsystem_graph_metrics,
        row_count_tables=("analytics.subsystem_graph_metrics",),
    )
)


def _plugin_config_graph_metrics(ctx: GraphMetricExecutionContext) -> GraphPluginResult | None:
    module = importlib.import_module("codeintel.analytics.graphs.config_graph_metrics")

    module.compute_config_graph_metrics(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
    )
    return GraphPluginResult(
        row_counts=_row_counts_for_tables(
            ctx,
            (
                "analytics.config_graph_metrics_keys",
                "analytics.config_graph_metrics_modules",
                "analytics.config_projection_key_edges",
                "analytics.config_projection_module_edges",
            ),
        )
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="config_graph_metrics",
        description="Config bipartite/projection graph metrics.",
        stage="config",
        enabled_by_default=True,
        run=_plugin_config_graph_metrics,
        row_count_tables=(
            "analytics.config_graph_metrics_keys",
            "analytics.config_graph_metrics_modules",
            "analytics.config_projection_key_edges",
            "analytics.config_projection_module_edges",
        ),
    )
)


def _plugin_graph_stats(ctx: GraphMetricExecutionContext) -> GraphPluginResult | None:
    module = importlib.import_module("codeintel.analytics.graphs.graph_stats")

    module.compute_graph_stats(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
    )
    return GraphPluginResult(row_counts=_row_counts_for_tables(ctx, ("analytics.graph_stats",)))


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="graph_stats",
        description="Global graph statistics for core graphs.",
        stage="stats",
        enabled_by_default=True,
        run=_plugin_graph_stats,
        row_count_tables=("analytics.graph_stats",),
    )
)


def _plugin_subsystem_agreement(ctx: GraphMetricExecutionContext) -> GraphPluginResult | None:
    module = importlib.import_module("codeintel.analytics.graphs.subsystem_agreement")

    module.compute_subsystem_agreement(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    return GraphPluginResult(row_counts=_row_counts_for_tables(ctx, ("analytics.subsystem_agreement",)))


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="subsystem_agreement",
        description="Check agreement between subsystem labels and import communities.",
        stage="subsystem",
        enabled_by_default=True,
        run=_plugin_subsystem_agreement,
        depends_on=("subsystem_graph_metrics", "graph_metrics_modules_ext"),
        row_count_tables=("analytics.subsystem_agreement",),
    )
)


DEFAULT_GRAPH_METRIC_PLUGINS: tuple[str, ...] = (
    "core_graph_metrics",
    "graph_metrics_functions_ext",
    "graph_metrics_modules_ext",
    "test_graph_metrics",
    "cfg_metrics",
    "dfg_metrics",
    "symbol_graph_metrics_modules",
    "symbol_graph_metrics_functions",
    "config_graph_metrics",
    "subsystem_graph_metrics",
    "subsystem_agreement",
    "graph_stats",
)
