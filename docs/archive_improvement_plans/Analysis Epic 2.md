Nice, congrats on shipping refactor 1 🎉 — let’s treat that as “done” and now layer a **generic analytics plugin harness** on top.

Below is a detailed, step-by-step plan for **Epic 2 — Generalize the plugin harness beyond graphs**, with concrete code sketches. I’ll structure it so you can implement in **phases**, without having to flip the entire repo at once.

---

## High-level goals

* Introduce a **generic plugin abstraction** for *all* analytics (functions, graphs, tests, subsystems, data models, history).
* Factor your **graph plugin machinery** into reusable pieces:

  * planning (deps, severity, timeouts)
  * execution (retries, isolation)
  * unchanged detection & manifests
  * telemetry
* Express non-graph steps as **plugins** with a **unified harness**, while keeping their existing “big step” functions intact.

We’ll do this in three big steps:

1. New generic layer: `analytics/plugins.py` + `analytics/plugin_runtime.py`.
2. Adapt **graph metrics** to sit on top of the generic layer (while keeping their current API).
3. Register **non-graph analytics** as plugins.

---

## Step 1 — Introduce generic plugin types (`analytics/plugins.py`)

### 1.1 Define `ResourceHints`

We generalize the “resource hints” you already use in graph plugins, but put them in a generic place:

```python
# analytics/plugins.py

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntime
from codeintel.config.steps_analytics import (
    FunctionAnalyticsStepConfig,
    TestProfileStepConfig,
    SubsystemsStepConfig,
    DataModelsStepConfig,
    DataModelUsageStepConfig,
    EntryPointsStepConfig,
    ProfilesAnalyticsStepConfig,
    HistoryTimeseriesStepConfig,
    # ...any other step configs you want to plug in
)
from codeintel.config.steps_graphs import GraphMetricsStepConfig, GraphRunScope
from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class ResourceHints:
    """
    Runtime resource hints for schedulers / harness.

    This is intentionally generic so both graph and non-graph analytics can use it.
    """

    max_runtime_ms: int | None = None
    max_memory_mb: int | None = None
    requires_gpu: bool = False
    priority: int = 0  # lower = more important
```

### 1.2 Define `AnalyticsExecutionContext`

This is the **generic runtime context** available to all plugins. Plugins that need more can wrap it.

```python
@dataclass
class AnalyticsExecutionContext:
    """
    Shared execution context for generic analytics plugins.

    For graphs we still build a GraphMetricExecutionContext on top of this;
    for functions/tests/subsystems we usually just use this directly.
    """

    gateway: StorageGateway
    analytics_context: AnalyticsContext | None
    repo: str
    commit: str

    # Optional graph runtime, populated when stages need it
    graph_runtime: GraphRuntime | None = None

    # Step configs for specific analytics families
    function_cfg: FunctionAnalyticsStepConfig | None = None
    function_history_cfg: HistoryTimeseriesStepConfig | None = None
    test_profile_cfg: TestProfileStepConfig | None = None
    subsystems_cfg: SubsystemsStepConfig | None = None
    data_models_cfg: DataModelsStepConfig | None = None
    data_model_usage_cfg: DataModelUsageStepConfig | None = None
    entrypoints_cfg: EntryPointsStepConfig | None = None
    profiles_cfg: ProfilesAnalyticsStepConfig | None = None
    history_cfg: HistoryTimeseriesStepConfig | None = None
    graph_cfg: GraphMetricsStepConfig | None = None

    # Per-plugin execution details
    options: object | None = None
    plugin_name: str | None = None
    scope: GraphRunScope = field(default_factory=GraphRunScope)  # reused shape
    run_id: str | None = None
    scratch: object | None = None      # will be an AnalyticsRuntimeScratch later

    # Escape hatch for future extensions
    extra: dict[str, Any] = field(default_factory=dict)
```

> For now `scope` reuses `GraphRunScope` because its `(paths, modules, time_window)` shape is generally useful. You can later lift it into a truly generic `AnalyticsScope` if you want.

### 1.3 Define `AnalyticsPlugin` (generic metadata + run callable)

Here we define a **generic plugin contract** that mirrors your `GraphMetricPlugin`, but without graph-specific bits:

```python
Severity = Literal["fatal", "soft_fail", "skip_on_error"]
Stage = Literal[
    "graph",
    "function",
    "function_history",
    "test",
    "subsystem",
    "data_model",
    "data_model_usage",
    "entrypoints",
    "profiles",
    "history",
    "other",
]

TContext = TypeVar("TContext", bound=AnalyticsExecutionContext)


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
    run: Callable[[TContext], object | None]

    # Planning / execution metadata
    severity: Severity = "fatal"
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    options_model: type[BaseModel] | None = None
    options_default: object | None = None
    resource_hints: ResourceHints | None = None
    version_hash: str | None = None
    row_count_tables: tuple[str, ...] = ()  # tables to use for unchanged detection

    # Optional context adapter:
    # if provided, harness will call this to build the per-plugin context,
    # otherwise it passes AnalyticsExecutionContext directly to `run`.
    context_factory: Callable[[AnalyticsExecutionContext], AnalyticsExecutionContext] | None = None
```

You can keep this **purely generic**; Graph metrics will plug in via an adapter.

### 1.4 Registry & planning for generic plugins

Very similar to `graphs/plugins.py`, but stage-independent:

```python
log = logging.getLogger(__name__)

_ANALYTICS_PLUGINS: dict[str, AnalyticsPlugin] = {}
_ANALYTICS_ENTRYPOINTS_LOADED = False


def register_analytics_plugin(plugin: AnalyticsPlugin) -> None:
    """
    Register an analytics plugin.

    Intended for module-level registration in analytics/functions/*,
    analytics/tests/*, analytics/subsystems/*, etc.
    """
    if plugin.name in _ANALYTICS_PLUGINS:
        msg = f"Duplicate analytics plugin name: {plugin.name}"
        raise ValueError(msg)
    _ANALYTICS_PLUGINS[plugin.name] = plugin
    log.debug("Registered analytics plugin %s (stage=%s)", plugin.name, plugin.stage)


def get_analytics_plugin(name: str) -> AnalyticsPlugin:
    try:
        return _ANALYTICS_PLUGINS[name]
    except KeyError as exc:
        msg = f"Unknown analytics plugin: {name}"
        raise KeyError(msg) from exc


@dataclass(frozen=True)
class AnalyticsPluginPlan:
    """Resolved execution plan for a set of analytics plugins."""

    plugins: tuple[AnalyticsPlugin, ...]
    plan_id: str
    skipped_plugins: tuple[AnalyticsSkippedStep, ...]
    dep_graph: dict[str, tuple[str, ...]]


def plan_analytics_plugins(
    plugin_names: Sequence[str] | None = None,
    *,
    enabled: Sequence[str] | None = None,
    disabled: Sequence[str] | None = None,
    defaults: Sequence[str] | None = None,
) -> AnalyticsPluginPlan:
    """
    Generic version of plan_graph_metric_plugins.

    - resolves requested plugins
    - applies depends_on graph
    - topologically sorts
    - records skipped plugins
    """
    # Very similar to your GraphMetricPluginPlan implementation:
    # - compute selection / skipped
    # - validate dependency graph
    # - topological sort
    # - construct AnalyticsPluginPlan
    ...
```

You don’t have to fully implement this immediately; you can literally clone the logic from `GraphMetricPluginPlan`, swapping types.

---

## Step 2 — Generic harness: `analytics/plugin_runtime.py`

This is the **shared execution harness** for all analytics plugins. It parallels the now-refactored graph runtime from Epic 1, but parameterized over `AnalyticsPlugin`.

Key pieces:

* `AnalyticsPluginRunOptions` (like GraphPluginRunOptions).
* `AnalyticsPluginRunRecord` / `AnalyticsPluginRunReport`.
* Planning step using `GraphPluginPolicy` (reused as generic policy).
* Execution using the **same isolation / retry / unchanged detection** machinery that graphs use.

### 2.1 Define run options & records

```python
# analytics/plugin_runtime.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, Sequence

from codeintel.analytics.plugins import AnalyticsPlugin, AnalyticsExecutionContext
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphPluginRetryPolicy, GraphRunScope
from codeintel.storage.gateway import StorageGateway
from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntime

from codeintel.analytics.runtime_manifest import (
    AnalyticsRunRecord,
    AnalyticsRunReport,
    AnalyticsPlanInfo,
    AnalyticsSkippedStep,
    AnalyticsScope,
    encode_manifest,
)
from codeintel.analytics.graphs.runtime.manifest import (
    load_prior_manifest,
    write_manifest,
    is_unchanged,
    compute_input_hash,
    compute_options_hash,
)
from codeintel.config.steps_analytics import (
    FunctionAnalyticsStepConfig,
    TestProfileStepConfig,
    SubsystemsStepConfig,
    # ...
)


@dataclass(frozen=True)
class AnalyticsPluginRunOptions:
    """
    Optional per-run controls for analytics plugins.
    """

    plugin_options: dict[str, dict[str, object]] | None = None
    manifest_path: Path | None = None
    scope: GraphRunScope | None = None
    dry_run: bool | None = None
```

We reuse `AnalyticsRunRecord` / `AnalyticsRunReport` from the generic manifest abstraction you just implemented.

### 2.2 Planning: `plan_analytics_plugin_run`

This is the **generic analogue** of `plan_graph_plugin_run`.

```python
@dataclass(frozen=True)
class AnalyticsPluginExecutionSettings:
    name: str
    severity: Literal["fatal", "soft_fail", "skip_on_error"]
    retry_cfg: GraphPluginRetryPolicy
    timeout_ms: int | None
    fail_fast: bool
    input_hash: str | None
    options_hash: str | None
    version_hash: str | None


@dataclass(frozen=True)
class AnalyticsPluginExecutionPlan:
    plan_id: str
    run_id: str
    repo: str
    commit: str
    policy: GraphPluginPolicy
    prior_manifest: Mapping[str, Mapping[str, object]] | None
    scope: GraphRunScope
    plugins: tuple[AnalyticsPlugin, ...]
    ordered_names: tuple[str, ...]
    skipped: tuple[AnalyticsSkippedStep, ...]
    dep_graph: dict[str, tuple[str, ...]]
    settings_by_plugin: dict[str, AnalyticsPluginExecutionSettings]
    options_by_plugin: dict[str, object | None]
```

```python
def plan_analytics_plugin_run(
    plugin_names: Sequence[str],
    *,
    policy: GraphPluginPolicy,
    repo: str,
    commit: str,
    scope: GraphRunScope,
    prior_manifest: Mapping[str, Mapping[str, object]] | None,
    cfg_options: Mapping[str, dict[str, object]] | None,
    runtime_options: Mapping[str, dict[str, object]] | None,
    run_id: str,
) -> AnalyticsPluginExecutionPlan:
    """
    Generic plugin planning:

    - select and order plugins via plan_analytics_plugins
    - resolve options
    - compute input/options hashes (reusing manifest utilities)
    - build per-plugin execution settings
    """
    plan = plan_analytics_plugins(plugin_names)
    plugins = plan.plugins

    # Resolve options (similar to graph)
    options_by_plugin = _resolve_analytics_options_map(
        plugins=plugins,
        cfg_options=cfg_options or {},
        runtime_options=runtime_options or {},
    )

    settings_by_plugin: dict[str, AnalyticsPluginExecutionSettings] = {}
    for plugin in plugins:
        options = options_by_plugin.get(plugin.name)
        severity = _effective_severity(plugin, policy)
        retry_cfg = policy.retries.get(plugin.name, GraphPluginRetryPolicy())
        timeout_ms = _effective_timeout(plugin, policy)
        input_hash = compute_input_hash(
            repo=repo,
            commit=commit,
            plugin_name=plugin.name,
            version_hash=plugin.version_hash,
            options=options,
            extra_scope=None,  # optional: serialize scope
        )
        options_hash = compute_options_hash(options)
        settings_by_plugin[plugin.name] = AnalyticsPluginExecutionSettings(
            name=plugin.name,
            severity=severity,
            retry_cfg=retry_cfg,
            timeout_ms=timeout_ms,
            fail_fast=policy.fail_fast,
            input_hash=input_hash,
            options_hash=options_hash,
            version_hash=plugin.version_hash,
        )

    skipped = tuple(
        AnalyticsSkippedStep(name=s.name, reason=s.reason, kind="analytics_plugin")
        for s in plan.skipped_plugins
    )

    return AnalyticsPluginExecutionPlan(
        plan_id=plan.plan_id,
        run_id=run_id,
        repo=repo,
        commit=commit,
        policy=policy,
        prior_manifest=prior_manifest,
        scope=scope,
        plugins=plugins,
        ordered_names=plan.ordered_names,
        skipped=skipped,
        dep_graph=plan.dep_graph,
        settings_by_plugin=settings_by_plugin,
        options_by_plugin=dict(options_by_plugin),
    )
```

### 2.3 Execution: `run_analytics_plugins`

We then **execute** the plan, with unchanged detection and retries, reusing your graph semantics:

```python
def run_analytics_plugins(
    *,
    plan: AnalyticsPluginExecutionPlan,
    gateway: StorageGateway,
    analytics_context: AnalyticsContext | None,
    graph_runtime: GraphRuntime | None,
    cfgs: dict[str, object],  # keyed by stage or plugin family
) -> AnalyticsRunReport:
    """
    Execute all plugins in `plan` using a shared harness.

    - builds AnalyticsExecutionContext per plugin
    - applies dry_run / skip_on_unchanged
    - handles retries/timeouts
    - records AnalyticsRunRecord for each plugin
    """
    records: list[AnalyticsRunRecord] = []
    scratch = GraphRuntimeScratch()  # or renamed AnalyticsRuntimeScratch

    scope = plan.scope
    analytics_scope = AnalyticsScope(
        paths=scope.paths,
        modules=scope.modules,
        time_window=scope.time_window,
        labels={"runtime": "analytics"},
    )

    for plugin in plan.plugins:
        settings = plan.settings_by_plugin[plugin.name]
        options = plan.options_by_plugin.get(plugin.name)

        ctx = AnalyticsExecutionContext(
            gateway=gateway,
            analytics_context=analytics_context,
            repo=plan.repo,
            commit=plan.commit,
            graph_runtime=graph_runtime if plugin.stage == "graph" else None,
            function_cfg=cfgs.get("function") if plugin.stage == "function" else None,
            test_profile_cfg=cfgs.get("test_profile") if plugin.stage == "test" else None,
            subsystems_cfg=cfgs.get("subsystems") if plugin.stage == "subsystem" else None,
            # ...etc
            graph_cfg=cfgs.get("graph") if plugin.stage == "graph" else None,
            options=options,
            plugin_name=plugin.name,
            scope=scope,
            run_id=plan.run_id,
            scratch=scratch,
        )

        if plugin.context_factory is not None:
            ctx = plugin.context_factory(ctx)  # type: ignore[assignment]

        started_at = datetime.now(tz=UTC)
        try:
            # unchanged detection
            if plan.policy.dry_run:
                status = "skipped"
                error = None
                duration_ms = 0.0
            elif plan.policy.skip_on_unchanged and is_unchanged(
                prior_manifest=plan.prior_manifest or {},
                plugin_name=plugin.name,
                row_count_tables=plugin.row_count_tables,
                gateway=gateway,
                repo=plan.repo,
                commit=plan.commit,
                input_hash=settings.input_hash,
                options_hash=settings.options_hash,
            ):
                status = "skipped"
                error = None
                duration_ms = 0.0
            else:
                status, error, duration_ms = _execute_with_retries(plugin, ctx, settings)
        except PluginFatalError as exc:
            # you can wrap this into AnalyticsRunRecord similarly to graphs
            status = "failed"
            error = str(exc)
            duration_ms = 0.0

        ended_at = datetime.now(tz=UTC)
        records.append(
            AnalyticsRunRecord(
                name=plugin.name,
                kind=plugin.stage,
                status=status,  # Literal-compatible
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=duration_ms,
                attempts=1,  # can be updated in _execute_with_retries
                partial=status != "succeeded",
                error=error,
                meta={
                    "severity": settings.severity,
                    "options_hash": settings.options_hash,
                    "version_hash": settings.version_hash,
                    # plus anything else you want
                },
            )
        )

    scratch.cleanup()

    report = AnalyticsRunReport(
        repo=plan.repo,
        commit=plan.commit,
        run_id=plan.run_id,
        scope=analytics_scope,
        records=tuple(records),
        plan=AnalyticsPlanInfo(
            plan_id=plan.plan_id,
            ordered_steps=plan.ordered_names,
            skipped_steps=plan.skipped,
            dep_graph=plan.dep_graph,
        ),
        tags={"runtime": "analytics"},
    )
    return report
```

Then writing manifests becomes a one-liner:

```python
if run_options.manifest_path is not None:
    payload = encode_manifest(report)
    write_manifest(run_options.manifest_path, payload)
```

---

## Step 3 — Bridge graph metrics into the generic harness

Now we **reuse** this generic harness for graphs, instead of having graphs be a bespoke special case.

### 3.1 Define an adapter from `GraphMetricPlugin` → `AnalyticsPlugin`

In `analytics/graphs/plugins.py`, add:

```python
from codeintel.analytics.plugins import AnalyticsPlugin, ResourceHints, AnalyticsExecutionContext


def graph_metric_plugin_to_analytics(plugin: GraphMetricPlugin) -> AnalyticsPlugin:
    """
    Wrap a GraphMetricPlugin as a generic AnalyticsPlugin.

    This DOES NOT change GraphMetricPlugin; it just provides an adapter. The
    generic harness will call plugin.run(GraphMetricExecutionContext) by way of
    a context_factory that converts AnalyticsExecutionContext → GraphMetricExecutionContext.
    """
    def context_factory(ctx: AnalyticsExecutionContext) -> GraphMetricExecutionContext:
        # Build the GraphMetricExecutionContext exactly as you do today:
        return GraphMetricExecutionContext(
            gateway=ctx.gateway,
            runtime=ctx.graph_runtime or resolve_graph_runtime(...),
            repo=ctx.repo,
            commit=ctx.commit,
            config=ctx.graph_cfg,
            analytics_context=ctx.analytics_context,
            catalog_provider=...,
            options=ctx.options,
            plugin_name=plugin.name,
            scope=ctx.scope,
            run_id=ctx.run_id,
            scratch=ctx.scratch or GraphRuntimeScratch(),
        )

    hints = None
    if plugin.resource_hints is not None:
        hints = ResourceHints(
            max_runtime_ms=plugin.resource_hints.max_runtime_ms,
            requires_gpu=getattr(plugin.resource_hints, "requires_gpu", False),
        )

    return AnalyticsPlugin(
        name=plugin.name,
        description=plugin.description,
        stage="graph",
        enabled_by_default=plugin.enabled_by_default,
        run=plugin.run,  # callable with GraphMetricExecutionContext
        severity=plugin.severity,
        depends_on=plugin.depends_on,
        provides=plugin.provides,
        requires=plugin.requires,
        options_model=plugin.options_model,
        options_default=plugin.options_default,
        resource_hints=hints,
        version_hash=plugin.version_hash,
        row_count_tables=plugin.row_count_tables,
        context_factory=context_factory,
    )
```

You can then register these adapter plugins in the analytics registry:

```python
# at the end of analytics/graphs/plugins.py

from codeintel.analytics.plugins import register_analytics_plugin

def _register_graph_plugins_as_analytics() -> None:
    for plugin in _PLUGINS.values():  # your internal graph plugin registry
        register_analytics_plugin(graph_metric_plugin_to_analytics(plugin))
```

Call `_register_graph_plugins_as_analytics()` in an import hook or within your analytics bootstrap path.

### 3.2 Make `GraphServiceRuntime` call the generic harness

`GraphServiceRuntime.run_plugins` becomes:

```python
# analytics/graph_service_runtime.py (new body of run_plugins)

from codeintel.analytics.plugin_runtime import (
    AnalyticsPluginRunOptions,
    plan_analytics_plugin_run,
    run_analytics_plugins,
)

class GraphServiceRuntime:
    ...

    def run_plugins(
        self,
        plugin_names: Sequence[str],
        *,
        cfg: GraphMetricsStepConfig | None = None,
        target: tuple[str, str] | None = None,
        run_options: AnalyticsPluginRunOptions | None = None,
    ) -> GraphPluginRunReport:
        policy = cfg.plugin_policy if cfg is not None else GraphPluginPolicy()
        if run_options is not None and run_options.dry_run is not None:
            policy = replace(policy, dry_run=run_options.dry_run)

        run_id = uuid.uuid4().hex
        manifest_path = run_options.manifest_path if run_options is not None else None
        prior_manifest = load_prior_manifest(manifest_path)

        if cfg is None and target is None and self.runtime.options.snapshot is None:
            msg = "Graph runtime missing snapshot; cannot derive repo/commit"
            raise ValueError(msg)
        repo, commit = _resolve_target(cfg, target, self.runtime)

        scope = (
            run_options.scope
            if run_options is not None and run_options.scope is not None
            else cfg.scope if cfg is not None
            else GraphRunScope()
        )

        cfg_options = cfg.plugin_options if cfg is not None else {}
        rt_options = (run_options.plugin_options if run_options is not None else {}) or {}

        plan = plan_analytics_plugin_run(
            plugin_names=plugin_names,
            policy=policy,
            repo=repo,
            commit=commit,
            scope=scope,
            prior_manifest=prior_manifest,
            cfg_options=cfg_options,
            runtime_options=rt_options,
            run_id=run_id,
        )

        analytics_report = run_analytics_plugins(
            plan=plan,
            gateway=self.gateway,
            analytics_context=self.analytics_context,
            graph_runtime=self.runtime,
            cfgs={"graph": cfg} if cfg is not None else {},
        )

        # map AnalyticsRunReport back to GraphPluginRunReport using your existing adapter
        return analytics_to_graph_run(analytics_report)
```

Where `analytics_to_graph_run` is the inverse of `graph_run_to_analytics` you already sketched.

---

## Step 4 — Express non-graph analytics as plugins

Finally, we wrap your existing “big step” functions as `AnalyticsPlugin`s.

### 4.1 Function metrics: `compute_function_metrics_and_types`

In `analytics/functions/__init__.py` (or a new `plugins.py` under `analytics/functions/`):

```python
# analytics/functions/plugins.py

from __future__ import annotations

from dataclasses import dataclass

from codeintel.analytics.plugins import AnalyticsPlugin, AnalyticsExecutionContext, register_analytics_plugin
from codeintel.analytics.functions.metrics import compute_function_metrics_and_types
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig
from codeintel.storage.gateway import StorageGateway


def function_metrics_run(ctx: AnalyticsExecutionContext) -> dict[str, int]:
    """
    Bridge from generic plugin context -> existing function metrics step.
    """
    assert ctx.function_cfg is not None, "FunctionAnalyticsStepConfig required"
    assert isinstance(ctx.gateway, StorageGateway)
    return compute_function_metrics_and_types(
        gateway=ctx.gateway,
        cfg=ctx.function_cfg,
        options=None,  # later you can pass ctx.options as FunctionAnalyticsOptions
    )


FUNCTION_METRICS_PLUGIN = AnalyticsPlugin(
    name="functions.metrics",
    description="Compute function metrics and typedness coverage.",
    stage="function",
    enabled_by_default=True,
    run=function_metrics_run,
    severity="fatal",
    depends_on=(),
    provides=("analytics.function_metrics", "analytics.function_types"),
    requires=("core.goids",),
    row_count_tables=("analytics.function_metrics", "analytics.function_types"),
)

register_analytics_plugin(FUNCTION_METRICS_PLUGIN)
```

You repeat this pattern for:

* `FunctionHistoryStepConfig` + `analytics/history/*`
* `TestProfileStepConfig` + `analytics/tests/profiles.build_test_profiles`
* `SubsystemsStepConfig` + `analytics/subsystems/materialize.*`
* `DataModelsStepConfig` + `analytics/data_models.compute_data_models`
* `DataModelUsageStepConfig` + `analytics/data_model_usage.compute_data_model_usage`
* `EntryPointsStepConfig` + `analytics/entrypoints.build_entrypoints`
* `ProfilesAnalyticsStepConfig` + `analytics/profiles.build_*` functions
* `HistoryTimeseriesStepConfig` + `analytics/history/timeseries.*`

Each plugin:

* Names its `stage` appropriately (`"test"`, `"subsystem"`, `"history"`, etc.).
* Sets `provides` to its main output tables.
* Sets `requires` to any upstream datasets or other plugin outputs (for future dependency graph).
* Sets `row_count_tables` for unchanged detection to work.

### 4.2 Feeding step configs into the harness

At call-sites (CLI, pipeline, tests), you already build step configs via `config/steps_analytics.py`.

You then pass them as the `cfgs` dict when calling `run_analytics_plugins`, keyed by stage:

```python
cfgs = {
    "function": function_cfg,
    "test": test_profile_cfg,
    "subsystem": subsystems_cfg,
    "data_model": data_models_cfg,
    # ...
}

analytics_report = run_analytics_plugins(
    plan=plan,
    gateway=gateway,
    analytics_context=analytics_context,
    graph_runtime=None,
    cfgs=cfgs,
)
```

Each plugin’s `run(ctx)` will pick up its config from the appropriate `ctx.*_cfg` field.

---

## Step 5 — Testing strategy

To keep this safe, I’d do tests in layers:

1. **Unit tests** for `analytics/plugins.py`:

   * register / get plugins
   * plan_analytics_plugins respects `depends_on` and disabled lists.

2. **Unit tests** for `analytics/plugin_runtime.py`:

   * A fake plugin with:

     * deterministic output and row counts.
     * `row_count_tables` set to a temp table.
   * Test:

     * dry_run → always skipped.
     * skip_on_unchanged → uses `is_unchanged` properly.
     * retry logic via raising once then succeeding.

3. **Thin integration tests** per domain:

   * `tests/analytics/test_function_metrics_plugin_runtime.py`:

     * Build a tiny repo snapshot with 1–2 functions.
     * Build `FunctionAnalyticsStepConfig`.
     * Register `functions.metrics` plugin.
     * Run via generic harness; assert rows in `analytics.function_metrics` and that the manifest is written.

4. **Graph regression tests**:

   * Reuse your existing `test_graph_service_runtime.py` and `test_graph_manifest_skip.py`, but point them at the new GraphServiceRuntime implementation that calls the generic harness.
   * Assert behavior of:

     * `skip_on_unchanged` for graph plugins.
     * severity / fail_fast semantics.

---

If you’d like, I can next:

* Pick **one non-graph plugin** (e.g. **function metrics**) and give you a **file-by-file patch plan** (exact diffs) to move it into this generic harness, or
* Focus on the **tests/profiles** slice and show how to express test profiling as an analytics plugin using this new layer.

# file by file patch for function metrics #

Great, let’s make “function metrics” a first-class citizen in the new generic plugin harness.

I’ll give you a **file-by-file patch plan**, with concrete code snippets, assuming:

* You’ve implemented (or are implementing) the **generic layer** from Epic 2:

  * `analytics/plugins.py` – `AnalyticsPlugin`, `AnalyticsExecutionContext`, registry, planning.
  * `analytics/plugin_runtime.py` – generic harness (planning + execution + manifest).
* We’ll now **plug `compute_function_metrics_and_types` into that harness**, and
* Update the pipeline to use the harness instead of calling the step directly.

I’ll keep the plan usable even if you’re still finishing the harness – the function side is cleanly separated.

---

## Overview of changes

**Files touched / added:**

1. `analytics/plugins.py`

   * Ensure `"function"` is a known stage.
2. `analytics/functions/plugins.py` **(new)**

   * Define `FunctionMetrics` as an `AnalyticsPlugin`.
3. `analytics/functions/__init__.py`

   * Optionally expose plugin name and keep public API stable.
4. `pipeline/orchestration/steps_analytics.py`

   * Make `FunctionAnalyticsStep.run` call the generic harness instead of `compute_function_metrics_and_types` directly.
5. `tests/analytics/test_function_metrics_plugin_runtime.py` **(new)**

   * Integration test harness for the plugin; checks unchanged detection & summary plumbing.

---

## 1. `analytics/plugins.py` — ensure `"function"` stage & registry

If you followed the earlier Epic-2 sketch, you should already have something like:

```python
# analytics/plugins.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Callable, TypeVar

from pydantic import BaseModel

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntime
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig
from codeintel.config.steps_graphs import GraphMetricsStepConfig, GraphRunScope
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)
```

### 1.1 Stage literal includes `"function"`

Make sure the `Stage` alias includes `"function"` (and whatever else you like):

```python
Stage = Literal[
    "graph",
    "function",
    "function_history",
    "test",
    "subsystem",
    "data_model",
    "data_model_usage",
    "entrypoints",
    "profiles",
    "history",
    "other",
]
```

If you don’t have `Stage` yet, add it with the values above.

### 1.2 `AnalyticsExecutionContext` includes `function_cfg`

Confirm your `AnalyticsExecutionContext` has `function_cfg: FunctionAnalyticsStepConfig | None`:

```python
@dataclass
class AnalyticsExecutionContext:
    gateway: StorageGateway
    analytics_context: AnalyticsContext | None
    repo: str
    commit: str

    graph_runtime: GraphRuntime | None = None

    function_cfg: FunctionAnalyticsStepConfig | None = None
    # ... other cfgs, e.g. test_profile_cfg, subsystems_cfg, etc.

    options: object | None = None
    plugin_name: str | None = None
    scope: GraphRunScope = field(default_factory=GraphRunScope)
    run_id: str | None = None
    scratch: object | None = None
    extra: dict[str, Any] = field(default_factory=dict)
```

If it’s missing, add `function_cfg`.

### 1.3 Registry is generic

Make sure you have registry helpers (or add them):

```python
_ANALYTICS_PLUGINS: dict[str, AnalyticsPlugin] = {}


def register_analytics_plugin(plugin: AnalyticsPlugin) -> None:
    if plugin.name in _ANALYTICS_PLUGINS:
        msg = f"Duplicate analytics plugin name: {plugin.name}"
        raise ValueError(msg)
    _ANALYTICS_PLUGINS[plugin.name] = plugin
    log.debug("Registered analytics plugin %s (stage=%s)", plugin.name, plugin.stage)


def get_analytics_plugin(name: str) -> AnalyticsPlugin:
    try:
        return _ANALYTICS_PLUGINS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown analytics plugin {name!r}") from exc
```

We’ll use `register_analytics_plugin` from the function side next.

---

## 2. `analytics/functions/plugins.py` — new plugin for function metrics

**New file:** `analytics/functions/plugins.py`

This file wraps your existing `compute_function_metrics_and_types` into an `AnalyticsPlugin`.

```python
# analytics/functions/plugins.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_metrics_and_types,
)
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig
from codeintel.storage.gateway import StorageGateway


def _function_metrics_run(ctx: AnalyticsExecutionContext) -> dict[str, int]:
    """
    Bridge from generic AnalyticsExecutionContext -> existing function metrics step.

    - Requires ctx.function_cfg to be populated by the harness caller.
    - Uses ctx.analytics_context for richer context (validation reporter, etc.).
    """
    if ctx.function_cfg is None:
        msg = "FunctionAnalyticsStepConfig is required in AnalyticsExecutionContext.function_cfg"
        raise ValueError(msg)
    if not isinstance(ctx.gateway, StorageGateway):
        msg = "AnalyticsExecutionContext.gateway must be a StorageGateway"
        raise TypeError(msg)

    cfg: FunctionAnalyticsStepConfig = ctx.function_cfg
    opts = FunctionAnalyticsOptions(context=ctx.analytics_context)

    # returns the same summary dict as the old pipeline call
    summary = compute_function_metrics_and_types(
        ctx.gateway,
        cfg,
        options=opts,
    )
    # The harness will attach this to the AnalyticsRunRecord.meta["result"]
    return summary


FUNCTION_METRICS_PLUGIN = AnalyticsPlugin(
    name="functions.metrics",
    description="Compute function metrics, complexity, and type annotations.",
    stage="function",
    enabled_by_default=True,
    run=_function_metrics_run,
    severity="fatal",
    depends_on=("goids",),
    provides=("analytics.function_metrics", "analytics.function_types"),
    requires=("core.goids",),
    options_model=None,
    options_default=None,
    resource_hints=ResourceHints(
        max_runtime_ms=60_000,
        requires_gpu=False,
        priority=10,
    ),
    version_hash=None,  # you can wire a real hash from git or a constant later
    row_count_tables=("analytics.function_metrics", "analytics.function_types"),
)

# Register at import time
register_analytics_plugin(FUNCTION_METRICS_PLUGIN)
```

Key points:

* We keep **all existing behavior** inside `_function_metrics_run`:

  * It calls `compute_function_metrics_and_types(gateway, cfg, options=...)`.
  * It returns the summary dict `{metrics_rows, types_rows, ...}`.
* The plugin metadata:

  * `name="functions.metrics"` – we’ll use this later from the harness/pipeline.
  * Stage `"function"`.
  * `row_count_tables` so unchanged detection can reuse `is_unchanged(...)`.

---

## 3. `analytics/functions/__init__.py` — optional small tweak

You don’t *have* to expose the plugin from the public API, but you might want to signal its name.

### 3.1 Option A — leave as-is (minimal change)

You can leave `__all__` and `_LAZY_ATTRS` untouched; the plugin lives in `analytics/functions/plugins.py` and is discovered via its registration. No change required.

### 3.2 Option B — explicitly expose plugin name (optional)

If you want, you can add the plugin name to `__all__` just for discoverability:

```python
__all__ = [
    "FunctionAnalyticsOptions",
    "FunctionAnalyticsStepConfig",
    "TypednessFlags",
    "compute_function_contracts",
    "compute_function_effects",
    "compute_function_history",
    "compute_function_metrics_and_types",
    # Optional:
    # "FUNCTION_METRICS_PLUGIN",
]
```

I’d keep it simple and **not** export the plugin from this module; it’s mostly internal to the harness.

---

## 4. `pipeline/orchestration/steps_analytics.py` — use the generic harness

We now update `FunctionAnalyticsStep.run` to **call the generic plugin harness** instead of invoking `compute_function_metrics_and_types` directly.

### 4.1 Imports

At the top of `pipeline/orchestration/steps_analytics.py`, adjust imports:

**Before:**

```python
from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_contracts,
    compute_function_effects,
    compute_function_history,
    compute_function_metrics_and_types,
)
from codeintel.analytics.graph_service_runtime import GraphPluginRunOptions, GraphServiceRuntime
```

**After:**

```python
from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_contracts,
    compute_function_effects,
    compute_function_history,
)
from codeintel.analytics.plugins import AnalyticsExecutionContext
from codeintel.analytics.plugin_runtime import (
    AnalyticsPluginRunOptions,
    plan_analytics_plugin_run,
    run_analytics_plugins,
)
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
```

We drop `compute_function_metrics_and_types` from here because the harness will call it via the plugin.

> If you already have `AnalyticsPluginRunOptions` under a different name, just align the import accordingly.

### 4.2 Rewrite `FunctionAnalyticsStep.run`

**Before (current):**

```python
@dataclass
class FunctionAnalyticsStep:
    """Build analytics.function_metrics and analytics.function_types."""

    name: str = "function_metrics"
    description: str = "Compute per-function metrics, complexity, and type annotations."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("goids",)

    def run(self, ctx: PipelineContext) -> None:
        """Compute per-function metrics and typedness."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = ctx.config_builder().function_analytics(
            fail_on_missing_spans=ctx.function_fail_on_missing_spans,
            parser=ctx.function_parser,
        )
        acx = _analytics_context(ctx)
        summary = compute_function_metrics_and_types(
            gateway,
            cfg,
            options=FunctionAnalyticsOptions(context=acx),
        )
        log.info(
            "function_metrics summary rows=%d types=%d validation=%d "
            "parse_failed=%d span_not_found=%d",
            summary["metrics_rows"],
            summary["types_rows"],
            summary["validation_total"],
            summary["validation_parse_failed"],
            summary["validation_span_not_found"],
        )
```

**After (use generic harness; keep logging):**

```python
@dataclass
class FunctionAnalyticsStep:
    """Build analytics.function_metrics and analytics.function_types via the generic harness."""

    name: str = "function_metrics"
    description: str = "Compute per-function metrics, complexity, and type annotations."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("goids",)

    def run(self, ctx: PipelineContext) -> None:
        """Compute per-function metrics and typedness via AnalyticsPlugin harness."""
        _log_step(self.name)

        gateway = ctx.gateway
        cfg = ctx.config_builder().function_analytics(
            fail_on_missing_spans=ctx.function_fail_on_missing_spans,
            parser=ctx.function_parser,
        )
        acx = _analytics_context(ctx)

        # Determine repo/commit
        repo = cfg.repo
        commit = cfg.commit

        # Policy: reuse GraphPluginPolicy for now (it already encodes severity / retries / skip_on_unchanged)
        policy = GraphPluginPolicy()

        # Scope: we can start with a default; you can later narrow to specific modules/paths.
        scope = GraphRunScope()

        # Manifest path: optional, but nice for "skip on unchanged" across runs
        manifest_path = ctx.build_dir / "manifests" / "function_metrics.json"

        # Per-run options (we're not using plugin-specific options yet)
        run_options = AnalyticsPluginRunOptions(
            plugin_options=None,
            manifest_path=manifest_path,
            scope=scope,
            dry_run=False,
        )

        prior_manifest = load_prior_manifest(manifest_path)

        # Plan: single plugin for now
        plugin_names = ("functions.metrics",)

        plan = plan_analytics_plugin_run(
            plugin_names=plugin_names,
            policy=policy,
            repo=repo,
            commit=commit,
            scope=scope,
            prior_manifest=prior_manifest or {},
            cfg_options={},      # no config-driven plugin options yet
            runtime_options={},  # no runtime-supplied options yet
            run_id=ctx.run_id,   # or uuid.uuid4().hex if you prefer
        )

        # Execute via generic harness
        analytics_report = run_analytics_plugins(
            plan=plan,
            gateway=gateway,
            analytics_context=acx,
            graph_runtime=None,
            cfgs={"function": cfg},
        )

        # Find our function metrics record and extract the summary dict
        summary: dict[str, int] = {
            "metrics_rows": 0,
            "types_rows": 0,
            "validation_total": 0,
            "validation_parse_failed": 0,
            "validation_span_not_found": 0,
        }
        for rec in analytics_report.records:
            if rec.name == "functions.metrics":
                result = rec.meta.get("result")
                if isinstance(result, dict):
                    summary = {
                        **summary,
                        **{k: int(v) for k, v in result.items() if isinstance(v, int)},
                    }
                break

        log.info(
            "function_metrics summary rows=%d types=%d validation=%d "
            "parse_failed=%d span_not_found=%d",
            summary["metrics_rows"],
            summary["types_rows"],
            summary["validation_total"],
            summary["validation_parse_failed"],
            summary["validation_span_not_found"],
        )
```

> This assumes your harness populates `rec.meta["result"]` with whatever `plugin.run(ctx)` returns. In the next mini-patch we’ll ensure `_execute_with_retries` does that.

---

## 5. `analytics/plugin_runtime.py` — capture plugin result into `meta["result"]`

In the generic harness you sketched earlier, `_execute_with_retries` (or equivalent) probably returns `(status, error, duration_ms, attempts)` and discards the plugin’s return value.

We want to **keep the plugin’s return value**, so we can stash it into `AnalyticsRunRecord.meta["result"]`.

### 5.1 Patch `_execute_with_retries`

**Before (conceptually):**

```python
def _execute_with_retries(
    plugin: AnalyticsPlugin,
    ctx: AnalyticsExecutionContext,
    settings: AnalyticsPluginExecutionSettings,
) -> tuple[AnalyticsStatus, str | None, float, int]:
    start = time.perf_counter()
    attempts = 0
    error: str | None = None
    status: AnalyticsStatus = "succeeded"

    while attempts < max(settings.retry_cfg.max_attempts, 1):
        attempts += 1
        try:
            plugin.run(ctx)
            status = "succeeded"
            error = None
            break
        except Exception as exc:
            # ... retry logic ...
            status = "failed"
            error = repr(exc)
            # fail_fast path etc.
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    return status, error, duration_ms, attempts
```

**After (capture `result`):**

```python
from typing import Any

def _execute_with_retries(
    plugin: AnalyticsPlugin,
    ctx: AnalyticsExecutionContext,
    settings: AnalyticsPluginExecutionSettings,
) -> tuple[AnalyticsStatus, str | None, float, int, Any]:
    """
    Execute plugin.run(ctx) with retries & timeouts.

    Returns (status, error, duration_ms, attempts, result_payload).
    """
    start = time.perf_counter()
    attempts = 0
    error: str | None = None
    status: AnalyticsStatus = "succeeded"
    result: Any = None

    while attempts < max(settings.retry_cfg.max_attempts, 1):
        attempts += 1
        try:
            result = plugin.run(ctx)
            status = "succeeded"
            error = None
            break
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            if settings.severity == "skip_on_error":
                status = "skipped"
                break
            if attempts < max(settings.retry_cfg.max_attempts, 1):
                # optional backoff
                if settings.retry_cfg.backoff_ms > 0:
                    time.sleep(settings.retry_cfg.backoff_ms / 1000)
                continue
            status = "failed"
            if settings.severity == "fatal" and settings.fail_fast:
                raise PluginFatalError(rec := _make_failure_record(...), exc) from exc
            break

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    return status, error, duration_ms, attempts, result
```

Then, when you build `AnalyticsRunRecord` in `run_analytics_plugins`, attach the result:

```python
status, error, duration_ms, attempts, result = _execute_with_retries(plugin, ctx, settings)

records.append(
    AnalyticsRunRecord(
        name=plugin.name,
        kind=plugin.stage,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        attempts=attempts,
        partial=status != "succeeded",
        error=error,
        meta={
            "severity": settings.severity,
            "options_hash": settings.options_hash,
            "version_hash": settings.version_hash,
            "result": result,
            # extra things if you want
        },
    )
)
```

This is what `FunctionAnalyticsStep.run` reads back as `rec.meta["result"]`.

---

## 6. Tests — `tests/analytics/test_function_metrics_plugin_runtime.py` (new)

Finally, add a **targeted test** to exercise the plugin + harness end-to-end.

**New file:** `tests/analytics/test_function_metrics_plugin_runtime.py`

```python
from __future__ import annotations

from pathlib import Path

from codeintel.analytics.functions.plugins import FUNCTION_METRICS_PLUGIN  # import ensures registration
from codeintel.analytics.plugin_runtime import (
    AnalyticsPluginRunOptions,
    plan_analytics_plugin_run,
    run_analytics_plugins,
)
from codeintel.config import ConfigBuilder
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import GoidRow, insert_goids


def _seed_single_function(
    gateway: StorageGateway,
    tmp_path: Path,
    *,
    rel_path: str = "mod.py",
    qualname: str = "pkg.mod.foo",
) -> None:
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def foo(x: int) -> int:\n    return x + 1\n", encoding="utf-8")

    insert_goids(
        gateway,
        GoidRow(
            repo="demo/repo",
            commit="abc123",
            relpath=rel_path,
            qualname=qualname,
            start_line=1,
            end_line=2,
        ),
    )


def test_function_metrics_plugin_via_harness(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    gateway = fresh_gateway
    _seed_single_function(gateway, tmp_path)

    builder = ConfigBuilder.from_snapshot(
        repo="demo/repo",
        commit="abc123",
        repo_root=tmp_path,
    )
    cfg = builder.function_analytics(
        fail_on_missing_spans=False,
        parser=None,
    )
    policy = GraphPluginPolicy()
    scope = GraphRunScope()

    run_id = "test-run-123"
    manifest_path = tmp_path / "manifest_function_metrics.json"
    prior_manifest = {}

    plan = plan_analytics_plugin_run(
        plugin_names=("functions.metrics",),
        policy=policy,
        repo=cfg.repo,
        commit=cfg.commit,
        scope=scope,
        prior_manifest=prior_manifest,
        cfg_options={},
        runtime_options={},
        run_id=run_id,
    )

    report = run_analytics_plugins(
        plan=plan,
        gateway=gateway,
        analytics_context=None,
        graph_runtime=None,
        cfgs={"function": cfg},
    )

    # Check we have exactly one record and it succeeded
    assert len(report.records) == 1
    rec = report.records[0]
    assert rec.name == "functions.metrics"
    assert rec.status == "succeeded"
    assert rec.error is None

    # Check result summary is present and contains reasonable rows
    summary = rec.meta.get("result")
    assert isinstance(summary, dict)
    assert summary["metrics_rows"] >= 1
    assert summary["types_rows"] >= 1
```

You can add a second test that:

* Runs once, captures manifest.
* Runs again with `policy.skip_on_unchanged = True` and the same manifest path.
* Asserts `rec.status == "skipped"` on the second run.

---

If you’d like, next I can do the same style of **patch-plan** for another non-graph plugin (e.g. test profiles or subsystems), which will further solidify the pattern and make it easy to roll across the rest of `analytics/`.


# zoom in on test profiles and behavioral coverage #

Awesome, let’s turn **test profiles + behavioral coverage** into first-class plugins like we did for function metrics.

I’ll follow the same structure: **file-by-file patch plan with code snippets**, assuming:

* You now have the **generic plugin harness** (`analytics/plugins.py`, `analytics/plugin_runtime.py`, `runtime_manifest`, `manifest.py`) roughly in place.
* Function metrics is already wired through the harness (`functions.metrics`.

We’ll now:

1. Extend the generic context to know about **test profile configs**.
2. Add **two plugins**:

   * `tests.profile` → `analytics.test_profile`.
   * `tests.behavioral_coverage` → `analytics.behavioral_coverage`.
3. Move `TestProfileStep` and `BehavioralCoverageStep` to call the harness.
4. Add a small harness test.

---

## 1. `analytics/plugins.py` — extend context for test configs

We want the generic `AnalyticsExecutionContext` to carry **test-specific configs** so plugins can grab them.

### 1.1 Imports

At the top of `analytics/plugins.py`, extend imports to include the test configs:

```python
from codeintel.config.steps_analytics import (
    FunctionAnalyticsStepConfig,
    TestProfileStepConfig,
    BehavioralCoverageStepConfig,
    # ... other step configs ...
)
```

### 1.2 Extend `AnalyticsExecutionContext`

Add two optional fields to the context:

```python
@dataclass
class AnalyticsExecutionContext:
    gateway: StorageGateway
    analytics_context: AnalyticsContext | None
    repo: str
    commit: str

    graph_runtime: GraphRuntime | None = None

    function_cfg: FunctionAnalyticsStepConfig | None = None

    # NEW: test-focused configs
    test_profile_cfg: TestProfileStepConfig | None = None
    behavioral_cfg: BehavioralCoverageStepConfig | None = None

    # ... other cfgs (subsystems_cfg, etc.) ...

    options: object | None = None
    plugin_name: str | None = None
    scope: GraphRunScope = field(default_factory=GraphRunScope)
    run_id: str | None = None
    scratch: object | None = None
    extra: dict[str, Any] = field(default_factory=dict)
```

Now any plugin with `stage="test"` can access `ctx.test_profile_cfg` / `ctx.behavioral_cfg`.

---

## 2. `analytics/tests/plugins.py` — new plugins for tests

**New file:** `analytics/tests/plugins.py`

We’ll wrap your existing entrypoints:

* `analytics/tests/profiles.build_test_profile`
* `analytics/tests/profiles.build_behavioral_coverage`

as `AnalyticsPlugin`s.

### 2.1 `tests.profile` plugin

```python
# analytics/tests/plugins.py

from __future__ import annotations

from typing import Any

from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.analytics.tests.profiles import build_test_profile
from codeintel.config.steps_analytics import TestProfileStepConfig
from codeintel.storage.gateway import StorageGateway


def _test_profile_run(ctx: AnalyticsExecutionContext) -> dict[str, int]:
    """
    Bridge from generic AnalyticsExecutionContext -> build_test_profile.

    Returns a small summary dict that we can log in the pipeline:
    number of tests and number of profile rows written.
    """
    if ctx.test_profile_cfg is None:
        msg = "TestProfileStepConfig is required in AnalyticsExecutionContext.test_profile_cfg"
        raise ValueError(msg)
    if not isinstance(ctx.gateway, StorageGateway):
        msg = "AnalyticsExecutionContext.gateway must be a StorageGateway"
        raise TypeError(msg)

    cfg: TestProfileStepConfig = ctx.test_profile_cfg
    gateway = ctx.gateway

    # Call existing implementation (side-effect: writes analytics.test_profile)
    build_test_profile(gateway, cfg)

    # Optional: compute a tiny summary for logging
    con = gateway.con
    (row_count,) = con.execute(
        """
        SELECT COUNT(*) FROM analytics.test_profile
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchone()
    return {
        "profile_rows": int(row_count),
    }


TEST_PROFILE_PLUGIN = AnalyticsPlugin(
    name="tests.profile",
    description="Build per-test profiles with coverage and subsystem context.",
    stage="test",
    enabled_by_default=True,
    run=_test_profile_run,
    severity="fatal",
    depends_on=(
        "tests.ingest",          # or whatever your plugin names become
        "coverage.functions",
        "coverage.edges",
        "subsystems.build",
        "graphs.metrics",
    ),
    provides=("analytics.test_profile",),
    requires=("core.goids", "coverage.test_edges"),
    options_model=None,
    options_default=None,
    resource_hints=ResourceHints(
        max_runtime_ms=60_000,
        requires_gpu=False,
        priority=20,
    ),
    version_hash=None,  # can be wired to a stable constant or git hash later
    row_count_tables=("analytics.test_profile",),
)

register_analytics_plugin(TEST_PROFILE_PLUGIN)
```

(You can tune `depends_on`/`requires`/`provides` names later to match the eventual plugin registry vocabulary; the important thing is the pattern.)

### 2.2 `tests.behavioral_coverage` plugin

`build_behavioral_coverage` needs:

* `BehavioralCoverageStepConfig`
* An **LLM runner** (`BehavioralLLMRunner | None`)
* The same gateway.

We’ll pass the config via `ctx.behavioral_cfg`, and LLM bits via **context.extra** and plugin options (for hashing).

```python
from codeintel.analytics.tests.profiles import build_behavioral_coverage
from codeintel.analytics.tests_profiles.types import BehavioralLLMRunner
from codeintel.config.steps_analytics import BehavioralCoverageStepConfig


def _behavioral_coverage_run(ctx: AnalyticsExecutionContext) -> dict[str, int]:
    """
    Bridge from AnalyticsExecutionContext -> build_behavioral_coverage.

    LLM knobs:
      - cfg.enable_llm / cfg.llm_model come from BehavioralCoverageStepConfig
      - actual llm_runner is passed through ctx.extra["behavioral_llm_runner"]
    """
    if ctx.behavioral_cfg is None:
        msg = "BehavioralCoverageStepConfig is required in AnalyticsExecutionContext.behavioral_cfg"
        raise ValueError(msg)
    if not isinstance(ctx.gateway, StorageGateway):
        msg = "AnalyticsExecutionContext.gateway must be a StorageGateway"
        raise TypeError(msg)

    cfg: BehavioralCoverageStepConfig = ctx.behavioral_cfg
    gateway = ctx.gateway

    llm_runner = ctx.extra.get("behavioral_llm_runner")
    if llm_runner is not None and not callable(llm_runner):
        raise TypeError("behavioral_llm_runner in ctx.extra must be callable or None")

    build_behavioral_coverage(
        gateway,
        cfg,
        llm_runner=llm_runner,  # type: ignore[arg-type]
    )

    con = gateway.con
    (row_count,) = con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.behavioral_coverage
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchone()
    return {"behavior_rows": int(row_count)}
```

Define the plugin:

```python
BEHAVIORAL_COVERAGE_PLUGIN = AnalyticsPlugin(
    name="tests.behavioral_coverage",
    description="Assign heuristic behavior tags to tests (unit/integration/etc.).",
    stage="test",
    enabled_by_default=True,
    run=_behavioral_coverage_run,
    severity="fatal",
    depends_on=("tests.profile",),
    provides=("analytics.behavioral_coverage",),
    requires=("analytics.test_profile",),
    # These options mirror the config for hashing / manifest purposes
    options_model=None,  # you could add a Pydantic model for enable_llm/llm_model
    options_default={"enable_llm": False, "llm_model": None},
    resource_hints=ResourceHints(
        max_runtime_ms=120_000,
        requires_gpu=False,
        priority=30,
    ),
    version_hash=None,
    row_count_tables=("analytics.behavioral_coverage",),
)

register_analytics_plugin(BEHAVIORAL_COVERAGE_PLUGIN)
```

Now both test-oriented operations are proper plugins in the **shared analytics registry**.

---

## 3. `analytics/plugin_runtime.py` — pass cfgs & extra to tests

Your generic harness already takes `cfgs` and builds `AnalyticsExecutionContext`.

We now just need to:

1. Pass the test configs into the context.
2. Propagate any `extra` dict (for LLMS) into `ctx.extra`.

### 3.1 Extend `run_analytics_plugins` signature

Change from:

```python
def run_analytics_plugins(
    *,
    plan: AnalyticsPluginExecutionPlan,
    gateway: StorageGateway,
    analytics_context: AnalyticsContext | None,
    graph_runtime: GraphRuntime | None,
    cfgs: dict[str, object],
) -> AnalyticsRunReport:
    ...
```

to:

```python
from typing import Any

def run_analytics_plugins(
    *,
    plan: AnalyticsPluginExecutionPlan,
    gateway: StorageGateway,
    analytics_context: AnalyticsContext | None,
    graph_runtime: GraphRuntime | None,
    cfgs: dict[str, object],
    extra: dict[str, Any] | None = None,
) -> AnalyticsRunReport:
    ...
```

### 3.2 Set the right cfg fields for test plugins

Inside the plugin loop:

```python
    extra_payload = extra or {}
    for plugin in plan.plugins:
        settings = plan.settings_by_plugin[plugin.name]
        options = plan.options_by_plugin.get(plugin.name)

        ctx = AnalyticsExecutionContext(
            gateway=gateway,
            analytics_context=analytics_context,
            repo=plan.repo,
            commit=plan.commit,
            graph_runtime=graph_runtime if plugin.stage == "graph" else None,
            function_cfg=cfgs.get("function") if plugin.stage == "function" else None,
            test_profile_cfg=cfgs.get("test_profile") if plugin.name == "tests.profile" else None,
            behavioral_cfg=cfgs.get("behavioral_coverage")
            if plugin.name == "tests.behavioral_coverage"
            else None,
            options=options,
            plugin_name=plugin.name,
            scope=plan.scope,
            run_id=plan.run_id,
            scratch=scratch,
            extra=dict(extra_payload),
        )
        ...
```

(You can refactor that mapping into a helper later; for now this explicit mapping makes the behavior very clear.)

### 3.3 Make sure you attach plugin result into `meta["result"]`

As in the previous function-metrics work, ensure `_execute_with_retries` returns a `result` and `run_analytics_plugins` stashes it in the `AnalyticsRunRecord.meta`:

```python
status, error, duration_ms, attempts, result = _execute_with_retries(plugin, ctx, settings)

records.append(
    AnalyticsRunRecord(
        name=plugin.name,
        kind=plugin.stage,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        attempts=attempts,
        partial=status != "succeeded",
        error=error,
        meta={
            "severity": settings.severity,
            "options_hash": settings.options_hash,
            "version_hash": settings.version_hash,
            "result": result,
        },
    )
)
```

Then both `function_metrics` and test plugins have their summary outputs accessible at `rec.meta["result"]`.

---

## 4. `pipeline/orchestration/steps_analytics.py` — wire steps to the harness

Now we make **TestProfileStep** and **BehavioralCoverageStep** use the generic harness instead of calling the functions directly.

### 4.1 Imports

At the top of `steps_analytics.py`, add:

```python
from codeintel.analytics.plugin_runtime import (
    AnalyticsPluginRunOptions,
    plan_analytics_plugin_run,
    run_analytics_plugins,
)
from codeintel.analytics.plugins import AnalyticsExecutionContext
from codeintel.analytics.graphs.runtime.manifest import load_prior_manifest
from codeintel.analytics.runtime_manifest import AnalyticsScope, encode_manifest
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
```

(If you already imported some of these for function metrics, just reuse.)

### 4.2 Patch `TestProfileStep.run`

**Before:**

```python
@dataclass
class TestProfileStep:
    ...

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.test_profile."""
        _log_step(self.name)
        cfg = ctx.config_builder().test_profile()
        build_test_profile(ctx.gateway, cfg)
        if cfg.refresh_subsystem_cache:
            refresh_subsystem_caches(
                ctx.gateway,
                repo=cfg.repo,
                commit=cfg.commit,
                benchmark=cfg.benchmark_subsystem_cache,
            )
```

**After (use plugin harness; keep refresh logic):**

```python
@dataclass
class TestProfileStep:
    ...

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.test_profile via AnalyticsPlugin harness."""
        _log_step(self.name)
        cfg = ctx.config_builder().test_profile()

        # Policy + scope
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        run_id = ctx.run_id

        manifest_path = ctx.build_dir / "manifests" / "test_profile.json"
        prior_manifest = load_prior_manifest(manifest_path)

        # Single plugin run
        plugin_names = ("tests.profile",)

        plan = plan_analytics_plugin_run(
            plugin_names=plugin_names,
            policy=policy,
            repo=cfg.repo,
            commit=cfg.commit,
            scope=scope,
            prior_manifest=prior_manifest or {},
            cfg_options={},      # no structured plugin options yet
            runtime_options={},  # same
            run_id=run_id,
        )

        analytics_report = run_analytics_plugins(
            plan=plan,
            gateway=ctx.gateway,
            analytics_context=_analytics_context(ctx),
            graph_runtime=None,
            cfgs={"test_profile": cfg},
            extra={},
        )

        # Optional: manifest for unchanged detection
        if manifest_path is not None:
            payload = encode_manifest(analytics_report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Keep subsystem cache refresh behavior as-is
        if cfg.refresh_subsystem_cache:
            refresh_subsystem_caches(
                ctx.gateway,
                repo=cfg.repo,
                commit=cfg.commit,
                benchmark=cfg.benchmark_subsystem_cache,
            )
```

You can optionally log the `result` summary:

```python
        summary = {}
        for rec in analytics_report.records:
            if rec.name == "tests.profile":
                if isinstance(rec.meta.get("result"), dict):
                    summary = rec.meta["result"]
                break
        log.info(
            "test_profile summary rows=%d",
            summary.get("profile_rows", 0),
        )
```

### 4.3 Patch `BehavioralCoverageStep.run`

**Before:**

```python
@dataclass
class BehavioralCoverageStep:
    ...

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.behavioral_coverage."""
        _log_step(self.name)
        enable_llm = bool(
            ctx.extra.get("enable_behavioral_llm")
            or os.getenv("CODEINTEL_BEHAVIORAL_LLM", "").lower() in {"1", "true", "yes"}
        )
        llm_model_raw = ctx.extra.get("behavioral_llm_model")
        llm_model = llm_model_raw if isinstance(llm_model_raw, str) else None
        llm_runner = ctx.extra.get("behavioral_llm_runner")
        cfg = ctx.config_builder().behavioral_coverage(
            enable_llm=enable_llm,
            llm_model=llm_model,
        )
        build_behavioral_coverage(ctx.gateway, cfg, llm_runner=llm_runner)  # type: ignore[arg-type]
```

**After (use plugin harness; keep env + extras logic):**

```python
@dataclass
class BehavioralCoverageStep:
    ...

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.behavioral_coverage via AnalyticsPlugin harness."""
        _log_step(self.name)

        # Existing env/extra logic
        enable_llm = bool(
            ctx.extra.get("enable_behavioral_llm")
            or os.getenv("CODEINTEL_BEHAVIORAL_LLM", "").lower() in {"1", "true", "yes"}
        )
        llm_model_raw = ctx.extra.get("behavioral_llm_model")
        llm_model = llm_model_raw if isinstance(llm_model_raw, str) else None

        cfg = ctx.config_builder().behavioral_coverage(
            enable_llm=enable_llm,
            llm_model=llm_model,
        )

        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        run_id = ctx.run_id

        manifest_path = ctx.build_dir / "manifests" / "behavioral_coverage.json"
        prior_manifest = load_prior_manifest(manifest_path)

        plugin_names = ("tests.behavioral_coverage",)

        # Use plugin options purely to reflect config knobs in the manifest hash
        cfg_options: dict[str, dict[str, object]] = {
            "tests.behavioral_coverage": {
                "enable_llm": cfg.enable_llm,
                "llm_model": cfg.llm_model,
            }
        }

        plan = plan_analytics_plugin_run(
            plugin_names=plugin_names,
            policy=policy,
            repo=cfg.repo,
            commit=cfg.commit,
            scope=scope,
            prior_manifest=prior_manifest or {},
            cfg_options=cfg_options,
            runtime_options={},
            run_id=run_id,
        )

        analytics_report = run_analytics_plugins(
            plan=plan,
            gateway=ctx.gateway,
            analytics_context=_analytics_context(ctx),
            graph_runtime=None,
            cfgs={"behavioral_coverage": cfg},
            extra={
                "behavioral_llm_runner": ctx.extra.get("behavioral_llm_runner"),
            },
        )

        if manifest_path is not None:
            payload = encode_manifest(analytics_report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Optional logging
        summary = {}
        for rec in analytics_report.records:
            if rec.name == "tests.behavioral_coverage":
                if isinstance(rec.meta.get("result"), dict):
                    summary = rec.meta["result"]
                break
        log.info(
            "behavioral_coverage summary rows=%d enable_llm=%s llm_model=%s",
            summary.get("behavior_rows", 0),
            cfg.enable_llm,
            cfg.llm_model,
        )
```

Now both steps are **thin wrappers** around the same generic harness used for functions and graphs.

---

## 5. Basic harness test for `tests.profile` (optional but recommended)

**New test:** `tests/analytics/test_tests_profile_plugin_runtime.py`

```python
from __future__ import annotations

from pathlib import Path

from codeintel.analytics.tests.plugins import TEST_PROFILE_PLUGIN  # noqa: F401 ensure registration
from codeintel.analytics.plugin_runtime import (
    plan_analytics_plugin_run,
    run_analytics_plugins,
)
from codeintel.config import ConfigBuilder
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from tests._helpers.fixtures import ProvisionedGateway


def test_tests_profile_plugin_smoke(provisioned_gateway: ProvisionedGateway, tmp_path: Path) -> None:
    gateway = provisioned_gateway.gateway
    snapshot = provisioned_gateway.snapshot

    builder = ConfigBuilder.from_snapshot(snapshot)
    cfg = builder.test_profile()

    policy = GraphPluginPolicy()
    scope = GraphRunScope()
    run_id = "test-tests-profile-1"

    plan = plan_analytics_plugin_run(
        plugin_names=("tests.profile",),
        policy=policy,
        repo=cfg.repo,
        commit=cfg.commit,
        scope=scope,
        prior_manifest={},
        cfg_options={},
        runtime_options={},
        run_id=run_id,
    )

    report = run_analytics_plugins(
        plan=plan,
        gateway=gateway,
        analytics_context=None,
        graph_runtime=None,
        cfgs={"test_profile": cfg},
        extra={},
    )

    assert len(report.records) == 1
    rec = report.records[0]
    assert rec.name == "tests.profile"
    assert rec.status == "succeeded"
    summary = rec.meta.get("result")
    assert isinstance(summary, dict)
    assert summary["profile_rows"] >= 0
```

You can add a second test that verifies `skip_on_unchanged` when re-running with a manifest, just like for graph and function plugins.

---

This gives you a **fully parallel story**:

* Graphs: `graph` stage, plugins, harness, manifest.
* Functions: `function` stage, `functions.metrics` plugin, harness.
* Tests: `test` stage, `tests.profile` + `tests.behavioral_coverage` plugins, harness.

If you want, next we can:

* Do a **clean “registry view”** doc: a table of all current plugins (graph, function, test) with their `stage`, `provides`, `requires`, and `row_count_tables`, or
* Tackle another analytics family (e.g. **subsystems/materialize** or **entrypoints**) with the same patch-plan pattern.
