You’re at the fun part now: you already have a nice graph runtime and orchestration; we’re basically going to bolt a clean plugin system on top of what you’ve built.

I’ll base this entirely on the *current* code you just uploaded (with `GraphServiceRuntime`, `graph_metrics.py`, `config_graph_metrics.py`, `cfg_dfg/materialize.py`, `tests/graph_metrics.py`, etc.).

Below is a full implementation plan, with concrete file-level changes and code snippets.

---

## 0. What we’re trying to achieve

**Today:**

* `GraphServiceRuntime` (`analytics/analytics/graph_service_runtime.py`) has hard-coded methods:

  * `compute_graph_metrics(cfg)`
  * `compute_graph_metrics_ext(repo, commit)`
  * `compute_symbol_metrics(repo, commit)`
  * `compute_subsystem_metrics(repo, commit)`
  * `compute_graph_stats(repo, commit)`
* `GraphMetricsStep.run` (`pipeline/pipeline/orchestration/steps_analytics.py`) calls:

  ```python
  service.compute_graph_metrics(cfg)
  service.compute_graph_metrics_ext(...)
  compute_test_graph_metrics(...)
  compute_cfg_metrics(...)
  compute_dfg_metrics(...)
  service.compute_symbol_metrics(...)
  compute_config_graph_metrics(...)
  compute_subsystem_agreement(...)
  service.compute_graph_stats(...)
  ```

**After refactor:**

* All these are **GraphMetricPlugin**s in `analytics.analytics.graphs.plugins`.
* `GraphMetricsStep` just says: “run these plugin names”.
* `GraphServiceRuntime.run_plugins(...)` executes plugins against a shared `GraphRuntime`.
* `GraphMetricsStepConfig` can enable/disable plugins declaratively.

---

## 1. Add plugin infrastructure (`analytics/analytics/graphs/plugins.py`)

### 1.1. New module: `analytics/analytics/graphs/plugins.py`

Create a new file:

```python
# analytics/analytics/graphs/plugins.py
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from codeintel.analytics.graph_runtime import GraphRuntime
from codeintel.analytics.context import AnalyticsContext
from codeintel.config import GraphMetricsStepConfig
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


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
    run: Callable[[GraphMetricExecutionContext], None]


_PLUGINS: dict[str, GraphMetricPlugin] = {}


def register_graph_metric_plugin(plugin: GraphMetricPlugin) -> None:
    """Register a graph metric plugin at import time."""
    if plugin.name in _PLUGINS:
        message = f"Duplicate graph metric plugin name: {plugin.name}"
        raise ValueError(message)
    _PLUGINS[plugin.name] = plugin
    log.debug("Registered graph metric plugin %s (stage=%s)", plugin.name, plugin.stage)


def get_graph_metric_plugin(name: str) -> GraphMetricPlugin:
    """Return a plugin by name or raise KeyError."""
    return _PLUGINS[name]


def list_graph_metric_plugins() -> tuple[GraphMetricPlugin, ...]:
    """Return all registered plugins."""
    return tuple(_PLUGINS.values())
```

We’ll also define a default ordering after we register built-ins.

---

## 2. Register built-in plugins (wrap the existing functions)

Now we wrap the existing metric functions with plugins; each plugin just calls into the module you already have.

### 2.1. Core function/module graph metrics

This wraps `analytics.analytics.graphs.graph_metrics.compute_graph_metrics`.

Add this near the bottom of `analytics/analytics/graphs/plugins.py`:

```python
# analytics/analytics/graphs/plugins.py (continued)

from pathlib import Path

from codeintel.analytics.graphs.graph_metrics import (
    GraphMetricFilters,
    GraphMetricsDeps,
    compute_graph_metrics,
)
from codeintel.analytics.cfg_dfg.materialize import (
    compute_cfg_metrics,
    compute_dfg_metrics,
)
from codeintel.analytics.graphs.config_graph_metrics import compute_config_graph_metrics
from codeintel.analytics.tests.graph_metrics import compute_test_graph_metrics
from codeintel.analytics.graphs.symbol_graph_metrics import (
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.analytics.graphs.subsystem_graph_metrics import (
    compute_subsystem_graph_metrics,
)
from codeintel.analytics.graphs.graph_stats import compute_graph_stats
from codeintel.config.primitives import SnapshotRef
from codeintel.analytics.graph_runtime import GraphRuntimeOptions


def _ensure_graph_metrics_cfg(
    ctx: GraphMetricExecutionContext,
) -> GraphMetricsStepConfig:
    """
    Resolve a GraphMetricsStepConfig from context.config or derive from runtime.

    This ensures that plugins which rely on GraphMetricsStepConfig can always
    obtain one, even when called from outside the pipeline.
    """
    if ctx.config is not None:
        return ctx.config

    # Fall back to runtime snapshot when config is not available.
    opts: GraphRuntimeOptions = ctx.runtime.options
    snapshot = opts.snapshot or SnapshotRef(
        repo=ctx.repo,
        commit=ctx.commit,
        repo_root=Path(),
    )
    return GraphMetricsStepConfig(snapshot=snapshot)


def _plugin_core_graph_metrics(ctx: GraphMetricExecutionContext) -> None:
    cfg = _ensure_graph_metrics_cfg(ctx)
    deps = GraphMetricsDeps(
        catalog_provider=ctx.catalog_provider,
        runtime=ctx.runtime,
        analytics_context=ctx.analytics_context,
        filters=None,  # could be extended later
    )
    compute_graph_metrics(ctx.gateway, cfg, deps=deps)
```

Then register it:

```python
register_graph_metric_plugin(
    GraphMetricPlugin(
        name="core_graph_metrics",
        description="Core function/module graph metrics (centrality, neighbors, components).",
        stage="core",
        enabled_by_default=True,
        run=_plugin_core_graph_metrics,
    )
)
```

### 2.2. CFG / DFG metrics

Wrap `compute_cfg_metrics` and `compute_dfg_metrics` as separate plugins:

```python
def _plugin_cfg_metrics(ctx: GraphMetricExecutionContext) -> None:
    compute_cfg_metrics(
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
    compute_dfg_metrics(
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
```

### 2.3. Test graph metrics

Wrap `analytics.analytics.tests.graph_metrics.compute_test_graph_metrics`:

```python
def _plugin_test_graph_metrics(ctx: GraphMetricExecutionContext) -> None:
    compute_test_graph_metrics(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="test_graph_metrics",
        description="Metrics over the test <-> function bipartite graph.",
        stage="test",
        enabled_by_default=True,
        run=_plugin_test_graph_metrics,
    )
)
```

### 2.4. Symbol graph metrics (functions & modules)

Wrap both functions from `symbol_graph_metrics.py`:

```python
def _plugin_symbol_graph_metrics_modules(ctx: GraphMetricExecutionContext) -> None:
    compute_symbol_graph_metrics_modules(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="symbol_graph_metrics_modules",
        description="Symbol graph metrics at the module level.",
        stage="symbol",
        enabled_by_default=True,
        run=_plugin_symbol_graph_metrics_modules,
    )
)


def _plugin_symbol_graph_metrics_functions(ctx: GraphMetricExecutionContext) -> None:
    compute_symbol_graph_metrics_functions(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="symbol_graph_metrics_functions",
        description="Symbol graph metrics at the function level.",
        stage="symbol",
        enabled_by_default=True,
        run=_plugin_symbol_graph_metrics_functions,
    )
)
```

### 2.5. Subsystem graph metrics

Wrap `compute_subsystem_graph_metrics`:

```python
def _plugin_subsystem_graph_metrics(ctx: GraphMetricExecutionContext) -> None:
    compute_subsystem_graph_metrics(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
        filters=None,
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="subsystem_graph_metrics",
        description="Subsystem-level condensed import graph metrics.",
        stage="subsystem",
        enabled_by_default=True,
        run=_plugin_subsystem_graph_metrics,
    )
)
```

### 2.6. Config graph metrics

Wrap `compute_config_graph_metrics`:

```python
def _plugin_config_graph_metrics(ctx: GraphMetricExecutionContext) -> None:
    compute_config_graph_metrics(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="config_graph_metrics",
        description="Config bipartite/projection graph metrics.",
        stage="config",
        enabled_by_default=True,
        run=_plugin_config_graph_metrics,
    )
)
```

### 2.7. Global graph stats

Wrap `compute_graph_stats`:

```python
def _plugin_graph_stats(ctx: GraphMetricExecutionContext) -> None:
    compute_graph_stats(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=ctx.runtime,
    )


register_graph_metric_plugin(
    GraphMetricPlugin(
        name="graph_stats",
        description="Global graph statistics for core graphs.",
        stage="stats",
        enabled_by_default=True,
        run=_plugin_graph_stats,
    )
)
```

### 2.8. Default plugin ordering

At the very end of `plugins.py`, define a default ordering:

```python
DEFAULT_GRAPH_METRIC_PLUGINS: tuple[str, ...] = (
    "core_graph_metrics",
    "cfg_metrics",
    "dfg_metrics",
    "test_graph_metrics",
    "symbol_graph_metrics_modules",
    "symbol_graph_metrics_functions",
    "subsystem_graph_metrics",
    "config_graph_metrics",
    "graph_stats",
)
```

---

## 3. Add plugin execution to `GraphServiceRuntime`

**File:** `analytics/analytics/graph_service_runtime.py`

We’ll add a generic `run_plugins` method and adapt the existing methods to call it.

### 3.1. Imports

At the top, add:

```python
from collections.abc import Sequence

from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    DEFAULT_GRAPH_METRIC_PLUGINS,
    get_graph_metric_plugin,
)
```

### 3.2. New `run_plugins` method

Inside `GraphServiceRuntime`:

```python
@dataclass
class GraphServiceRuntime:
    """Lightweight orchestrator for graph analytics using a shared runtime."""

    gateway: StorageGateway
    runtime: GraphRuntime
    analytics_context: AnalyticsContext | None = None
    catalog_provider: FunctionCatalogProvider | None = None

    def run_plugins(
        self,
        plugin_names: Sequence[str],
        *,
        cfg: GraphMetricsStepConfig | None = None,
    ) -> None:
        """
        Execute a sequence of graph metric plugins against this runtime.

        Parameters
        ----------
        plugin_names:
            Names of plugins to execute, in order.
        cfg:
            Optional graph metrics configuration; provided to plugins via
            GraphMetricExecutionContext.
        """
        if cfg is not None:
            repo = cfg.repo
            commit = cfg.commit
        else:
            opts = self.runtime.options
            snapshot = opts.snapshot
            if snapshot is None:
                message = "Graph runtime missing snapshot; cannot derive repo/commit"
                raise ValueError(message)
            repo = snapshot.repo
            commit = snapshot.commit

        ctx = GraphMetricExecutionContext(
            gateway=self.gateway,
            runtime=self.runtime,
            repo=repo,
            commit=commit,
            config=cfg,
            analytics_context=self.analytics_context,
            catalog_provider=self.catalog_provider,
        )

        for name in plugin_names:
            plugin = get_graph_metric_plugin(name)
            log.info(
                "graph_runtime.plugin.start name=%s repo=%s commit=%s",
                name,
                repo,
                commit,
            )
            self._observe(name, lambda p=plugin: p.run(ctx))
```

> Note the `lambda p=plugin: ...` trick to avoid late binding of `plugin` in the loop.

### 3.3. Keep existing methods as thin wrappers (optional but nice)

For compatibility, re-implement:

```python
    def compute_graph_metrics(
        self, cfg: GraphMetricsStepConfig, *, filters: object | None = None
    ) -> None:
        """Compute core function/module graph metrics."""
        # filters is currently unused by plugins; kept for signature compatibility.
        self.run_plugins(["core_graph_metrics"], cfg=cfg)

    def compute_graph_metrics_ext(self, *, repo: str, commit: str) -> None:
        """Compute extended function and module graph metrics."""
        # We'll treat the ext metrics as just more plugins; they will derive cfg.
        self.run_plugins(
            ["symbol_graph_metrics_modules", "symbol_graph_metrics_functions"],
            cfg=None,
        )

    def compute_symbol_metrics(self, *, repo: str, commit: str) -> None:
        """Compute symbol graph metrics for modules and functions."""
        self.run_plugins(
            ["symbol_graph_metrics_modules", "symbol_graph_metrics_functions"],
            cfg=None,
        )

    def compute_subsystem_metrics(self, *, repo: str, commit: str) -> None:
        """Compute subsystem-level graph metrics."""
        self.run_plugins(["subsystem_graph_metrics"], cfg=None)

    def compute_graph_stats(self, *, repo: str, commit: str) -> None:
        """Compute global graph statistics."""
        self.run_plugins(["graph_stats"], cfg=None)
```

You may choose to drop some of these later, but leaving them as wrappers keeps internal callers working during the transition.

---

## 4. Extend `GraphMetricsStepConfig` to carry plugin toggles

**File:** `config/config/steps_graphs.py`

In `GraphMetricsStepConfig`, add optional plugin configuration:

```python
@dataclass
class GraphMetricsStepConfig:
    """Configuration for graph metrics analytics."""

    snapshot: SnapshotRef
    max_betweenness_sample: int | None = 200
    eigen_max_iter: int = 200
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"
    seed: int = 0

    # NEW: plugin selection
    enabled_plugins: tuple[str, ...] = ()
    disabled_plugins: tuple[str, ...] = ()

    @property
    def repo(self) -> str:
        ...
```

These fields are **data only**; we’ll handle interpretation in the orchestration step to avoid `config` importing `analytics`.

---

## 5. Make `GraphMetricsStep` plugin-driven

**File:** `pipeline/pipeline/orchestration/steps_analytics.py`

We’ll adjust `GraphMetricsStep.run` to:

* Build `GraphMetricsStepConfig` as before.
* Construct `GraphServiceRuntime`.
* Determine plugin list from:

  * `DEFAULT_GRAPH_METRIC_PLUGINS` (from plugins module),
  * applying `enabled_plugins` / `disabled_plugins`.
* Call `service.run_plugins(...)` instead of manually calling each metric.

### 5.1. Imports

At the top of `steps_analytics.py`, extend imports:

```python
from codeintel.analytics.graph_service_runtime import GraphServiceRuntime
from codeintel.analytics.graphs.plugins import DEFAULT_GRAPH_METRIC_PLUGINS
```

### 5.2. Helper to resolve plugin list

Optionally add a small helper near `GraphMetricsStep`:

```python
def _resolve_graph_plugins(
    cfg: GraphMetricsStepConfig,
    default_plugins: Sequence[str],
) -> tuple[str, ...]:
    """
    Resolve effective graph metric plugins from config and defaults.

    Rules:
    - If cfg.enabled_plugins is non-empty, use that list exactly (in order).
    - Otherwise, start from default_plugins and drop any in cfg.disabled_plugins.
    """
    if cfg.enabled_plugins:
        return tuple(cfg.enabled_plugins)

    result: list[str] = []
    disabled = set(cfg.disabled_plugins)
    for name in default_plugins:
        if name not in disabled:
            result.append(name)
    return tuple(result)
```

### 5.3. Rewrite `GraphMetricsStep.run`

Current (simplified):

```python
class GraphMetricsStep:
    ...
    def run(self, ctx: PipelineContext) -> None:
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = ctx.config_builder().graph_metrics()
        acx = _analytics_context(ctx)
        runtime = ensure_graph_runtime(ctx, acx=acx)
        service = GraphServiceRuntime(
            gateway=gateway,
            runtime=runtime,
            analytics_context=acx,
            catalog_provider=acx.catalog,
        )
        service.compute_graph_metrics(cfg)
        service.compute_graph_metrics_ext(repo=ctx.repo, commit=ctx.commit)
        compute_test_graph_metrics(..., runtime=runtime)
        compute_cfg_metrics(...)
        compute_dfg_metrics(...)
        service.compute_symbol_metrics(repo=ctx.repo, commit=ctx.commit)
        compute_config_graph_metrics(..., runtime=runtime)
        compute_subsystem_agreement(...)
        service.compute_subsystem_metrics(repo=ctx.repo, commit=ctx.commit)
        service.compute_graph_stats(repo=ctx.repo, commit=ctx.commit)
```

New version:

```python
class GraphMetricsStep:
    """Compute graph metrics for functions and modules."""

    name: str = "graph_metrics"
    description: str = "Compute centrality, coupling, and graph metrics for functions and modules."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("callgraph", "import_graph", "symbol_uses", "cfg", "test_coverage_edges")

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.graph_metrics_* tables."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = ctx.config_builder().graph_metrics()
        acx = _analytics_context(ctx)
        runtime = ensure_graph_runtime(ctx, acx=acx)
        service = GraphServiceRuntime(
            gateway=gateway,
            runtime=runtime,
            analytics_context=acx,
            catalog_provider=acx.catalog,
        )

        # Determine plugin list from config + defaults.
        plugin_names = _resolve_graph_plugins(cfg, DEFAULT_GRAPH_METRIC_PLUGINS)
        log.info(
            "graph_metrics.plugins repo=%s commit=%s plugins=%s",
            ctx.repo,
            ctx.commit,
            plugin_names,
        )
        service.run_plugins(plugin_names, cfg=cfg)

        # Subsystem agreement is not directly a graph metric; you can either
        # keep it here or make it a separate plugin later.
        compute_subsystem_agreement(gateway, repo=ctx.repo, commit=ctx.commit)
```

> Note:
>
> * All of the work previously done by direct calls (`compute_graph_metrics`, `compute_test_graph_metrics`, `compute_cfg_metrics`, etc.) is now handled by plugins.
> * You keep `compute_subsystem_agreement` as a separate (non-plugin) operation for now, or you can add a plugin `subsystem_agreement` that wraps it.

---

## 6. Optional: expose plugins to CLI / agents

If you want agents or users to choose plugins at runtime, you can add a CLI subcommand and an MCP endpoint.

### 6.1. CLI: list graph plugins

**File:** `cli/cli/main.py`

Add a simple handler that prints plugin metadata:

```python
from codeintel.analytics.graphs.plugins import list_graph_metric_plugins

def cmd_list_graph_plugins(args: argparse.Namespace) -> int:
    from textwrap import indent

    for plugin in list_graph_metric_plugins():
        print(f"- {plugin.name} [{plugin.stage}]")
        print(indent(plugin.description, "    "))
    return 0
```

Wire it into your CLI argument parser under something like `graph plugins`.

### 6.2. Serving: list graph plugins

**File:** `serving/serving/mcp/models.py`

Add a simple descriptor:

```python
class GraphPluginDescriptor(BaseModel):
    name: str
    stage: str
    description: str
    enabled_by_default: bool
```

**File:** `serving/serving/mcp/query_service.py`

Add method:

```python
from codeintel.analytics.graphs.plugins import list_graph_metric_plugins

    def list_graph_plugins(self) -> list[GraphPluginDescriptor]:
        """
        List available graph metric plugins.

        Useful for agent configuration and debugging.
        """
        return [
            GraphPluginDescriptor(
                name=p.name,
                stage=p.stage,
                description=p.description,
                enabled_by_default=p.enabled_by_default,
            )
            for p in list_graph_metric_plugins()
        ]
```

---

## 7. Testing & validation

### 7.1. Unit tests for plugins

**New file:** `analytics/analytics/tests/test_graph_plugins.py`

```python
from __future__ import annotations

from codeintel.analytics.graphs.plugins import (
    DEFAULT_GRAPH_METRIC_PLUGINS,
    list_graph_metric_plugins,
    get_graph_metric_plugin,
)


def test_default_plugins_registered() -> None:
    names = {p.name for p in list_graph_metric_plugins()}
    for name in DEFAULT_GRAPH_METRIC_PLUGINS:
        assert name in names


def test_get_graph_metric_plugin_round_trip() -> None:
    for name in DEFAULT_GRAPH_METRIC_PLUGINS:
        plugin = get_graph_metric_plugin(name)
        assert plugin.name == name
        assert plugin.description
```

### 7.2. Integration test for `GraphMetricsStep`

In your existing analytics pipeline tests, add a scenario that:

* Runs the `graph_metrics` pipeline step for a small test repo.
* Asserts that key result tables or views are populated:

  * `analytics.graph_metrics_functions`
  * `analytics.graph_metrics_modules`
  * `analytics.test_graph_metrics_*`
  * `analytics.cfg_function_metrics`
  * etc.

This is mostly a regression check to ensure we didn’t forget a plugin.

### 7.3. Config toggles test

Add a small test that disables a plugin and asserts its table is not touched:

```python
def test_disable_cfg_metrics_plugin(tmp_gateway, pipeline_runner) -> None:
    # Build a config where GraphMetricsStepConfig disables cfg_metrics.
    cfg = build_config_with_overrides(
        graph_metrics={"disabled_plugins": ("cfg_metrics",)}
    )
    pipeline_runner.run_steps(["graph_metrics"], config=cfg)

    # Assert cfg tables are empty while others are populated.
    con = tmp_gateway.con
    assert con.execute("SELECT COUNT(*) FROM analytics.cfg_function_metrics").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM analytics.graph_metrics_functions").fetchone()[0] > 0
```

(Adapt to your actual testing harness.)

---

## 8. Implementation order (for you / your agent)

1. **Add `analytics/analytics/graphs/plugins.py`**:

   * `GraphMetricExecutionContext`, `GraphMetricPlugin`, registry functions.
   * Built-in plugin definitions and `DEFAULT_GRAPH_METRIC_PLUGINS`.

2. **Modify `analytics/analytics/graph_service_runtime.py`**:

   * Import plugin helpers.
   * Add `run_plugins(...)`.
   * Reimplement `compute_*` methods as thin wrappers (optional but recommended).

3. **Extend `GraphMetricsStepConfig`** in `config/config/steps_graphs.py` with `enabled_plugins` / `disabled_plugins`.

4. **Update `GraphMetricsStep.run`** in `pipeline/pipeline/orchestration/steps_analytics.py`:

   * Replace direct compute calls with plugin execution via `GraphServiceRuntime.run_plugins`.

5. **Add tests** for plugin registry & plugin-driven `graph_metrics` step.

After this, adding a new graph metric is literally:

* Implement it in a module (e.g. `analytics/analytics/graphs/my_metric.py`).
* Wrap it in a `GraphMetricPlugin` in `plugins.py` and register it.
* Optionally add it to `DEFAULT_GRAPH_METRIC_PLUGINS`.

And GraphMetricsStep + GraphServiceRuntime will pick it up without further wiring.
