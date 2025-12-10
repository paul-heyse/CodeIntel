Here’s a concrete “unfork the graph stack” plan that:

* **Keeps graphs as the canonical place for graph *compute* (including metrics)**
* **Keeps analytics as the canonical plugin / pipeline runtime**
* **Removes the orphaned graph plugin runtime and duplication of metrics logic** 

I’ll structure it as:

1. Target architecture (what we want the world to look like)
2. Step A – Make `graphs.compute.metrics` the canonical metrics library
3. Step B – Rewrite analytics graph metrics to use `graphs.compute.metrics`
4. Step C – Remove the graph plugin runtime (`graphs.plugins.*` + registry/executor)
5. Step D – Clean up config/pipeline/CLI references
6. Final “am I done?” checklist

---

## 1. Target architecture

**Today (simplified)**

* Analytics pipeline path:

  ```text
  pipeline → analytics.core.plugins.graphs.CoreGraphMetricsPlugin
           → analytics.graphs.graph_metrics
           → analytics.graph_service
           → analytics.graph_metrics.metrics (centrality, components, etc)
           → NetworkX
  ```

* Orphan graph plugin stack:

  ```text
  graphs.plugins.builders.* / graphs.plugins.metrics.*
       → graphs.compute.metrics.*
       → NetworkX
  graphs.runtime.registry / graphs.runtime.executor
  ```

The plugin stack in `graphs.plugins.*` is effectively **unused** outside `graphs/` itself; analytics has its own plugin system that drives metrics.

**Target state**

* **One metrics implementation** in `graphs.compute.metrics.*`
* **One plugin runtime** – the *analytics* plugin system
* Analytics graph metrics plugin calls into `graphs.compute.metrics`, not its own parallel implementation
* No `GraphPluginRegistry`, `GraphPluginProtocol`, or `graphs.plugins.metrics/*` in the live code paths

Visually:

```text
pipeline → analytics.core.plugins.graphs.CoreGraphMetricsPlugin
         → analytics.graphs.graph_metrics (or similar)
         → graphs.compute.metrics
         → NetworkX
```

…and `graphs.plugins.*` + graph plugin registry/executor are gone.

---

## 2. Step A – Make `graphs.compute.metrics` canonical

### A.1. Inventory and stabilize metrics API in `graphs.compute.metrics`

Open `graphs/compute/metrics/*.py` and identify the functions that are conceptually “public”:

* e.g.:

  ```python
  # graphs/compute/metrics/centrality.py
  def compute_core_centralities(G: nx.Graph, *, caps: CentralityCaps) -> CoreCentralityResult: ...
  def compute_extended_centralities(G: nx.Graph, *, caps: CentralityCaps) -> ExtendedCentralityResult: ...

  # graphs/compute/metrics/coupling.py
  def compute_coupling_metrics(G: nx.DiGraph, ...) -> CouplingMetricsResult: ...

  # graphs/compute/metrics/components.py
  def compute_component_stats(G: nx.Graph, ...) -> ComponentStatsResult: ...
  ```

Create a **single façade module** inside `graphs.compute.metrics` (if it doesn’t already exist) that re‑exports the canonical surface:

```python
# graphs/compute/metrics/__init__.py

from .centrality import compute_core_centralities, compute_extended_centralities
from .components import compute_component_stats
from .coupling import compute_coupling_metrics
# ...any others

__all__ = [
    "compute_core_centralities",
    "compute_extended_centralities",
    "compute_component_stats",
    "compute_coupling_metrics",
    # ...
]
```

This gives analytics a single, stable import surface:

```python
from codeintel.graphs.compute.metrics import (
    compute_core_centralities,
    compute_component_stats,
    compute_coupling_metrics,
)
```

> If some of the functions are currently only wired through `graphs.plugins.metrics`, keep them – we’ll call them directly from analytics in the next step instead of through the plugin layer.

---

## 3. Step B – Rewire analytics metrics to use `graphs.compute.metrics`

### B.1. Replace analytics’ metrics implementation with thin facades

In `analytics/graph_metrics/metrics.py` (and any sibling modules), you likely have code like:

```python
# analytics/graph_metrics/metrics.py (today)

def compute_core_graph_metrics(G: nx.Graph, caps: GraphMetricsCaps) -> CoreGraphMetrics:
    # … direct NetworkX calls here …
    betweenness = nx.betweenness_centrality(G, k=caps.betweenness_sample, ...)
    pagerank = nx.pagerank(G, alpha=caps.pagerank_alpha)
    eigen = nx.eigenvector_centrality(G, max_iter=caps.eigen_max_iter)
    # …pack into CoreGraphMetrics...
    return CoreGraphMetrics(
        betweenness=betweenness,
        pagerank=pagerank,
        eigen=eigen,
        # ...
    )
```

We want to turn this into a **thin wrapper** over `graphs.compute.metrics`, then eventually delete it.

**After:**

```python
# analytics/graph_metrics/metrics.py (new, thin façade)
from codeintel.graphs.compute.metrics import compute_core_centralities
from codeintel.graphs.compute.metrics import compute_component_stats
from codeintel.graphs.compute.metrics import compute_coupling_metrics

from .types import CoreGraphMetrics, GraphMetricsCaps  # existing analytics types

def compute_core_graph_metrics(G: nx.Graph, caps: GraphMetricsCaps) -> CoreGraphMetrics:
    result = compute_core_centralities(
        G,
        caps=_to_centrality_caps(caps),  # small adapter if needed
    )
    return _from_core_centrality_result(result)


def compute_component_metrics(G: nx.Graph, caps: GraphMetricsCaps) -> ComponentMetrics:
    result = compute_component_stats(G, caps=_to_component_caps(caps))
    return _from_component_stats_result(result)

def compute_coupling_metrics(G: nx.DiGraph, caps: GraphMetricsCaps) -> CouplingMetrics:
    result = compute_coupling_metrics(G, caps=_to_coupling_caps(caps))
    return _from_coupling_metrics_result(result)
```

Where:

* `_to_centrality_caps`, `_from_core_centrality_result`, etc. are small adapter functions that map between the *analytics* types and the *graphs* types (if they differ). If the types already align, you can delete the adapters and pass/return directly.

The same pattern applies for:

* Any subsystem graph metrics (`analytics/graphs/subsystem_graph_metrics.py`)
* Any symbol graph metrics (`analytics/graphs/symbol_graph_metrics.py`)
* Etc.

### B.2. Update `analytics/graphs/graph_metrics.py` to call the façade

In `analytics/graphs/graph_metrics.py` (the orchestrator used by the analytics plugin), you should already call into `analytics.graph_metrics.metrics`:

```python
from codeintel.analytics.graph_metrics.metrics import compute_core_graph_metrics

def build_function_graph_metrics(...):
    ...
    metrics = compute_core_graph_metrics(G, caps)
    ...
```

With the façade in place, you don’t have to change call sites here – they’re automatically routed to `graphs.compute.metrics`. The only change is that the *implementation* in `analytics.graph_metrics.metrics` is now just a delegating layer.

### B.3. Optional: move more orchestration logic into graphs

If some orchestration logic (like building the appropriate subgraph or normalizing weights) is duplicated between analytics and `graphs.compute.metrics`, you can:

* Move the shared parts into helper functions in `graphs.compute.metrics` (or a sibling module `graphs.compute.views`), and
* Have `analytics.graphs.graph_metrics` call those helpers instead of copy/pasting.

Do that *after* the initial delegation is working and tests pass.

---

## 4. Step C – Remove the graph plugin runtime and metrics plugins

Now that analytics plugins are the only ones invoking metrics, we can delete the **orphaned graph plugin layer**.

### C.1. Remove graph metrics plugins in `graphs/plugins/metrics/*`

Search for graph metrics plugins, e.g.:

```bash
rg "class .*GraphMetricsPlugin" src/graphs/graphs/plugins
rg "plugins.metrics" src/graphs
```

You should see modules like:

* `graphs/plugins/metrics/core.py`
* `graphs/plugins/metrics/secondary.py`
* Maybe others (symbol, subsystem, config, etc.)

They’ll look roughly like:

```python
# graphs/plugins/metrics/core.py

from codeintel.graphs.core.protocol import GraphPluginProtocol, GraphPluginMetadata
from codeintel.graphs.compute.metrics import compute_core_centralities

class CoreGraphMetricsPlugin(GraphPluginProtocol):
    metadata = GraphPluginMetadata(
        name="core_graph_metrics",
        ...
    )

    def execute(self, ctx: GraphPluginContext) -> GraphPluginResult:
        engine = ctx.require_graph_engine()
        G = engine.get_graph("callgraph")
        caps = ctx.require_caps()

        result = compute_core_centralities(G, caps=caps)
        # write to DB / return result
```

These are now **redundant**: analytics’ graph metrics plugin is the only registered one.

**Actions:**

1. Delete the graph metrics plugin modules:

   * `graphs/plugins/metrics/core.py`
   * `graphs/plugins/metrics/secondary.py`
   * any other `graphs/plugins/metrics/*.py`

2. Remove any imports / re‑exports of these plugins from:

   * `graphs/plugins/__init__.py`
   * `graphs/runtime/manifest.py` (if it lists default plugins)
   * `graphs/runtime/__init__.py` (if it re‑exports “core graph metrics plugin” getters)

If you find any helper functions like:

```python
def get_core_graph_metrics_plugin() -> GraphPluginProtocol:
    """Backward-compatible getter for core graph metrics plugin."""
    return CoreGraphMetricsPlugin()
```

Delete them too.

### C.2. Remove builder/validation plugins in `graphs/plugins/builders/*` **if** they’re unused

Repeat the same analysis for builder/validation plugins:

```bash
rg "graphs.plugins.builders" src
rg "GraphPluginRegistry" src
```

If they are **only** registered and executed inside the graph plugin runtime (and analytics/pipeline never import `graphs.plugins.builders` directly), you have two options:

* **Minimalist:** keep the pure *compute* functions in `graphs.compute.*` (e.g., callgraph construction, CFG/DFG building) but remove the *plugin wrappers* around them.

  * E.g. keep `graphs.compute.callgraph.build_callgraph(...)`, but delete `CallgraphBuilderPlugin` class and its registration in `graphs/plugins/builders/callgraph.py`.

* **Or keep builder plugins** if the analytics pipeline or the CLI **directly** instantiates them (unlikely given earlier inspection). In that case, they aren’t orphaned and you’d leave them for now.

Given the earlier analysis, it looked like these builder plugins are only referenced from the graph plugin registry/executor; if that holds, you can safely delete `graphs/plugins/builders/*` and keep the compute modules.

### C.3. Remove the graph plugin registry & executor

Now strip the plugin runtime itself:

Files to inspect and likely remove:

* `graphs/core/protocol.py` – `GraphPluginProtocol`, `GraphPluginMetadata`, `GraphPluginMetaOptions`, `GraphPluginResult`
* `graphs/core/registry.py` – `GraphPluginRegistry`, `GraphPluginPlan`, plugin registration helpers
* `graphs/runtime/manifest.py` – plugin manifests / plugin lists
* `graphs/runtime/planning.py` – plugin planning / scheduling logic
* `graphs/runtime/executor.py` – plugin executor
* `graphs/runtime/telemetry.py` – tracing/metrics for GraphPluginExecutor

For each:

1. **Search for external uses**, e.g.:

   ```bash
   rg "GraphPluginRegistry" src
   rg "GraphPluginProtocol" src
   rg "graph_plugin" src
   rg "run_graph_plugins" src
   ```

   If all hits are inside `graphs/*` (and not e.g. in `pipeline/` or `cli/`), you’re safe to delete.

2. Remove:

   * These modules entirely, or
   * Reduce them to a very small stub if some constants are still imported but easily replaced.

3. If any external module imports something from them, replace those imports with direct calls to **compute** or **engine** functions instead:

   ```python
   # BEFORE
   from codeintel.graphs.runtime.executor import run_graph_plugins

   # AFTER – call compute directly or use analytics.GraphRuntime
   from codeintel.graphs.compute.callgraph import build_callgraph
   ```

Once this is done, `graphs` becomes:

* **Engine + compute** only (no plugin registry/executor),
* Used by analytics plugins as the sole consumer of metrics.

---

## 5. Step D – Clean up config / pipeline / CLI references

After removing `graphs.plugins.*` and the plugin registry, we need to ensure no other package thinks graph plugins exist.

### D.1. Pipeline / config

Search for references in `pipeline/` and `config/`:

```bash
rg "plugins.metrics" src/pipeline src/config
rg "GraphPlugin" src/pipeline src/config
rg "graph_plugins" src/pipeline src/config
```

If you find something like:

```python
# pipeline/steps_graphs.py
from codeintel.graphs.runtime.executor import run_graph_plugins

def run_graph_metrics(...):
    ...
    run_graph_plugins(manifest=graph_manifest, ...)
```

Replace that call with **the analytics plugin pipeline** you already have:

* Either:

  ```python
  from codeintel.analytics.core.pipeline_bridge import run_analytics_graphs_step

  run_analytics_graphs_step(...)
  ```

* Or call your analytics graph plugin directly, depending on how the rest of your pipeline is wired (you already use analytics plugins for metrics, so there should be a standard hook to run them).

The important bit: **no pipeline stage should import or call the removed graph plugin executor.**

### D.2. CLI

Search for graph plugin mentions in `cli/`:

```bash
rg "GraphPlugin" src/cli
rg "graphs.plugins" src/cli
```

If there’s any CLI command that directly exposes graph plugins (e.g., a debug tool that allows running arbitrary graph plugins), either:

* Remove that feature, or
* Rewrite it to call analytics graph metrics or specific `graphs.compute` helpers instead.

Given previous inspection, the CLI primarily talks to the pipeline and serving layers, not graph plugins directly, so this should be minimal or nonexistent.

---

## 6. Final checklist: no more orphan graph plugin + metrics duplication

You’re done with this cluster when:

* [ ] `rg "GraphPluginProtocol" src` returns **no results**.
* [ ] `rg "GraphPluginRegistry" src` returns **no results**.
* [ ] `rg "graphs.plugins.metrics" src` and `rg "graphs.plugins.builders" src` return no results.
* [ ] All graph metrics functions used by analytics live in `graphs.compute.metrics` (and are imported from there).
* [ ] `analytics.graph_metrics.metrics` is either:

  * a **thin façade** over `graphs.compute.metrics`, or
  * fully deleted, with analytics calling `graphs.compute.metrics` directly.
* [ ] `analytics.graphs.graph_metrics` and other analytics graph modules do **not** call NetworkX directly; they go through `graphs.compute.metrics` (or a minimal shared helper).
* [ ] Pipeline and CLI only use:

  * analytics plugins,
  * `GraphRuntime` / engine factory, and
  * `graphs.compute` helpers – **not** graph plugins.

At that point you have:

* **Exactly one plugin system** (analytics) driving graph metrics.
* **Exactly one metrics implementation** (in `graphs.compute.metrics`).
* The `graphs.plugins.*` layer and its registry/executor no longer exist, so there’s no parallel/orphan graph plugin architecture to maintain.
