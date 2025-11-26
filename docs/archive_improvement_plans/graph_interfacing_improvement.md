Love it, and nice work getting the analytics re-shape done 👌 — that makes this step a lot cleaner.

Let’s now make **graphs** into a proper “engine layer” and have analytics talk to it through a stable interface.

I’ll structure this like before:

1. Goals & invariants
2. Inventory: how graphs & analytics interact today
3. Define the `GraphEngine` interface
4. Implement `NxGraphEngine` on top of `graphs.nx_views`
5. Integrate `GraphEngine` with `GraphRuntimeOptions` / `AnalyticsContext`
6. Refactor `analytics.graph_metrics` to use `GraphEngine`
7. Refactor the other analytics modules (module/symbol/config/subsystem/tests)
8. Optional: CLI / future backends
9. Tests & guardrails

---

## 1. Goals & invariants

**End state:**

* There is a small, explicit **graph engine interface** that data consumers depend on:

  ```python
  GraphEngine -> nx.Graph / nx.DiGraph
  ```

* All code in `analytics.*` that needs graphs uses **`GraphEngine`**, not `graphs.nx_views` directly.

* All **NetworkX + DuckDB wiring** (how to build call/import/symbol graphs from DuckDB) lives in the graphs layer (e.g. `graphs.nx_views` and `graphs.engine`), not sprinkled through analytics.

* `AnalyticsContext` + `GraphRuntimeOptions` act as **hints and caches** for the engine, not as parallel “graph loaders”.

This gives you:

* Clean separation: graph construction vs. graph analytics.
* The ability to later plug in a different engine (GPU, alternative graph library, duckdb-native graphs) with minimal analytics changes.

---

## 2. Inventory: current state (from your zips)

### Graph building / views

In `graphs/`:

* `graphs/nx_views.py` builds NetworkX graphs directly from DuckDB, e.g.:

  ```python
  def load_call_graph(gateway: StorageGateway, repo: str, commit: str, *, use_gpu: bool = False) -> nx.DiGraph
  def load_import_graph(...)
  def load_test_function_bipartite(...)
  def load_config_module_bipartite(...)
  def load_symbol_module_graph(...)
  def load_symbol_function_graph(...)
  ```

* `graphs/callgraph_builder.py`, `call_ast.py`, `call_cst.py`, `import_graph.py`, `symbol_uses.py`, `cfg_builder.py`, etc. build the **underlying tables** in `graph.*` that `nx_views` consumes.

These are the core “graph construction” primitives.

### Analytics graph consumers

In `analytics/`:

* `analytics/graph_metrics.py`:

  ```python
  from codeintel.graphs.nx_views import load_call_graph, load_import_graph
  from codeintel.analytics.graph_service import GraphBundle, GraphContext, neighbor_stats, ...

  def compute_graph_metrics(..., runtime: GraphRuntimeOptions | None = None):
      runtime = runtime or GraphRuntimeOptions()
      context = runtime.context
      graph_ctx = runtime.graph_ctx
      use_gpu = runtime.use_gpu

      ctx = graph_ctx or build_graph_context(cfg, now=..., use_gpu=use_gpu)
      call_graph_cached = context.call_graph if context is not None else None
      import_graph_cached = context.import_graph if context is not None else None

      def _call_graph_loader() -> nx.DiGraph:
          return call_graph_cached or load_call_graph(gateway, cfg.repo, cfg.commit, use_gpu=use_gpu)

      def _import_graph_loader() -> nx.DiGraph:
          return import_graph_cached or load_import_graph(...)
      
      bundle = GraphBundle(ctx=ctx, loaders={"call_graph": _call_graph_loader, "import_graph": _import_graph_loader})
      _compute_function_graph_metrics(..., ctx=ctx, bundle=bundle)
      _compute_module_graph_metrics(..., ctx=ctx, bundle=bundle, ...)
  ```

* `analytics/module_graph_metrics_ext.py`:

  ```python
  from codeintel.graphs.nx_views import load_import_graph

  def compute_module_graph_metrics_ext(..., runtime: GraphRuntimeOptions | None = None):
      runtime = runtime or GraphRuntimeOptions()
      use_gpu = runtime.use_gpu
      import_graph_cached = runtime.context.import_graph if runtime.context is not None else None
      graph = import_graph_cached or load_import_graph(gateway, repo, commit, use_gpu=use_gpu)
      # compute metrics...
  ```

* `analytics/symbol_graph_metrics.py`:

  ```python
  from codeintel.graphs.nx_views import load_symbol_function_graph, load_symbol_module_graph
  from codeintel.analytics.graph_service import GraphBundle, GraphContext, ...

  bundle = GraphBundle(
      ctx=ctx,
      loaders={
          "symbol_function_graph": lambda: load_symbol_function_graph(...),
      },
  )
  graph = bundle.get("symbol_function_graph")
  # compute metrics...
  ```

* `analytics/config_graph_metrics.py`:

  ```python
  from codeintel.graphs.nx_views import load_config_module_bipartite
  graph = load_config_module_bipartite(gateway, repo, commit, use_gpu=use_gpu)
  ```

* `analytics/subsystem_graph_metrics.py`:

  ```python
  from codeintel.graphs.nx_views import load_import_graph
  graph = cached_import or load_import_graph(...)
  ```

* `analytics/tests/graph_metrics.py`:

  ```python
  from codeintel.graphs.nx_views import load_test_function_bipartite
  graph: nx.Graph = load_test_function_bipartite(gateway, repo, commit, use_gpu=use_gpu)
  ```

Also:

* `analytics/context.py` preloads graphs with `load_call_graph`, `load_import_graph`, `load_symbol_*` and keeps them as cached attributes.
* `analytics/graph_runtime.py` provides `GraphRuntimeOptions`, bundling `AnalyticsContext`, `GraphContext` and `GraphBackendConfig` (use_gpu hint).

So: analytics is currently tightly coupled to `graphs.nx_views` and the idea of “cached NetworkX graphs hanging off `AnalyticsContext`”.

---

## 3. Step 1 – Introduce the `GraphEngine` interface

We’ll create a small, NetworkX-oriented interface that analytics can depend on, and that `graphs` will implement.

### 3.1: New `graphs/engine.py`

Create `src/codeintel/graphs/engine.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, Callable, Mapping

import networkx as nx

from codeintel.storage.gateway import StorageGateway
```

Define a `GraphKind` enum for clarity:

```python
class GraphKind(Enum):
    CALL_GRAPH = auto()
    IMPORT_GRAPH = auto()
    SYMBOL_MODULE_GRAPH = auto()
    SYMBOL_FUNCTION_GRAPH = auto()
    CONFIG_MODULE_BIPARTITE = auto()
    TEST_FUNCTION_BIPARTITE = auto()
```

Define the **protocol**:

```python
class GraphEngine(Protocol):
    """
    Backend-agnostic interface for building and caching graphs used in analytics.

    All methods return NetworkX graphs for now, but this boundary allows us to
    swap in a GPU-backed or alternative implementation later.
    """

    def call_graph(self) -> nx.DiGraph: ...
    def import_graph(self) -> nx.DiGraph: ...

    def symbol_module_graph(self) -> nx.Graph: ...
    def symbol_function_graph(self) -> nx.Graph: ...

    def config_module_bipartite(self) -> nx.Graph: ...
    def test_function_bipartite(self) -> nx.Graph: ...
```

You can keep it minimal at first (call/import only) and extend later, but given you already have NxViews for all those, it’s straightforward to include them now.

The key point: **analytics code will depend only on this interface**, not on `nx_views`.

---

## 4. Step 2 – Implement `NxGraphEngine` on top of `nx_views`

Still in `graphs/engine.py`:

```python
from codeintel.graphs import nx_views
from codeintel.utils.paths import normalize_rel_path  # if needed
```

Implement the engine:

```python
@dataclass
class NxGraphEngine:
    """
    GraphEngine implementation backed by DuckDB + NetworkX.

    All graph construction details (SQL, views, GPU vs CPU) are encapsulated here.
    """

    gateway: StorageGateway
    repo: str
    commit: str
    use_gpu: bool = False

    # simple in-memory cache keyed by GraphKind.name
    _cache: dict[GraphKind, nx.Graph] = field(default_factory=dict)

    def _get(self, kind: GraphKind, loader: Callable[[], nx.Graph]) -> nx.Graph:
        graph = self._cache.get(kind)
        if graph is None:
            graph = loader()
            self._cache[kind] = graph
        return graph

    # ---- Core graphs ----

    def call_graph(self) -> nx.DiGraph:
        graph = self._get(
            GraphKind.CALL_GRAPH,
            lambda: nx_views.load_call_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )
        # mypy / type checkers don't know it's DiGraph, but we do
        return graph  # type: ignore[return-value]

    def import_graph(self) -> nx.DiGraph:
        graph = self._get(
            GraphKind.IMPORT_GRAPH,
            lambda: nx_views.load_import_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )
        return graph  # type: ignore[return-value]

    # ---- Symbol graphs ----

    def symbol_module_graph(self) -> nx.Graph:
        return self._get(
            GraphKind.SYMBOL_MODULE_GRAPH,
            lambda: nx_views.load_symbol_module_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )

    def symbol_function_graph(self) -> nx.Graph:
        return self._get(
            GraphKind.SYMBOL_FUNCTION_GRAPH,
            lambda: nx_views.load_symbol_function_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )

    # ---- Config + test graphs ----

    def config_module_bipartite(self) -> nx.Graph:
        return self._get(
            GraphKind.CONFIG_MODULE_BIPARTITE,
            lambda: nx_views.load_config_module_bipartite(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )

    def test_function_bipartite(self) -> nx.Graph:
        return self._get(
            GraphKind.TEST_FUNCTION_BIPARTITE,
            lambda: nx_views.load_test_function_bipartite(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )
```

This is now the **only place** that imports `graphs.nx_views` in analytics-land.

---

## 5. Step 3 – Integrate `GraphEngine` into `GraphRuntimeOptions` / `AnalyticsContext`

We want analytics to be able to reuse preloaded graphs (from `AnalyticsContext`) and GPU preferences (`GraphBackendConfig`) when constructing an engine.

### 5.1: Extend `GraphRuntimeOptions`

In `analytics/graph_runtime.py`:

Current (from your zip):

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.analytics.graph_service import GraphContext
from codeintel.config.models import GraphBackendConfig

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext


@dataclass(frozen=True)
class GraphRuntimeOptions:
    """
    Runtime options for graph-based analytics functions.

    Attributes
    ----------
    context :
        Optional shared AnalyticsContext containing cached graphs.
    graph_ctx :
        Optional GraphContext carrying weights and sampling limits.
    graph_backend :
        Optional backend selection used to derive GPU preferences.
    """

    context: AnalyticsContext | None = None
    graph_ctx: GraphContext | None = None
    graph_backend: GraphBackendConfig | None = None

    @property
    def use_gpu(self) -> bool:
        ...
```

Modify it to also carry an optional **engine**:

```python
from codeintel.graphs.engine import GraphEngine, NxGraphEngine
from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class GraphRuntimeOptions:
    ...
    engine: GraphEngine | None = None

    @property
    def use_gpu(self) -> bool:
        ...
```

Then add a helper to build an `NxGraphEngine` using:

* The current `GraphBackendConfig` (for `use_gpu`).
* `AnalyticsContext` (for seeded cache).
* Repo/commit from whatever config the analytics function passes in.

```python
    def build_engine(
        self,
        gateway: StorageGateway,
        repo: str,
        commit: str,
    ) -> GraphEngine:
        """
        Construct or reuse a GraphEngine for the given snapshot.

        Respects GPU preferences and seeds caches from AnalyticsContext graphs
        when available.
        """
        # If caller provided a concrete engine, reuse it
        if self.engine is not None:
            return self.engine

        use_gpu = self.use_gpu
        engine = NxGraphEngine(
            gateway=gateway,
            repo=repo,
            commit=commit,
            use_gpu=use_gpu,
        )

        # Seed cache from AnalyticsContext if available & matching snapshot
        if self.context is not None:
            ctx = self.context
            if ctx.repo == repo and ctx.commit == commit:
                if ctx.call_graph is not None:
                    engine._cache[GraphKind.CALL_GRAPH] = ctx.call_graph
                if ctx.import_graph is not None:
                    engine._cache[GraphKind.IMPORT_GRAPH] = ctx.import_graph
                if ctx.symbol_module_graph is not None:
                    engine._cache[GraphKind.SYMBOL_MODULE_GRAPH] = ctx.symbol_module_graph
                if ctx.symbol_function_graph is not None:
                    engine._cache[GraphKind.SYMBOL_FUNCTION_GRAPH] = ctx.symbol_function_graph
                # Tests/config graphs are rarely pre-loaded; you can add them later if needed.

        return engine
```

> You can make `_cache` protected and provide a method on `NxGraphEngine` to seed it more cleanly; I’m keeping it simple here so you can see the wiring.

---

## 6. Step 4 – Refactor `analytics.graph_metrics` to use `GraphEngine`

We’ll stop importing `nx_views` and `GraphBundle` from analytics.

### 6.1: Update imports

In `analytics/graph_metrics.py`, replace:

```python
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.graphs.nx_views import load_call_graph, load_import_graph
from codeintel.analytics.graph_service import (
    GraphBundle,
    GraphContext,
    centrality_directed,
    component_metadata,
    neighbor_stats,
)
```

with:

```python
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.graphs.engine import GraphEngine
from codeintel.analytics.graph_service import (
    GraphContext,
    centrality_directed,
    component_metadata,
    neighbor_stats,
)
```

(We’ll keep `GraphContext` and the centrality helpers; we’re just removing `GraphBundle` and direct `nx_views` imports.)

### 6.2: Change `compute_graph_metrics` to build an engine

Current skeleton:

```python
def compute_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    runtime: GraphRuntimeOptions | None = None,
) -> None:
    runtime = runtime or GraphRuntimeOptions()
    context = runtime.context
    graph_ctx = runtime.graph_ctx
    ensure_schema(...)
    use_gpu = runtime.use_gpu
    ctx = graph_ctx or build_graph_context(cfg, now=datetime.now(tz=UTC), use_gpu=use_gpu)
    ...
    call_graph_cached = context.call_graph if context is not None else None
    import_graph_cached = context.import_graph if context is not None else None

    def _call_graph_loader() -> nx.DiGraph: ...
    def _import_graph_loader() -> nx.DiGraph: ...

    bundle: GraphBundle[nx.DiGraph] = GraphBundle(
        ctx=ctx,
        loaders={
            "call_graph": _call_graph_loader,
            "import_graph": _import_graph_loader,
        },
    )
    _compute_function_graph_metrics(..., ctx=ctx, bundle=bundle)
    _compute_module_graph_metrics(..., ctx=ctx, bundle=bundle, ...)
```

Refactor:

```python
from datetime import UTC, datetime
from codeintel.graphs.engine import GraphEngine
from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import build_graph_context, GraphContext, ...

def compute_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    runtime: GraphRuntimeOptions | None = None,
) -> None:
    runtime = runtime or GraphRuntimeOptions()
    context = runtime.context
    graph_ctx = runtime.graph_ctx

    con = gateway.con
    ensure_schema(con, "analytics.graph_metrics_functions")
    ensure_schema(con, "analytics.graph_metrics_modules")

    use_gpu = runtime.use_gpu
    ctx = graph_ctx or build_graph_context(
        cfg,
        now=datetime.now(tz=UTC),
        use_gpu=use_gpu,
    )
    if ctx.repo != cfg.repo or ctx.commit != cfg.commit or ctx.use_gpu != use_gpu:
        ctx = replace(ctx, repo=cfg.repo, commit=cfg.commit, use_gpu=use_gpu)

    # NEW: build engine via runtime helper
    engine: GraphEngine = runtime.build_engine(gateway, cfg.repo, cfg.commit)

    _compute_function_graph_metrics(
        gateway,
        cfg,
        ctx=ctx,
        engine=engine,
    )

    module_by_path = None
    if context is not None:
        module_by_path = context.module_map
    elif catalog_provider is not None:
        module_by_path = catalog_provider.catalog().module_by_path

    _compute_module_graph_metrics(
        gateway,
        cfg,
        ctx=ctx,
        engine=engine,
        module_by_path=module_by_path,
    )
```

### 6.3: Update internal helpers to take `GraphEngine` instead of `GraphBundle`

Change signatures:

```python
def _compute_function_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsConfig,
    *,
    ctx: GraphContext,
    engine: GraphEngine,
) -> None:
    con = gateway.con
    graph = engine.call_graph()
    stats = neighbor_stats(graph, weight=ctx.betweenness_weight)
    centrality = centrality_directed(graph, ctx)
    components = component_metadata(graph)
    created_at = ctx.resolved_now()
    ...
```

```python
def _compute_module_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsConfig,
    *,
    ctx: GraphContext,
    engine: GraphEngine,
    module_by_path: dict[str, str] | None,
) -> None:
    con = gateway.con
    graph = engine.import_graph()
    symbol_modules, symbol_inbound, symbol_outbound = _load_symbol_module_edges(
        gateway, module_by_path
    )
    ...
```

Remove all references to `GraphBundle` and the local `*_graph_loader` functions.

---

## 7. Step 5 – Refactor other analytics modules to use `GraphEngine`

### 7.1: `analytics/module_graph_metrics_ext.py`

Current pattern:

```python
from codeintel.graphs.nx_views import load_import_graph
from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import GraphContext, CentralityBundle, ...

def compute_module_graph_metrics_ext(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    *,
    runtime: GraphRuntimeOptions | None = None,
) -> None:
    runtime = runtime or GraphRuntimeOptions()
    context = runtime.context
    use_gpu = runtime.use_gpu

    import_graph_cached = context.import_graph if context is not None else None
    if import_graph_cached is not None:
        graph: nx.DiGraph = import_graph_cached
    else:
        graph = load_import_graph(gateway, repo, commit, use_gpu=use_gpu)

    # compute centrality, components, etc.
```

Refactor:

* Replace `load_import_graph` import with `GraphEngine`.

```python
from codeintel.graphs.engine import GraphEngine
```

* Use `runtime.build_engine`:

```python
def compute_module_graph_metrics_ext(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    *,
    runtime: GraphRuntimeOptions | None = None,
) -> None:
    runtime = runtime or GraphRuntimeOptions()
    engine: GraphEngine = runtime.build_engine(gateway, repo, commit)
    graph: nx.DiGraph = engine.import_graph()

    # rest of the function is unchanged
```

No more direct `nx_views` usage.

### 7.2: `analytics/symbol_graph_metrics.py`

Current:

```python
from codeintel.analytics.graph_service import GraphBundle, GraphContext, ...
from codeintel.graphs.nx_views import load_symbol_function_graph, load_symbol_module_graph

bundle: GraphBundle[nx.Graph] = GraphBundle(
    ctx=ctx,
    loaders={
        "symbol_function_graph": lambda: load_symbol_function_graph(...),
    },
)
graph = bundle.get("symbol_function_graph")
...
```

Refactor:

* Imports:

  ```python
  from codeintel.graphs.engine import GraphEngine
  from codeintel.analytics.graph_runtime import GraphRuntimeOptions
  ```

* Build engine and fetch graph:

  ```python
  def compute_symbol_graph_metrics(
      gateway: StorageGateway,
      repo: str,
      commit: str,
      *,
      runtime: GraphRuntimeOptions | None = None,
  ) -> None:
      runtime = runtime or GraphRuntimeOptions()
      use_gpu = runtime.use_gpu
      ctx = runtime.graph_ctx or GraphContext(
          repo=repo,
          commit=commit,
          use_gpu=use_gpu,
          ...
      )
      if ctx.use_gpu != use_gpu:
          ctx = replace(ctx, use_gpu=use_gpu)
      if runtime.context is not None and (
          runtime.context.repo != repo or runtime.context.commit != commit
      ):
          return

      engine: GraphEngine = runtime.build_engine(gateway, repo, commit)
      graph = engine.symbol_function_graph()
      
      if graph.number_of_nodes() == 0:
          ...
  ```

If you also need symbol-module graph somewhere in this module, call `engine.symbol_module_graph()`.

### 7.3: `analytics/config_graph_metrics.py`

Current:

```python
from codeintel.graphs.nx_views import load_config_module_bipartite

graph = load_config_module_bipartite(gateway, repo, commit, use_gpu=use_gpu)
...
```

Refactor:

```python
from codeintel.graphs.engine import GraphEngine
from codeintel.analytics.graph_runtime import GraphRuntimeOptions

def compute_config_graph_metrics(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    *,
    runtime: GraphRuntimeOptions | None = None,
) -> None:
    runtime = runtime or GraphRuntimeOptions()
    engine: GraphEngine = runtime.build_engine(gateway, repo, commit)
    graph = engine.config_module_bipartite()
    ...
```

### 7.4: `analytics/subsystem_graph_metrics.py`

Current:

```python
from codeintel.graphs.nx_views import load_import_graph

import_graph_cached = runtime.context.import_graph if runtime.context else None
graph = import_graph_cached or load_import_graph(gateway, repo, commit, use_gpu=runtime.use_gpu)
...
```

Refactor:

```python
from codeintel.graphs.engine import GraphEngine

def compute_subsystem_graph_metrics(..., runtime: GraphRuntimeOptions | None = None) -> None:
    runtime = runtime or GraphRuntimeOptions()
    engine: GraphEngine = runtime.build_engine(gateway, repo, commit)
    graph = engine.import_graph()
    ...
```

### 7.5: `analytics/tests/graph_metrics.py`

Current:

```python
from codeintel.graphs.nx_views import load_test_function_bipartite

graph: nx.Graph = load_test_function_bipartite(gateway, repo, commit, use_gpu=use_gpu)
```

Refactor similarly:

```python
from codeintel.graphs.engine import GraphEngine
from codeintel.analytics.graph_runtime import GraphRuntimeOptions

def compute_test_graph_metrics(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    *,
    runtime: GraphRuntimeOptions | None = None,
) -> None:
    runtime = runtime or GraphRuntimeOptions()
    engine: GraphEngine = runtime.build_engine(gateway, repo, commit)
    graph: nx.Graph = engine.test_function_bipartite()
    ...
```

---

## 8. Optional – Use `GraphEngine` in `AnalyticsContext` and CLI

You don’t *have to* touch these now, but medium-term it’s nice.

### 8.1: `analytics/context.py`

This module preloads graphs via `nx_views.load_*` and stores them on `AnalyticsContext`.

Optional refactor:

* Either:

  * Keep it as-is and let `GraphRuntimeOptions.build_engine` seed the engine cache from `AnalyticsContext`.
* Or:

  * Have `AnalyticsContext` **own** a `GraphEngine` and fill its cache eagerly.

For now, your `build_engine` seeding approach (section 5.1) is enough.

### 8.2: CLI (nx backend)

In `cli/nx_backend.py` (not in the zips but in your repo), you can:

* Replace direct calls to `graphs.nx_views` with `NxGraphEngine`.
* Provide CLI flags to choose backend later (e.g., GPU vs CPU engine).

---

## 9. Tests & guardrails

### 9.1: GraphEngine correctness tests

Create `tests/graphs/test_engine_nx.py`:

```python
from codeintel.graphs.engine import NxGraphEngine, GraphKind
from codeintel.graphs import nx_views

def test_nx_engine_matches_nx_views_for_call_graph(gateway, repo, commit) -> None:
    engine = NxGraphEngine(gateway=gateway, repo=repo, commit=commit, use_gpu=False)

    direct = nx_views.load_call_graph(gateway, repo, commit, use_gpu=False)
    via_engine = engine.call_graph()

    assert set(direct.nodes) == set(via_engine.nodes)
    assert set(direct.edges) == set(via_engine.edges)
```

Similar tests for import graph and symbol graphs; you don’t need full metric equality, just structural equality.

### 9.2: Analytics regression tests

You likely already have tests under `tests/analytics/test_graph_metrics.py` (and friends). After refactor, rerun them to ensure metrics haven’t changed.

Optionally, add explicit tests for:

* `compute_graph_metrics` with `runtime.context` set (cached graphs used).
* `compute_graph_metrics` with `runtime.context=None` (engine builds graphs itself).

### 9.3: Architecture test: no direct `nx_views` imports in analytics

Add `tests/architecture/test_analytics_graph_boundary.py`:

```python
from pathlib import Path

def test_analytics_does_not_import_nx_views() -> None:
    root = Path("src/codeintel/analytics")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "codeintel.graphs.nx_views" not in text, f"{path} should use GraphEngine instead"
```

If you want to allow `analytics.context.py` or some other specific file to still use `nx_views` for bootstrapping, just special-case that path in the test.

---

