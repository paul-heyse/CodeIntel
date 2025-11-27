#Epic B — Converge on a single graph runtime abstraction#

Nice, glad Epic A landed cleanly. Let’s now give graphs the same treatment and make the graph runtime something you can “just plug in” everywhere.

I’ll structure this similarly:

1. **Target end-state** (what the graph layer looks like when you’re done)
2. **Module layout & responsibilities**
3. **Step-by-step refactor plan**
4. **Call-site migration map**
5. **Testing & rollout**

I’ll keep it aligned with the Epic A changes (StorageGateway + repositories, GraphRepository, etc.).

---

## 1. Target end-state

Conceptually, the “end state” of Epic B is:

* There is **one canonical abstraction** for graph access:

  * `GraphRuntimeOptions` – *configuration* for how to build/use graphs.
  * `GraphRuntime` – *live runtime* with a `GraphEngine` and caches.

* Only **one place in the code is allowed to build a `GraphEngine`**:

  * `ensure_graph_runtime(...)` in `pipeline/orchestration/core.py` (or a very thin wrapper around it).

* Every consumer that needs graphs gets one of:

  * a `GraphRuntime` (for analytics, validation, pipeline steps), or
  * a `GraphEngine` injected into it (e.g. serving `DuckDBQueryService`).

* `analytics/graph_service.py` is de-god-moduled:

  * `graph_service_runtime.py` (wiring + caching)
  * `graph_metrics_*.py` (pure “graph → metrics” functions)
  * `graph_rows_*.py` (translate metrics into row dicts / models)

* The **relational enrichment** part (mapping GOIDs to function metadata, subsystems, tests, etc.) goes through **repositories** from Epic A (esp. `GraphRepository`, `FunctionRepository`, `SubsystemRepository`), not raw SQL.

So the flow becomes:

> config/env → `GraphBackendConfig` → `GraphRuntimeOptions` → `ensure_graph_runtime()` → `GraphRuntime(engine + caches)` → analytics / serving / CLI / validation

No more rogue calls to `build_graph_engine`.

---

## 2. Module layout & responsibilities

### Graphs package

**Today (approx):**

```text
graphs/
    __init__.py
    engine.py          # GraphEngine, NxGraphEngine
    engine_factory.py  # build_graph_engine(...)
    nx_backend.py      # Nx-specific helpers
    cfg_builder.py
    callgraph_builder.py
    importgraph_builder.py
    symbol_uses.py
    nx_views.py
    validation.py
```

**Target:**

```text
graphs/
    __init__.py        # re-exports GraphEngine, GraphKind, etc.

    engine.py          # GraphEngine protocol + NxGraphEngine impl
    engine_factory.py  # build_graph_engine(backend_config, storage, snapshot)
    nx_backend.py      # Nx-specific glue

    cfg_builder.py
    callgraph_builder.py
    importgraph_builder.py
    symbol_uses.py
    nx_views.py

    validation.py      # small validation functions that accept GraphRuntime or GraphEngine
```

The big change is **not new files here**, but **who is allowed to call `build_graph_engine`** (only `ensure_graph_runtime`).

---

### Analytics graph runtime + services

**Today (approx):**

```text
analytics/
    context.py         # AnalyticsContext (storage, config, graph_engine?, etc.)
    graph_runtime.py   # GraphRuntimeOptions, maybe small helper(s)
    graph_service.py   # giant: builds graphs, calls algorithms, writes analytics tables
    ...
```

**Target:**

```text
analytics/
    context.py
    graph_runtime.py        # GraphRuntimeOptions + GraphRuntime dataclasses + helpers
    graph_service_runtime.py# GraphRuntimeService: wiring, caching, storage integration

    graph_metrics/
        __init__.py
        centrality.py       # degree, betweenness, pagerank, etc.
        structure.py        # SCCs, biconnected comps, articulation points, etc.
        paths.py            # shortest paths, distances, radii, eccentricity
        communities.py      # (if you do Louvain/Infomap/SArF here)

    graph_rows/
        __init__.py
        hotspots.py         # build goid_hotspots rows from metrics + repos
        subsystems.py       # subsystem graph / boundary rows
        tests.py            # test ↔ graph coverage overlays
        graph_stats.py      # generic graph metrics → analytics.graph_stats rows
```

You don’t have to create **all** of these submodules on day one – but the structure makes it obvious where things go, and makes `graph_service.py` shrink to something like:

```python
# graph_service_runtime.py
def compute_hotspot_metrics(runtime: GraphRuntime, repos: RepositoriesBundle) -> list[GraphStatsRow]: ...
```

---

### Pipeline & orchestration

**Today (approx):**

```text
pipeline/orchestration/core.py
    # PipelineContext, StepRegistry, ensure_graph_engine, ensure_graph_runtime

cli/main.py
    # some commands call build_graph_engine() directly

serving/mcp/query_service.py
    # constructs NxGraphEngine directly for callgraph-related tooling
```

**Target:**

* `pipeline/orchestration/core.py` is the **only** module that calls `build_graph_engine`.

* CLI commands that need graphs:

  * use `PipelineContext` → `ensure_graph_runtime(ctx)` → `ctx.graph_runtime`
  * or use a small `build_graph_runtime_for_snapshot(snapshot_ref, backend_config)` helper that delegates to `ensure_graph_runtime` semantics.

* `DuckDBQueryService` gets either:

  * a `GraphEngine` injected (so it never constructs one), or
  * a lightweight wrapper `GraphRuntimeAdapter` with just the methods it needs.

---

## 3. Step-by-step refactor plan

### Step 0 — Inventory & guardrails

1. **Search for current entrypoints** for building graphs:

   * `build_graph_engine` in `graphs/engine_factory.py`
   * `NxGraphEngine(...)` direct constructors
   * `ensure_graph_engine` in `pipeline/orchestration/core.py`

   Expect usages in:

   * `pipeline/orchestration/core.py`
   * `cli/main.py`
   * `graphs/validation.py`
   * `serving/mcp/query_service.py`
   * possibly some ad-hoc analytics/debug scripts.

2. Add **smoke tests** (or enhance existing ones) that validate:

   * graph-dependent analytics steps (e.g. `graph_stats`, `subsystem detection`) produce same row counts / key metrics before/after.
   * `DuckDBQueryService` graph-based endpoints (callgraph neighborhood, import boundary) behave as expected.

These will be your “no behavioural change” tripwires.

---

### Step 1 — Stabilize `GraphRuntimeOptions` and define `GraphRuntime`

You already have `analytics/graph_runtime.py` with `GraphRuntimeOptions`. We make it the canonical configuration object and add a first-class `GraphRuntime`.

#### 1.1 Define GraphKind / GraphFlags (if not already)

In `analytics/graph_runtime.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any

from codeintel.config.primitives import GraphBackendConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.engine import GraphEngine

class GraphKind(Flag):
    NONE = 0
    CALL = auto()
    IMPORT = auto()
    CFG = auto()
    SYMBOL = auto()
    ALL = CALL | IMPORT | CFG | SYMBOL
```

If you already have an equivalent, re-use it; don’t duplicate.

#### 1.2 Lock down `GraphRuntimeOptions`

Make it **explicit** about what it configures:

```python
@dataclass(frozen=True)
class GraphRuntimeOptions:
    snapshot: SnapshotRef                 # repo + commit + root path
    backend: GraphBackendConfig            # Nx vs GPU, on-disk vs in-memory, etc.
    graphs: GraphKind = GraphKind.ALL     # which graphs to materialize / cache
    eager: bool = False                   # whether to build on construction
    validate: bool = False                # run validation.y checks
    cache_key: str | None = None          # optional key for cross-step caching
```

If you already have fields like this, reconcile them into something close to this shape.

#### 1.3 Introduce `GraphRuntime`

Also in `graph_runtime.py`:

```python
@dataclass
class GraphRuntime:
    options: GraphRuntimeOptions
    engine: GraphEngine

    # optional caches – types can be networkx.DiGraph / Graph etc.
    call_graph: Any | None = None
    import_graph: Any | None = None
    cfg_graph: Any | None = None
    symbol_graph: Any | None = None

    def ensure_call_graph(self) -> Any:
        if self.call_graph is None:
            self.call_graph = self.engine.load_call_graph(self.options.snapshot)
        return self.call_graph

    def ensure_import_graph(self) -> Any:
        ...
```

The important bit:

* **All graph building is delegated to `GraphEngine`**.
* `GraphRuntime` only caches / multiplexes.

Later, you can make `engine.load_*` consult `GraphBackendConfig` (e.g., load from DuckDB / Parquet / in-memory).

---

### Step 2 — Centralize engine construction in `ensure_graph_runtime`

In `pipeline/orchestration/core.py` you already have something like:

```python
def ensure_graph_engine(ctx: PipelineContext) -> GraphEngine: ...
def ensure_graph_runtime(ctx: PipelineContext) -> GraphRuntimeOptions | GraphRuntime: ...
```

We make this the only place that calls `build_graph_engine`.

#### 2.1 Create a thin `build_graph_runtime` helper

In `analytics/graph_runtime.py` (or `pipeline/orchestration/core.py` if you prefer), define:

```python
from codeintel.graphs.engine_factory import build_graph_engine
from codeintel.config.primitives import GraphBackendConfig

def build_graph_runtime(
    snapshot: SnapshotRef,
    backend: GraphBackendConfig,
    *,
    graphs: GraphKind = GraphKind.ALL,
    eager: bool = False,
    validate: bool = False,
) -> GraphRuntime:
    options = GraphRuntimeOptions(
        snapshot=snapshot,
        backend=backend,
        graphs=graphs,
        eager=eager,
        validate=validate,
    )
    engine = build_graph_engine(backend, snapshot=snapshot)
    runtime = GraphRuntime(options=options, engine=engine)

    if eager:
        if graphs & GraphKind.CALL:
            runtime.ensure_call_graph()
        if graphs & GraphKind.IMPORT:
            runtime.ensure_import_graph()
        if graphs & GraphKind.CFG:
            runtime.ensure_cfg_graph()
        if graphs & GraphKind.SYMBOL:
            runtime.ensure_symbol_graph()
    return runtime
```

**Key rule:** *`build_graph_engine` must not be called anywhere else* after this step.

#### 2.2 Wire `ensure_graph_runtime(ctx)` through `build_graph_runtime`

In `pipeline/orchestration/core.py`:

* `PipelineContext` should already carry:

  * `snapshot_ref: SnapshotRef`
  * `graph_backend: GraphBackendConfig`
  * maybe `graph_runtime: GraphRuntime | None`

Implement:

```python
from codeintel.analytics.graph_runtime import (
    build_graph_runtime,
    GraphRuntime,
    GraphKind,
)

def ensure_graph_runtime(ctx: PipelineContext) -> GraphRuntime:
    if ctx.graph_runtime is not None:
        return ctx.graph_runtime

    runtime = build_graph_runtime(
        snapshot=ctx.snapshot_ref,
        backend=ctx.graph_backend,
        graphs=GraphKind.ALL,
        eager=False,
        validate=False,  # or driven by config
    )
    ctx.graph_runtime = runtime
    return runtime
```

Also make `ensure_graph_engine` a thin wrapper (or deprecate it):

```python
def ensure_graph_engine(ctx: PipelineContext) -> GraphEngine:
    return ensure_graph_runtime(ctx).engine
```

Now **every pipeline step** that needs graphs should accept a `GraphRuntime` (or derive `GraphEngine` from it), not call `build_graph_engine` directly.

---

### Step 3 — Move consumers onto `GraphRuntime` instead of building engines

#### 3.1 Analytics context

In `analytics/context.py`, you likely have something like:

```python
@dataclass
class AnalyticsContext:
    gateway: StorageGateway
    config: AnalyticsConfig
    graph_engine: GraphEngine | None = None
    ...
```

Refactor to:

```python
from codeintel.analytics.graph_runtime import GraphRuntime

@dataclass
class AnalyticsContext:
    gateway: StorageGateway
    config: AnalyticsConfig
    graph_runtime: GraphRuntime | None = None

    @property
    def graph_engine(self) -> GraphEngine:
        if self.graph_runtime is None:
            raise RuntimeError("graph_runtime not initialized")
        return self.graph_runtime.engine
```

And in whatever code currently populates `AnalyticsContext` (likely pipeline steps / CLI), use `ensure_graph_runtime(ctx)` and set `analytics_ctx.graph_runtime`.

#### 3.2 CLI commands

In `cli/main.py`, search for any direct uses of:

* `build_graph_engine`
* `NxGraphEngine(...)`
* manual backend selection / env detection.

For each graph-aware CLI command:

1. Make sure it constructs or can access a `PipelineContext` (if it doesn’t yet, add a small helper to do so).

2. Replace any direct engine creation with:

   ```python
   from codeintel.pipeline.orchestration.core import ensure_graph_runtime

   runtime = ensure_graph_runtime(pipeline_ctx)
   engine = runtime.engine
   ```

3. If the CLI only needs a specific graph (e.g. debugging call graph), call `runtime.ensure_call_graph()` rather than reloading from storage.

This reuses caching and backend configuration consistently.

#### 3.3 Validation code (`graphs/validation.py`)

If `graphs/validation.py` currently accepts a `GraphEngine` or calls `build_graph_engine` themselves, normalize to:

* The **public** functions accept `GraphRuntime`:

  ```python
  def validate_call_graph(runtime: GraphRuntime) -> list[GraphValidationIssue]:
      G = runtime.ensure_call_graph()
      ...
  ```

* For internal unit tests where you want to bypass the runtime, you can still call `build_graph_runtime` in the test code — but not in production code.

Pipeline steps that run validation should call:

```python
runtime = ensure_graph_runtime(ctx)
issues = validate_call_graph(runtime)
```

---

### Step 4 — Refactor `DuckDBQueryService` & serving to use `GraphRuntime`

You’ve already done a lot of work in Epic A to make `DuckDBQueryService` use repositories. Now we “plug in” graph runtime instead of building its own engine.

#### 4.1 Update `DuckDBQueryService` constructor

Today it likely looks like:

```python
@dataclass
class DuckDBQueryService:
    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    engine: NxGraphEngine | None = None
```

Change it to **depend on `GraphEngine` only, injected by caller**:

```python
from codeintel.graphs.engine import GraphEngine

@dataclass
class DuckDBQueryService:
    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    graph_engine: GraphEngine | None = None
```

(You can also accept a `GraphRuntime` instead, but in practice the query service just needs methods like “get callgraph neighborhood”; the full runtime is overkill.)

#### 4.2 Use `GraphEngine` methods instead of building graph anew

Where you currently do something like:

```python
engine = self.engine or build_graph_engine(...)
callgraph = engine.load_call_graph(...)
neighbors = callgraph.neighbors(node)
```

Refactor:

* Assume `graph_engine` is already set (enforce via error if missing):

  ```python
  def _require_graph_engine(self) -> GraphEngine:
      if self.graph_engine is None:
          raise ProblemError(
              ProblemDetail.from_internal(
                  "graph-engine-not-configured",
                  "Graph engine was not provided for this query service.",
              )
          )
      return self.graph_engine
  ```

* Then:

  ```python
  def get_callgraph_neighbors(...):
      engine = self._require_graph_engine()
      callgraph = engine.load_call_graph(snapshot=self._snapshot_ref)
      # or, if you want caching, pass in a GraphRuntime instead and call runtime.ensure_call_graph()
      ...
  ```

For extra consistency, you can introduce a tiny adapter:

```python
@dataclass
class GraphQueryContext:
    runtime: GraphRuntime

    @property
    def engine(self) -> GraphEngine:
        return self.runtime.engine

    def call_graph(self) -> Any:
        return self.runtime.ensure_call_graph()
```

…and inject that into `DuckDBQueryService` instead, but it’s optional.

#### 4.3 Where does serving get the graph engine from?

In your serving wiring (likely `serving/services/wiring.py` or `serving/http/fastapi.py`):

* You already fetch a `StorageGateway` + `ServingConfig`.

Extend this wiring to:

1. Build a minimal **serving-time pipeline context**:

   ```python
   snapshot = SnapshotRef(repo=config.repo, commit=config.commit, root=config.repo_root)
   backend = config.graph_backend  # from ServingConfig/GraphBackendConfig

   runtime = build_graph_runtime(
       snapshot=snapshot,
       backend=backend,
       graphs=GraphKind.CALL | GraphKind.IMPORT,   # only what serving needs
       eager=False,
       validate=False,
   )
   ```

2. Create `DuckDBQueryService` with:

   ```python
   query_service = DuckDBQueryService(
       gateway=gateway,
       repo=config.repo,
       commit=config.commit,
       limits=config.backend_limits,
       graph_engine=runtime.engine,
   )
   ```

3. Optionally keep `runtime` in the DI container/context so **other** services (e.g. HTTP debugging endpoints) can use graphs too.

Note: this means **serving now depends on the same graph backend config as pipeline** (`GraphBackendConfig`), which is what we want for consistency.

---

### Step 5 — Refactor `analytics/graph_service.py` into runtime + metrics

This is the god-module part.

#### 5.1 Introduce `graph_service_runtime.py`

Create `analytics/graph_service_runtime.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from codeintel.analytics.graph_runtime import GraphRuntime
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.subsystems import SubsystemRepository
from codeintel.storage.repositories.graphs import GraphRepository

@dataclass
class GraphServiceRuntime:
    runtime: GraphRuntime
    functions: FunctionRepository
    subsystems: SubsystemRepository
    graphs: GraphRepository

    def compute_graph_stats(self) -> list[GraphStatsRow]:
        G = self.runtime.ensure_call_graph()
        # delegate to metrics/rows modules
        from codeintel.analytics.graph_metrics.centrality import compute_centrality_metrics
        from codeintel.analytics.graph_rows.graph_stats import build_graph_stats_rows

        metrics = compute_centrality_metrics(G)
        return build_graph_stats_rows(metrics, self.functions, self.subsystems)
```

`GraphServiceRuntime` should:

* never embed raw SQL – use repos from Epic A.
* handle **wiring**: “given runtime + repos, call metrics + row builders”.

#### 5.2 Extract pure metrics into `analytics/graph_metrics/*`

From `graph_service.py`, locate sections like:

* computing degree / betweenness / pagerank
* computing SCCs, biconnected components
* computing “hotspot” scores from combinations of metrics

Move them into pure functions that:

* **take a Graph** (nx.DiGraph / GraphEngine),
* **return typed metrics objects** or dictionaries.

Example (`centrality.py`):

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

@dataclass
class CentralityMetrics:
    degree: dict[Any, float]
    betweenness: dict[Any, float]
    pagerank: dict[Any, float]

def compute_centrality_metrics(G: nx.DiGraph) -> CentralityMetrics:
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    return CentralityMetrics(
        degree=degree,
        betweenness=betweenness,
        pagerank=pagerank,
    )
```

Similar for:

* `structure.py` – SCCs, articulation points, etc.
* `paths.py` – shortest path lengths, distances.
* `communities.py` – community detection if present.

These functions **should not** know about GOIDs, storage, or repos; only graph topology.

#### 5.3 Extract row builders into `analytics/graph_rows/*`

After metrics, `graph_service.py` probably had code like:

* join metrics with `goid_metadata` / `function_profile` using GOID keys.
* compute normalized scores and buckets.
* emit `analytics.graph_stats` row dicts.

Move that to `graph_rows`:

```python
# analytics/graph_rows/graph_stats.py
from __future__ import annotations

from collections.abc import Iterable

from codeintel.analytics.graph_metrics.centrality import CentralityMetrics
from codeintel.storage.repositories.functions import FunctionRepository

@dataclass
class GraphStatsRow:
    function_goid_h128: int
    degree: float
    pagerank: float
    betweenness: float
    # maybe extra normalized scores

def build_graph_stats_rows(
    metrics: CentralityMetrics,
    functions: FunctionRepository,
) -> list[GraphStatsRow]:
    rows: list[GraphStatsRow] = []
    # use functions.repository to map node -> GOID + metadata if needed
    ...
    return rows
```

Now the big `graph_service.py` can be largely replaced by `GraphServiceRuntime` orchestrating:

* `runtime.ensure_call_graph()`
* `compute_*` metrics
* `build_*_rows`, then writing rows using your **ingest helpers** (`macro_insert_rows`) and `StorageGateway`.

---

### Step 6 — Hook graph runtime into pipeline steps

Now that `GraphRuntime` and `GraphServiceRuntime` exist, we make pipeline steps use them.

In `pipeline/orchestration/steps_analytics.py` (or whichever file handles graph analytics steps):

1. Ensure the step function signature receives `PipelineContext`:

   ```python
   def step_graph_stats(ctx: PipelineContext) -> None:
       runtime = ensure_graph_runtime(ctx)

       repos = RepositoriesBundle.from_gateway(
           ctx.gateway, repo=ctx.snapshot_ref.repo, commit=ctx.snapshot_ref.commit
       )
       graph_service = GraphServiceRuntime(
           runtime=runtime,
           functions=repos.functions,
           subsystems=repos.subsystems,
           graphs=repos.graphs,
       )

       rows = graph_service.compute_graph_stats()
       # use StorageGateway / ingest_helpers.macro_insert_rows to write into analytics.graph_stats
   ```

2. Remove any local calls to `build_graph_engine` or direct `NxGraphEngine` construction.

3. If a step only needs the engine and not the full runtime, call `ensure_graph_runtime(ctx).engine`.

This way, **every step uses the same engine instance**, caching, and backend config.

---

### Step 7 — Tests & rollout

#### 7.1 Unit tests for metrics & row builders

* For `graph_metrics/*`:

  * Construct a tiny `networkx.DiGraph` in-memory (no storage required).
  * Assert centrality / structure metrics match expected values.

* For `graph_rows/*`:

  * Use fixture `FunctionRepository` / `GraphRepository` with an in-memory DuckDB and tiny test tables (`call_graph_edges`, `function_profile`).
  * Assert row models tie metrics to the right GOIDs and fields.

#### 7.2 Integration tests for `GraphServiceRuntime`

* Build a real snapshot DB for a tiny test repo (you likely already have this).
* Create `GraphRuntime` via `build_graph_runtime`.
* Construct actual repos from Epic A (`FunctionRepository`, `SubsystemRepository`, `GraphRepository`).
* Run the `GraphServiceRuntime` methods and:

  * assert row counts,
  * assert some key fields have expected ranges (e.g. degree > 0, pagerank ≈ 1.0 sum).

#### 7.3 End-to-end smoke tests

* Run the full pipeline (ingestion + graphs + analytics).
* Run CLI commands that depend on graphs (e.g., graph stats dump).
* Hit serving endpoints / MCP tools for:

  * callgraph neighbors,
  * import boundary,
  * any “graph neighborhood” introspection.

Check that:

* No code path calls `build_graph_engine` directly except `build_graph_runtime`.
* Graph-dependent behaviour hasn’t regressed (within expected variance if algorithms changed slightly).

---

## 4. Call-site migration cheat-sheet

Here’s the “who should now call what” table for your LLM agent:

| Old pattern / location                                 | New pattern                                                                         |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `build_graph_engine(backend, snapshot)` everywhere     | `build_graph_runtime(snapshot, backend)` **only** in orchestration / serving wiring |
| `ensure_graph_engine(ctx)` in analytics steps          | `ensure_graph_runtime(ctx).engine` (or just `ensure_graph_runtime(ctx)`)            |
| `NxGraphEngine(...` direct constructor                 | Never in app code; only via `build_graph_engine` inside `build_graph_runtime`       |
| CLI commands building engines directly                 | Build `PipelineContext` → `ensure_graph_runtime(ctx)`                               |
| `graphs.validation.validate_*` accepting engine        | `validate_*(runtime: GraphRuntime)`                                                 |
| `DuckDBQueryService` building engine internally        | Accept `graph_engine: GraphEngine` (or `GraphRuntime`) injected by wiring           |
| `analytics.graph_service` doing everything in one file | `GraphServiceRuntime` + `graph_metrics.*` + `graph_rows.*`                          |

---


