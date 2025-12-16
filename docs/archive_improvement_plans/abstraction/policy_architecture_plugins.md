
# high level architecture overview #

Awesome, I’m glad this is landing for you. Let’s make that chain very concrete.

I’ll stay in **design/spec space** (not full code), but I’ll be explicit about:

* What each plugin **requires** and **guarantees**.
* What its **PluginMetadata** should say.
* What its **PluginResult** should contain.

We’ll do this for the six plugins in the chain:

1. `ingest.repo_scan`
2. `ingest.scip_python`
3. `graphs.goid_builder`
4. `graphs.callgraph_builder`
5. `analytics.function_metrics`
6. `analytics.function_hotspots`

I’ll assume the unified plugin types:

* `PluginMetadata` with fields like:

  * `name`, `kind`, `stage`, `version`
  * `description`
  * `provides: set[PluginCapability]`
  * `requires: set[PluginCapability]`
  * `produces_tables: set[str]`
  * `consumes_tables: set[str]`
  * `supports_incremental: bool`
  * `options_model: type | None`
  * `resource_hints: dict[str, Any]`
* `PluginResult` with fields like:

  * `status` (`SUCCESS | SKIPPED | FAILED | PARTIAL`)
  * `row_counts: dict[str, int]`
  * `outputs: dict[str, Any]` (optional domain objects / summaries)
  * `metrics: dict[str, float | int]`
  * `warnings: list[str]`
  * `timing: ExecutionTiming` (wall/CPU)
  * `extra: dict[str, Any]`

---

## 1. `ingest.repo_scan` – discover modules

### 1.1 Functional role

**Preconditions**

* A `Repository` + `commit` exist (from `RunContext`).
* Underlying source is accessible (filesystem or git checkout).

**Postconditions / guarantees**

* You have a complete set of **modules** for this repo/commit:

  * Each module has:

    * `repo`, `commit`, `rel_path`, `language`, `is_test`, etc.
* These modules are persisted to a table like `ingest.modules`.
* A capability like `"ingest.modules"` is now satisfied for this repo/commit.

### 1.2 `PluginMetadata` for `ingest.repo_scan`

Conceptually, you’d define:

* `name`: `"ingest.repo_scan"`

* `kind`: `INGEST` (or `"ingest"`)

* `stage`: `"ingest"`

* `version`: implementation version/hash

* `description`: e.g. `"Scan repository for source modules for ingestion."`

* `provides`:

  * `"ingest.modules"`

* `requires`:

  * *empty* (root plugin)

* `produces_tables`:

  * `"ingest.modules"`

* `consumes_tables`:

  * optionally `"ingest.modules"` for incremental diff, but not required for correctness

* `supports_incremental`: `True`

* `options_model` (optional, but useful):

  * `RepoScanOptions` with fields like:

    * `include_globs: list[str]`
    * `exclude_globs: list[str]`
    * `languages: list[str] | None`

* `resource_hints`:

  * `{"expected_cost": "low", "io_intensive": True}`

### 1.3 `PluginResult` for `ingest.repo_scan`

When `execute(IngestExecutionContext)` returns, the `PluginResult` should contain:

* `status`: usually `SUCCESS`, `FAILED` or `SKIPPED`
* `row_counts`:

  * `{"ingest.modules": <int modules_written>}`
* `outputs`:

  * Optionally:

    * `"modules": list[CodeModule]` (domain objects) if you want to reuse them in-memory.
* `metrics`:

  * `"modules_total": <int>`
  * `"modules_changed": <int>` (for incremental)
* `warnings`:

  * e.g. `"ignored 15 files due to unsupported language"`, etc.
* `timing`:

  * `wall_time_ms`, `cpu_time_ms`, etc.
* `extra`:

  * `"scan_mode": "full" | "incremental"`
  * `"root_paths": [...]`

---

## 2. `ingest.scip_python` – SCIP indexing

### 2.1 Functional role

**Preconditions**

* Capability `"ingest.modules"` is available (from `ingest.repo_scan`).
* A Python SCIP indexer (`scip-python`) is available in the environment.

**Postconditions / guarantees**

* A SCIP index exists for Python modules in this repo/commit.
* Symbol information is materialized into tables like:

  * `ingest.scip_index`
  * `core.symbols` (or similar)
* Parsed module/function info is available to graphs & analytics (either persisted or reconstructible).

### 2.2 `PluginMetadata` for `ingest.scip_python`

* `name`: `"ingest.scip_python"`

* `kind`: `INGEST`

* `stage`: `"ingest"`

* `version`: version/hash of integration with `scip-python`

* `description`: `"Run scip-python to index Python code and persist symbol/index data."`

* `provides`:

  * `"ingest.scip_index"`
  * `"core.symbols"`
  * optionally `"core.parsed_code"`

* `requires`:

  * `"ingest.modules"` (needs the list of modules to index)

* `produces_tables`:

  * `"ingest.scip_index"`
  * `"core.symbols"`

* `consumes_tables`:

  * `"ingest.modules"`

* `supports_incremental`: `True`

* `options_model`:

  * `ScipIngestOptions` with fields such as:

    * `index_all: bool`
    * `extra_args: list[str]`
    * `max_workers: int`

* `resource_hints`:

  * `{"expected_cost": "high", "cpu_intensive": True, "io_intensive": True}`

### 2.3 `PluginResult` for `ingest.scip_python`

* `status`
* `row_counts`:

  * `{"ingest.scip_index": N_index_rows, "core.symbols": N_symbols}`
* `outputs`:

  * Could be:

    * `"indexed_modules": list[CodeModule]` (successfully indexed)
    * `"failed_modules": list[CodeModule]` (if any)
* `metrics`:

  * `"modules_indexed": <int>`
  * `"index_size_bytes": <int>` (if available)
* `warnings`:

  * any non-fatal tool errors or partial failures
* `timing`:

  * wall/CPU
* `extra`:

  * `"tool_version": "scip-python X.Y.Z"`
  * `"invocations": <count or structured info>`

---

## 3. `graphs.goid_builder` – assign function GOIDs

### 3.1 Functional role

**Preconditions**

* `"ingest.scip_index"` and/or `"core.symbols"` capability is available.

**Postconditions / guarantees**

* Every function has a stable **GOID** and associated metadata.
* Tables:

  * `core.goids` – one row per function.
  * `core.goid_crosswalk` – mapping between GOIDs and changing identities (paths, commits, etc).

### 3.2 `PluginMetadata` for `graphs.goid_builder`

* `name`: `"graphs.goid_builder"`

* `kind`: `GRAPH` (or `BUILDER` with `stage="goid"`)

* `stage`: `"goid"`

* `version`: GOID algorithm version/hash

* `description`: `"Assign stable GOIDs to functions using symbol information."`

* `provides`:

  * `"core.goids"`
  * `"core.goid_crosswalk"`

* `requires`:

  * `"ingest.scip_index"` *or* `"core.symbols"` (depending on implementation)

* `produces_tables`:

  * `"core.goids"`
  * `"core.goid_crosswalk"`

* `consumes_tables`:

  * `"core.symbols"` (likely),
  * `ingest.scip_index` for context.

* `supports_incremental`: `True`

* `options_model`:

  * optional `GoidOptions`, e.g.:

    * `hash_algorithm: Literal["xxh128", "sha256"]`
    * `include_signature: bool`

* `resource_hints`:

  * `{"expected_cost": "medium", "cpu_intensive": True}`

### 3.3 `PluginResult` for `graphs.goid_builder`

* `status`
* `row_counts`:

  * `{"core.goids": N_functions, "core.goid_crosswalk": N_crosswalk}`
* `outputs` (optional):

  * `"functions": list[Function]` (domain objects with GOIDs)
* `metrics`:

  * `"functions_total": <int>`
  * `"functions_new": <int>` – newly assigned GOIDs
  * `"functions_updated": <int>` – changed metadata for existing GOIDs
* `warnings`:

  * e.g., functions that could not be assigned GOIDs
* `timing`
* `extra`:

  * `"goid_version": <hash or semantic version>`

---

## 4. `graphs.callgraph_builder` – build call graph

### 4.1 Functional role

**Preconditions**

* `"core.goids"` capability is available (functions exist with GOIDs).
* There is a way to compute call edges:

  * via SCIP index (`ingest.scip_index`) and symbol data, and/or
  * via parsed code.

**Postconditions / guarantees**

* Callgraph nodes and edges are built for the requested scope.
* Tables:

  * `graph.call_graph_nodes` – one node per function in scope.
  * `graph.call_graph_edges` – one row per call edge (caller → callee).

### 4.2 `PluginMetadata` for `graphs.callgraph_builder`

* `name`: `"graphs.callgraph_builder"`

* `kind`: `GRAPH`

* `stage`: `"graph"`

* `version`: callgraph algorithm version

* `description`: `"Build call graph edges between functions."`

* `provides`:

  * `"graph.callgraph"`

* `requires`:

  * `"core.goids"`
  * plus one of:

    * `"ingest.scip_index"`, `"core.symbols"`, `"core.parsed_code"`

* `produces_tables`:

  * `"graph.call_graph_nodes"`
  * `"graph.call_graph_edges"`

* `consumes_tables`:

  * `"core.goids"`
  * `"ingest.scip_index"`/`"core.symbols"` as needed

* `supports_incremental`: `True`

* `options_model`:

  * `CallGraphOptions`, e.g.:

    * `include_external_calls: bool`
    * `scope_paths: list[str] | None`
    * `max_depth: int | None`

* `resource_hints`:

  * `{"expected_cost": "medium-high", "cpu_intensive": True}`

* (If you keep graph-specific metadata:

  * `produces_graph_kinds = {"callgraph"}`
  * `requires_graph_kinds = set()`)

### 4.3 `PluginResult` for `graphs.callgraph_builder`

* `status`
* `row_counts`:

  * `{"graph.call_graph_nodes": N_nodes, "graph.call_graph_edges": N_edges}`
* `outputs` (optional):

  * `"callgraph": CallGraph` domain object (might be too big for huge repos, so you may only include summaries).
* `metrics`:

  * `"nodes": <int>`
  * `"edges": <int>`
  * `"avg_out_degree": <float>`
* `warnings`:

  * e.g. truncated results due to scope or resource limits
* `timing`
* `extra`:

  * `"scope_paths": [...]`
  * `"algorithm": "scip-based" | "hybrid"`

---

## 5. `analytics.function_metrics` – compute per-function metrics

This is where `FunctionAnalyticsOptions` comes in.

### 5.1 Functional role

**Preconditions**

* `"core.goids"` is available (functions).
* `"graph.callgraph"` is available (for call-based metrics).
* Optional capabilities if metrics use them:

  * `"history.git_churn"`
  * `"coverage.function"`
  * `"core.parsed_code"` (if computing AST-based metrics on demand).

**Postconditions / guarantees**

* For each function in scope, you have a set of metrics.
* Tables:

  * `analytics.function_metrics`
  * `analytics.function_types` (if you separate type information)

### 5.2 `PluginMetadata` for `analytics.function_metrics`

* `name`: `"analytics.function_metrics"`

* `kind`: `ANALYTICS`

* `stage`: `"function_metrics"`

* `version`: metrics algorithm version/hash

* `description`: `"Compute function-level metrics (complexity, size, types, graph centrality, etc.)."`

* `provides`:

  * `"analytics.function_metrics"`
  * `"analytics.function_types"`

* `requires`:

  * `"core.goids"`
  * `"graph.callgraph"`
  * optionally:

    * `"history.git_churn"`
    * `"coverage.function"`
    * `"core.parsed_code"`

* `produces_tables`:

  * `"analytics.function_metrics"`
  * `"analytics.function_types"`

* `consumes_tables`:

  * `"core.goids"`
  * `"graph.call_graph_nodes"`
  * `"graph.call_graph_edges"`
  * plus coverage/history tables as needed

* `supports_incremental`: `True`

* `options_model`:

  * **Here you plug your existing `FunctionAnalyticsOptions`**.
  * It might include fields like:

    * `include_graph_metrics: bool`
    * `include_coverage: bool`
    * `metric_whitelist: list[str] | None`
    * `complexity_thresholds: ...`

* `resource_hints`:

  * `{"expected_cost": "medium", "cpu_intensive": True}`

### 5.3 `PluginResult` for `analytics.function_metrics`

* `status`
* `row_counts`:

  * `{"analytics.function_metrics": N_metric_rows, "analytics.function_types": N_type_rows}`
* `outputs` (optional):

  * e.g. a small summary object:

    * `"metric_summary": {"complexity": {"mean": ..., "p95": ...}, ...}`
* `metrics` (meta-metrics about the run, not per-function metrics):

  * `"functions_analyzed": <int>`
  * `"functions_skipped": <int>`
* `warnings`:

  * e.g. metrics skipped due to missing dependencies (no coverage)
* `timing`
* `extra`:

  * `"options_effective": <serialized subset of FunctionAnalyticsOptions>`
  * `"scope": e.g. paths/modules filter`

The **actual per-function metrics** live in the table `analytics.function_metrics` via row mappers; they’re not all packed into `PluginResult.metrics` (which is for run-level stats).

---

## 6. `analytics.function_hotspots` – compute hotspots

### 6.1 Functional role

**Preconditions**

* `"analytics.function_metrics"` capability is available (metrics computed).
* `"history.git_churn"` capability is available (churn computed by some history/analytics plugin).
* Optional:

  * coverage capabilities,
  * subsystem/grouping capabilities.

**Postconditions / guarantees**

* Each function has a hotspot score (or flagged/unflagged).
* Table:

  * `analytics.function_hotspots`

### 6.2 `PluginMetadata` for `analytics.function_hotspots`

* `name`: `"analytics.function_hotspots"`

* `kind`: `ANALYTICS`

* `stage`: `"hotspots"`

* `version`: hotspot algorithm version/hash

* `description`: `"Compute function hotspots based on metrics and history (churn, complexity, coverage)."`

* `provides`:

  * `"analytics.function_hotspots"`

* `requires`:

  * `"analytics.function_metrics"`
  * `"history.git_churn"`
  * optionally:

    * `"coverage.function"`
    * `"analytics.subsystem_profiles"` (if used in weighting/grouping)

* `produces_tables`:

  * `"analytics.function_hotspots"`

* `consumes_tables`:

  * `"analytics.function_metrics"`
  * `"history.git_churn"`
  * plus optional coverage/history tables

* `supports_incremental`: `True`

* `options_model`:

  * `HotspotOptions`, e.g.:

    * `weight_complexity: float`
    * `weight_churn: float`
    * `weight_coverage: float`
    * `hotspot_threshold: float`
    * `top_k: int | None`

* `resource_hints`:

  * `{"expected_cost": "low-medium", "cpu_intensive": False}`

### 6.3 `PluginResult` for `analytics.function_hotspots`

* `status`
* `row_counts`:

  * `{"analytics.function_hotspots": N_hotspots}`
* `outputs` (optional and trimmed):

  * `"top_hotspots": list[{"function_goid": ..., "score": ..., "path": ..., "qualname": ...}]` – top K hotspots for convenience.
* `metrics`:

  * `"functions_scored": <int>`
  * `"hotspots_above_threshold": <int>`
* `warnings`:

  * e.g. “churn data missing for X functions; scores degraded”
* `timing`
* `extra`:

  * `"hotspot_formula": description or version id`
  * `"threshold": <float>`
  * `"top_k": <int | None>`

---

## 7. Why this is useful for migration

With this level of specificity, you can now:

* For each existing plugin in your codebase, map:

  * **What it currently reads/writes** → `requires`/`consumes_tables` and `produces_tables`.
  * **What other code depends on it** → `provides` capabilities.
  * **Its options class** → `options_model`.
  * **Its current result/telemetry** → fields inside `PluginResult`.

* For the **six critical plugins above**, you have:

  * A clear contract in terms of:

    * capabilities,
    * datasets,
    * incremental behavior,
    * options.
  * A clear result shape that:

    * drives manifests (via `row_counts`, `status`),
    * powers telemetry/serving (via `metrics`, `top_hotspots`).

Everything else (CLI, build planner, manifest logic, policies) becomes plumbing around these very explicit contracts.

If you’d like next, we can:

* take one of these plugins (say `analytics.function_metrics`) and “diff” from your current implementation: what to keep, what to rename, what to add, so it cleanly slots into this new PluginMetadata / PluginResult world with minimal surprises for your current code.


# graph example #

Let’s do the “graph version” of what we just did for `analytics.function_metrics`, but this time for:

> `graphs.plugins.builders.callgraph.CallGraphPlugin`

Goal: make it very clear what you have today, what the **unified plugin** version looks like, and what you’d keep / change / add to slot it cleanly into the new architecture.

---

## 1. Snapshot: what `CallGraphPlugin` does today

### 1.1 Current role and shape

File: `graphs/plugins/builders/callgraph.py`

Today it is a **build TargetPlugin**:

```python
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import CallGraphStepConfig
...

class CallGraphPlugin(TargetPlugin):
    """Build call graph nodes and edges."""
    plugin_name: ClassVar[str] = "callgraph"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build call graph nodes and edges."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        ...
```

Key behaviors inside `execute`:

* Builds `CallGraphStepConfig(snapshot=ctx.snapshot)`:

  * Gets `repo`, `commit`, and (maybe) `repo_root`.
* Uses `ctx.gateway` (a `StorageGateway`) to:

  * Log repo state (`core.modules`, `core.goids` counts).
  * Load function index:

    ```python
    function_index = load_function_index(gateway, repo=repo, commit=commit)
    paths = function_index.paths()
    ```
* If there are no functions (`paths` empty), it returns a **successful** result with 0 rows.
* Computes lookup structures:

  * `global_callees = _build_global_callee_lookup(...)` (qualname → GOID).
  * `def_goids = _build_def_goids_by_path(...)` (path → module GOID).
  * `source_root = ctx.snapshot.repo_root or _get_source_root(...) or Path.cwd()`.
* For each Python file path:

  * Uses `normalize_rel_path` to map repo rel paths to filesystem paths.
  * Reads the file, parses with LibCST (fallback to AST).
  * Collects import aliases (`collect_aliases`).
  * Uses `EdgeResolutionContext` and `collect_edges_cst` / `collect_edges_ast` to generate **call edges**.
* Persists:

  * Nodes via `_persist_nodes`:

    * uses `IngestStorageService.from_gateway(gateway)` and `run_batch(
        "graph.call_graph_nodes", [call_graph_node_to_tuple(node)], delete_params=[repo, commit])`.
  * Edges via `_persist_edges`:

    * dedupe via `dedupe_edge_rows`,
    * serialize `evidence_json`,
    * then `run_batch("graph.call_graph_edges", ...)`.
* Returns:

  * `TargetResult.succeeded(row_counts={"graph.call_graph_nodes": node_count, "graph.call_graph_edges": edge_count})`
  * or `TargetResult.failed(...)` on `RuntimeError`, `ValueError`, `OSError`.

So: **CallGraphPlugin is a thin build wrapper** around well-factored graph-building code, analogous to `FunctionMetricsPlugin` on the analytics side.

### 1.2 How graphs runtime sees it today

Even though you now have a **unified graph plugin runtime**, `CallGraphPlugin` is still a `TargetPlugin`. You bridge it via:

* `graphs/core/adapters.py → TargetPluginAdapter`

  * Wraps a `TargetPlugin` and exposes `GraphPluginProtocol`:

    * builds `GraphPluginMetadata` via `create_graph_metadata` using:

      * `plugin.plugin_name` (`"callgraph"`),
      * `plugin.plugin_description`,
      * `_PLUGIN_KIND_STAGE_MAP["callgraph"] = ("builder", "edges")`,
      * `version_hash=plugin.plugin_version`.
    * builds a `TargetExecutionContext` from `GraphPluginExecutionContext`.
    * runs `asyncio.run(self._plugin.execute(target_ctx))`.
    * converts `TargetResult` → `PluginResult`.

And `graphs/plugins/__init__.py` registers:

```python
target_plugins = [
    GoidBuilderPlugin(),
    CallGraphPlugin(),
    ImportGraphPlugin(),
    CfgDfgPlugin(),
    SymbolUsesPlugin(),
    ...
]

for plugin in target_plugins:
    adapter = TargetPluginAdapter(plugin)
    registry.register(adapter)
```

So:

* Build system talks to `CallGraphPlugin` as a **TargetPlugin**.
* Graph runtime talks to it through **TargetPluginAdapter → GraphPluginProtocol**.

---

## 2. Target shape: unified graph plugin for callgraph

In the unified architecture, we want `CallGraphPlugin` to be a **first-class graph plugin**, not a TargetPlugin that needs adapting.

That means:

* Implement `GraphPluginProtocol` (which itself extends the unified `PluginProtocol` semantics).
* Provide **rich `GraphPluginMetadata`** directly on the plugin.
* Use `GraphPluginExecutionContext` as the context type.
* Return `PluginResult`.

Conceptually (sketch):

```python
from codeintel.graphs.core.protocol import (
    GraphPluginMetadata,
    GraphPluginProtocol,
    GraphPluginKind,
    GraphPluginStage,
    create_graph_metadata,
)
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.graphs.engine import GraphKind
from codeintel.core.plugins.types.result import PluginResult

class CallGraphPlugin(GraphPluginProtocol):
    """Build call graph nodes and edges."""

    _METADATA = create_graph_metadata(
        name="graphs.callgraph",  # or just "callgraph" if you want to keep old id
        description="Build call graph nodes and edges.",
        kind="builder",
        stage="edges",
        severity="fatal",
        enabled_by_default=True,
        produces_tables=("graph.call_graph_nodes", "graph.call_graph_edges"),
        produces_graph_kinds=(GraphKind.CALLGRAPH,),
        requires=("core.goids", "ingest.scip_index"),
        provides=("graph.callgraph",),
        supports_incremental=True,
        scope_aware=True,  # if you add scope support
        # options_model=CallGraphOptions,
    )

    @property
    def metadata(self) -> GraphPluginMetadata:
        return self._METADATA

    def execute(self, ctx: GraphPluginExecutionContext) -> PluginResult:
        ...
```

The **core building logic** (`load_function_index`, `_collect_all_edges`, `_persist_nodes`, `_persist_edges`) is unchanged; we’re just:

* normalizing the plugin interface,
* explicitly describing its dependencies & outputs in metadata,
* and using the graph execution context.

---

## 3. Diff by concern (current vs target)

### 3.1 Execution interface & context

**Current**

* Base class: `TargetPlugin`

* Signature:

  ```python
  async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
  ```

* Context is build-oriented:

  * `ctx.snapshot` (repo, commit, repo_root)
  * `ctx.gateway` (`StorageGateway`)

**Target**

* Implements `GraphPluginProtocol` (just a typing protocol; you don’t subclass it).

* Signature:

  ```python
  def execute(self, ctx: GraphPluginExecutionContext) -> PluginResult:
  ```

* `GraphPluginExecutionContext` gives you:

  * Everything from `PluginExecutionContext` (run info, config, logging, scratch space).
  * A **graph resource** that bundles:

    * `StorageGateway` (or a storage port),
    * the `GraphEngine` (to query existing graphs),
    * scope (`GraphRunScope`) for path/module scoping.
  * Plus helpers like `require_graphs(...)` if you start composing graphs.

**What to keep**

* Use of snapshot-like data:

  * `repo`, `commit`, `repo_root` (these can be accessed from `ctx.run` or a small `GraphStepConfig` built from `ctx`).
* Use of `StorageGateway` for DB access.

**What to change**

* `TargetExecutionContext` → `GraphPluginExecutionContext`.

  * Instead of directly doing `gateway = ctx.gateway`, you may do:

    * `gateway = ctx.graph_resource.gateway` or similar (depending on how you finalize `GraphPluginExecutionContext`).
* `async def execute` → **synchronous** `def execute`:

  * Your body is already synchronous; dropping `async` is mechanical and removes the need for `asyncio.run` in the adapter.

**What to add**

* Optionally, use graph-context helpers:

  * `ctx.require_graphs({"goid"})` before you read `core.goids`.
  * Use `ctx.scope` (paths/modules) to limit which files you iterate over.

---

### 3.2 Metadata & identification

**Current**

Metadata is implicit and split:

* On the plugin class:

  ```python
  plugin_name = "callgraph"
  plugin_version = "3.0.0"
  plugin_description = "Build call graph nodes and edges."
  ```

* On the adapter, via `_PLUGIN_KIND_STAGE_MAP`:

  ```python
  _PLUGIN_KIND_STAGE_MAP = {
      "callgraph": ("builder", "edges"),
      ...
  }
  ```

The adapter then creates metadata:

```python
create_graph_metadata(
    name=plugin_name,                     # "callgraph"
    description=description,
    kind=kind,                            # "builder"
    stage=stage,                          # "edges"
    severity="fatal",
    enabled_by_default=True,
    version_hash=plugin.plugin_version,
)
```

But this metadata **does not** currently include:

* `produces_tables` = `("graph.call_graph_nodes", "graph.call_graph_edges")`
* graph capabilities (`provides="graph.callgraph"`)
* explicit `requires` capabilities (`"core.goids"`, `"ingest.scip_index"`).

**Target**

Move all that knowledge *onto the plugin* via `GraphPluginMetadata`:

* Name & description:

  * Could keep `"callgraph"`, or make it `"graphs.callgraph"` for clarity.
  * You probably want to align with the capability name `graph.callgraph`.

* Graph-specific fields:

  * `kind="builder"`
  * `stage="edges"`
  * `produces_graph_kinds=(GraphKind.CALLGRAPH,)`

* Cross-domain capabilities:

  * `provides=("graph.callgraph",)`
  * `requires=("core.goids", "ingest.scip_index")` (or `"core.symbols"`/`"core.parsed_code"` depending on what you really use).

* Datasets:

  * `produces_tables=("graph.call_graph_nodes", "graph.call_graph_edges")`
  * `consumes_tables=("core.goids", "ingest.scip_index", "core.modules", "core.snapshots")` as appropriate.

* Behavior:

  * `supports_incremental=True` (you can re-run only for changed paths).
  * `scope_aware=True` if you decide to honor scopes.

**What to keep**

* `plugin_version` (3.0.0) → `version_hash` or `version` inside metadata.
* Description text.

**What to change**

* Stop relying on `_PLUGIN_KIND_STAGE_MAP` in `TargetPluginAdapter` for kind/stage.
* Declare **all** of callgraph’s contract (tables, capabilities, requirements) directly in `GraphPluginMetadata`.

**What to add**

* Capabilities: `provides`, `requires`.
* Tables: `produces_tables`, `consumes_tables`.
* GraphKinds: `produces_graph_kinds=(GraphKind.CALLGRAPH,)`.
* Flags: `supports_incremental`, `scope_aware`, etc.
* `options_model` (see next section).

---

### 3.3 Options & step configs

**Current**

* Uses `CallGraphStepConfig` from `config/steps_graphs.py`:

  ```python
  cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
  gateway, repo, commit = ctx.gateway, cfg.repo, cfg.commit
  ```

* Options are effectively **hard-coded** inside the implementation:

  * Always uses LibCST when available.
  * Always processes all functions.
  * No explicit scope/limit options at the plugin level.

**Target**

* `CallGraphStepConfig` can stay as your **internal domain config**, but you’ll probably want to:

  * Construct it from `GraphPluginExecutionContext` instead of `TargetExecutionContext`.
  * Eventually slim it down, since the comments in `steps_graphs.py` say those configs are being deprecated in favor of build/graph runtime config.

* Introduce a **`CallGraphOptions`** model (Pydantic or dataclass) advertised via metadata:

  * Examples:

    ```python
    class CallGraphOptions(BaseModel):
        scope_paths: list[str] | None = None
        include_external_calls: bool = False
        use_ast_fallback: bool = True
        max_module_size_lines: int | None = None
    ```

  * Then set `options_model=CallGraphOptions` in `GraphPluginMetadata`.

* `GraphPluginExecutionContext` can expose:

  * `ctx.options` / `ctx.get_options(CallGraphOptions)` so the plugin can see:

    * scopes (paths/modules),
    * configuration for parsing strategy,
    * performance tuning flags.

**What to keep**

* The idea of a **step config** object bundling repo/commit/snapshot info; you just move its construction from build context to graph context.

**What to change**

* Instead of deriving everything purely from `ctx.snapshot` and implicit behavior, you:

  * pass down structured `CallGraphOptions` from CLI/build/policy → engine → context → plugin.

**What to add**

* `CallGraphOptions` as the canonical options model for this plugin.
* `options_model=CallGraphOptions` in metadata.
* Use of `ctx` to retrieve effective options (resolved from policies/profiles).

---

### 3.4 IO / datasets

**Current**

Data contracts are clear but implicit:

* **Reads** (via raw SQL and helper functions):

  * `core.modules` (for logging repo state).
  * `core.goids` (for building node list, function spans).
  * `core.snapshots` (to try to get `source_root`).
  * `ingest.scip_index` / `core.symbols` indirectly via helpers (depending on how `load_function_index` is implemented).

* **Writes**:

  * `graph.call_graph_nodes` via `_persist_nodes`:

    * uses `CallGraphNodeRow`, `call_graph_node_to_tuple`.
    * `IngestStorageService.from_gateway(gateway).run_batch("graph.call_graph_nodes", ...)`.
  * `graph.call_graph_edges` via `_persist_edges`:

    * deduplicates with `dedupe_edge_rows`.
    * uses `CallGraphEdgeRow`, `call_graph_edge_to_tuple`.
    * persists via `IngestStorageService.run_batch("graph.call_graph_edges", ...)`.

These are exactly the tables you want in the unified architecture.

**Target**

You **keep** the same IO behavior, but:

* Make the dataset dependencies explicit in metadata:

  * `produces_tables=("graph.call_graph_nodes", "graph.call_graph_edges")`.
  * `consumes_tables` listing the ones you use for reading.

* Later, you can refactor `_persist_nodes/_persist_edges` to use shared **RowMappers** instead of `call_graph_*_to_tuple`, but that’s a separate concern.

**What to keep**

* Use of `IngestStorageService` / `StorageGateway` for writing batched rows.
* The dataset names & schemata (`CallGraphNodeRow`, `CallGraphEdgeRow`).

**What to change**

* Nothing in IO for the first migration step; just describe it formally in metadata.

**What to add**

* Eventually: shared `RowMapper[CallGraphNode]` / `RowMapper[CallGraphEdge]`, so domain code can work on graph objects; but that’s optional for this plugin migration.

---

### 3.5 Result type & manifest integration

**Current**

* Returns a `TargetResult`:

  ```python
  return TargetResult.succeeded(
      row_counts={
          "graph.call_graph_nodes": node_count,
          "graph.call_graph_edges": edge_count,
      }
  )
  ```

* The `TargetPluginAdapter` converts:

  * `TargetResult` → `PluginResult`,
  * `plugin_name` / `plugin_version` → `GraphPluginMetadata.version_hash`.

* Build + graph runtime each have slightly different manifest/tracking logic.

**Target**

* `CallGraphPlugin.execute(...)` returns a **`PluginResult`** directly:

  * e.g.:

    ```python
    return PluginResult.ok(
        row_counts={
            "graph.call_graph_nodes": node_count,
            "graph.call_graph_edges": edge_count,
        }
    )
    ```

* The unified manifest layer (`PluginExecutionRecord` etc.) uses:

  * `plugin_name` / `version_hash`,
  * `row_counts` and `status`,
  * `input_hash` / `options_hash` (filled in by the engine).

**What to keep**

* Row counts as the primary success metric (for both manifests and telemetry).

**What to change**

* `TargetResult` → `PluginResult`.
* Error path: return `PluginResult.fail("Call graph build failed: ...")` instead of `TargetResult.failed(...)`.

**What to add**

* Optional run-level metrics in `PluginResult.metrics`, e.g.:

  * `"nodes": node_count`,
  * `"edges": edge_count`,
  * `"avg_out_degree": ...`.

---

### 3.6 Registration & planning

**Current**

* Registration:

  * `graphs/plugins/__init__.py` builds a list of `TargetPlugin` instances, including `CallGraphPlugin()`.
  * `TargetPluginAdapter` wraps each into `GraphPluginProtocol` and registers with `GraphPluginRegistry`.

* Build planning:

  * Build system has a **target graph** where the “call graph” target has:

    * a mapping to plugin name `"callgraph"`,
    * its dataset outputs (`graph.call_graph_*`),
    * dependencies (`goids`, etc.).
  * Build executor calls the plugin via `TargetPlugin` API.

**Target**

* Graph plugin registration:

  * `CallGraphPlugin` is itself a `GraphPluginProtocol`:

    * `GraphPluginRegistry.register(CallGraphPlugin())` directly.
  * `TargetPluginAdapter` is **no longer needed** for this plugin (though you might keep it for legacy ones until they’re migrated).

* Build planner:

  * Reads `GraphPluginMetadata` for `CallGraphPlugin`:

    * sees:

      * kind=`"builder"`, stage=`"edges"`,
      * `provides=("graph.callgraph",)`,
      * `requires=("core.goids", "ingest.scip_index")`,
      * `produces_tables=("graph.call_graph_nodes", "graph.call_graph_edges")`.
    * uses those to:

      * build the execution DAG,
      * schedule callgraph after GOIDs and ingestion,
      * match operations like “build-graphs” / “compute-hotspots” to required capabilities.

**What to keep**

* The fact that the callgraph builder is a graph **builder** at the `"edges"` stage.
* The existence of a planner that uses dataset dependencies and capabilities.

**What to change**

* Stop generating `GraphPluginMetadata` via `TargetPluginAdapter`; let the plugin define its own metadata.
* Let build/graph runtime both rely on this one metadata object, rather than separate maps.

**What to add**

* Map CLI operations → required capabilities (`"graph.callgraph"`, etc.), so the planner naturally includes `CallGraphPlugin` in execution plans.

---

## 4. Pragmatic migration path for `CallGraphPlugin`

Here’s a very similar phased approach to what we did for `analytics.function_metrics`:

### Phase 0 – Confirm core building logic is “pure”

* Make sure helper functions:

  * `load_function_index`, `_build_global_callee_lookup`, `_build_def_goids_by_path`,
  * `_collect_all_edges`, `_persist_nodes`, `_persist_edges`,

  are **decoupled from build**:

  * they depend only on:

    * `StorageGateway` / `IngestStorageService`,
    * config/dataclasses (CallGraphNodeRow/EdgeRow),
    * not on `TargetExecutionContext`.

* This is mostly already true in your current code.

### Phase 1 – Add unified graph plugin behavior alongside the old TargetPlugin API

**Goal:** Implement `GraphPluginProtocol` on `CallGraphPlugin` while keeping existing build behavior working.

Concretely:

1. Add a `_graph_metadata` field:

   * Build a `GraphPluginMetadata` via `create_graph_metadata`.
   * Include:

     * name, description, kind, stage,
     * `produces_tables`, `produces_graph_kinds`,
     * `provides`, `requires`,
     * `supports_incremental`, `scope_aware`, `options_model`.

2. Add a `metadata` property returning that metadata.

3. Add a “new world” execute method that uses `GraphPluginExecutionContext` and returns `PluginResult`.

4. Keep the existing `async def execute(self, ctx: TargetExecutionContext) -> TargetResult` as a **compat shim**, and call into the new method from there via a small adapter.

   * So the real logic lives in `_execute_unified(ctx: GraphPluginExecutionContext) -> PluginResult`.
   * Current build executor still uses `TargetPlugin.execute`, which internally builds a temporary graph context and calls `_execute_unified`.

This gives you:

* one “source of truth” for execution logic,
* one “source of truth” for callgraph metadata,
* no behavioral change for build initially.

### Phase 2 – Switch graph runtime & build to use unified plugin directly

**Goal:** Graph runtime + unified ExecutionEngine invoke callgraph as a first-class GraphPluginProtocol; build becomes a planner.

* Update `graphs/plugins/__init__.py`:

  * Instead of wrapping with `TargetPluginAdapter`, register `CallGraphPlugin()` directly with `GraphPluginRegistry`.

* Update the planner:

  * Use `GraphPluginMetadata` to infer:

    * dependencies (`requires`),
    * DAG stage (`stage`),
    * outputs (`produces_tables`, `produces_graph_kinds`).

* Update build:

  * Instead of calling `TargetPlugin.execute` directly, hand off to the unified `ExecutionEngine` which uses `GraphPluginExecutionContext` and `PluginResult`.

* Once everything uses the new execution path, you can:

  * Delete the old `TargetPlugin` base for `CallGraphPlugin`.
  * Drop the async wrapper (make `execute` fully synchronous).
  * Remove the `TargetPluginAdapter` entry for `callgraph`.

---

## 5. How this compares to the analytics plugin

Side-by-side with `analytics.function_metrics`:

* Both are **thin plugin wrappers** over substantial domain logic (`compute_function_metrics_and_types` vs callgraph helpers).
* Both:

  * inherit from `TargetPlugin` today,
  * use build contexts (`TargetExecutionContext`),
  * return `TargetResult` with row counts.
* In the unified world:

  * `FunctionMetricsPlugin` becomes a **generic analytics plugin** (`PluginProtocol` + `AnalyticsExecutionContext`).
  * `CallGraphPlugin` becomes a **graph plugin** (`GraphPluginProtocol` + `GraphPluginExecutionContext`).
* Both declare:

  * capabilities (`analytics.function_metrics` vs `graph.callgraph`),
  * dataset outputs (analytics vs graph tables),
  * explicit dependencies (`core.goids`, graphs, history, coverage, ingestion).

The patterns and migration strategy are essentially the same, which is exactly what you want: you’ll have one mental model for “how a plugin looks and behaves” whether it’s ingest, graph, or analytics.

---

If you’d like to go one level deeper next, we could:

* pick one of the **shared concerns**—for example, how `ExecutionEngine` + manifests would handle skip/rehash for callgraph—and spell out exactly which fields on `GraphPluginMetadata`, `PluginResult`, and `PluginExecutionRecord` you’ll rely on for cache correctness.


# call graph example #

For callgraph, the “skip vs re-run” story really lives in three places:

* **`GraphPluginMetadata`** – what the plugin *claims* it needs and produces.
* **`PluginResult`** – what the plugin *actually did* in a run.
* **`PluginExecutionRecord`** – what the engine *remembers* to decide if it can safely skip next time.

I’ll walk through:

1. What fields on each of those matter specifically for callgraph.
2. How the **ExecutionEngine** computes hashes for callgraph.
3. The exact conditions under which callgraph can be **skipped** (and when it must re-run).
4. Optional refinements for **incremental** / scoped runs.

I’ll stick to the concrete plugin we’ve been using:

* plugin: `graphs.callgraph` (or `graphs.callgraph_builder`)
* capability it provides: `"graph.callgraph"`
* tables: `graph.call_graph_nodes`, `graph.call_graph_edges`

---

## 1. GraphPluginMetadata: what matters for caching callgraph

Think of `GraphPluginMetadata` as the contract that tells the engine:

> “These are the things that affect my output. If any of them change, my results are no longer guaranteed valid.”

For callgraph, the key metadata fields are:

### 1.1 Identity and version

* `name: "graphs.callgraph"`
  Used as the **primary key** for manifest lookups and execution records.

* `version: "3.0.0"` (or `version_hash`)
  Represents the version of the *algorithm*. A change here should *always* invalidate old results.

You want **plugin version** included in whatever hash you use to decide “same inputs”.

### 1.2 Capabilities and dependencies

* `provides = {"graph.callgraph"}`
  Tells the engine: “I am the thing that satisfies the callgraph capability.”

* `requires = {"core.goids", "ingest.scip_index"}` (or `{"core.goids", "core.symbols"}` depending on implementation)
  This is crucial for caching because:

  * It tells the planner which **upstream plugins** must have run first.
  * It tells the engine which **upstream execution records** to look at when building the callgraph’s input hash.

If an upstream provider changes its input/options, callgraph should see that through its hash.

### 1.3 Datasets

* `produces_tables = {"graph.call_graph_nodes", "graph.call_graph_edges"}`
* `consumes_tables = {"core.goids", "ingest.scip_index", "core.modules", "core.snapshots", ...}`

For caching, `consumes_tables` aren’t hashed directly, but they:

* tell the engine which **data sources** the plugin logically depends on,
* can be used to enforce “don’t skip callgraph if dependent tables were mutated outside the plugin system”.

### 1.4 Behavior flags

* `supports_incremental = True`
  Lets the engine know that partial or path-level invalidation is *allowed* and that a previous record might still be useful if only part of the repo changed.

* `scope_aware = True` (if you add this)
  Indicates that callgraph results depend on a `GraphRunScope` (paths/modules).
  This must be reflected in the input hash (see below).

**Metadata fields that don’t matter for caching**:

* `kind` / `stage` – useful for planning, not for cache correctness.
* `description`, `resource_hints` – docs/planning only.
* `enabled_by_default`, `severity` – controls *whether/how* to run, but not “is my output still valid”.

---

## 2. PluginResult: what matters for caching callgraph

`PluginResult` is the **output** from a particular execution of callgraph.

Most of it is for telemetry and debugging; a few pieces are relevant for caching:

### 2.1 Status

* `status: SUCCESS | FAILED | SKIPPED | PARTIAL`

The **skip logic** should only reuse a record if:

* `status == SUCCESS` (or possibly `PARTIAL` in some advanced incremental schemes, but assume just SUCCESS for now).

If the last run failed, you never treat callgraph as “up to date”, regardless of hashes.

### 2.2 Row counts

* `row_counts = {
    "graph.call_graph_nodes": N_nodes,
    "graph.call_graph_edges": N_edges,
  }`

These aren’t strictly needed for correctness, but:

* They are stored into the `PluginExecutionRecord` and can be used as a sanity check.
* They are useful in dashboards/CLI to show “last time we built callgraph, it had 100k nodes/300k edges”.

They **do not** need to be part of the hash; they are *derived* from inputs. If the inputs don’t change, row counts should remain stable.

### 2.3 Everything else

* `outputs` (e.g., a `CallGraph` in-memory object),
* `metrics` (summary stats),
* `warnings`,
* `extra` (debug info),

are all useful but **not required for caching**. They get written into the record or logs; the engine doesn’t need them to decide skip vs re-run.

---

## 3. PluginExecutionRecord: the critical fields for cache correctness

`PluginExecutionRecord` is the “ledger entry” for one run of callgraph. For caching, this is the key object.

A good shape for callgraph might be:

```python
@dataclass(frozen=True)
class PluginExecutionRecord:
    plugin_name: str              # "graphs.callgraph"
    version: str                  # "3.0.0"
    repo: str                     # "owner/repo"
    commit: str                   # "abc123"
    scope_id: str | None          # hashed GraphRunScope (paths/modules)
    variant: str | None           # e.g. "full" vs "fast"

    status: PluginStatus          # SUCCESS/FAILED/SKIPPED...

    input_hash: str               # primary cache key for "what went in"
    options_hash: str | None      # hash of CallGraphOptions

    row_counts: dict[str, int]
    started_at: datetime
    finished_at: datetime

    upstream_state: dict[str, str]  # capability -> state signature
    extra: dict[str, Any]          # debugging / extra fields only
```

The **fields that matter for skip/rehash** are:

1. `plugin_name` – identifies which plugin this record is for.
2. `version` – plugin code version.
3. `repo`, `commit` – which snapshot we ran on.
4. `scope_id` – a stable hash representing the graph scope (paths/modules) used.
5. `variant` – if you have variants (e.g. “fast”), this must be part of the identity.
6. `status` – must be `SUCCESS` to reuse.
7. `input_hash` – hash of all logical inputs to callgraph.
8. `options_hash` – hash of `CallGraphOptions`.
9. `upstream_state` – mapping from required capabilities to **state signatures** of their providers; these should be included in `input_hash` but are also stored explicitly for debugging.

Row counts, timestamps, and `extra` are useful but not part of the *decision*.

---

## 4. How ExecutionEngine computes hashes for callgraph

The engine needs a function like:

```python
compute_callgraph_input_hash(
    *,
    run: RunContext,
    plugin_meta: GraphPluginMetadata,
    options: CallGraphOptions,
    scope: GraphRunScope,
    upstream_records: dict[PluginCapability, PluginExecutionRecord],
) -> tuple[str, str, dict[str, str]]
```

Which returns:

* `input_hash` – overall signature of the inputs.
* `options_hash` – hash of the options model.
* `upstream_state` – a map of capability → state signature (for record.extra/debugging).

### 4.1 Step 1: options_hash

Use a generic `compute_options_hash`:

* Input: `(plugin_name, options_instance)`.
* Implementation: serialize options to JSON with sorted keys, then SHA256 and truncate.

For callgraph:

* `options_model = CallGraphOptions`.
* Example serialized shape:

  ```json
  {
    "scope_paths": ["src/"],
    "include_external_calls": false,
    "use_ast_fallback": true,
    "max_module_size_lines": null
  }
  ```

If any of those fields change → `options_hash` changes.

### 4.2 Step 2: upstream state signatures

For each **required capability** in metadata:

* `requires = {"core.goids", "ingest.scip_index"}`

The engine:

1. Finds which plugin(s) provide that capability from metadata registry.
2. Fetches their latest successful `PluginExecutionRecord` for this `(repo, commit, scope?)`.
3. Derives a **state signature** from the record.

The simplest state signature is just:

* `state_sig = record.input_hash`

You could get fancier (include row counts, output hash, etc.), but if each upstream plugin’s input hash already includes its own upstream dependencies, you effectively get a **transitive** view of state.

So for callgraph you might end up with:

```python
upstream_state = {
    "core.goids": "abcd1234...",        # from graphs.goid_builder.input_hash
    "ingest.scip_index": "f00dbabe...", # from ingest.scip_python.input_hash
}
```

### 4.3 Step 3: include scope

If callgraph is **scope-aware** (e.g. it only builds graphs for certain paths/modules):

* You need a deterministic representation of scope in the hash.

Create something like:

```python
scope_id = sha256(
    json.dumps(
        {
            "paths": sorted(scope.paths),
            "modules": sorted(scope.modules),
        },
        sort_keys=True,
    )
).hexdigest()[:16]
```

This `scope_id` goes both into:

* `PluginExecutionRecord.scope_id`, and
* the `input_hash` payload.

If scope is always “entire repo”, `scope_id` can be `None` or a constant.

### 4.4 Step 4: build the input hash payload

The engine builds a dict like:

```python
payload = {
    "repo": run.repository.name,
    "commit": run.commit,
    "plugin_name": plugin_meta.name,           # "graphs.callgraph"
    "plugin_version": plugin_meta.version,     # "3.0.0"
    "options_hash": options_hash,
    "scope_id": scope_id,
    "upstream_state": upstream_state,          # {"core.goids": "...", "ingest.scip_index": "..."}
}
```

Then uses a generic `compute_input_hash(payload)`:

* Serialize with `json.dumps(..., sort_keys=True)`.
* SHA256, truncated.

Result is the `input_hash` that goes into `PluginExecutionRecord.input_hash`.

---

## 5. How the engine uses records to skip or re-run callgraph

When it reaches the callgraph step in the `ExecutionPlan`, the engine:

1. **Compute `options_hash`, `input_hash`, `upstream_state`, `scope_id`** as above.

2. **Lookup previous record**:

   * Query manifest store for the **latest** `PluginExecutionRecord` with:

     * `plugin_name == "graphs.callgraph"`
     * `repo == run.repository.name`
     * `commit == run.commit`
     * `scope_id == computed_scope_id`
     * `variant == step.variant` (if variants exist)

3. If no record: → **must run**.

4. If record exists:

   * Check:

     * `record.status == SUCCESS`
     * `record.version == plugin_meta.version`
     * `record.input_hash == input_hash`
     * `record.options_hash == options_hash`
   * If all true → **safe to skip**.
   * Otherwise → **must run**.

When skipping:

* Engine may emit events like:

  * `PluginStarted(..., skipped=True)`
  * `PluginFinished(..., status=SKIPPED, reason="cache-hit")`

* Returns a synthetic `PluginResult` with:

  * `status=SKIPPED`
  * `row_counts` copied from last `PluginExecutionRecord`.

When running:

* Engine:

  * Builds `GraphPluginExecutionContext`.
  * Calls `CallGraphPlugin.execute(ctx)` → gets `PluginResult`.
  * Writes a new `PluginExecutionRecord` with:

    * `input_hash`, `options_hash`, `upstream_state`, `row_counts`, etc.

---

## 6. When callgraph must re-run (even if repo/commit look the same)

With this design, callgraph’s results are invalidated whenever *any* of the following change:

1. **Code or data changes**

   * `repo` or `commit` change.
   * Upstream providers’ `input_hash` change:

     * GOIDs were rebuilt with different options/version.
     * Ingestion changed (SCIP index, symbols).
   * Scope changes (different paths/modules) → different `scope_id`.

2. **Algorithm or config changes**

   * Callgraph plugin’s `version` changes.
   * `CallGraphOptions` changes → different `options_hash`.

     * Example: toggling `include_external_calls`, adjusting `max_module_size_lines`.

3. **Variant changes**

   * If you have `variant="fast"` vs `variant="full"`, switching variants must be treated either as:

     * different `plugin_name` (e.g. `"graphs.callgraph_full"`), or
     * part of the input payload (e.g. `variant` included in `payload`), or
     * part of the record key (`PluginExecutionRecord.variant`).

   Any of those ensures variant changes cause re-run.

4. **Policy changes that affect behavior**

   * For cache correctness, policies that only affect runtime behavior (timeouts, concurrency) generally **should not** be included in the `input_hash`.
   * If a policy changes **what** gets computed (e.g. a policy-driven cutoff for file size), then that policy value needs to be part of either:

     * `CallGraphOptions`, or
     * the `extra_fields` included in `input_hash`.

---

## 7. Incremental / partial runs (optional refinement)

So far, the design assumes a single `input_hash` per `(repo, commit, scope)`, and callgraph either runs or skips.

Because `supports_incremental = True`, you can go further:

* Track **per-path** or **per-module** state in `PluginExecutionRecord.extra`, e.g.:

  ```python
  extra = {
      "paths": {
          "src/foo.py": {"last_built_commit": "abc123"},
          "src/bar.py": {"last_built_commit": "abc123"},
      }
  }
  ```

* Use `IngestExecutionContext.change_tracker` to know which paths changed since last callgraph run.

Then:

* If only a subset of paths changed, callgraph can:

  * read old callgraph edges,
  * delete/rebuild edges for changed paths only,
  * leave unaffected parts intact.

From the engine’s perspective:

* The **run-level input_hash** is still recomputed based on upstream state and full scope, so you’ll still store a new `PluginExecutionRecord` when anything changes.
* The **implementation** of callgraph uses its own incremental strategy internally.

This keeps the caching model simple (one hash per `(repo, commit, scope)`) while letting the plugin implementation be smarter about the work it does when re-run.

---

## 8. Quick summary of “which field is used where” for callgraph

To tie it together:

### From `GraphPluginMetadata`

* `name` → record key + hash payload.
* `version` → hash payload; ensures new algorithm invalidates old cache.
* `requires` → which upstream `PluginExecutionRecord`s to include in hash.
* `provides` → which capability this plugin satisfies; used for planning.
* `produces_tables` / `consumes_tables` → **not hashed**, but:

  * used for planning and sanity checks (e.g., no skipping if someone mutated those tables manually).
* `supports_incremental` / `scope_aware` → control behavior, not hashed directly (but `scope` is hashed via `scope_id`).

### From `PluginResult`

* `status` → must be `SUCCESS` for cache reuse.
* `row_counts` → stored in record; not used in hash.
* `metrics`, `outputs`, `warnings`, `extra` → not relevant to skip vs run; useful for telemetry/debug.

### From `PluginExecutionRecord`

* `plugin_name`, `repo`, `commit`, `scope_id`, `variant` → uniqueness key for “this execution”.
* `version` → ensures you don’t reuse old results after plugin upgrade.
* `input_hash` → main “did anything that matters change?” signal.
* `options_hash` → changes when `CallGraphOptions` changes.
* `status` → must be `SUCCESS` to consider record valid.
* `upstream_state` → included in `input_hash` and good for debugging “why did callgraph re-run?”.
* `row_counts`, `started_at`, `finished_at`, `extra` → visible in CLI/telemetry; not part of the cache decision.

That’s the full picture of how callgraph caching/rehash works in this architecture—and how the three layers (metadata, result, manifest record) cooperate to give you correct and debuggable skip behavior.
