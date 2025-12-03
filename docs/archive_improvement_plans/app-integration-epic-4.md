Here’s a full Epic 10 implementation plan, picking up from the earlier thinking but aligned with your **actual** Epic 9 code (spec/planner/executor, OperationCatalog, dataset contracts, etc.). 

I’ll organize it as:

1. Functional intent recap
2. 10.1 – `op_planner.py`: from operation → `PipelineSpec`
3. 10.2 – Orchestrator API: `ensure_prerequisites_for_operation(...)`
4. 10.3 – Tests and fixtures
5. Optional Phase 2: fine-grained plugin selection (future-friendly design)

---

## 1. Functional intent (Epic 10)

**Goal:** for a given serving operation `op_id`, answer:

> “What do I have to run (ingest / graphs / analytics) to make operation X safe to execute?”

using the *canonical* metadata you already have:

* **OperationCatalog:** every `Operation` has:

  * `required_datasets: tuple[str, ...]` – dataset table_keys.
  * `required_graphs: tuple[str, ...]` – graph runtimes (e.g. `"callgraph"`). 

* **Dataset contracts (`codeintel.config.datasets`):**

  * `DATASET_CONTRACTS_BY_TABLE_KEY: dict[str, DatasetContract]`
  * Each `DatasetContract` includes `table_key`, `owner_package`, `schema_version`, `upstream_dependencies`, etc. 

* **Plugin metadata:**

  * Ingestion plugins have `output_tables: ClassVar[tuple[str, ...]]` (e.g. `AstExtractPlugin.output_tables = ("core.ast_nodes", "core.ast_metrics")`).
  * Graph builder plugins have `produces_tables`, `produces_graphs`, `provides`, etc. 

* **Unified pipeline:** Epic 9 gave you:

  * `PipelineSpec` + `PipelineStage` in `codeintel.pipeline.spec`.
  * `build_pipeline_plan(...)` in `codeintel.pipeline.planner`.
  * `run_pipeline(...)` in `codeintel.pipeline.executor`.

Epic 10’s job is to sit **on top** of this and:

1. Inspect the operation → infer which stage types are needed (ingestion / graphs / analytics).
2. Choose an appropriate `PipelineSpec` (initially from your canned specs).
3. Provide a **one-liner** orchestration API for callers: “run whatever is needed so op X is safe.”

---

## 2. 10.1 – Planner: from operation → `PipelineSpec`

### 10.1.1 New module skeleton

Create **`codeintel/pipeline/op_planner.py`**:

```python
# src/codeintel/pipeline/op_planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from codeintel.config.datasets import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    DatasetContract,
)
from codeintel.pipeline.spec import (
    PipelineSpec,
    PipelineStage,
    FULL_PIPELINE,
    INGEST_ONLY,
    GRAPHS_ONLY,
    ANALYTICS_ONLY,
)
from codeintel.serving.operations.catalog import (
    Operation,
    get_operation,
)

# Optional: will be handy later if you want to expose more detail
@dataclass(frozen=True)
class OpPrereqSummary:
    op: Operation
    required_tables: frozenset[str]
    expanded_tables: frozenset[str]
    core_tables: frozenset[str]
    graph_tables: frozenset[str]
    analytics_tables: frozenset[str]
    required_graphs: frozenset[str]
```

> **Note**: `DATASET_CONTRACTS` is keyed by dataset *name*, `DATASET_CONTRACTS_BY_TABLE_KEY` by `table_key`. Operation’s `required_datasets` are *table_keys*, so you’ll need both. 

### 10.1.2 Lookup operation + base prerequisites

Add helper to fetch an operation and its direct requirements:

```python
def _get_required_from_operation(op_id: str) -> tuple[Operation, set[str], set[str]]:
    """
    Look up an operation and return its direct dataset + graph requirements.

    Returns
    -------
    (op, required_tables, required_graphs)
    """
    op = get_operation(op_id)
    if op is None:
        raise ValueError(f"Unknown operation id: {op_id}")

    required_tables = set(op.required_datasets)
    required_graphs = set(op.required_graphs)
    return op, required_tables, required_graphs
```

### 10.1.3 Expand dataset upstream dependencies (transitive closure)

We want: if an operation requires dataset `analytics.function_profile`, and that contract declares `upstream_dependencies=("function_metrics", "graph_metrics", ...)`, we pull those in too.

Assuming:

* `DATASET_CONTRACTS: dict[str, DatasetContract]` keyed by dataset **name**.
* Each `DatasetContract.name` is the dataset name referred to by `upstream_dependencies`.
* `DATASET_CONTRACTS_BY_TABLE_KEY: dict[str, DatasetContract]` keyed by `table_key`.

We can build a name→contract map and compute closure:

```python
from codeintel.config.datasets import DATASET_CONTRACTS

def _build_contract_index() -> tuple[
    dict[str, DatasetContract],  # by table_key
    dict[str, DatasetContract],  # by name
]:
    by_table = DATASET_CONTRACTS_BY_TABLE_KEY
    by_name = {contract.name: contract for contract in DATASET_CONTRACTS.values()}
    return by_table, by_name


def _expand_dataset_dependencies(required_tables: set[str]) -> set[str]:
    """
    Expand required dataset table_keys by following upstream_dependencies transitively.

    Parameters
    ----------
    required_tables
        Directly required dataset table_keys from Operation.required_datasets.

    Returns
    -------
    set[str]
        All dataset table_keys needed, including upstream dependencies.
    """
    by_table, by_name = _build_contract_index()

    needed: set[str] = set(required_tables)
    queue: list[str] = list(required_tables)

    while queue:
        table_key = queue.pop()
        contract = by_table.get(table_key)
        if contract is None:
            continue

        deps = contract.upstream_dependencies or ()
        for upstream_name in deps:
            upstream_contract = by_name.get(upstream_name)
            if upstream_contract is None:
                continue
            upstream_table = upstream_contract.table_key
            if upstream_table not in needed:
                needed.add(upstream_table)
                queue.append(upstream_table)

    return needed
```

### 10.1.4 Partition datasets by owner package

Per your high-level plan:

* `owner_package == "core"` → ingestion.
* `owner_package == "graphs"` → graph pipeline.
* `owner_package == "analytics"` → analytics pipeline.

Implementation:

```python
def _partition_by_owner_package(
    table_keys: Iterable[str],
) -> tuple[set[str], set[str], set[str]]:
    """
    Partition dataset table_keys into core / graphs / analytics buckets.

    Returns
    -------
    (core_tables, graph_tables, analytics_tables)
    """
    by_table, _ = _build_contract_index()

    core_tables: set[str] = set()
    graph_tables: set[str] = set()
    analytics_tables: set[str] = set()

    for table_key in table_keys:
        contract = by_table.get(table_key)
        if contract is None:
            continue

        owner = contract.owner_package or ""
        if owner == "core":
            core_tables.add(table_key)
        elif owner == "graphs":
            graph_tables.add(table_key)
        elif owner == "analytics":
            analytics_tables.add(table_key)
        else:
            # For now, ignore or attribute to core; can log for diagnostics.
            core_tables.add(table_key)

    return core_tables, graph_tables, analytics_tables
```

### 10.1.5 Decide which *stages* are needed (first cut based on packages + graphs)

Now we combine:

* `core_tables`, `graph_tables`, `analytics_tables`
* `required_graphs` from the operation

into booleans:

```python
def _compute_stage_flags(
    *,
    core_tables: set[str],
    graph_tables: set[str],
    analytics_tables: set[str],
    required_graphs: set[str],
    include_analytics: bool,
) -> tuple[bool, bool, bool]:
    """
    Decide whether ingestion / graphs / analytics stages are needed.

    Returns
    -------
    (need_ingestion, need_graphs, need_analytics)
    """
    need_ingestion = bool(core_tables) or bool(graph_tables) or bool(required_graphs)
    # Graphs stage is needed if:
    # - any graph-owned tables, or
    # - operation explicitly requires graph runtimes (callgraph, importgraph, ...)
    need_graphs = bool(graph_tables) or bool(required_graphs)

    # Analytics stage if:
    # - any analytics tables, OR
    # - caller opted into always running analytics (include_analytics=True)
    need_analytics = bool(analytics_tables) or include_analytics

    return need_ingestion, need_graphs, need_analytics
```

> For example, `function.summary` has `required_graphs=("callgraph",)` and no datasets, so:
>
> * core_tables = ∅
> * graph_tables = ∅
> * analytics_tables = ∅
> * required_graphs = {"callgraph"}
>   → ingestion + graphs are needed; analytics is included if `include_analytics=True` (default). That naturally resolves to **FULL_PIPELINE**.

For `datasets.list`, both `required_datasets` and `required_graphs` are empty and `data_source=COMPUTED`, so no stages are required; we’ll treat this as a **no-op** pipeline. 

### 10.1.6 Choosing a `PipelineSpec`

For **Phase 1**, we use the existing canonical specs (`FULL_PIPELINE`, `INGEST_ONLY`, `GRAPHS_ONLY`, `ANALYTICS_ONLY`) and a small new `NOOP` spec.

Add to `codeintel.pipeline.spec`:

```python
# spec.py

NOOP_PIPELINE = PipelineSpec(
    id="noop",
    description="No-op pipeline for operations with no prerequisites.",
    stages=(),
)

PIPELINE_SPECS: dict[str, PipelineSpec] = {
    spec.id: spec
    for spec in (
        FULL_PIPELINE,
        INGEST_ONLY,
        GRAPHS_ONLY,
        ANALYTICS_ONLY,
        NOOP_PIPELINE,
    )
}
```

Then implement the mapping logic in `op_planner.py`:

```python
from codeintel.pipeline.spec import (
    FULL_PIPELINE,
    INGEST_ONLY,
    GRAPHS_ONLY,
    ANALYTICS_ONLY,
    NOOP_PIPELINE,
)

def _choose_spec(
    *,
    need_ingestion: bool,
    need_graphs: bool,
    need_analytics: bool,
) -> PipelineSpec:
    """Map stage flags to one of the canonical PipelineSpecs."""
    if not (need_ingestion or need_graphs or need_analytics):
        return NOOP_PIPELINE

    # Simple first cut — prefer full pipeline when any combination is needed.
    if need_ingestion and need_graphs and need_analytics:
        return FULL_PIPELINE
    if need_ingestion and not need_graphs and not need_analytics:
        return INGEST_ONLY
    if need_graphs and not need_ingestion and not need_analytics:
        return GRAPHS_ONLY
    if need_analytics and not need_ingestion and not need_graphs:
        return ANALYTICS_ONLY

    # Mixed but not all three (e.g. ingest+graphs only)
    # For now, treat as FULL_PIPELINE; can be refined later
    return FULL_PIPELINE
```

### 10.1.7 Public API: `build_pipeline_for_operation(...)`

Finally, expose the main planner function:

```python
from codeintel.config.primitives import SnapshotRef

def build_pipeline_for_operation(
    op_id: str,
    snapshot: SnapshotRef,
    *,
    include_analytics: bool = True,
) -> PipelineSpec:
    """
    Build a PipelineSpec representing the minimal stages needed for an operation.

    This function:
    - Looks up the operation from the canonical catalog.
    - Expands required datasets via dataset contracts and upstream dependencies.
    - Partitions tables by owner_package into ingestion / graphs / analytics.
    - Uses required_graphs to force graph stage when necessary.
    - Chooses a canonical PipelineSpec based on the inferred stage needs.

    Parameters
    ----------
    op_id
        Operation identifier (e.g. "function.summary").
    snapshot
        Repository snapshot (currently used for logging / future hints).
    include_analytics
        If True, always include analytics stage when any datasets/graphs are needed.

    Returns
    -------
    PipelineSpec
        Canonical pipeline spec to execute for prereqs.
    """
    op, required_tables, required_graphs = _get_required_from_operation(op_id)

    expanded_tables = _expand_dataset_dependencies(required_tables)

    core_tables, graph_tables, analytics_tables = _partition_by_owner_package(
        expanded_tables,
    )

    need_ingestion, need_graphs, need_analytics = _compute_stage_flags(
        core_tables=core_tables,
        graph_tables=graph_tables,
        analytics_tables=analytics_tables,
        required_graphs=required_graphs,
        include_analytics=include_analytics,
    )

    spec = _choose_spec(
        need_ingestion=need_ingestion,
        need_graphs=need_graphs,
        need_analytics=need_analytics,
    )

    # Optional: debug logging
    # log.info("op_planner: op=%s tables=%s graphs=%s spec=%s", ...)

    return spec
```

You can optionally return an `OpPrereqSummary` alongside the spec if you want richer introspection for debugging or tests.

---

## 3. 10.2 – Orchestrator API

High-level requirement:

> “A one-liner any caller (CLI, HTTP, MCP, tests) can use to ‘do whatever work is necessary before serving operation X’.”

### 3.2.1 Align with **current** `run_pipeline` API

Your actual `run_pipeline` is:

```python
def run_pipeline(
    *,
    spec: PipelineSpec,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    gateway: StorageGateway,
    tools: ToolsConfig,
    trigger: TriggerKind = "cli",
) -> PipelineRunRecord:
    ...
```

So we should define the new API in terms of **gateway, snapshot, paths, tools**, not raw `DuckDBConnection` + `RunContext` (that was the earlier high-level sketch).

### 3.2.2 Optional: enrich RunContext with requested operation/datasets

If you want full observability, we can lightly extend **`build_pipeline_plan`** in `pipeline/planner.py` to pass `requested_operation` and `requested_datasets` into `new_run_context`:

```python
# pipeline/planner.py

from typing import Sequence

def build_pipeline_plan(
    *,
    spec: PipelineSpec,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    gateway: StorageGateway,
    tools: ToolsConfig,
    trigger: TriggerKind = "cli",
    requested_operation: str | None = None,
    requested_datasets: Sequence[str] | None = None,
) -> PipelinePlan:
    ...
    run_kind = _infer_run_kind(spec)
    run_ctx = new_run_context(
        snapshot=snapshot,
        kind=run_kind,
        trigger=trigger,
        requested_operation=requested_operation,
        requested_datasets=requested_datasets or (),
    )
    ...
```

Then `run_pipeline(...)` can accept the extra kwargs and pass them through when building the plan (keeping its existing signature backwards-compatible by making the new args optional).

### 3.2.3 `ensure_prerequisites_for_operation(...)` function

Add this to **`codeintel.pipeline.op_planner`** (or a tiny `op_orchestrator.py` if you prefer to separate concerns, but keeping it in `op_planner` is fine):

```python
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.pipeline.executor import run_pipeline
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.run_tracking import PipelineRunRecord
from codeintel.runtime import TriggerKind


def ensure_prerequisites_for_operation(
    *,
    op_id: str,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    gateway: StorageGateway,
    tools: ToolsConfig,
    include_analytics: bool = True,
    trigger: TriggerKind = "api",
) -> PipelineRunRecord:
    """
    Run whatever work is necessary before serving operation `op_id`.

    This function:
    - Builds an operation-driven PipelineSpec via build_pipeline_for_operation.
    - Executes that spec using the unified pipeline executor.
    - Records run + step metadata to pipeline_runs / pipeline_steps.

    Parameters
    ----------
    op_id
        Operation id (e.g. "function.summary", "datasets.list").
    snapshot
        Repository snapshot reference.
    paths
        Build paths for this run.
    gateway
        Storage gateway for DuckDB and metadata tables.
    tools
        Tools configuration (used by ingestion and analytics where relevant).
    include_analytics
        Whether to include analytics stage even if not strictly required by contracts.
    trigger
        How this run was triggered (default: "api").

    Returns
    -------
    PipelineRunRecord
        Run record describing the prereq pipeline execution.
    """
    spec = build_pipeline_for_operation(
        op_id=op_id,
        snapshot=snapshot,
        include_analytics=include_analytics,
    )

    # Optional: compute requested_datasets for observability
    op, required_tables, _ = _get_required_from_operation(op_id)
    expanded_tables = _expand_dataset_dependencies(required_tables)

    # If you extended run_pipeline / build_pipeline_plan with requested_operation/datasets:
    return run_pipeline(
        spec=spec,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        trigger=trigger,
        # Only if you wired these through:
        # requested_operation=op_id,
        # requested_datasets=sorted(expanded_tables),
    )
```

Callers (HTTP handlers, MCP tools, tests) can now do:

```python
run = ensure_prerequisites_for_operation(
    op_id="function.summary",
    snapshot=snapshot,
    paths=paths,
    gateway=gateway,
    tools=cfg.tools,
    trigger="http",
)
```

---

## 4. 10.3 – Tests

### 4.3.1 Unit tests for planner (`op_planner.py`)

New file **`tests/pipeline/test_op_planner.py`**:

1. **Simple mapping tests**:

   * `function.summary`:

     ```python
     from codeintel.pipeline.op_planner import build_pipeline_for_operation
     from codeintel.pipeline.spec import FULL_PIPELINE

     def test_function_summary_maps_to_full_pipeline(snapshot):
         spec = build_pipeline_for_operation("function.summary", snapshot)
         assert spec.id == FULL_PIPELINE.id
         assert {stage.module for stage in spec.stages} == {
             "ingestion",
             "graphs",
             "analytics",
         }
     ```

     (Because it requires `callgraph` and we opt into analytics by default.)

   * `datasets.list`:

     ```python
     from codeintel.pipeline.op_planner import build_pipeline_for_operation
     from codeintel.pipeline.spec import NOOP_PIPELINE

     def test_datasets_list_is_noop(snapshot):
         spec = build_pipeline_for_operation("datasets.list", snapshot)
         assert spec.id == NOOP_PIPELINE.id
         assert spec.stages == ()
     ```

2. **Dataset dependency expansion test**:

   * Pick an analytics dataset that has `upstream_dependencies` (e.g. `analytics.function_profile` if configured that way) and assert that `_expand_dataset_dependencies` pulls in upstreams (`analytics.function_metrics`, `analytics.graph_metrics`, etc.) by name and table_key.

   * Use the real `DATASET_CONTRACTS_BY_TABLE_KEY` and `DATASET_CONTRACTS` to ensure you’re not relying on test-only fixtures.

### 4.3.2 Integration tests for orchestrator

New file **`tests/pipeline/test_operation_prereqs.py`**:

Assume you have small fixture repos + gateways similar to `tests/test_pipeline_smoke.py`.

**Test 1 – `function.summary` runs ingest + graphs + analytics**

```python
from codeintel.config.primitives import SnapshotRef, BuildPaths
from codeintel.storage.gateway import open_memory_gateway, StorageConfig
from codeintel.pipeline.op_planner import ensure_prerequisites_for_operation
from codeintel.config.models import ToolsConfig

def test_function_summary_prereqs_runs_all_modules(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    # ... populate tiny repo if needed ...

    snapshot = SnapshotRef(
        repo="test/repo",
        commit="deadbeef",
        repo_root=repo_root,
    )
    paths = BuildPaths.from_layout(repo_root=repo_root)
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    tools = ToolsConfig()  # or from a fixture

    run = ensure_prerequisites_for_operation(
        op_id="function.summary",
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        include_analytics=True,
        trigger="api",
    )

    assert run.status == "succeeded"

    steps = gateway.runs.fetch_steps(run.run_id)
    modules = {step.module for step in steps}
    # We expect orchestrator + engine-level steps for each module
    assert {"ingestion", "graphs", "analytics"}.issubset(modules)
```

**Test 2 – `datasets.list` is effectively a no-op**

```python
def test_datasets_list_prereqs_is_noop(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    snapshot = SnapshotRef(
        repo="test/repo",
        commit="deadbeef",
        repo_root=repo_root,
    )
    paths = BuildPaths.from_layout(repo_root=repo_root)
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    tools = ToolsConfig()

    run = ensure_prerequisites_for_operation(
        op_id="datasets.list",
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        include_analytics=False,
    )

    assert run.status == "succeeded"

    steps = gateway.runs.fetch_steps(run.run_id)
    # No steps recorded – nothing had to be computed
    assert steps == []
```

You can add more cases (profiles, risk factors, graph-only operations, etc.) as you refine the mapping heuristics.

---

## 5. Phase 2 (optional) – fine-grained plugin selection

The above gives you **correct stage selection** and a clean orchestration API, but still uses coarse specs (FULL_PIPELINE, INGEST_ONLY, etc.). The high-level sketch you wrote also hints at a more advanced step:

> For each package, consult plugin metadata (ingest output_tables, graphs produces_tables / produces_graphs, analytics datasets) and plan a *minimal* plugin subset.

Design outline for that refinement (once you’re ready):

1. **Plugin introspection helpers**

   * In ingestion:

     * Iterate over registry: `get_ingest_registry().all_plugins()`.
     * For each plugin, inspect `.output_tables`.
   * In graphs:

     * Use `GraphPluginRegistry.instance()` and plugin metadata (`produces_tables`, `produces_graphs`).
   * In analytics:

     * The `analytics.core.plugins` registration + `analytics.datasets` lets you map from plugin outputs to dataset table_keys.

2. **Operation-specific plugin sets**

   * Given `expanded_tables` and `required_graphs`, compute:

     * `required_ingest_plugins`.
     * `required_graph_plugins`.
     * `required_analytics_plugins`.

3. **Extend `PipelineSpec` / planner to support “op-prereqs” dialect**

   * Introduce new stage names like:

     * `"op_prereqs.ingestion"`, `"op_prereqs.graphs"`, `"op_prereqs.analytics"`.
   * Extend `pipeline.planner._resolve_ingest_recipe` / `_plan_graphs_stage` / `_plan_analytics_stage` to:

     * When stage name starts with `"op_prereqs."`, use the plugin sets computed by op_planner instead of builtin defaults.

4. **Mark RunContext.kind as `"op_prereqs"`**

   * Extend `build_pipeline_plan` to accept `run_kind_override="op_prereqs"` when called from `ensure_prerequisites_for_operation`, so that run IDs clearly separate prereq runs from arbitrary full pipeline runs.

That’s a natural follow-on once the first phase is in, but you don’t *need* it to get a working Epic 10 – the phase 1 plan above already gives you operation-driven orchestration with consistent semantics, leveraging your Epic 9 pipeline stack.
