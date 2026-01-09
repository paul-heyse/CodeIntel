# Core Compute Acero DSL Maximal Adoption Plan

## Goal
Make the Arrow Acero and DSL surface the default compute path across
`src/codeintel/build` and `src/codeintel/ingestion`, with a strict
"plan -> execute -> finalize" boundary, consistent determinism policies,
and centralized scan, explode, and validation helpers.

## Guiding Principles
- Plan lane handles scan, filter, project, join, aggregate using Acero declarations.
- Kernel lane handles row-count-changing operations (explode, dedupe, canonical sort).
- Finalize gate owns schema alignment, invariants, dedupe, ordering, and artifacts.
- No raw `pyarrow.compute` in nodes; expressions and kernels live in core helpers.
- Dataset scan options and thread profiles are centrally managed.

## Scope Items

### 1) Core DSL IR consolidation + guardrails
Pattern (ExecutionContext, Plan, QuerySpec)
```python
from dataclasses import dataclass

import pyarrow.compute as pc


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    use_threads: bool
    combine_chunks: bool
    determinism: str  # "canonical" | "stable_set" | "best_effort"
    provenance: bool


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    base_cols: tuple[str, ...]
    computed: tuple[tuple[str, pc.Expression], ...] = ()


@dataclass(frozen=True, slots=True)
class QuerySpec:
    predicate: pc.Expression | None
    pushdown_predicate: pc.Expression | None
    projection: ProjectionSpec
```

Target files
- src/codeintel/core/columnar/plan_ops.py
- src/codeintel/core/columnar/arrowdsl.py
- src/codeintel/core/columnar/expr_vocab.py
- src/codeintel/core/columnar/streaming.py
- src/codeintel/core/columnar/queryspec.py (new)
- src/codeintel/core/columnar/__init__.py
- tests/core/columnar/test_no_raw_pc_imports.py

Checklist
- [x] Add `QuerySpec` and `ProjectionSpec` in a dedicated module.
- [x] Extend `ExecutionContext` to carry `provenance` and determinism tier.
- [x] Ensure `Plan` usage is canonical across build pipelines; ingestion adoption tracked in item 6.
- [x] Add guardrail test to ban `pyarrow.compute` imports in nodes outside core helpers.
Status: Complete for core DSL + guardrails.


### 2) QuerySpec-driven scan control plane + provenance
Pattern (compile query to scan + plan nodes)
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec

spec = QuerySpec(
    predicate=E.and_(E.eq("kind", "call"), E.ge("confidence", 0.8)),
    pushdown_predicate=E.eq("kind", "call"),
    projection=ProjectionSpec(
        base_cols=("repo_id", "caller_id", "callee_id", "kind", "confidence"),
    ),
)

plan = Plan.scan(
    dataset,
    columns=spec.projection.base_cols,
    filter_expr=spec.pushdown_predicate,
).filter(spec.predicate)
```

Target files
- src/codeintel/core/columnar/streaming.py
- src/codeintel/core/columnar/plan_ops.py
- src/codeintel/core/columnar/queryspec.py (new)
- src/codeintel/core/columnar/compute_config.py
- src/codeintel/core/config/settings.py
- src/codeintel/build/graphs/engine/datasets.py

Checklist
- [x] Compile QuerySpec to both scan options and filter/project nodes.
- [ ] Wire provenance columns into scan projections when `ctx.provenance=True`.
- [x] Add scan telemetry hooks (fragment counts, estimated rows).
- [x] Define named scan profiles (dev, CI, prod) and apply in one place.
Status: Partial — provenance is supported in QuerySpec/build_query_plan but not yet wired from callers.


### 3) Build graph pipelines: plan lane adoption (Acero-first)
Pattern (scan + join + finalize boundary)
```python
from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import FinalizeSpec
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan

left = Plan.table(call_sites).project({"caller_id": E.field("caller_id")})
right = Plan.table(symbols).project({"symbol_id": E.field("symbol_id")})

join_spec = HashJoinSpec(left_keys=["callee_id"], right_keys=["symbol_id"])
joined = left.hash_join(right=right, spec=join_spec)

result = run_pipeline(
    plan=ExecutionPlan(inner=joined.declaration),
    finalize=FinalizeSpec(table_key="graph.cpg_edges_calls", mode="tolerant"),
    ctx=ExecutionContext(use_threads=True, combine_chunks=True, determinism="canonical"),
)
```

Target files
- src/codeintel/build/hamilton/native/graphs/call_wiring.py
- src/codeintel/build/hamilton/native/graphs/cfg_dfg.py
- src/codeintel/build/hamilton/native/graphs/cdg.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/*.py
- src/codeintel/build/graphs/assembly/ids.py
- src/codeintel/build/analytics/cfg_dfg/helpers.py

Checklist
- [x] Replace ad hoc `pc.*` use with `Plan` + `HashJoinSpec` pipelines.
- [x] Pre-project and pre-cast join keys before `hash_join`.
- [x] Enforce join-safe schemas (no list payloads) before joins.
- [x] Route outputs through `FinalizeSpec` with table-specific invariants.
Status: Complete for the listed build graph targets.


### 4) Kernel lane consolidation: explode + dedupe + canonical sort
Pattern (explode list payloads with aligned lists)
```python
from codeintel.core.columnar.explode_ops import explode_edges_with_aligned_lists

result = explode_edges_with_aligned_lists(
    table,
    src_col="src_id",
    dst_list_col="callee_ids",
    aligned_list_cols=("callsite_spans",),
    repeat_cols=("repo_id", "file_id"),
)
edges = result.good
errors = result.errors
```

Target files
- src/codeintel/core/columnar/kernels.py
- src/codeintel/core/columnar/explode_ops.py
- src/codeintel/core/columnar/nested_ops.py
- src/codeintel/build/tabular/explode_ops.py
- src/codeintel/build/tabular/array_ops.py

Checklist
- [x] Standardize explode helpers with list-alignment checks.
- [x] Reuse parent indices for repeating scalar columns.
- [x] Add null-list policies (error vs empty) in kernel helpers.
- [x] Ensure dedupe and canonical sort live in shared kernels.
Status: Complete; aligned explode helper lives in explode_ops and is re-exported for build usage.


### 5) Finalize gate: determinism tiers + structured artifacts
Pattern (FinalizeSpec with determinism and artifacts)
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec

spec = FinalizeSpec(
    table_key="graph.cpg_edges_calls",
    mode="tolerant",
    required_non_null=("repo_id", "src_id", "dst_id"),
    order_by=(("repo_id", "ascending"), ("src_id", "ascending"), ("dst_id", "ascending")),
    emit_artifacts=True,
)
```

Target files
- src/codeintel/core/columnar/finalize_ops.py
- src/codeintel/core/columnar/dedupe_ops.py
- src/codeintel/core/columnar/arrowdsl.py
- src/codeintel/core/columnar/schema_alignment.py

Checklist
- [x] Encode determinism tiers in dedupe and ordering policy.
- [x] Require canonical sort keys when determinism is canonical.
- [x] Emit `good`, `errors`, `alignment`, `stats` artifacts consistently.
- [x] Standardize error codes and stages for nested invariant failures.
Status: Complete.


### 6) Ingestion pipelines: scan + finalize + typed extras
Pattern (dataset scan to plan, then finalize)
```python
from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import FinalizeSpec
from codeintel.core.columnar.plan_ops import Plan

plan = Plan.scan(dataset, columns=["repo_id", "path", "extras"])
result = run_pipeline(
    plan=ExecutionPlan(inner=plan.declaration),
    finalize=FinalizeSpec(table_key="ingestion.repo_scan", mode="tolerant"),
    ctx=ExecutionContext(use_threads=True, combine_chunks=True, determinism="stable_set"),
)
```

Target files
- src/codeintel/ingestion/compute/base.py
- src/codeintel/ingestion/compute/repo_scan.py
- src/codeintel/ingestion/compute/ast_extract.py
- src/codeintel/ingestion/compute/cst_extract.py
- src/codeintel/ingestion/compute/inspect_extract.py
- src/codeintel/ingestion/compute/tree_sitter_index.py
- src/codeintel/ingestion/compute/*_extract.py

Checklist
- [ ] Replace direct table materialization with QuerySpec + Plan.scan.
- [ ] Emit typed `extras` struct plus optional `extras_kv` for long-tail fields.
- [x] Route ingestion outputs through finalize for schema alignment and artifacts.
Status: Partial — ingestion finalize gates exist; QuerySpec scan + typed extras adoption still pending.
- [ ] Apply provenance columns in tolerant modes for error traceability.


### 7) Streaming safety: remove to_pylist/to_numpy/to_pydict
Pattern (streaming-safe iteration)
```python
from codeintel.core.columnar.iter import iter_array_values

values = list(iter_array_values(table["count"]))
```

Target files
- src/codeintel/build/exports/writers.py
- src/codeintel/build/tabular/arrow_ops.py
- src/codeintel/core/columnar/iter.py
- src/codeintel/core/columnar/stream.py

Checklist
- [x] Replace `to_pydict` with column-wise iteration helpers.
- [x] Replace `to_numpy` with `iter_array_values` or compute kernels.
- [x] Keep reader surfaces streaming until finalize boundaries.
- [ ] Add small regression tests for streaming-safe helpers.
Status: Partial — streaming-safe replacements are done; regression tests still pending.


### 8) Optional escape hatches: external plans + serving alignment
Pattern (external plan runner)
```python
from codeintel.core.columnar.plan_ops import ExternalPlanRequest, ExternalPlanSpec, run_external_plan

request = ExternalPlanRequest(
    spec=ExternalPlanSpec(engine="substrait", payload=plan_bytes),
    dataset=dataset,
    filter_expr=None,
    columns=None,
    scan_options=None,
    use_threads=True,
)
reader = run_external_plan(request)
```

Target files
- src/codeintel/core/columnar/plan_ops.py
- src/codeintel/serving/semantic/arrow_plan_builder.py
- src/codeintel/serving/semantic/engines/arrow_engine.py
- src/codeintel/serving/semantic/engines/protocol.py

Checklist
- [x] Register external plan runners behind one interface.
- [ ] Keep finalize gate invariant across external engines.
- [ ] Reuse QuerySpec and expr vocab for serving plan translation.
- [ ] Add small integration tests for Arrow plan execution parity.
Status: Partial — external runners registered; serving QuerySpec wiring + finalize invariants + tests pending.


## Sequencing Recommendation
1) Core DSL IR + QuerySpec + scan profiles (items 1-2).
2) Build graph pipelines plan lane adoption + kernel consolidation (items 3-4).
3) Finalize gate determinism + artifacts (item 5).
4) Ingestion plan adoption + typed extras (item 6).
5) Streaming safety cleanup (item 7).
6) Optional external plan escape hatch (item 8).
