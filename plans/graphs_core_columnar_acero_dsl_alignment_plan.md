# Graph Core Columnar Acero/DSL Alignment Plan

## Objective
Align graph assembly, validation, and rustworkx integration to the centralized
Acero/DSL compute surface, maximizing reuse of plan-builder and kernel-lane
primitives while preserving determinism, extensibility, and performance.

## Design Principles (Non-Negotiable)
- Plan lane vs kernel lane stays explicit, with plan metadata preserved end-to-end.
- QuerySpec is the single source of truth for scan + projection + predicates.
- ExecutionContext and RuntimeProfile own determinism, threading, and provenance.
- Materialization is always mediated by ExecutionPlan + run_pipeline.
- Schema metadata is the only authority for finalize policies and canonical ordering.

---

## Scope 01 - Plan Builder Unification for Graph Scans and Producers
**Goal**
Route all graph scans and graph producer plans through the plan-builder API so
QuerySpec and schema defaults are authoritative and ad-hoc column selection is
eliminated.

**Code patterns**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.plan_builder import (
    build_grouped_rollup_plan,
    build_snapshot_plan,
    build_snapshot_query_spec,
)

execution_ctx = resolve_execution_context(None)
query_spec = build_snapshot_query_spec(
    base_cols=("repo", "commit", "caller_goid_h128", "callee_goid_h128"),
    repo=repo,
    commit=commit,
)
plan = build_snapshot_plan(table=table, spec=query_spec, ctx=execution_ctx)
plan = build_grouped_rollup_plan(
    plan,
    keys=("repo", "commit", "caller_goid_h128", "callee_goid_h128"),
    aggregates=(("caller_goid_h128", "count", None, "weight"),),
)
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan),
    finalize=finalize_spec_for_table("graph.call_graph_edges", mode="tolerant"),
    options=PipelineRunOptions(ctx=execution_ctx),
)
```

```python
from codeintel.core.columnar.plan_builder import (
    SchemaPlanDefaultsRequest,
    plan_from_schema_defaults,
)
from codeintel.core.schemas.service import get_schema_service

schema_service = get_schema_service()
plan = plan_from_schema_defaults(
    schema_service=schema_service,
    request=SchemaPlanDefaultsRequest(
        table_key="graph.call_graph_edges",
        dataset=dataset,
        predicate=query_spec.predicate,
        columns=query_spec.projection.columns(),
        ctx=execution_ctx,
    ),
)
```

**Target files**
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/assembly/plan_surface.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/hamilton/native/graphs/call_graph.py`
- `src/codeintel/build/hamilton/native/graphs/import_graph.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/graphs/cdg.py`
- `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
- `src/codeintel/build/hamilton/native/graphs/goids.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/goids.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`

**Implementation checklist**
- [ ] Replace remaining ad-hoc Plan assembly in graph producers with plan-builder helpers.
- [ ] Ensure graph scans use QuerySpec and schema-driven defaults (no re-derived columns).
- [ ] Collapse graph-specific plan surface APIs into thin plan-builder wrappers.
- [ ] Route materialization through ExecutionPlan + run_pipeline at all boundaries.

---

## Scope 02 - Kernel Lane Wrappers for Row-Changing Ops
**Goal**
Centralize explode/dedupe/rollup patterns in kernel wrappers to guarantee
join-safe projections and deterministic tie handling across graph paths.

**Code patterns**
```python
from codeintel.core.columnar.plan_kernels import (
    grouped_rollup_table,
    stable_dedupe_with_ties,
)

rollup = grouped_rollup_table(
    table=table,
    keys=("repo", "commit"),
    aggregates=(("severity", "count", None, "diagnostic_count"),),
    order_by=(("repo", "ascending"), ("commit", "ascending")),
)
deduped = stable_dedupe_with_ties(
    table=table,
    keys=("repo", "commit", "path"),
    order_by=(("score", "descending"),),
    tie_breakers=(("rel_path", "ascending"),),
)
```

```python
from codeintel.core.columnar.plan_kernels import explode_edges_for_join
from codeintel.core.columnar.explode_ops import ExplodeSpec

exploded = explode_edges_for_join(
    table=edges,
    spec=ExplodeSpec(parent_column="parents", child_column="children"),
    allowed_columns=("repo", "commit", "edge_id", "parents", "children"),
    schema_allowlist=table_schema,
)
```

**Target files**
- `src/codeintel/core/columnar/plan_kernels.py`
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/build/graphs/assembly/kernels.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py`
- `src/codeintel/build/tabular/explode_ops.py`
- `src/codeintel/build/tabular/arrow_ops.py`

**Implementation checklist**
- [ ] Replace bespoke explode helpers with `explode_edges_for_join` and schema allowlists.
- [ ] Replace ad-hoc group_by rollups with `grouped_rollup_table`.
- [ ] Use `stable_dedupe_with_ties` for order-dependent dedupe paths.
- [ ] Ensure all kernel wrappers accept schema-derived allowlists for join safety.

---

## Scope 03 - Ordering and Determinism as First-Class Plan Metadata
**Goal**
Propagate ordering metadata through plan nodes and enforce canonical ordering at
finalize boundaries for graph inputs.

**Code patterns**
```python
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan

plan = build_table_plan(
    table=table,
    options=TablePlanOptions(
        order_by=(("repo", "ascending"), ("commit", "ascending")),
    ),
)
```

```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table

finalize = finalize_spec_for_table("graph.import_graph_edges", mode="tolerant")
result = run_pipeline(plan=ExecutionPlan.from_plan(plan), finalize=finalize)
```

**Target files**
- `src/codeintel/core/columnar/ordering.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/assembly/finalize.py`

**Implementation checklist**
- [ ] Preserve ordering metadata across filter/project/join/aggregate plan nodes.
- [ ] Enforce canonical ordering at finalize when determinism tier is canonical.
- [ ] Remove implicit ordering reliance in graph view materialization paths.
- [ ] Use contract-derived canonical keys for stable ordering at finalize gates.

---

## Scope 04 - Schema-Driven Plan Defaults and Join-Safe Policies
**Goal**
Drive projection, ordering, and join-safe allowlists from schema metadata to
avoid re-declaring policy at call sites.

**Code patterns**
```python
from codeintel.core.schemas.primitives import PlanPolicy

TableSchema(
    ...,
    plan_policy=PlanPolicy(
        default_projection=("repo", "commit", "src", "dst", "edge_kind"),
        join_safe_columns=("repo", "commit", "src", "dst"),
    ),
)
```

```python
from codeintel.core.columnar.plan_builder import (
    SchemaPlanDefaultsRequest,
    plan_from_schema_defaults,
)

plan = plan_from_schema_defaults(
    schema_service=schema_service,
    request=SchemaPlanDefaultsRequest(
        table_key="graph.symbol_use_edges",
        dataset=dataset,
        predicate=query_spec.predicate,
        columns=query_spec.projection.columns(),
        ctx=execution_ctx,
    ),
)
```

**Target files**
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/schemas/view_registry.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/engine/views.py`

**Implementation checklist**
- [ ] Add default projections and join-safe allowlists for graph outputs.
- [ ] Use schema-driven defaults in graph scan entrypoints.
- [ ] Preserve plan_policy for schema overrides and view schemas.
- [ ] Align canonical sort keys with finalize ordering expectations.

---

## Scope 05 - External Plan Runner Unification
**Goal**
Keep external plan execution inside the plan lane and use ExecutionPlan to
preserve ordering/determinism metadata.

**Code patterns**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline
from codeintel.core.columnar.plan_ops import ExternalPlanRequest

execution_plan = ExecutionPlan.from_external_plan(request)
result = run_pipeline(plan=execution_plan, finalize=finalize_spec, options=options)
```

**Target files**
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/external_plans.py`
- `src/codeintel/build/tabular/plan_ops.py`
- `src/codeintel/build/tabular/substrait_ops.py`
- `src/codeintel/build/tabular/datafusion_ops.py`

**Implementation checklist**
- [ ] Register default external plan runners in the core columnar entrypoints.
- [ ] Ensure runners return Plan or ReaderThunk (no raw tables).
- [ ] Replace direct external plan reads with ExecutionPlan + run_pipeline.

---

## Scope 06 - Telemetry and Manifest Consolidation
**Goal**
Standardize scan telemetry and run manifest emission for graph pipelines,
including ordering and determinism metadata.

**Code patterns**
```python
from codeintel.core.columnar.arrowdsl import PipelineRunOptions, run_pipeline
from codeintel.core.columnar.run_manifest import run_manifest_options_for_context
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec

telemetry = scan_telemetry_for_queryspec(dataset, spec=query_spec)
options = PipelineRunOptions(
    ctx=execution_ctx,
    scan_telemetry=telemetry,
    manifest_dir=manifest_dir,
    manifest_options=run_manifest_options_for_context(
        ctx=execution_ctx,
        ordering=plan.ordering,
        scan_telemetry=telemetry,
    ),
)
result = run_pipeline(plan=plan, finalize=finalize_spec, options=options)
```

**Target files**
- `src/codeintel/core/columnar/run_manifest.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/graphs/assembly/finalize.py`
- `src/codeintel/build/graphs/validation/runner.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py`

**Implementation checklist**
- [ ] Emit plan-phase timings and determinism metadata in run manifests.
- [ ] Propagate scan telemetry from dataset scans into PipelineRunOptions.
- [ ] Ensure graph finalize wrappers accept manifest options and telemetry.

---

## Scope 07 - Rustworkx Boundary Contracts
**Goal**
Ensure rustworkx graph assembly consumes finalized, canonical-tier tables with
explicit ordering, and keeps graph algorithms isolated from plan/kernel details.

**Code patterns**
```python
from codeintel.build.graphs.assembly.finalize import finalize_graph_plan
from codeintel.build.graphs.rx.build_from_edges import EdgeBuildSpec, build_store_from_edge_tuples

finalized = finalize_graph_plan(
    plan,
    table_key="graph.call_graph_edges",
    determinism="canonical",
    ctx=execution_ctx,
    artifacts=artifacts,
).good
store = build_store_from_edge_tuples(
    iter_tuples(finalized.to_reader(), columns=("caller_goid_h128", "callee_goid_h128")),
    spec=EdgeBuildSpec(directed=True, weight_policy=weight_policy, numeric_policy=numeric_policy),
)
```

**Target files**
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/assembly/collectors.py`
- `src/codeintel/build/graphs/assembly/finalize.py`
- `src/codeintel/build/graphs/rx/build_from_edges.py`
- `src/codeintel/build/graphs/compute/metrics/*`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`

**Implementation checklist**
- [ ] Always finalize graph inputs before rustworkx ingestion.
- [ ] Use canonical ordering keys for stable node and edge ordering.
- [ ] Keep algorithm modules rustworkx-only (no plan or kernel logic inside).

---

## Sequencing Recommendation
1) Scope 01 - Plan builder unification sweep
2) Scope 02 - Kernel wrappers and row-changing ops consolidation
3) Scope 03 - Ordering and determinism propagation
4) Scope 04 - Schema-driven defaults
5) Scope 05 - External plan runner unification
6) Scope 06 - Telemetry and manifest consolidation
7) Scope 07 - Rustworkx boundary contracts

## Validation Gates (Guardrails Deferred)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted pytest subsets for graph/columnar modules once tests resume
