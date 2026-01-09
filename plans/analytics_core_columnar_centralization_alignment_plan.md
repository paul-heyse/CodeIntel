# Analytics Core Columnar Centralization Alignment Plan

## Objective
Align analytics compute with the centralized core columnar Acero/DSL surface while
preserving the goals in `plans/analytics_acero_dsl_intensification_plan.md`. This
plan maximizes plan-lane reuse, kernel-lane consolidation, schema-driven policy,
and unified telemetry across analytics modules.

## Inputs (Reference Plans)
- `plans/core_columnar_acero_dsl_centralization_followup_plan.md`
- `plans/analytics_acero_dsl_intensification_plan.md`
- `docs/python_library_reference/arrow_acero_dsl_guide.md`

## Design Principles (Non-Negotiable)
- Plan lane vs kernel lane stays explicit, with plan metadata preserved end-to-end.
- QuerySpec is the single source of truth for scan + projection + predicates.
- ExecutionContext and RuntimeProfile own determinism, threading, and provenance.
- Materialization is always mediated by ExecutionPlan + run_pipeline.
- Schema metadata is the only authority for finalize policies and canonical ordering.

---

## Scope 01 - Plan Builder Adoption for Analytics
**Goal**  
Replace ad-hoc plan construction in analytics with centralized plan_builder
helpers so QuerySpec remains authoritative.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.plan_builder import build_grouped_rollup_plan, build_snapshot_plan
from codeintel.core.columnar.queryspec import QuerySpec

execution_ctx = resolve_execution_context(ctx)
plan = build_snapshot_plan(table=table, spec=query_spec, ctx=execution_ctx)
plan = build_grouped_rollup_plan(
    plan,
    keys=("repo", "commit", "severity"),
    aggregates=(("severity", "count", None, "diagnostic_count"),),
)
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan),
    finalize=finalize_spec_for_table("analytics.scip_diagnostics_summary", mode="tolerant"),
    options=PipelineRunOptions(ctx=execution_ctx),
)
```

**Target files**
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/build/analytics/utilities/snapshot.py`
- `src/codeintel/build/analytics/scip_diagnostics_rollups.py`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/analytics/functions/*`

**Implementation checklist**
- [ ] Introduce or extend plan_builder helpers for snapshot + rollup patterns.
- [ ] Remove direct `Plan.table(...)` usage from analytics modules.
- [ ] Route all analytics plan construction through QuerySpec + plan_builder.

---

## Scope 02 - Unified Analytics Pipeline Runner (run_pipeline)
**Goal**  
Ensure analytics execution goes through a single runner that uses `run_pipeline`
and emits manifests/artifacts consistently.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.run_manifest import run_manifest_options_for_context
from codeintel.core.columnar.execution_context import resolve_execution_context

execution_ctx = resolve_execution_context(ctx)
finalize = finalize_spec_for_table(table_key, mode="tolerant")
options = PipelineRunOptions(
    ctx=execution_ctx,
    manifest_options=run_manifest_options_for_context(ctx=execution_ctx),
)
result = run_pipeline(plan=ExecutionPlan.from_plan(plan), finalize=finalize, options=options)
```

**Target files**
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/analytics/utilities/datasets.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py`
- `src/codeintel/build/analytics/py_cpg_quality_report.py`
- `src/codeintel/build/analytics/scip_diagnostics_rollups.py`

**Implementation checklist**
- [ ] Use `run_pipeline` inside the analytics runner (no direct `plan.to_reader`).
- [ ] Persist finalize artifacts (errors/alignment/stats) in analytics outputs.
- [ ] Ensure manifests include execution metadata and ordering/determinism.

---

## Scope 03 - Kernel Lane Wrappers for Row-Changing Ops
**Goal**  
Centralize dedupe/explode/rollup logic in kernel wrappers so analytics modules
only assemble inputs and decode outputs.

**Code pattern**
```python
from codeintel.core.columnar.plan_kernels import group_by_max_join_back, stable_dedupe_with_ties

deduped = stable_dedupe_with_ties(
    table=table,
    keys=("repo", "commit", "path"),
    order_by=(("score", "descending"),),
    tie_breakers=(("rel_path", "ascending"),),
)
winner_rows = group_by_max_join_back(
    table=table,
    key_columns=("repo", "commit"),
    score_column="score",
)
```

**Target files**
- `src/codeintel/core/columnar/plan_kernels.py`
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/build/analytics/subsystems/affinity.py`
- `src/codeintel/build/analytics/semantic_roles/core.py`

**Implementation checklist**
- [ ] Add kernel wrappers for stable dedupe, explode, and rollup patterns.
- [ ] Replace analytics-local dedupe/explode with kernel wrappers.
- [ ] Ensure wrappers enforce join-safe projections or schema allowlists.

---

## Scope 04 - Ordering and Determinism via Schema Policy
**Goal**  
Use ordering only for semantic list ordering; determinism comes exclusively
from schema-driven finalize policy.

**Code pattern**
```python
from codeintel.build.analytics.utilities.finalize import finalize_analytics_result

result = finalize_analytics_result(table_key, table)
return result.good
```

**Target files**
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`
- `src/codeintel/build/analytics/cfg_dfg/cfg_core.py`
- `src/codeintel/build/analytics/cfg_dfg/dfg_core.py`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/analytics/functions/*`

**Implementation checklist**
- [ ] Remove ad-hoc `order_by`/sorting used only for determinism.
- [ ] Keep `order_by` only when list ordering affects semantics.
- [ ] Ensure outputs always flow through finalize helpers.

---

## Scope 05 - Schema-Driven Defaults (Projection + Join-Safety)
**Goal**  
Drive projection lists, join-safe allowlists, and canonical ordering from schema
metadata instead of call-site logic.

**Code pattern**
```python
from codeintel.core.columnar.plan_builder import plan_from_schema_defaults
from codeintel.core.schemas.service import get_schema_service

schema_service = get_schema_service()
plan = plan_from_schema_defaults(
    schema_service=schema_service,
    table_key="analytics.scip_diagnostics_summary",
    dataset=dataset,
    ctx=execution_ctx,
)
```

**Target files**
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/build/analytics/utilities/snapshot.py`
- `src/codeintel/build/analytics/utilities/catalogs.py`

**Implementation checklist**
- [ ] Add schema metadata for default projections and join-safe allowlists.
- [ ] Use schema defaults in analytics snapshot builders.
- [ ] Ensure serialization preserves new metadata fields.

---

## Scope 06 - Rowset-First Analytics Boundaries
**Goal**  
Standardize rowset aggregation using Plan.aggregate(list) and decode lists only
at the final boundary (graph/AST construction).

**Code pattern**
```python
plan = build_snapshot_plan(table=table, spec=query_spec, ctx=execution_ctx)
plan = plan.aggregate(
    keys=[E.field("function_goid_h128")],
    aggregates=[
        ("src_id", "list", None, "src_id"),
        ("dst_id", "list", None, "dst_id"),
    ],
)
plan = plan.order_by(sort_keys=[("function_goid_h128", "ascending")])
rowset = ExecutionPlan.from_plan(plan).to_table(ctx=execution_ctx)
```

**Target files**
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`
- `src/codeintel/build/analytics/cfg_dfg/cfg_core.py`
- `src/codeintel/build/analytics/cfg_dfg/dfg_core.py`
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/build/analytics/subsystems/affinity.py`

**Implementation checklist**
- [ ] Use list-aggregate rowsets for adjacency and worklists.
- [ ] Apply ordering only when list semantics require it.
- [ ] Decode lists only at the final graph/AST boundary.

---

## Scope 07 - Telemetry and Run Manifest Unification
**Goal**  
Ensure analytics runs emit consistent telemetry, ordering metadata, and
determinism tier via run manifests.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import PipelineRunOptions, run_pipeline
from codeintel.core.columnar.run_manifest import run_manifest_options_for_context

options = PipelineRunOptions(
    ctx=execution_ctx,
    manifest_options=run_manifest_options_for_context(ctx=execution_ctx),
)
result = run_pipeline(plan=ExecutionPlan.from_plan(plan), finalize=finalize, options=options)
```

**Target files**
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/analytics/utilities/datasets.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py`

**Implementation checklist**
- [ ] Emit run manifests for all analytics pipelines.
- [ ] Include ordering/determinism metadata in manifests.
- [ ] Ensure scan telemetry is collected via QuerySpec pushdown.

---

## Sequencing Recommendation
1) Scope 01 (plan_builder adoption)
2) Scope 02 (runner consolidation via run_pipeline)
3) Scope 03 (kernel wrappers for row-changing ops)
4) Scope 04 (ordering + determinism via finalize_policy)
5) Scope 05 (schema-driven defaults)
6) Scope 06 (rowset boundaries)
7) Scope 07 (telemetry unification)
