# Analytics Core Columnar Centralization Followup Implementation Plan

## Objective
Align the analytics compute footprint with the centralized core columnar Acero/DSL
surface (plan lane vs kernel lane), while preserving analytics functionality and
extensibility. This plan translates the core followup design into concrete
analytics migrations and guardrails.

## Inputs (Reference Plans)
- `plans/core_columnar_acero_dsl_centralization_followup_plan.md`
- `plans/analytics_core_columnar_centralization_alignment_plan.md`
- `docs/python_library_reference/arrow_acero_dsl_guide.md`

## Design Principles (Non-Negotiable)
- Plan lane vs kernel lane stays explicit, with plan metadata preserved end-to-end.
- QuerySpec is the single source of truth for scan + projection + predicates.
- ExecutionContext and RuntimeProfile own determinism, threading, and provenance.
- Materialization is always mediated by ExecutionPlan + run_pipeline.
- Schema metadata is the only authority for finalize policies and canonical ordering.

---

## Scope 01 - Centralized Plan Builder Adoption (Analytics)
**Goal**
Route all analytics plan construction through plan_builder + QuerySpec so
projection/predicate logic is centralized and schema policies are applied.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.plan_builder import build_snapshot_plan
from codeintel.core.columnar.queryspec import QuerySpec

execution_ctx = resolve_execution_context(ctx)
plan = build_snapshot_plan(table=table, spec=query_spec, ctx=execution_ctx)
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan),
    finalize=finalize_spec_for_table("analytics.config_graph_metrics_keys", mode="tolerant"),
    options=PipelineRunOptions(ctx=execution_ctx),
)
```

**Target files**
- `src/codeintel/build/analytics/utilities/snapshot.py`
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/analytics/scip_diagnostics_rollups.py`
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/graphs/config_references.py`
- `src/codeintel/build/analytics/functions/function_effects.py`
- `src/codeintel/build/analytics/functions/function_contracts.py`
- `src/codeintel/build/analytics/functions/metrics.py`
- `src/codeintel/build/analytics/semantic_roles/core.py`
- `src/codeintel/build/analytics/entrypoints/core.py`
- `src/codeintel/build/analytics/compute/data_models/usage.py`
- `src/codeintel/build/analytics/compute/dependencies/compute.py`
- `src/codeintel/build/analytics/py_cpg_quality_report.py`

**Implementation checklist**
- [ ] Replace ad-hoc plan construction with `build_snapshot_plan` or
      `plan_from_schema_defaults` when scanning datasets.
- [ ] Ensure `QuerySpec` is the sole source of predicate/projection.
- [ ] Remove direct `Plan.table(...)` or `Plan.scan(...)` usage in analytics modules.

---

## Scope 02 - Kernel Lane Wrappers for Row-Changing Ops
**Goal**
Centralize explode/dedupe/rollup patterns into kernel wrappers so analytics
modules only assemble inputs and decode outputs.

**Code patterns**
```python
from codeintel.core.columnar.plan_kernels import stable_dedupe_with_ties

deduped = stable_dedupe_with_ties(
    table=table,
    keys=("repo", "commit", "path"),
    order_by=(("score", "descending"),),
    tie_breakers=(("rel_path", "ascending"),),
)
```

```python
from codeintel.core.columnar.plan_kernels import grouped_rollup_table

rollup = grouped_rollup_table(
    table=table,
    keys=("repo", "commit"),
    aggregates=(("severity", "count", None, "diagnostic_count"),),
    order_by=(("repo", "ascending"), ("commit", "ascending")),
)
```

**Target files**
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/build/analytics/subsystems/affinity.py`
- `src/codeintel/build/analytics/semantic_roles/core.py`
- `src/codeintel/build/analytics/functions/function_effects.py`
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`

**Implementation checklist**
- [ ] Replace analytics-local dedupe with `stable_dedupe_with_ties`.
- [ ] Replace analytics-local rollups with `grouped_rollup_table`.
- [ ] Apply schema allowlists or join-safe projections in kernel wrappers.

---

## Scope 03 - Ordering + Determinism via Schema Policy
**Goal**
Ensure determinism is driven by schema finalize policy, not ad-hoc sorting; keep
`order_by` only when required for list semantics.

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
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/functions/function_effects.py`
- `src/codeintel/build/analytics/functions/metrics.py`

**Implementation checklist**
- [ ] Remove manual sorting used only for output determinism.
- [ ] Keep `order_by` only where list ordering is semantically required.
- [ ] Ensure outputs are finalized via schema policy (`finalize_analytics_result`).

---

## Scope 04 - Schema-Driven Defaults (Projection + Join-Safe)
**Goal**
Drive default projection and join-safe allowlists from schema metadata rather
than call sites.

**Code patterns**
```python
from codeintel.core.columnar.plan_builder import plan_from_schema_defaults
from codeintel.core.columnar.plan_builder import SchemaPlanDefaultsRequest
from codeintel.core.schemas.service import get_schema_service

schema_service = get_schema_service()
plan = plan_from_schema_defaults(
    schema_service=schema_service,
    request=SchemaPlanDefaultsRequest(
        table_key="analytics.config_references",
        dataset=dataset,
        predicate=spec.predicate,
        columns=spec.projection.columns(),
        ctx=execution_ctx,
    ),
)
```

```python
from codeintel.core.schemas.primitives import PlanPolicy

TableSchema(
    ...,
    plan_policy=PlanPolicy(
        default_projection=("repo", "commit", "config_path", "key", "extras"),
        join_safe_columns=("repo", "commit", "config_path", "key"),
    ),
)
```

**Target files**
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/build/analytics/utilities/snapshot.py`
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/analytics/utilities/catalogs.py`

**Implementation checklist**
- [ ] Expand `PlanPolicy` coverage for analytics tables with list payloads or joins.
- [ ] Use schema-driven default projection in snapshot helpers.
- [ ] Ensure schema serialization preserves plan policy fields.

---

## Scope 05 - ExecutionContext Unification (Runtime -> Columnar)
**Goal**
Centralize conversion from runtime ExecutionContext to columnar ExecutionContext
and remove per-module conversion helpers.

**Code pattern**
```python
from codeintel.core.columnar.execution_context import resolve_columnar_context

execution_ctx = resolve_columnar_context(runtime_ctx)
plan = build_snapshot_plan(table=table, spec=spec, ctx=execution_ctx)
```

**Target files**
- `src/codeintel/core/columnar/execution_context.py` (add helper)
- `src/codeintel/build/analytics/utilities/snapshot.py`
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/graphs/config_references.py`
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/build/analytics/functions/function_effects.py`
- `src/codeintel/build/analytics/functions/metrics.py`
- `src/codeintel/build/analytics/compute/functions/goids.py`
- `src/codeintel/build/analytics/semantic_roles/core.py`

**Implementation checklist**
- [ ] Add a single `resolve_columnar_context` utility in core.
- [ ] Replace `_resolve_columnar_context` helpers in analytics modules.
- [ ] Keep runtime context support on analytics-facing APIs.

---

## Scope 06 - Telemetry + Manifest Unification
**Goal**
Ensure analytics pipelines emit consistent run manifests with ordering metadata
and scan telemetry.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import PipelineRunOptions, run_pipeline
from codeintel.core.columnar.run_manifest import run_manifest_options_for_context
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec

telemetry = scan_telemetry_for_queryspec(dataset, spec=query_spec)
options = PipelineRunOptions(
    ctx=execution_ctx,
    scan_telemetry=telemetry,
    manifest_options=run_manifest_options_for_context(
        ctx=execution_ctx,
        ordering=plan.ordering,
        scan_telemetry=telemetry,
    ),
)
result = run_pipeline(plan=ExecutionPlan.from_plan(plan), finalize=finalize, options=options)
```

**Target files**
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/analytics/utilities/datasets.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py`

**Implementation checklist**
- [ ] Attach scan telemetry to `PipelineRunOptions`.
- [ ] Include ordering/determinism metadata in run manifests.
- [ ] Persist finalize artifacts alongside analytics outputs.

---

## Sequencing Recommendation
1) Scope 01 (plan builder adoption)
2) Scope 02 (kernel wrappers for row-changing ops)
3) Scope 03 (ordering + determinism via schema policy)
4) Scope 04 (schema-driven defaults)
5) Scope 05 (ExecutionContext unification)
6) Scope 06 (telemetry + manifests)
