# Core Columnar Acero DSL Centralization Followup Plan

## Objective
Extend the centralized Acero/DSL architecture to a maximally deduplicated, modular,
and high-performance compute surface, building on the completed scope in:
- `plans/core_columnar_acero_dsl_centralization_plan.md`
- `docs/python_library_reference/arrow_acero_dsl_guide.md`

## Design Principles (Non-Negotiable)
- Plan lane vs kernel lane stays explicit, with plan metadata preserved end-to-end.
- QuerySpec remains the single source of truth for scan + projection + predicates.
- ExecutionContext and RuntimeProfile own determinism, threading, and provenance.
- Materialization is always mediated by ExecutionPlan + run_pipeline.
- Schema metadata is the only authority for finalize policies and canonical ordering.

---

## Scope 01 - Single Plan Construction API (Plan Builder)
**Goal**
Create a single, reusable plan-construction surface that encapsulates the
canonical Acero node patterns and reduces ad-hoc plan building.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.plan_builder import (
    build_grouped_rollup_plan,
    build_snapshot_plan,
)

execution_ctx = resolve_execution_context(None)
plan = build_snapshot_plan(
    table=table,
    spec=query_spec,
    ctx=execution_ctx,
)
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
- `src/codeintel/core/columnar/plan_builder.py` (new)
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/build/analytics/utilities/snapshot.py`
- `src/codeintel/build/analytics/scip_diagnostics_rollups.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/engine/datasets.py`

**Checklist**
- [ ] Introduce `plan_builder` with canonical plan fragments (scan, filter, project,
      aggregate, join).
- [ ] Replace remaining ad-hoc plan composition with plan_builder helpers in
      analytics + graph paths.
- [ ] Keep QuerySpec as the sole input for scan/filter/projection construction.

---

## Scope 02 - Kernel Lane Wrappers for Row-Changing Ops
**Goal**
Centralize explode, dedupe, and rollup patterns into reusable kernel wrappers
that ensure join-safe projections and deterministic tie handling.

**Code pattern**
```python
from codeintel.core.columnar.plan_kernels import (
    group_by_max_join_back,
    stable_dedupe_with_ties,
)

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
- `src/codeintel/core/columnar/plan_kernels.py` (new)
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py`
- `src/codeintel/build/analytics/semantic_roles/core.py`

**Checklist**
- [ ] Create kernel wrappers for order-independent dedupe, explode, and rollups.
- [ ] Ensure wrappers always use join-safe projections or explicit allowlists.
- [ ] Enforce deterministic tie handling via stable sort + hash ordinal fallback.

---

## Scope 03 - Ordering and Determinism as First-Class Plan Metadata
**Goal**
Make ordering propagation explicit and enforce canonical ordering rules for all
materialization boundaries.

**Code pattern**
```python
from codeintel.core.columnar.plan_ops import Plan

plan = Plan.table(table).filter(expr)
plan = plan.order_by(sort_keys=(("repo", "ascending"), ("commit", "ascending")))
```

**Target files**
- `src/codeintel/core/columnar/ordering.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/storage/queries/parquet.py`
- `src/codeintel/build/graphs/engine/datasets.py`

**Checklist**
- [ ] Add ordering propagation rules for filter/project/join/aggregate nodes.
- [ ] Require explicit ordering for canonical determinism at finalize boundaries.
- [ ] Default to unordered when plan nodes cannot preserve ordering semantics.

---

## Scope 04 - External Plan Runner Unification
**Goal**
Centralize external plan registration and ensure fallback behavior preserves the
plan lane vs reader lane boundary.

**Code pattern**
```python
from codeintel.core.columnar.plan_ops import (
    ExternalPlanRequest,
    ExternalPlanSpec,
    register_default_external_plan_runners,
    run_external_plan,
)

register_default_external_plan_runners()
request = ExternalPlanRequest(
    spec=ExternalPlanSpec(engine="substrait", payload=substrait_bytes),
    dataset=dataset,
    filter_expr=None,
    columns=None,
    scan_options=None,
    use_threads=None,
)
reader = run_external_plan(request)
```

**Target files**
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/external_plans.py` (new, optional)
- `src/codeintel/build/tabular/plan_ops.py`
- `src/codeintel/build/tabular/substrait_ops.py`
- `src/codeintel/build/tabular/datafusion_ops.py`

**Checklist**
- [ ] Move default external plan runner registration into core columnar.
- [ ] Ensure external runners return Plan or ReaderThunk, never raw tables.
- [ ] Provide a single registry surface for external plan usage.

---

## Scope 05 - Schema-Driven Plan Defaults
**Goal**
Extend schema metadata to drive plan defaults (projection, ordering, join-safe
columns) so compute policies are not redefined in call sites.

**Code pattern**
```python
from codeintel.core.columnar.plan_builder import plan_from_schema_defaults
from codeintel.core.schemas.service import get_schema_service

schema_service = get_schema_service()
plan = plan_from_schema_defaults(
    schema_service=schema_service,
    table_key="core.scip_diagnostics",
    dataset=dataset,
    ctx=execution_ctx,
)
```

**Target files**
- `src/codeintel/core/schemas/primitives.py`
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/core/columnar/finalize_ops.py`

**Checklist**
- [ ] Add schema metadata fields for default projection + join-safe allowlists.
- [ ] Drive plan defaults from schema metadata (not ad-hoc caller logic).
- [ ] Ensure serialization preserves new schema metadata fields.

---

## Scope 06 - Performance and Telemetry Consolidation
**Goal**
Capture plan-level metrics and scan telemetry uniformly, with deterministic
execution profiles applied at all entrypoints.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import PipelineRunOptions, run_pipeline
from codeintel.core.columnar.run_manifest import run_manifest_options_for_context

options = PipelineRunOptions(
    ctx=execution_ctx,
    manifest_dir=manifest_dir,
    manifest_options=run_manifest_options_for_context(ctx=execution_ctx),
)
result = run_pipeline(plan=plan, finalize=finalize_spec, options=options)
```

**Target files**
- `src/codeintel/core/columnar/run_manifest.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/graphs/validation/runner.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py`

**Checklist**
- [ ] Add plan-level timing hooks to run manifests.
- [ ] Include runtime profile, determinism tier, and ordering metadata.
- [ ] Standardize scan telemetry collection across all dataset scans.

---

## Sequencing Recommendation
1) Scope 01 (plan builder and migration sweep)
2) Scope 02 (kernel wrappers for row-changing ops)
3) Scope 03 (ordering and determinism propagation)
4) Scope 05 (schema-driven defaults)
5) Scope 04 (external plan runner unification)
6) Scope 06 (performance + telemetry consolidation)

## Validation Gates (Guardrails Deferred)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted pytest subsets for modified modules (columnar, queries, build pipelines)
