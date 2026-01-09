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

**Code patterns**
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
        table_key="core.scip_diagnostics",
        dataset=dataset,
        predicate=spec.predicate,
        columns=spec.projection.columns(),
        ctx=execution_ctx,
    ),
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
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/hamilton/native/ingestion/pipelines.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- `src/codeintel/build/hamilton/native/graphs/filter_helpers.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/goids.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`
- `src/codeintel/serving/semantic/engines/arrow_engine.py`
- `src/codeintel/storage/queries/parquet.py`

**Checklist**
- [ ] Replace direct `Plan.table`/`Plan.scan` usage with `build_table_plan`,
      `build_query_plan`, and `plan_from_schema_defaults`.
- [ ] Migrate `materialize_plan`/`Plan.to_table` paths to `ExecutionPlan` +
      `run_pipeline` to preserve ordering/determinism metadata.
- [ ] Ensure all scan entrypoints accept `QuerySpec` as the sole predicate/projection
      source and stop re-deriving projection from callers.
- [ ] Use schema-driven default projections via `SchemaPlanDefaultsRequest` for
      datasets with plan policies.

---

## Scope 02 - Kernel Lane Wrappers for Row-Changing Ops
**Goal**
Centralize explode, dedupe, and rollup patterns into reusable kernel wrappers
that ensure join-safe projections and deterministic tie handling.

**Code patterns**
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

```python
from codeintel.core.columnar.plan_kernels import explode_edges_for_join

exploded = explode_edges_for_join(
    table=edges,
    spec=ExplodeSpec(parent_column="parents", child_column="children"),
    allowed_columns=("repo", "commit", "edge_id", "parents", "children"),
    schema_allowlist=table_schema,
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
- `src/codeintel/core/columnar/plan_kernels.py` (new)
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py`
- `src/codeintel/build/analytics/semantic_roles/core.py`
- `src/codeintel/core/columnar/kernels.py`
- `src/codeintel/build/graphs/assembly/kernels.py`
- `src/codeintel/build/tabular/explode_ops.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`
- `src/codeintel/build/graphs/engine/views.py`

**Checklist**
- [ ] Replace raw explode helpers with `explode_edges_for_join` and schema-driven
      allowlists; retire duplicated wrappers in assembly/tabular kernels.
- [ ] Replace ad-hoc rollups (`table.group_by().aggregate(...)`) with
      `grouped_rollup_table` where ordering stability is required.
- [ ] Route order-independent dedupe through `stable_dedupe_with_ties` +
      `group_by_max_join_back` across graph/ingestion pipelines.
- [ ] Ensure all row-changing kernels apply join-safe projection or allowlists
      derived from schema metadata.

---

## Scope 03 - Ordering and Determinism as First-Class Plan Metadata
**Goal**
Make ordering propagation explicit and enforce canonical ordering rules for all
materialization boundaries.

**Code patterns**
```python
from codeintel.core.columnar.plan_ops import Plan

plan = Plan.table(table).filter(expr)
plan = plan.order_by(sort_keys=(("repo", "ascending"), ("commit", "ascending")))
```

```python
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table

finalize = finalize_spec_for_table("core.scip_diagnostics", mode="tolerant")
result = run_pipeline(plan=ExecutionPlan.from_plan(plan), finalize=finalize)
```

**Target files**
- `src/codeintel/core/columnar/ordering.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/storage/queries/parquet.py`
- `src/codeintel/build/graphs/engine/datasets.py`

**Checklist**
- [ ] Add explicit ordering propagation rules for join/aggregate/order_by to
      preserve ordering metadata and pipeline-breaker semantics.
- [ ] Enforce canonical ordering requirements at finalize boundaries when
      determinism is canonical, including explicit order_by when stable_sort_keys is empty.
- [ ] Remove unordered materialization paths in Parquet query utilities and
      dataset scan paths; require explicit ordering when requested by schema.
- [ ] Ensure plan builder applies canonical order_by when schema policy requires it.

---

## Scope 04 - External Plan Runner Unification
**Goal**
Centralize external plan registration and ensure fallback behavior preserves the
plan lane vs reader lane boundary.

**Code patterns**
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

```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan

plan = ExecutionPlan.from_external_plan(request)
result = run_pipeline(plan=plan, finalize=finalize_spec)
```

**Target files**
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/external_plans.py` (new, optional)
- `src/codeintel/build/tabular/plan_ops.py`
- `src/codeintel/build/tabular/substrait_ops.py`
- `src/codeintel/build/tabular/datafusion_ops.py`

**Checklist**
- [ ] Ensure default external plan runners are registered from core pipeline entrypoints.
- [ ] Standardize external plan runners to return ReaderThunk or ExecutionPlan
      (no raw `pa.Table` returns).
- [ ] Replace direct `run_external_plan` call sites with `ExecutionPlan.from_external_plan`
      and `run_pipeline` to preserve determinism/ordering metadata.

---

## Scope 05 - Schema-Driven Plan Defaults
**Goal**
Extend schema metadata to drive plan defaults (projection, ordering, join-safe
columns) so compute policies are not redefined in call sites.

**Code patterns**
```python
from codeintel.core.columnar.plan_builder import plan_from_schema_defaults
from codeintel.core.schemas.service import get_schema_service

schema_service = get_schema_service()
plan = plan_from_schema_defaults(
    schema_service=schema_service,
    request=SchemaPlanDefaultsRequest(
        table_key="core.scip_diagnostics",
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
        default_projection=("repo", "commit", "path", "severity"),
        join_safe_columns=("repo", "commit", "path"),
    ),
)
```

**Target files**
- `src/codeintel/core/schemas/primitives.py`
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/schemas/view_registry.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/validation/runner.py`
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/ingestion/compute/queryspecs.py`

**Checklist**
- [ ] Expand `PlanPolicy` coverage in `output_registry.py` for tables with
      list payloads, join-safe allowlists, and default projections.
- [ ] Replace ad-hoc projection defaults (e.g., snapshot utilities) with
      schema-driven defaults via `plan_from_schema_defaults`.
- [ ] Ensure derived/view schemas preserve `plan_policy` during schema overrides.
- [ ] Verify schema serialization preserves plan policy and join-safe allowlists.

---

## Scope 06 - Performance and Telemetry Consolidation
**Goal**
Capture plan-level metrics and scan telemetry uniformly, with deterministic
execution profiles applied at all entrypoints.

**Code patterns**
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

```python
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
```

**Target files**
- `src/codeintel/core/columnar/run_manifest.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/graphs/validation/runner.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py`
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/storage/queries/parquet.py`

**Checklist**
- [ ] Add plan-phase timings (scan/build/execute/finalize) to run manifests.
- [ ] Include runtime profile, determinism tier, and ordering metadata in every manifest.
- [ ] Standardize scan telemetry collection and propagation into
      `PipelineRunOptions` across all entrypoints.

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
