# Ingestion Acero DSL Centralization Alignment Implementation Plan

## Objective
Align ingestion pipelines with the centralized Acero/DSL core (plan builder, kernel
lane, schema-driven defaults, ordering metadata, and telemetry) while preserving
ingestion-specific carve-outs for parsing and tool execution. The goal is maximum
modularity, performance, determinism, and maintainability across ingestion.

This plan unifies ingestion with:
- `plans/core_columnar_acero_dsl_centralization_followup_plan.md`
- `plans/ingestion_acero_dsl_centralization_followup_implementation_plan.md`

## Design Principles (Non-Negotiable)
- Plan lane vs kernel lane stays explicit and metadata is preserved end-to-end.
- QuerySpec is the single source of truth for scan + projection + predicates.
- ExecutionContext and RuntimeProfile own determinism, threading, and provenance.
- Materialization is always mediated by ExecutionPlan + run_pipeline.
- Schema metadata is the only authority for finalize policies and canonical ordering.
- Ingestion persistence is dataset-first (Parquet), not storage adapters.

---

## Scope 01 - Plan Builder Unification for Ingestion
**Goal**
Ensure ingestion uses a single plan-construction surface and avoids ad-hoc
Plan.table/Plan.scan usage.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.plan_builder import build_snapshot_plan
from codeintel.core.columnar.queryspec import QuerySpec

ctx = resolve_execution_context(None)
spec = QuerySpec(
    projection=("repo", "commit", "path", "language", "module"),
    predicate=None,
    pushdown_predicate=None,
)
plan = build_snapshot_plan(table=table, spec=spec, ctx=ctx)
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan),
    finalize=finalize_spec_for_table("core.modules", mode="tolerant"),
    options=PipelineRunOptions(ctx=ctx),
)
```

**Target files**
- `src/codeintel/build/hamilton/native/ingestion/pipelines.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- `src/codeintel/ingestion/compute/plan_surface.py`

**Checklist**
- [ ] Replace direct Plan.table/Plan.scan usage with plan_builder helpers.
- [ ] Route all ingestion plan construction through plan_surface facade.
- [ ] Ensure ordering metadata is preserved through ExecutionPlan.

---

## Scope 02 - QuerySpec Control Plane + Schema Defaults
**Goal**
Make QuerySpec the only scan surface and apply schema-driven defaults for
projection and join-safe allowlists.

**Code pattern**
```python
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.plan_builder import SchemaPlanDefaultsRequest, plan_from_schema_defaults
from codeintel.core.schemas.service import get_schema_service

schema_service = get_schema_service()
ctx = resolve_execution_context(None)
plan = plan_from_schema_defaults(
    schema_service=schema_service,
    request=SchemaPlanDefaultsRequest(
        table_key="core.scip_diagnostics",
        dataset=dataset,
        predicate=spec.predicate,
        columns=spec.projection.columns(),
        ctx=ctx,
    ),
)
```

**Target files**
- `src/codeintel/ingestion/compute/queryspecs.py`
- `src/codeintel/ingestion/compute/plan_surface.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/schemas/output_registry.py`

**Checklist**
- [ ] Ensure ingestion QuerySpec projection/predicate is built once, centrally.
- [ ] Replace ad-hoc projection defaults with schema-driven defaults.
- [ ] Enforce QuerySpec as the only scan entrypoint in ingestion.

---

## Scope 03 - Kernel Lane Consolidation for Row-Changing Ops
**Goal**
Route all row-count-changing operations through kernel wrappers with join-safe
projection and deterministic tie handling.

**Code pattern**
```python
from codeintel.core.columnar.explode_ops import ExplodeSpec
from codeintel.core.columnar.plan_kernels import explode_edges_for_join

exploded = explode_edges_for_join(
    table=edges,
    spec=ExplodeSpec(
        src_col="src_id",
        dst_list_col="dst_ids",
        aligned_list_cols=("dst_spans",),
        repeat_cols=("repo", "commit", "rel_path"),
        null_list_policy="error",
    ),
    table_key="core.syntax_edges",
    schema_service=schema_service,
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
- `src/codeintel/core/columnar/plan_kernels.py`
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`

**Checklist**
- [ ] Replace raw explode helpers with `explode_edges_for_join`.
- [ ] Use kernel rollups for aggregate paths that must be ordered.
- [ ] Route dedupe through `stable_dedupe_with_ties` and `group_by_max_join_back`.
- [ ] Enforce join-safe projections using schema metadata allowlists.

---

## Scope 04 - Ordering and Determinism as First-Class Metadata
**Goal**
Make ordering transitions explicit and enforce canonical ordering at finalize
boundaries for ingestion outputs.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table

finalize = finalize_spec_for_table("core.syntax_edges", mode="tolerant")
result = run_pipeline(
    plan=ExecutionPlan.from_reader(reader),
    finalize=finalize,
)
```

**Target files**
- `src/codeintel/core/columnar/ordering.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`

**Checklist**
- [ ] Propagate ordering metadata across joins/aggregates/order_by.
- [ ] Enforce canonical order_by when determinism is canonical.
- [ ] Require finalize boundaries to include schema-provided canonical keys.

---

## Scope 05 - Schema-Driven Plan Defaults
**Goal**
Extend schema metadata so ingestion never encodes defaults in call sites.

**Code pattern**
```python
from codeintel.core.schemas.primitives import PlanPolicy

TableSchema(
    ...,
    plan_policy=PlanPolicy(
        default_projection=("repo", "commit", "path"),
        join_safe_columns=("repo", "commit", "path"),
    ),
)
```

**Target files**
- `src/codeintel/core/schemas/primitives.py`
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/queryspec.py`

**Checklist**
- [ ] Expand `PlanPolicy` for ingestion tables with list payloads.
- [ ] Ensure schema serialization retains plan policy fields.
- [ ] Replace call-site defaults with `plan_from_schema_defaults`.

---

## Scope 06 - Telemetry + Run Manifest Unification
**Goal**
Ensure every ingestion finalize emits deterministic manifests with scan telemetry.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import PipelineRunOptions, run_pipeline
from codeintel.core.columnar.run_manifest import run_manifest_options_for_context
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec

telemetry = scan_telemetry_for_queryspec(dataset, spec=query_spec)
options = PipelineRunOptions(
    ctx=ctx,
    scan_telemetry=telemetry,
    manifest_dir=manifest_dir,
    manifest_options=run_manifest_options_for_context(
        ctx=ctx,
        ordering=plan.ordering,
        scan_telemetry=telemetry,
    ),
)
result = run_pipeline(plan=plan, finalize=finalize_spec, options=options)
```

**Target files**
- `src/codeintel/build/hamilton/native/ingestion/manifesting.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- `src/codeintel/core/columnar/run_manifest.py`
- `src/codeintel/core/columnar/streaming.py`

**Checklist**
- [ ] Attach scan telemetry at all dataset scan entrypoints.
- [ ] Emit run manifests in every ingestion finalize path.
- [ ] Include determinism tier, ordering state, and profile name in manifests.

---

## Scope 07 - Ingestion Preprocessing Pipeline Centralization
**Goal**
Keep ingestion row cleanup in plan lane and avoid ad-hoc compute in DAG nodes.

**Code pattern**
```python
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_ops import materialize_plan
from codeintel.core.columnar.expr_vocab import E

plan = build_table_plan(
    table=table,
    options=TablePlanOptions(filter_expr=E.and_(E.is_valid("repo"), E.is_valid("commit"))),
)
cleaned = materialize_plan(plan, ctx=ctx)
```

**Target files**
- `src/codeintel/build/hamilton/native/ingestion/pipelines.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`

**Checklist**
- [ ] Replace `build.tabular.expr_vocab` usage with core `expr_vocab` in ingestion.
- [ ] Keep row-cleanup steps in plan lane via plan_builder helpers.
- [ ] Ensure ordering metadata is preserved when cleaning rows.

---

## Scope 08 - Finalize-First Orchestration and Reader Boundaries
**Goal**
Ensure compute stages return readers only and finalization happens centrally.

**Code pattern**
```python
readers = {
    "core.ast_nodes": collectors.ast_nodes.to_reader(),
    "core.ast_metrics": collectors.metrics.to_reader(),
}
finalized = {
    key: finalize_ingest_reader_with_manifest(
        env=env,
        table_key=key,
        reader=reader,
        target_name="ast",
    )
    for key, reader in readers.items()
}
```

**Target files**
- `src/codeintel/ingestion/compute/ast_extract.py`
- `src/codeintel/ingestion/compute/cst_extract.py`
- `src/codeintel/ingestion/compute/tree_sitter_index.py`
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`

**Checklist**
- [ ] Keep ingestion compute pure: return readers only.
- [ ] Centralize finalization and manifest emission in targets.
- [ ] Avoid `to_table()`/`read_all()` before finalize boundaries.

---

## Sequencing Recommendation
1) Scope 01 + 02 (plan builder + QuerySpec control plane)
2) Scope 03 (kernel lane consolidation)
3) Scope 04 + 05 (ordering/determinism + schema defaults)
4) Scope 06 + 07 + 08 (telemetry, preprocessing, finalize orchestration)

## Validation Gates (Non-Pytest)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Runtime validation via run manifests + scan telemetry for modified paths.
