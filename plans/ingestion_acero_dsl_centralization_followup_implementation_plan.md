# Ingestion Acero DSL Centralization Followup Implementation Plan

## Objective
Align ingestion pipelines with the centralized Acero/DSL core while preserving
ingestion-specific carve-outs for parsing, and achieve maximum modularity,
performance, determinism, and maintainability.

This plan unifies ingestion with:
- `plans/core_columnar_acero_dsl_centralization_followup_plan.md`
- `plans/ingestion_acero_dsl_unified_best_in_class_implementation_plan.md`

## Design Principles (Non-Negotiable)
- Plan lane vs kernel lane stays explicit, with plan metadata preserved end-to-end.
- QuerySpec remains the single source of truth for scan, projection, predicates.
- ExecutionContext and RuntimeProfile own determinism, threading, provenance.
- Materialization is always mediated by ExecutionPlan + run_pipeline.
- Schema metadata drives finalize policies and canonical ordering.
- Ingestion persistence is dataset-first (Parquet), not storage adapters.

---

## Scope 01 - Plan Builder Adoption for Ingestion
**Goal**
Route all ingestion plan construction through a single core plan builder surface
so ingestion does not build plans ad hoc.

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
plan = build_snapshot_plan(dataset=dataset, spec=spec, ctx=ctx)
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan),
    finalize=finalize_spec_for_table("core.modules", mode="tolerant"),
    options=PipelineRunOptions(ctx=ctx),
)
```

**Target files**
- `src/codeintel/core/columnar/plan_builder.py` (new)
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/ingestion/compute/plan_surface.py` (new or existing facade)
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`

**Checklist**
- [ ] Introduce `plan_builder` with canonical plan fragments (scan, filter, project, join).
- [ ] Update ingestion facade to call plan_builder instead of custom plan assembly.
- [ ] Ensure plan ordering metadata is preserved through ExecutionPlan.

---

## Scope 02 - QuerySpec Control Plane + Ingestion Facade
**Goal**
Standardize ingestion scans on QuerySpec only, with ingestion-specific defaults
coming from schema metadata and runtime profiles.

**Code pattern**
```python
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.plan_builder import plan_from_schema_defaults
from codeintel.core.schemas.service import get_schema_service

schema_service = get_schema_service()
ctx = ExecutionContext()
plan = plan_from_schema_defaults(
    schema_service=schema_service,
    table_key="core.modules",
    dataset=dataset,
    ctx=ctx,
)
```

**Target files**
- `src/codeintel/ingestion/compute/base.py`
- `src/codeintel/ingestion/compute/plan_surface.py`
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`

**Checklist**
- [ ] Ensure ingestion QuerySpec projection/predicate is built once, centrally.
- [ ] Remove ad hoc scoping/projection in ingestion call sites.
- [ ] Enforce QuerySpec as the only scan entrypoint.

---

## Scope 03 - Kernel Lane Consolidation + Columnar Buffers
**Goal**
Move all row-count-changing transforms into kernel wrappers and standardize row
assembly on columnar buffers or batch collectors.

**Code pattern**
```python
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges_with_aligned_lists
from codeintel.core.columnar.rows import columnar_batch_collector_for_table_key

spec = ExplodeSpec(
    src_col="src_id",
    dst_list_col="callee_ids",
    aligned_list_cols=("callsite_spans",),
    repeat_cols=("repo", "commit", "rel_path"),
    null_list_policy="error",
)
result = explode_edges_with_aligned_lists(table, spec=spec)

collector = columnar_batch_collector_for_table_key("core.syntax_nodes", batch_size=4096)
collector.append({"repo": repo, "commit": commit, "node_id": node_id, "kind": kind})
reader = collector.to_reader()
```

**Target files**
- `src/codeintel/core/columnar/plan_kernels.py` (new)
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/ingestion/compute/ast_extract.py`
- `src/codeintel/ingestion/compute/cst_extract.py`
- `src/codeintel/ingestion/compute/tree_sitter_index.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`

**Checklist**
- [ ] Replace `pa.Table.from_pylist` and row loops with columnar collectors.
- [ ] Use kernel wrappers for explode/dedupe/rollup patterns.
- [ ] Enforce list-aligned validation and null list policies consistently.

---

## Scope 04 - Ordering and Determinism as First-Class Metadata
**Goal**
Make ordering transitions explicit and require canonical order keys at finalize
boundaries for ingestion outputs.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.finalize_ops import FinalizeSpec

ctx = ExecutionContext()
finalize_spec = FinalizeSpec(
    table_key="core.syntax_edges",
    mode="tolerant",
    order_by=(("repo", "ascending"), ("commit", "ascending"), ("src_id", "ascending")),
    emit_artifacts=True,
)
result = run_pipeline(
    plan=ExecutionPlan.from_reader(reader),
    finalize=finalize_spec,
    options=PipelineRunOptions(ctx=ctx),
)
```

**Target files**
- `src/codeintel/core/columnar/ordering.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/ingestion/compute/base.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`

**Checklist**
- [ ] Propagate ordering metadata through Plan and ExecutionPlan.
- [ ] Require canonical order keys for determinism tiers.
- [ ] Ensure finalize artifacts always emit in tolerant mode.

---

## Scope 05 - Schema-Driven Plan Defaults
**Goal**
Drive projection, join-safe columns, and canonical ordering from schema metadata
so ingestion call sites never encode defaults directly.

**Code pattern**
```python
from codeintel.core.columnar.plan_builder import plan_from_schema_defaults
from codeintel.core.schemas.service import get_schema_service

schema_service = get_schema_service()
plan = plan_from_schema_defaults(
    schema_service=schema_service,
    table_key="core.file_state",
    dataset=dataset,
    ctx=ctx,
)
```

**Target files**
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/schemas/primitives.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/queryspec.py`

**Checklist**
- [ ] Add schema metadata fields for default projection and join-safe allowlists.
- [ ] Update plan_builder to consume schema defaults.
- [ ] Ensure serialization persists new schema metadata.

---

## Scope 06 - Telemetry and Run Manifest Unification
**Goal**
Capture plan-level timing, ordering, determinism, and scan telemetry for all
ingestion runs without pytest.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import PipelineRunOptions, run_pipeline
from codeintel.core.columnar.run_manifest import run_manifest_options_for_context

options = PipelineRunOptions(
    ctx=ctx,
    manifest_dir=manifest_dir,
    manifest_options=run_manifest_options_for_context(ctx=ctx),
    scan_telemetry=scan_telemetry,
)
result = run_pipeline(plan=plan, finalize=finalize_spec, options=options)
```

**Target files**
- `src/codeintel/core/columnar/run_manifest.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/hamilton/diagnostics.py`
- `src/codeintel/build/graphs/validation/runner.py`

**Checklist**
- [ ] Record plan-level timing and ordering metadata in run manifests.
- [ ] Require scan telemetry for ingestion dataset scans.
- [ ] Emit manifests in validation and ingestion entrypoints.

---

## Scope 07 - Extraction Targets Orchestration (Finalize-First)
**Goal**
Have extraction targets return readers only and finalize/persist in a shared
layer to keep ingestion compute steps pure and consistent.

**Code pattern**
```python
readers = {
    "core.ast_nodes": collectors.ast_nodes.to_reader(),
    "core.ast_metrics": collectors.metrics.to_reader(),
}
finalized = finalize_arrow_readers(readers, ctx=ctx)
write_dataset_outputs(dataset_root, finalized, snapshot=scope)
```

**Target files**
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- `src/codeintel/ingestion/compute/ast_extract.py`
- `src/codeintel/ingestion/compute/cst_extract.py`
- `src/codeintel/ingestion/compute/symtable_extract.py`
- `src/codeintel/ingestion/compute/inspect_extract.py`
- `src/codeintel/ingestion/compute/dis_extract.py`

**Checklist**
- [ ] Return readers from compute steps; no `to_table()` before finalize.
- [ ] Centralize finalize + dataset write for all extraction targets.
- [ ] Propagate warnings and artifacts consistently from finalize.

---

## Sequencing Recommendation
1) Plan builder adoption + QuerySpec control plane (Scopes 01-02).
2) Kernel lane consolidation + columnar buffers (Scope 03).
3) Ordering/determinism + schema defaults (Scopes 04-05).
4) Telemetry + orchestration consolidation (Scopes 06-07).

## Validation Gates (Non-Pytest)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Runtime validation via run manifests + scan telemetry for modified paths.
