# Core Columnar Acero DSL Centralization Followup Implementation Plan

## Objective
Deliver a maximally centralized Acero/DSL compute surface that keeps plan-lane
construction and kernel-lane operations reusable across analytics and graph
pipelines, with explicit ordering/determinism, schema-driven defaults, and
unified telemetry. This plan applies the followup centralization scope to the
graph stack (rustworkx-driven) while preserving flexibility and performance.

## Design Principles (Non-Negotiable)
- Plan lane vs kernel lane stays explicit, with plan metadata preserved end-to-end.
- QuerySpec is the single source of truth for scan + projection + predicates.
- ExecutionContext and RuntimeProfile own determinism, threading, and provenance.
- Materialization is always mediated by ExecutionPlan + run_pipeline.
- Schema metadata is the only authority for finalize policies and canonical ordering.

---

## Scope 01 - Single Plan Construction API (Plan Builder)
**Goal**
Replace ad-hoc Plan construction in graph pipelines with the core plan builder,
so QuerySpec remains authoritative and plan assembly stays uniform across modules.

**Code pattern**
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
    base_cols=("repo", "commit", "path", "module"),
    repo=repo,
    commit=commit,
)
plan = build_snapshot_plan(table=table, spec=query_spec, ctx=execution_ctx)
plan = build_grouped_rollup_plan(
    plan,
    keys=("repo", "commit", "module"),
    aggregates=(("module", "count", None, "module_count"),),
)
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan),
    finalize=finalize_spec_for_table("analytics.graph_metrics_modules", mode="tolerant"),
    options=PipelineRunOptions(ctx=execution_ctx),
)
```

**Target files**
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/build/graphs/assembly/plan_surface.py` (deprecate or thin wrapper)
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/hamilton/native/graphs/call_graph.py`
- `src/codeintel/build/hamilton/native/graphs/import_graph.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/graphs/cdg.py`
- `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
- `src/codeintel/build/hamilton/native/graphs/goids.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`

**Implementation checklist**
- [ ] Replace direct `Plan.table(...)` usage in graph producers with plan_builder helpers.
- [ ] Route graph view assembly through `build_snapshot_query_spec` + `build_snapshot_plan`.
- [ ] Remove or collapse graph-specific plan surface APIs to plan_builder wrappers.
- [ ] Ensure QuerySpec is the sole source for scan/filter/projection behavior.

---

## Scope 02 - Kernel Lane Wrappers for Row-Changing Ops
**Goal**
Centralize explode/dedupe/rollup patterns into kernel wrappers that enforce
join-safe projections and deterministic tie handling across graph pipelines.

**Code pattern**
```python
from codeintel.core.columnar.plan_kernels import (
    group_by_max_join_back,
    stable_dedupe_with_ties,
)

winner_rows = group_by_max_join_back(
    table=table,
    key_columns=("repo", "commit"),
    score_column="score",
)

stable_rows = stable_dedupe_with_ties(
    table=table,
    keys=("repo", "commit", "path"),
    order_by=(("score", "descending"),),
    tie_breakers=(("rel_path", "ascending"),),
)
```

**Target files**
- `src/codeintel/core/columnar/plan_kernels.py` (new)
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/build/graphs/assembly/kernels.py` (re-export or thin wrapper)
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py`
- `src/codeintel/build/analytics/semantic_roles/core.py`

**Implementation checklist**
- [ ] Add kernel wrappers for dedupe with stable ties + hash ordinal fallback.
- [ ] Add rollup helpers that return join-safe tables (projection allowlists enforced).
- [ ] Replace per-module explode/dedupe logic with kernel wrappers in graph paths.
- [ ] Ensure all kernel wrappers accept explicit allowlists or schema-derived defaults.

---

## Scope 03 - Ordering and Determinism as First-Class Plan Metadata
**Goal**
Propagate ordering through plan nodes and enforce canonical ordering at finalize
boundaries, with deterministic tie-breaking based on contract keys.

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
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/engine/views.py`

**Implementation checklist**
- [ ] Ensure ordering metadata is propagated by filter/project/join/aggregate nodes.
- [ ] Require explicit ordering when determinism tier is canonical at finalize.
- [ ] Default plan ordering to unordered when ordering cannot be preserved.
- [ ] Use contract-derived canonical keys for ordering in graph materialization paths.

---

## Scope 04 - External Plan Runner Unification
**Goal**
Centralize external plan registration so Substrait/DataFusion usage returns
Plan or ReaderThunk and stays inside the plan lane.

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

**Implementation checklist**
- [ ] Move default runner registration into core columnar.
- [ ] Ensure external plan runners return Plan or ReaderThunk, not raw tables.
- [ ] Replace any graph-local external plan usage with the unified runner.

---

## Scope 05 - Schema-Driven Plan Defaults
**Goal**
Drive projections, ordering, and join-safe allowlists from schema metadata to
avoid re-specifying compute policy at call sites.

**Code pattern**
```python
from codeintel.core.columnar.plan_builder import plan_from_schema_defaults
from codeintel.core.schemas.service import get_schema_service

schema_service = get_schema_service()
plan = plan_from_schema_defaults(
    schema_service=schema_service,
    table_key="graph.call_graph_edges",
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
- `src/codeintel/build/graphs/engine/views.py`

**Implementation checklist**
- [ ] Add schema metadata for default projections + join-safe allowlists.
- [ ] Add schema metadata for canonical ordering keys and determinism tier hints.
- [ ] Replace graph view projections with schema-driven defaults.
- [ ] Ensure schema serialization includes new metadata fields.

---

## Scope 06 - Performance and Telemetry Consolidation
**Goal**
Standardize plan execution telemetry with run manifests, and enforce deterministic
execution profiles across all graph entry points.

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
- `src/codeintel/build/graphs/assembly/finalize.py`
- `src/codeintel/build/graphs/validation/runner.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py`

**Implementation checklist**
- [ ] Add plan-level timing hooks and emit them through run manifests.
- [ ] Propagate runtime profile, determinism tier, and ordering metadata.
- [ ] Standardize scan telemetry collection for graph datasets and artifacts.
- [ ] Ensure graph finalize wrapper can accept manifest options.

---

## Sequencing Recommendation
1) Scope 01 - Plan builder adoption sweep
2) Scope 02 - Kernel wrappers and row-changing ops consolidation
3) Scope 03 - Ordering and determinism propagation
4) Scope 05 - Schema-driven defaults
5) Scope 04 - External plan runner unification
6) Scope 06 - Telemetry consolidation

## Validation Gates (Guardrails Deferred)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted pytest subsets for modified graph/columnar modules once tests resume
