# Core Columnar Acero DSL Centralization Plan

## Objective
Deliver a best-in-class, centralized PyArrow Acero + DSL compute architecture that is
high-performance, modular, extensible, deterministic, and maintainable. This plan
aligns with the compute and Acero/DSL capabilities described in:
- `docs/python_library_reference/compute_improvement_deepdive.md`
- `docs/python_library_reference/arrow_acero_dsl_guide.md`

## Design Principles (Non-Negotiable)
- Plan lane vs kernel lane is explicit and enforced by API, not by convention.
- The finalize gate is the only materialization boundary and always returns artifacts.
- QuerySpec is the single source of truth for scan + plan predicates/projections.
- RuntimeProfile owns threading and determinism; nodes never decide ad hoc.
- Streaming and pipeline breakers are explicit in plan metadata and execution options.

## Deferred (Explicitly Out of Scope for Now)
- Guardrails and lint-based enforcement are deferred per request.

---

## Scope 01 - Centralized Plan Lane and Runner
**Goal**
Make `plan_ops` + `arrowdsl` the single authoring/execution surface for Acero plans,
and remove legacy plan helpers that bypass ordering and finalize.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.plan_ops import build_query_plan_for_context
from codeintel.core.columnar.queryspec import QuerySpec

plan = build_query_plan_for_context(dataset, spec=query_spec, ctx=execution_ctx)
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan, determinism="canonical"),
    finalize=finalize_spec_for_table("core.example_table", mode="tolerant"),
    ctx=execution_ctx,
)
```

**Target files**
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/acero_ops.py`
- `src/codeintel/core/datasets/scanner_ops.py`
- `src/codeintel/build/tabular/plan_ops.py`

**Checklist**
- [x] Route all Acero plan construction through `plan_ops.Plan`.
- [x] Deprecate `acero_ops.build_exec_plan` and redirect callers to `Plan`.
- [x] Consolidate scanner construction into `columnar.streaming` APIs.
- [x] Ensure all plan execution uses `ExecutionPlan` + `run_pipeline`.

---

## Scope 02 - QuerySpec + Filter Compiler Unification
**Goal**
Make `QuerySpec` the canonical query object across Arrow/DuckDB/Polars, and
compile filters once to avoid predicate drift.

**Code pattern**
```python
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec
from codeintel.core.queries.filter_compiler import (
    arrow_filter_expression,
    compile_filter_predicates,
)

predicates = compile_filter_predicates(filters, allowed_columns=allowed, column_types=types)
predicate = arrow_filter_expression(predicates)
projection = ProjectionSpec(base_cols=("repo", "commit", "path"))
query_spec = QuerySpec(
    predicate=predicate,
    pushdown_predicate=predicate,
    projection=projection,
)
```

**Target files**
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/core/queries/filter_compiler.py`
- `src/codeintel/core/datasets/scanning.py`
- `src/codeintel/storage/queries/parquet.py`
- `src/codeintel/serving/semantic/duckdb_relation_builder.py`

**Checklist**
- [x] Add helper(s) to build `QuerySpec` from filter specs + projection.
- [x] Update scan builders to accept `QuerySpec` directly where possible.
- [x] Ensure pushdown and post-filter paths use the same predicate source.

---

## Scope 03 - Kernel Lane Standardization (Explode + Dedupe)
**Goal**
Centralize row-count-changing operations in kernel helpers and remove bespoke
Python loops for explode or dedupe behavior.

**Code pattern**
```python
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline

exploded = explode_edges(
    table,
    spec=ExplodeSpec(
        src_col="src_id",
        dst_list_col="dst_ids",
        aligned_list_cols=("edge_kinds",),
        repeat_cols=("repo", "commit"),
    ),
)
result = run_pipeline(
    plan=ExecutionPlan.from_table(exploded.good),
    finalize=finalize_spec_for_table("graph.edges", mode="tolerant"),
)
```

**Target files**
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/core/columnar/kernels.py`
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py`

**Checklist**
- [x] Ensure explode usage always flows through `explode_ops` or `kernels`.
- [x] Replace ad-hoc dedupe logic with `dedupe_ops` in finalize or kernel lane.
- [x] Use join-safe projections when joining exploded results.

---

## Scope 04 - Finalize Gate Enforcement Everywhere
**Goal**
Guarantee that every pipeline ends with `FinalizeResult` and structured artifacts,
and that finalize specs are contract-driven.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table

finalize_spec = finalize_spec_for_table(
    "analytics.metric_rollups",
    mode="tolerant",
    emit_artifacts=True,
)
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan),
    finalize=finalize_spec,
    ctx=execution_ctx,
)
good_rows = result.good
```

**Target files**
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/serving/export/ndjson.py`
- `src/codeintel/storage/queries/parquet.py`

**Checklist**
- [x] Replace direct `Plan.to_table()` usage in pipelines with `run_pipeline`.
- [x] Use `finalize_spec_for_table` for contract-driven defaults.
- [x] Ensure tolerant mode emits artifacts consistently where required.

---

## Scope 05 - Runtime Profiles as the Control Plane
**Goal**
Make runtime profile resolution and thread pool selection consistent at all
entrypoints, including scans and plan execution.

**Code pattern**
```python
from codeintel.build.settings import get_columnar_runtime_settings
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.execution_context import runtime_profile_from_settings

profile = runtime_profile_from_settings(get_columnar_runtime_settings())
execution_ctx = ExecutionContext(runtime_profile=profile, determinism="stable_set")
```

**Target files**
- `src/codeintel/core/columnar/execution_context.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/datasets/scanning.py`
- `src/codeintel/core/config/settings.py`

**Checklist**
- [x] Resolve runtime profiles from settings at all columnar entrypoints.
- [x] Apply `configure_arrow_threading_for_context` for scans and plans.
- [x] Ensure `determinism` and `provenance` defaults flow from profile.

---

## Scope 06 - Observability (Run Manifest + Artifacts)
**Goal**
Centralize run manifest emission in the pipeline runner and tie it to finalize
artifacts and scan telemetry.

**Code pattern**
```python
from codeintel.core.columnar.run_manifest import RunManifestOptions, write_run_manifest
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec

telemetry = scan_telemetry_for_queryspec(dataset, spec=query_spec)
write_run_manifest(
    output_dir,
    options=RunManifestOptions(
        determinism=execution_ctx.resolve_determinism(),
        ordering=plan.ordering,
        scan_telemetry=telemetry,
        profile_name=execution_ctx.runtime_profile.name if execution_ctx.runtime_profile else None,
        scan_profile=execution_ctx.runtime_profile.scan_profile if execution_ctx.runtime_profile else None,
    ),
)
```

**Target files**
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/run_manifest.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py`
- `src/codeintel/build/graphs/validation/runner.py`

**Checklist**
- [x] Add a runner-level path to emit run manifests after finalize.
- [x] Include scan telemetry and runtime profile details in manifests.
- [x] Ensure artifacts are persisted or returned for tolerant runs.

---

## Scope 07 - Build Layer Consolidation to Core Columnar
**Goal**
Reduce duplicate Arrow compute surfaces in build modules and rely on core
columnar primitives to prevent semantic drift.

**Code pattern**
```python
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.kernels import explode_edges
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
```

**Target files**
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/tabular/array_ops.py`
- `src/codeintel/build/tabular/dedupe_ops.py`
- `src/codeintel/build/analytics/scip_diagnostics_rollups.py`
- `src/codeintel/build/graphs/engine/datasets.py`

**Checklist**
- [x] Replace build-layer Arrow helpers with core columnar calls or re-exports.
- [x] Remove duplicate dedupe and explode logic from build modules.
- [x] Standardize plan execution on `ExecutionPlan` + `run_pipeline`.

---

## Sequencing Recommendation
1) Scope 01 (central plan lane + runner)
2) Scope 02 (QuerySpec + filter compiler unification)
3) Scope 04 (finalize enforcement)
4) Scope 03 (kernel lane standardization)
5) Scope 05 (runtime profiles)
6) Scope 06 (observability)
7) Scope 07 (build-layer consolidation)

## Validation Gates (No Guardrails Yet)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted pytest subsets for modified modules (columnar, queries, build pipelines)

## Remaining Scope (From This Plan)
All scope items in this plan have been completed. Guardrails remain explicitly deferred.
