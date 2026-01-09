# Core Compute Acero DSL Comprehensive Implementation Plan (Build + Ingestion)

## Purpose
Deliver a consolidated, end-to-end implementation plan that extends Arrow Acero
and DSL adoption across `src/codeintel/build` and `src/codeintel/ingestion`.
This plan is additive and aligned with:
- `plans/core_compute_acero_dsl_maximal_adoption_plan.md`
- `plans/core_compute_acero_dsl_maximal_adoption_extension_plan.md`

The plan below assumes the core DSL surface (ExecutionContext, Plan, QuerySpec,
FinalizeSpec, kernel lane helpers) is available or is being implemented per the
two plans above. Each scope item includes:
- Representative pattern snippet
- Required DSL extensions (if any)
- Target file list
- Implementation checklist

## Assumed Baseline (from the two prerequisite plans)
The following are treated as present or in-flight and are not redefined here:
- `codeintel.core.columnar.arrowdsl` (ExecutionContext, ExecutionPlan, run_pipeline)
- `codeintel.core.columnar.plan_ops` (Plan, HashJoinSpec, build_query_plan)
- `codeintel.core.columnar.queryspec` (QuerySpec, ProjectionSpec, provenance fields)
- `codeintel.core.columnar.finalize_ops` (FinalizeSpec, finalize_table, finalize_reader)
- `codeintel.core.columnar.explode_ops` (explode_edges, explode_edges_with_aligned_lists)
- `codeintel.core.columnar.dedupe_ops` and `core.columnar.kernels` (dedupe, stable sort)
- `codeintel.core.columnar.expr_vocab` (E expression helpers)

## Scope Items

### Scope 01 - DSL IR v2 + guardrails (ordering-aware plans)
Status: Completed
Description:
Establish an ordering-aware DSL IR that keeps plan lane logic declarative, prevents
compute sprawl, and encodes plan metadata (ordering + pipeline breakers) explicitly.
Guardrails enforce "no raw pc in nodes" and "no materialize outside finalize."
Representative pattern:
```python
from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, run_pipeline
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.finalize_ops import FinalizeSpec
from codeintel.core.columnar.ordering import OrderingSpec
from codeintel.core.columnar.plan_ops import Plan

ctx = ExecutionContext(use_threads=True, determinism="canonical", provenance=True)
plan = Plan.scan(dataset, columns=["repo", "commit"], filter_expr=E.is_valid("repo"))
exec_plan = ExecutionPlan(inner=plan.declaration, ordering=OrderingSpec.implicit())
result = run_pipeline(
    plan=exec_plan,
    finalize=FinalizeSpec(table_key="core.modules", mode="tolerant", emit_artifacts=True),
    ctx=ctx,
)
```
DSL extensions:
- Add `OrderingSpec` metadata (unordered/implicit/explicit) plus pipeline breaker tags.
- Add guardrails for "no raw pc in nodes" and "no to_table outside finalize boundaries."
Target files:
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/execution_context.py`
- `src/codeintel/core/columnar/ordering.py` (new)
- `tools/lint_no_raw_pyarrow_compute_in_nodes.py`
- `tools/lint_no_materialize_in_nodes.py` (new)
Implementation checklist:
- [x] Introduce `OrderingSpec` and attach to `ExecutionPlan` / plan metadata.
- [x] Annotate ordering effects on Plan ops (scan implicit, join unordered, order_by explicit).
- [x] Enforce guardrails for raw compute and materialization boundaries.
- [x] Wire guardrail lints into the quality report suite.
- [x] Re-export updated DSL surface from `codeintel.core.columnar.__init__`.

### Scope 02 - QuerySpec + scan control plane + provenance
Status: In progress (profile defaults + entrypoint migrations started)
Description:
Make QuerySpec the single source of truth for scans, pushdown, and projection, and
enforce a scan control plane driven by named runtime profiles. Provenance columns
must be opt-in by profile and flow to errors for reproducibility.
Representative pattern:
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import build_query_plan_for_context, QueryPlanOptions
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec

spec = QuerySpec(
    predicate=E.eq("kind", "call"),
    pushdown_predicate=E.eq("kind", "call"),
    projection=ProjectionSpec(base_cols=("repo", "commit", "rel_path", "kind")),
)
plan = build_query_plan_for_context(
    dataset,
    spec=spec,
    ctx=ctx,
    options=QueryPlanOptions(provenance=True, implicit_ordering=True),
)
telemetry = scan_telemetry_for_queryspec(dataset, spec=spec)
```
DSL extensions:
- Add `RuntimeProfile` (scan profile + plan threading + determinism defaults).
- Enforce scanner construction via `build_scanner_for_queryspec_ctx`.
Target files:
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/core/columnar/compute_config.py`
- `src/codeintel/core/columnar/execution_context.py`
Implementation checklist:
- [x] Add RuntimeProfile defaults and propagate into scan/plan helpers.
- [x] Wire provenance columns to errors when enabled.
- [x] Emit scan telemetry + run manifests for post-run quality outputs.
- [ ] Route remaining dataset scans through QuerySpec + scan profile helpers.

### Scope 03 - Determinism budget + ordering enforcement
Status: Completed
Description:
Align determinism tiers with the DSL guide (canonical, stable_set, best_effort),
and enforce canonical ordering and dedupe policy at finalize boundaries. Ordering
is a contract requirement, not an incidental property.
Representative pattern:
```python
from codeintel.core.columnar.finalize_ops import FinalizeDedupe, FinalizeSpec

dedupe = FinalizeDedupe(
    keys=("repo", "src_id", "dst_id"),
    tie_breakers=(("confidence", "descending"),),
    tier="canonical",
    strategy="keep_best_by_score",
)
spec = FinalizeSpec(
    table_key="graph.cpg_edges",
    mode="tolerant",
    dedupe=dedupe,
    order_by=(("repo", "ascending"), ("src_id", "ascending"), ("dst_id", "ascending")),
)
```
DSL extensions:
- Expand determinism tiers to `canonical`, `stable_set`, `best_effort`.
- Add `keep_best_by_score` and `keep_arbitrary` strategies for dedupe.
Target files:
- `src/codeintel/core/columnar/execution_context.py`
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/columnar/arrowdsl.py`
Implementation checklist:
- [x] Update determinism tiers and default policy mapping.
- [x] Enforce canonical ordering and tie-breakers for canonical outputs.
- [x] Add score-based dedupe strategy for order-independent winners.

### Scope 04 - Kernel lane consolidation (explode + alignment + null policy)
Status: Completed
Description:
Centralize row-count-changing transforms in kernel helpers and keep them out of
plan lane nodes. Explode must validate list alignment and null list policies with
structured error codes.
Representative pattern:
```python
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges_with_aligned_lists

spec = ExplodeSpec(
    src_col="src_id",
    dst_list_col="callee_ids",
    aligned_list_cols=("callsite_spans", "call_kinds"),
    repeat_cols=("repo", "commit", "rel_path"),
    null_list_policy="error",
)
result = explode_edges_with_aligned_lists(table, spec=spec)
```
DSL extensions:
- Keep list alignment and null list handling centralized in kernel helpers.
- Standardize list-aligned error codes and stages for finalize.
Target files:
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/core/columnar/kernels.py`
- `src/codeintel/core/columnar/nested_ops.py`
- `src/codeintel/build/tabular/explode_ops.py`
Implementation checklist:
- [x] Use list_parent_indices + take for repeat columns.
- [x] Validate aligned list lengths before explode.
- [x] Emit structured errors for null list policy violations.

### Scope 05 - Finalize gate + observability artifacts
Status: In progress (core artifacts + post-run manifest wiring complete)
Description:
Finalize is the single correctness boundary and the primary observability surface.
Tolerant mode must always emit good/errors/alignment/stats and attach provenance
fields and run metadata to enable fast repro and diagnostics.
Representative pattern:
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_reader
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec

telemetry = scan_telemetry_for_queryspec(dataset, spec=spec)
result = finalize_reader(
    reader,
    spec=FinalizeSpec(table_key="graph.cpg_edges", mode="tolerant", emit_artifacts=True),
)
```
DSL extensions:
- Add run-manifest emission (profile, determinism, scan telemetry, Arrow version).
- Ensure errors include provenance fields when enabled.
Target files:
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py`
- `tools/arrowdsl/run_manifest.py` (new)
Implementation checklist:
- [x] Emit run manifest alongside post-run finalize artifacts.
- [x] Attach provenance columns to errors when provenance is enabled.
- [x] Always emit alignment/stats artifacts in tolerant finalize mode.
- [ ] Wire run manifests + artifact persistence into remaining validation outputs.

### Scope 06 - Build graph pipelines: ordering-aware plan lane
Status: In progress
Description:
Continue migration of build graph pipelines to Acero plan lane with ordering-aware
metadata, join prechecks, and finalize determinism enforcement. Remove ad hoc masks
and keep key casting inside the Plan layer.
Representative pattern:
```python
from codeintel.core.columnar.arrowdsl import JoinPrecheckSpec, precheck_join_keys
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan

left = Plan.table(left_table).project({"src_id": E.cast(E.field("src_id"), "int64")})
right = Plan.table(right_table).project({"dst_id": E.cast(E.field("dst_id"), "int64")})
left_ok = precheck_join_keys(left_table, spec=JoinPrecheckSpec(required_non_null=("src_id",)))
joined = left.hash_join(
    right=right,
    spec=HashJoinSpec(left_keys=["src_id"], right_keys=["dst_id"]),
)
```
DSL extensions:
- Join precheck helpers and join-safe projection for list-bearing inputs.
- Ordering metadata propagation for joins and aggregates.
Target files:
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/graphs/cdg.py`
- `src/codeintel/build/hamilton/native/graphs/call_graph.py`
- `src/codeintel/build/hamilton/native/graphs/goids.py`
- `src/codeintel/build/hamilton/native/graphs/import_graph.py`
- `src/codeintel/build/hamilton/native/graphs/pdg.py`
- `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
Implementation checklist:
- [ ] Replace remaining ad hoc filters with Plan.filter/project nodes.
- [ ] Apply join_safe_projection and join prechecks for all joins.
- [ ] Route graph outputs through FinalizeSpec with canonical order keys.

### Scope 07 - Graph relation builders + CPG2 assembly helpers
Status: In progress
Description:
Standardize edge construction via explode + struct projection helpers and ensure
canonical ordering for graph outputs. Join inputs must be list-free or exploded
before hash joins.
Representative pattern:
```python
from codeintel.core.columnar.arrowdsl import project_struct_fields
from codeintel.build.tabular.explode_ops import ExplodeSpec, explode_edges

exploded = explode_edges(parent_table, spec=ExplodeSpec(src_col="edge_id", dst_list_col="edge"))
projected = Plan.table(exploded.good).project(project_struct_fields("edge", edge_fields))
```
DSL extensions:
- `project_struct_fields` for struct projections in relation builders.
- Enforce `FinalizeSpec.order_by` for CPG edge/node outputs.
Target files:
- `src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/assemble.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/*.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/ids.py`
Implementation checklist:
- [ ] Replace any remaining struct-field projections with `project_struct_fields`.
- [ ] Enforce canonical ordering at finalize for graph outputs.
- [ ] Ensure joins only consume list-free inputs.

### Scope 08 - Ingestion pipelines (Hamilton): scan + finalize_reader
Status: Planned
Description:
Adopt QuerySpec + plan lane for ingestion DAGs, enforce join prechecks, and keep
readers streaming until finalize boundaries. Provenance should flow for repro.
Representative pattern:
```python
from codeintel.core.columnar.plan_ops import build_query_plan_for_context
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_reader

plan = build_query_plan_for_context(dataset, spec=spec, ctx=ctx)
reader = plan.to_reader(use_threads=ctx.use_threads)
result = finalize_reader(reader, spec=FinalizeSpec(table_key="ingestion.repo_scan", mode="tolerant"))
```
DSL extensions:
- Join precheck helpers for ingestion joins with standard error codes.
- Profile-driven scan defaults for ingestion pipelines.
Target files:
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/tree_sitter.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
Implementation checklist:
- [ ] Replace ad hoc scans with QuerySpec + build_query_plan_for_context.
- [ ] Use finalize_reader for streaming ingestion outputs.
- [ ] Record join precheck errors with provenance context.

### Scope 09 - Ingestion compute (non-Hamilton): ColumnarRowBuffer
Status: Planned
Description:
Replace dict-list assembly with ColumnarRowBuffer and typed extras builders, and
route outputs through finalize for alignment and structured artifacts.
Representative pattern:
```python
from codeintel.core.columnar.rows import columnar_buffer_for_table_key, table_for_columnar_rows
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table

buffer = columnar_buffer_for_table_key("core.typing_diagnostics")
buffer.append(row_payload)
table = table_for_columnar_rows(buffer, table_key="core.typing_diagnostics")
result = finalize_table(table, spec=FinalizeSpec(table_key="core.typing_diagnostics", mode="tolerant"))
```
DSL extensions:
- Add `reader_for_columnar_rows` to avoid table materialization.
- Add typed extras struct builder for ingestion metadata.
Target files:
- `src/codeintel/ingestion/compute/config_ingest.py`
- `src/codeintel/ingestion/compute/typing_ingest.py`
- `src/codeintel/ingestion/compute/tests_ingest.py`
- `src/codeintel/ingestion/compute/docstrings_extract.py`
Implementation checklist:
- [ ] Replace dict-list assembly with ColumnarRowBuffer.
- [ ] Enforce typed extras and finalize for all ingestion compute outputs.
- [ ] Prefer reader-based finalization when possible.

### Scope 10 - Analytics pipelines: plan-first aggregates + deterministic outputs
Status: Planned
Description:
Migrate analytics to Plan.aggregate and deterministic finalize ordering. Replace
row-wise loops with group_by kernels and list aggregations where needed.
Representative pattern:
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table

plan = Plan.table(metrics).aggregate(
    keys=[E.field("repo"), E.field("commit")],
    aggregates=[(E.field("goid_h128"), "count", None, "goid_count")],
)
table = plan.to_table(use_threads=True)
result = finalize_table(table, spec=FinalizeSpec(table_key="analytics.graph_metrics", mode="tolerant"))
```
DSL extensions:
- Deterministic list aggregation helpers and score-based dedupe for winners.
- Canonical ordering for analytics outputs under canonical determinism.
Target files:
- `src/codeintel/build/analytics/graphs/*.py`
- `src/codeintel/build/analytics/functions/*.py`
- `src/codeintel/build/analytics/subsystems/*.py`
- `src/codeintel/build/analytics/cfg_dfg/cfg_core.py`
- `src/codeintel/build/analytics/cfg_dfg/dfg_core.py`
Implementation checklist:
- [ ] Replace Python loops with Plan.aggregate or kernels.
- [ ] Apply FinalizeSpec ordering for canonical analytics outputs.
- [ ] Add score-based winner selection where needed.

### Scope 11 - Graph engine, validation, and diagnostics: reader-first scans
Status: In progress
Description:
Move graph engine scans to reader-first plans and record run telemetry. Validation
outputs should include provenance and use bounded sampling helpers instead of full
materialization.
Representative pattern:
```python
from codeintel.core.columnar.plan_ops import build_query_plan_for_context
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_reader

plan = build_query_plan_for_context(dataset, spec=spec, ctx=ctx)
reader = plan.to_reader(use_threads=ctx.use_threads)
result = finalize_reader(reader, spec=FinalizeSpec(table_key="graph.call_graph_edges", mode="tolerant"))
```
DSL extensions:
- Canonical bounded sampling helper for diagnostics (iter_rows_limit).
- Run manifest emission with scan telemetry and profile metadata.
Target files:
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/validation/runner.py`
- `src/codeintel/build/hamilton/diagnostics.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py`
Implementation checklist:
- [x] Migrate graph engine snapshot scans to QuerySpec + plan readers.
- [x] Emit run manifests for post-run quality scans.
- [ ] Replace remaining materialized scans with plan readers and finalize_reader.
- [ ] Emit run manifests with scan telemetry for validation runs.
- [ ] Use bounded sampling helpers for diagnostics output.

### Scope 12 - Exports and materializers: finalize before write
Status: Planned
Description:
Ensure all export and materialization paths finalize outputs before writing, with
canonical ordering for deterministic artifacts. Streaming writes should avoid
materializing full tables unless explicitly required.
Representative pattern:
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table

finalized = finalize_table(table, spec=FinalizeSpec(table_key=table_key, mode="strict"))
writer.write_table(table_key, finalized.good)
```
DSL extensions:
- Add a shared `materialize_and_finalize` helper for exports.
- Use preserve_order only when explicitly required for contract outputs.
Target files:
- `src/codeintel/build/exports/common.py`
- `src/codeintel/build/exports/engine.py`
- `src/codeintel/build/exports/validation.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/hamilton/materializers/arrow_parquet_cache.py`
Implementation checklist:
- [ ] Finalize outputs prior to export and cache writes.
- [ ] Enforce canonical ordering for deterministic artifacts.
- [ ] Keep export writers streaming where possible.

### Scope 13 - Causal audits: anti-joins + deterministic samples
Status: Planned
Description:
Replace row-wise audit logic with anti-joins and deterministic sampling pipelines.
Emit counts and samples via group_by and stable sorting for reproducibility.
Representative pattern:
```python
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan
from codeintel.core.columnar.kernels import stable_sort_table

missing = Plan.table(edges).hash_join(
    right=Plan.table(nodes),
    spec=HashJoinSpec(left_keys=["dst_id"], right_keys=["cpg_node_id"], how="left anti"),
)
sample = stable_sort_table(missing.to_table(), sort_keys=[("dst_id", "ascending")]).slice(0, 50)
```
DSL extensions:
- Add `sample_top_k` and `value_counts` helpers for audit outputs.
Target files:
- `src/codeintel/build/causal_analysis/cpg_edge_integrity.py`
- `src/codeintel/build/causal_analysis/cpg_symbol_destination_audit.py`
Implementation checklist:
- [ ] Replace row-wise missing checks with anti-joins.
- [ ] Use stable sort and bounded sampling for deterministic reports.
- [ ] Emit counts by error code/stage as structured outputs.

### Scope 14 - Golden pipeline + runtime validation/telemetry (no pytest)
Status: Planned
Description:
Deliver a golden end-to-end pipeline that exercises scan, join, deterministic
finalize, and observability artifacts. Validation is embedded in the run and
emits a run manifest and telemetry rather than pytest suites.
Representative pattern:
```python
from codeintel.core.columnar.kernels import stable_sort_table
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from tools.arrowdsl.run_manifest import write_run_manifest

finalized = finalize_table(table, spec=FinalizeSpec(table_key="graph.cpg_edges", mode="tolerant"))
canonical = stable_sort_table(finalized.good, sort_keys=[("repo", "ascending")])
write_run_manifest(output_dir, telemetry=telemetry, determinism="canonical")
```
DSL extensions:
- Add a run harness that executes pipelines in validation mode and emits telemetry.
- Add a repro extractor that uses provenance fields to materialize minimal inputs.
Target files:
- `tools/arrowdsl/run_pipeline.py` (new)
- `tools/arrowdsl/run_manifest.py` (new)
- `tools/arrowdsl/repro_extract.py` (new)
- `src/codeintel/core/columnar/streaming.py`
Implementation checklist:
- [ ] Build a golden pipeline runner with deterministic output hashing.
- [ ] Emit run manifest, scan telemetry, and finalize artifacts per run.
- [ ] Provide repro extraction from error tables using provenance fields.


## Sequencing Recommendation
1) DSL IR + scan control plane + determinism budget (Scopes 01-03).
2) Kernel lane consolidation + finalize/observability artifacts (Scopes 04-05).
3) Build graph pipelines + relation builders + CPG2 assembly (Scopes 06-07).
4) Ingestion pipelines + ingestion compute (Scopes 08-09).
5) Analytics + graph engine validation/diagnostics (Scopes 10-11).
6) Exports/materializers + causal audits + golden pipeline (Scopes 12-14).

## Expected Outcome
After this plan, the majority of build and ingestion compute paths will:
- Use Acero plans for scan/filter/project/join/aggregate.
- Use kernel lane helpers for explode, dedupe, and deterministic ordering.
- Route outputs through finalize gates with structured artifacts.
- Avoid row-wise Python loops except at explicit boundary points.
- Emit run manifests and scan telemetry for reproducibility and diagnostics.
