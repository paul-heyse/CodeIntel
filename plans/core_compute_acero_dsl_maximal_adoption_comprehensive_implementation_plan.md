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

### Scope 01 - DSL IR consolidation and guardrails
Representative pattern:
```python
from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import FinalizeSpec
from codeintel.core.columnar.plan_ops import Plan

plan = Plan.table(input_table).project({"repo": E.field("repo")})
result = run_pipeline(
    plan=ExecutionPlan(inner=plan.declaration),
    finalize=FinalizeSpec(table_key="core.modules", mode="tolerant"),
    ctx=ExecutionContext(use_threads=True, determinism="canonical"),
)
```
DSL extensions:
- Add a guardrail test to ban `pyarrow.compute` usage outside core DSL modules.
- Extend `ExecutionContext` defaults and expose in `codeintel.core.columnar.__init__`.
Target files:
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/__init__.py`
- `src/codeintel/build/tabular/expr_vocab.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `tests/arrowdsl/test_no_raw_compute_in_nodes.py`
Implementation checklist:
- [ ] Add a `rg`-based guardrail test for `pyarrow.compute` imports in build/ingestion.
- [ ] Ensure DSL surface is re-exported from `codeintel.core.columnar.__init__`.
- [ ] Update docs or comments to enforce "plan -> execute -> finalize".

### Scope 02 - QuerySpec-driven scan control plane + provenance
Representative pattern:
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import build_query_plan, QueryPlanOptions
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec

spec = QuerySpec(
    predicate=E.and_(E.eq("kind", "call"), E.ge("confidence", 0.8)),
    pushdown_predicate=E.eq("kind", "call"),
    projection=ProjectionSpec(base_cols=("repo", "commit", "rel_path", "kind", "confidence")),
)
plan = build_query_plan(dataset, spec=spec, options=QueryPlanOptions(provenance=True))
```
DSL extensions:
- Add scan telemetry helper (fragment counts, row estimates) in `streaming.py`.
- Standardize named scan profiles (dev/ci/prod) in `compute_config.py`.
Target files:
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/core/columnar/compute_config.py`
- `src/codeintel/core/columnar/queryspec.py`
Implementation checklist:
- [ ] Compile QuerySpec into scan + filter + project nodes everywhere.
- [ ] Attach provenance columns when `ctx.provenance=True`.
- [ ] Record scan telemetry per dataset scan for diagnostics.

### Scope 03 - Build graph pipelines: plan lane adoption (Acero-first)
Representative pattern:
```python
from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, run_pipeline
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan
from codeintel.core.columnar.finalize_ops import FinalizeSpec

left = Plan.table(call_sites).project({"callee_id": E.field("callee_id")})
right = Plan.table(symbols).project({"symbol_id": E.field("symbol_id")})
joined = left.hash_join(
    right=right,
    spec=HashJoinSpec(left_keys=["callee_id"], right_keys=["symbol_id"]),
)
result = run_pipeline(
    plan=ExecutionPlan(inner=joined.declaration),
    finalize=FinalizeSpec(table_key="graph.cpg_edges_calls", mode="tolerant"),
    ctx=ExecutionContext(determinism="canonical"),
)
```
DSL extensions:
- Use `join_safe_projection` and `require_join_safe_schema` before joins.
- Add `JoinPrecheckSpec` to enforce non-null join keys.
Target files:
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/graphs/cdg.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/*.py`
- `src/codeintel/build/graphs/assembly/ids.py`
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`
Implementation checklist:
- [ ] Replace ad hoc `pc.*` logic with Plan.filter/project/join pipelines.
- [ ] Pre-project and pre-cast join keys in Plan nodes.
- [ ] Enforce join-safe schemas to avoid list payload join failures.
- [ ] Route outputs through FinalizeSpec with invariants and canonical order keys.

### Scope 04 - Kernel lane consolidation: explode + dedupe + canonical sort
Representative pattern:
```python
from codeintel.core.columnar.explode_ops import explode_edges_with_aligned_lists
from codeintel.core.columnar.kernels import stable_sort_table

exploded = explode_edges_with_aligned_lists(
    table,
    src_col="src_id",
    dst_list_col="callee_ids",
    aligned_list_cols=("callsite_spans",),
    repeat_cols=("repo", "commit", "rel_path"),
)
ordered = stable_sort_table(exploded.good, sort_keys=[("src_id", "ascending")])
```
DSL extensions:
- Add a shared `explode_list_struct` helper for list<struct> payloads.
- Add deterministic dedupe wrapper that sorts before `hash_first`.
Target files:
- `src/codeintel/core/columnar/kernels.py`
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/core/columnar/nested_ops.py`
- `src/codeintel/build/tabular/explode_ops.py`
- `src/codeintel/build/tabular/array_ops.py`
Implementation checklist:
- [ ] Standardize explode helpers with list alignment checks.
- [ ] Reuse parent indices to repeat scalar columns efficiently.
- [ ] Add null-list policies (error vs empty) in explode specs.
- [ ] Centralize dedupe and canonical sort in shared kernels.

### Scope 05 - Finalize gate: determinism tiers + structured artifacts
Representative pattern:
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec

spec = FinalizeSpec(
    table_key="graph.cpg_edges_calls",
    mode="tolerant",
    required_non_null=("repo", "src_id", "dst_id"),
    order_by=(("repo", "ascending"), ("src_id", "ascending"), ("dst_id", "ascending")),
    emit_artifacts=True,
)
```
DSL extensions:
- Ensure `FinalizeSpec` enforces canonical sort keys for deterministic tiers.
- Standardize error codes for nested list alignment failures.
Target files:
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/schema_alignment.py`
Implementation checklist:
- [ ] Encode determinism tiers in dedupe and ordering policies.
- [ ] Emit `good/errors/alignment/stats` for all finalize operations.
- [ ] Add nested invariant error codes and stage labeling.

### Scope 06 - Ingestion pipelines: scan + finalize via Plan + QuerySpec
Representative pattern:
```python
from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import FinalizeSpec
from codeintel.core.columnar.plan_ops import build_query_plan, QueryPlanOptions

plan = build_query_plan(dataset, spec=spec, options=QueryPlanOptions(provenance=True))
result = run_pipeline(
    plan=ExecutionPlan(inner=plan.declaration),
    finalize=FinalizeSpec(table_key="ingestion.repo_scan", mode="tolerant"),
    ctx=ExecutionContext(determinism="stable_set"),
)
```
DSL extensions:
- Provide ingestion-specific QuerySpec templates (repo, commit, rel_path).
- Add `finalize_reader` usage for ingestion streaming paths.
Target files:
- `src/codeintel/ingestion/compute/base.py`
- `src/codeintel/ingestion/compute/repo_scan.py`
- `src/codeintel/ingestion/compute/ast_extract.py`
- `src/codeintel/ingestion/compute/cst_extract.py`
- `src/codeintel/ingestion/compute/inspect_extract.py`
- `src/codeintel/ingestion/compute/tree_sitter_index.py`
- `src/codeintel/ingestion/compute/*_extract.py`
Implementation checklist:
- [ ] Replace direct table materialization with QuerySpec + Plan.scan.
- [ ] Keep readers streaming until finalize boundaries.
- [ ] Route ingestion outputs through finalize for schema alignment and artifacts.

### Scope 07 - Ingestion compute (non-Hamilton): ColumnarRowBuffer and typed extras
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
- Add a `reader_for_columnar_rows` helper to avoid table materialization.
- Add typed `extras` struct builder for ingestion payloads with dynamic metadata.
Target files:
- `src/codeintel/ingestion/compute/config_ingest.py`
- `src/codeintel/ingestion/compute/typing_ingest.py`
- `src/codeintel/ingestion/compute/tests_ingest.py`
- `src/codeintel/ingestion/compute/docstrings_extract.py`
Implementation checklist:
- [ ] Replace dict-list assembly with ColumnarRowBuffer or batch collectors.
- [ ] Enforce typed `extras` struct where present.
- [ ] Route outputs through finalize for alignment + error artifacts.

### Scope 08 - Streaming safety and row-iteration boundaries
Representative pattern:
```python
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.finalize_ops import finalize_reader

finalized = finalize_reader(reader, spec=spec)
for row in iter_rows(finalized.good):
    handle(row)
```
DSL extensions:
- Add `iter_rows_limit` helper for bounded sampling without full materialization.
- Add `finalize_reader_batches` usage patterns in writers/materializers.
Target files:
- `src/codeintel/build/exports/writers.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/tabular/scoping.py`
- `src/codeintel/build/tabular/table_ops.py`
Implementation checklist:
- [ ] Replace `to_pylist`/`to_pydict` usage with `iter_rows` at boundaries only.
- [ ] Keep readers streaming for bulk operations and use `finalize_reader`.
- [ ] Add small tests for iter helpers where behavior differs by chunking.

### Scope 09 - Graph relation builders and CPG2 assembly helpers
Representative pattern:
```python
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges

exploded = explode_edges(parent_table, spec=ExplodeSpec(src_col="edge_id", dst_list_col="edge"))
projected = Plan.table(exploded.good).project({name: E.field(("edge", name)) for name in edge_fields})
```
DSL extensions:
- Add a small helper for struct-field projection (`project_struct_fields`).
- Add `join_safe_projection` for list payload removal before hash joins.
Target files:
- `src/codeintel/build/hamilton/native/graphs/call_graph.py`
- `src/codeintel/build/hamilton/native/graphs/goids.py`
- `src/codeintel/build/hamilton/native/graphs/import_graph.py`
- `src/codeintel/build/hamilton/native/graphs/pdg.py`
- `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
- `src/codeintel/build/graphs/compute/goid.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/ids.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`
Implementation checklist:
- [ ] Replace `pa.Table.from_pylist` edge builds with explode + project helpers.
- [ ] Enforce canonical ordering in FinalizeSpec for graph outputs.
- [ ] Ensure join inputs are list-free or exploded before hash joins.

### Scope 10 - Hamilton ingestion pipelines: join prechecks + finalize_reader
Representative pattern:
```python
from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, precheck_join_keys, run_pipeline
from codeintel.core.columnar.finalize_ops import FinalizeSpec
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan

precheck = precheck_join_keys(left_table, spec=JoinPrecheckSpec(required_non_null=join_keys))
plan = Plan.table(precheck.good).hash_join(right=Plan.table(right_table), spec=HashJoinSpec(...))
result = run_pipeline(
    plan=ExecutionPlan(inner=plan.declaration),
    finalize=FinalizeSpec(table_key="core.syntax_xref", mode="tolerant"),
    ctx=ExecutionContext(determinism="canonical"),
)
```
DSL extensions:
- Add join precheck helpers for ingestion pipelines with standardized error codes.
- Add `finalize_reader` usage in ingestion DAG nodes that already stream.
Target files:
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_proto.py`
- `src/codeintel/build/hamilton/native/ingestion/tree_sitter.py`
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/file_line_index.py`
- `src/codeintel/build/hamilton/native/ingestion/frame_utils.py`
- `src/codeintel/build/hamilton/native/ingestion/pipelines.py`
Implementation checklist:
- [ ] Replace ad hoc joins with HashJoinSpec + precheck_join_keys.
- [ ] Use Plan.filter for join key validation and deterministic inference.
- [ ] Prefer finalize_reader when a reader is already produced upstream.

### Scope 11 - Analytics pipelines: plan-first aggregates + deterministic outputs
Representative pattern:
```python
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import FinalizeSpec

plan = Plan.table(input_table).aggregate(
    keys=[E.field("repo"), E.field("commit")],
    aggregates=[("goid_h128", "count", None, "goid_count")],
).order_by(sort_keys=[("repo", "ascending"), ("commit", "ascending")])
result = run_pipeline(
    plan=ExecutionPlan(inner=plan.declaration),
    finalize=FinalizeSpec(table_key="analytics.graph_metrics", mode="tolerant"),
)
```
DSL extensions:
- Add list-aggregate helper for deterministic `hash_list` and `hash_distinct`.
- Add `stable_dedupe_by_score` helper for deterministic winner selection.
Target files:
- `src/codeintel/build/hamilton/native/analytics/*.py`
- `src/codeintel/build/analytics/graphs/*.py`
- `src/codeintel/build/analytics/functions/*.py`
- `src/codeintel/build/analytics/subsystems/*.py`
- `src/codeintel/build/analytics/compute/*.py`
- `src/codeintel/build/analytics/entrypoints/*.py`
- `src/codeintel/build/analytics/semantic_roles/*.py`
- `src/codeintel/build/analytics/cfg_dfg/cfg_core.py`
- `src/codeintel/build/analytics/cfg_dfg/dfg_core.py`
- `src/codeintel/build/analytics/data_models/core.py`
- `src/codeintel/build/analytics/utilities/catalogs.py`
Implementation checklist:
- [ ] Replace `iter_rows` loops with Plan.aggregate/group_by kernels.
- [ ] Use Plan.order_by + FinalizeSpec for deterministic analytics outputs.
- [ ] Introduce list aggregation for per-key payloads instead of Python lists.

### Scope 12 - Graph engine, validation, and diagnostics: reader-first scans
Representative pattern:
```python
from codeintel.core.columnar.plan_ops import build_query_plan, QueryPlanOptions
from codeintel.core.columnar.finalize_ops import finalize_reader, FinalizeSpec

plan = build_query_plan(dataset, spec=spec, options=QueryPlanOptions(provenance=True))
reader = plan.to_reader(use_threads=True)
finalized = finalize_reader(reader, spec=FinalizeSpec(table_key="graph.call_graph_edges", mode="tolerant"))
```
DSL extensions:
- Add scan telemetry hooks for validation scans (fragment counts and estimates).
- Add canonical sampling helper for deterministic error samples.
Target files:
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/validation/checks/*.py`
- `src/codeintel/build/graphs/validation/runner.py`
- `src/codeintel/build/hamilton/diagnostics.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py`
- `src/codeintel/build/hamilton/join_precheck_issues.py`
Implementation checklist:
- [ ] Replace materialized scans with Plan scans + finalize_reader.
- [ ] Attach provenance fields in validation outputs for traceability.
- [ ] Convert diagnostics row collection to list aggregation or bounded sampling.

### Scope 13 - Exports and materializers: finalize before write
Representative pattern:
```python
from codeintel.core.columnar.plan_ops import build_query_plan
from codeintel.core.columnar.finalize_ops import finalize_table, FinalizeSpec

plan = build_query_plan(dataset, spec=spec)
table = plan.to_table(use_threads=True)
finalized = finalize_table(table, spec=FinalizeSpec(table_key=table_key, mode="strict"))
writer.write_table(table_key, finalized.good)
```
DSL extensions:
- Add `materialize_and_finalize` helper to centralize table writes.
- Add standard stable sort keys from schema contracts before write.
Target files:
- `src/codeintel/build/exports/common.py`
- `src/codeintel/build/exports/engine.py`
- `src/codeintel/build/exports/validation.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/hamilton/materializers/arrow_parquet_cache.py`
Implementation checklist:
- [ ] Apply QuerySpec for projection/filter consistency in export paths.
- [ ] Finalize outputs prior to writing artifacts and caches.
- [ ] Preserve canonical ordering for exported datasets.

### Scope 14 - Causal analysis audits: anti-joins and deterministic samples
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
- Add `sample_top_k` helper that sorts then slices deterministically.
- Add `value_counts` helper for error rollups (group_by + count).
Target files:
- `src/codeintel/build/causal_analysis/cpg_edge_integrity.py`
- `src/codeintel/build/causal_analysis/cpg_symbol_destination_audit.py`
Implementation checklist:
- [ ] Replace row-wise missing checks with anti-join plans.
- [ ] Compute counts and samples via group_by and stable sort helpers.
- [ ] Keep row iteration only at final reporting boundaries.

## Sequencing Recommendation
1) DSL guardrails + QuerySpec-driven scan control plane (Scopes 01-02).
2) Kernel lane consolidation + finalize gate determinism (Scopes 04-05).
3) Build graph pipelines + relation builders + CPG2 assembly (Scopes 03, 09).
4) Ingestion pipelines and ingestion compute steps (Scopes 06-07, 10).
5) Analytics and graph validation expansions (Scopes 11-12).
6) Exports/materializers + causal audits (Scopes 13-14).

## Expected Outcome
After this plan, the majority of build and ingestion compute paths will:
- Use Acero plans for scan/filter/project/join/aggregate.
- Use kernel lane helpers for explode, dedupe, and deterministic ordering.
- Route outputs through finalize gates with structured artifacts.
- Avoid row-wise Python loops except at explicit boundary points.
