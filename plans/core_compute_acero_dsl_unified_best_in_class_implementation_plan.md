# Core Compute Acero DSL Unified Best-In-Class Implementation Plan (Code Only)

## Purpose
Deliver a unified, best-in-class implementation plan for Arrow Acero + DSL
adoption across core, build, analytics, and ingestion. This plan aligns with:
- `plans/core_columnar_acero_unified_best_in_class_plan.md`
- `plans/analytics_core_columnar_alignment_plan.md`

## Exclusions (explicit)
- Text scope: docs, guides, benchmarks, and narrative-only artifacts.
- Guardrails: new lint rules or guardrail tooling changes.
- Pytest: no new test scope; use runtime validation and telemetry instead.

## Assumed Baseline
The following surfaces exist or are in-flight and will be extended, not replaced:
- `codeintel.core.columnar.arrowdsl` (ExecutionPlan, run_pipeline)
- `codeintel.core.columnar.plan_ops` (Plan, build_query_plan)
- `codeintel.core.columnar.queryspec` (QuerySpec, ProjectionSpec)
- `codeintel.core.columnar.finalize_ops` (FinalizeSpec, finalize_table/reader)
- `codeintel.core.columnar.explode_ops` (explode helpers)
- `codeintel.core.columnar.dedupe_ops` and `core.columnar.kernels` (dedupe, sort)
- `codeintel.core.columnar.expr_vocab` (E expression helpers)

## Scope Items

### Scope 01 - Runtime profiles and global thread pools
Description:
Make runtime profiles the single control plane for CPU/I/O pools, scan defaults,
plan threading, and determinism tier defaults.
Pattern:
```python
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.profiles import RuntimeProfile
from codeintel.core.columnar.runtime import apply_runtime_profile

profile = RuntimeProfile(
    name="PROD_THROUGHPUT",
    cpu_threads=32,
    io_threads=64,
    plan_use_threads=True,
    scan_profile="prod_default",
    determinism="stable_set",
    provenance=True,
)
apply_runtime_profile(profile)
ctx = ExecutionContext(runtime_profile=profile)
```
Target files:
- `src/codeintel/core/columnar/execution_context.py`
- `src/codeintel/core/columnar/compute_config.py`
- `src/codeintel/core/columnar/profiles.py` (new if needed)
- `src/codeintel/core/columnar/runtime.py`
- `src/codeintel/core/config/settings.py`
Implementation checklist:
- [ ] Define RuntimeProfile and ScanProfile with explicit defaults.
- [ ] Apply CPU/I/O pools once per process and record in run metadata.
- [ ] Plumb profile defaults into ExecutionContext and scan helpers.

### Scope 02 - QuerySpec control plane completion
Description:
Ensure all dataset scans use QuerySpec + profile defaults, with provenance and
pushdown behavior centralized.
Pattern:
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import build_query_plan_for_context
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec

spec = QuerySpec(
    predicate=E.eq("kind", "call"),
    pushdown_predicate=E.eq("kind", "call"),
    projection=ProjectionSpec(base_cols=("repo", "commit", "path", "kind")),
)
plan = build_query_plan_for_context(dataset, spec=spec, ctx=ctx)
reader = plan.to_reader(use_threads=ctx.runtime_profile.plan_use_threads)
```
Target files:
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/core/datasets/scanning.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/validation/runner.py`
Implementation checklist:
- [ ] Route all remaining scans through QuerySpec helpers.
- [ ] Enforce provenance columns via profile defaults.
- [ ] Record scan telemetry (fragment count, row estimates) for runs.

### Scope 03 - Legacy surface retirement (plan-first execution)
Description:
Retire legacy plan/scan surfaces and consolidate around Plan + ExecutionPlan.
Pattern:
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.plan_ops import Plan

plan = Plan.scan(dataset, spec=spec, ctx=ctx)
exec_plan = ExecutionPlan(inner=plan.declaration, ordering=plan.ordering)
table = exec_plan.to_table(use_threads=ctx.runtime_profile.plan_use_threads)
```
Target files:
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/acero_ops.py`
- `src/codeintel/core/datasets/scanner_ops.py`
Implementation checklist:
- [ ] Deprecate legacy plan runner entrypoints in favor of ExecutionPlan.
- [ ] Replace legacy scan helpers with QuerySpec + Plan.
- [ ] Ensure ordering metadata is preserved across all plan surfaces.

### Scope 04 - Finalize gate upgrades (nested, determinism, artifacts)
Description:
Make finalize the universal correctness boundary, including deep casting for
nested types, deterministic ordering tiers, and structured artifacts.
Pattern:
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_reader

spec = FinalizeSpec(
    table_key="graph.cpg_edges",
    mode="tolerant",
    emit_artifacts=True,
    order_by=(("repo", "ascending"), ("src_id", "ascending"), ("dst_id", "ascending")),
)
result = finalize_reader(reader, spec=spec)
```
Target files:
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/core/columnar/nested_ops.py`
- `src/codeintel/core/columnar/kernels.py`
Implementation checklist:
- [ ] Implement deep cast helpers for list/struct/map columns.
- [ ] Enforce determinism tiers and canonical ordering at finalize boundaries.
- [ ] Emit good/errors/alignment/stats artifacts in tolerant mode.

### Scope 05 - Kernel lane consolidation (explode + struct projection)
Description:
Centralize all row-count-changing transforms and struct projections in kernel
helpers, with aligned list validation and null list policies.
Pattern:
```python
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges_with_aligned_lists

spec = ExplodeSpec(
    src_col="src_id",
    dst_list_col="callee_ids",
    aligned_list_cols=("callsite_spans",),
    repeat_cols=("repo", "commit"),
    null_list_policy="error",
)
result = explode_edges_with_aligned_lists(table, spec=spec)
```
Target files:
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/core/columnar/kernels.py`
- `src/codeintel/core/columnar/nested_ops.py`
- `src/codeintel/build/tabular/explode_ops.py`
Implementation checklist:
- [ ] Use list_parent_indices + take for repeat columns.
- [ ] Validate aligned list lengths before explode.
- [ ] Expose struct projection helpers for relation builders.

### Scope 06 - Build graph pipelines (plan-first joins and filters)
Description:
Complete migration of graph pipelines to Plan.filter/project/join with join
prechecks and finalize ordering for all outputs.
Pattern:
```python
from codeintel.core.columnar.arrowdsl import JoinPrecheckSpec, precheck_join_keys
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan

left = Plan.table(nodes).project({"src_id": "src_id"})
right = Plan.table(edges).project({"dst_id": "dst_id"})
_ = precheck_join_keys(edges, spec=JoinPrecheckSpec(required_non_null=("dst_id",)))
joined = left.hash_join(
    right=right,
    spec=HashJoinSpec(left_keys=("src_id",), right_keys=("dst_id",)),
)
```
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
- [ ] Replace remaining ad hoc filters/masks with Plan.filter/project.
- [ ] Apply join prechecks for all join inputs.
- [ ] Route graph outputs through FinalizeSpec with canonical order_by.

### Scope 07 - CPG2 assembly helpers and relation builders
Description:
Replace manual edge construction with explode + struct projection helpers and
enforce list-free joins.
Pattern:
```python
from codeintel.build.tabular.explode_ops import ExplodeSpec, explode_edges
from codeintel.core.columnar.arrowdsl import project_struct_fields
from codeintel.core.columnar.plan_ops import Plan

edges = explode_edges(table, spec=ExplodeSpec(src_col="node_id", dst_list_col="edges"))
projected = Plan.table(edges.good).project(project_struct_fields("edges", ("kind", "span")))
```
Target files:
- `src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/assemble.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/*.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/ids.py`
Implementation checklist:
- [ ] Replace pa.Table.from_pylist edge builds with explode helpers.
- [ ] Use project_struct_fields for struct projections.
- [ ] Ensure joins consume list-free inputs or explode beforehand.

### Scope 08 - Ingestion pipelines (Hamilton) to QuerySpec + finalize_reader
Description:
Adopt QuerySpec plans for ingestion DAGs and keep readers streaming to finalize.
Pattern:
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_reader
from codeintel.core.columnar.plan_ops import build_query_plan_for_context

plan = build_query_plan_for_context(dataset, spec=spec, ctx=ctx)
reader = plan.to_reader(use_threads=ctx.runtime_profile.plan_use_threads)
result = finalize_reader(reader, spec=FinalizeSpec(table_key="ingestion.repo_scan", mode="tolerant"))
```
Target files:
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/tree_sitter.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
Implementation checklist:
- [ ] Replace ad hoc scans with QuerySpec + plan helpers.
- [ ] Use finalize_reader for streaming ingestion outputs.
- [ ] Propagate provenance to errors when enabled.

### Scope 09 - Ingestion compute (non-Hamilton) with ColumnarRowBuffer
Description:
Replace dict-list assembly with ColumnarRowBuffer and typed extras builders, then
finalize via reader or table helpers.
Pattern:
```python
from codeintel.core.columnar.rows import columnar_buffer_for_table_key, table_for_columnar_rows
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table

buffer = columnar_buffer_for_table_key("core.typing_diagnostics")
buffer.append(payload)
table = table_for_columnar_rows(buffer, table_key="core.typing_diagnostics")
result = finalize_table(table, spec=FinalizeSpec(table_key="core.typing_diagnostics", mode="tolerant"))
```
Target files:
- `src/codeintel/ingestion/compute/config_ingest.py`
- `src/codeintel/ingestion/compute/typing_ingest.py`
- `src/codeintel/ingestion/compute/tests_ingest.py`
- `src/codeintel/ingestion/compute/docstrings_extract.py`
- `src/codeintel/core/columnar/rows.py`
Implementation checklist:
- [ ] Replace dict-list assembly with ColumnarRowBuffer.
- [ ] Add typed extras struct builders for ingestion metadata.
- [ ] Prefer reader-based finalization when possible.

### Scope 10 - Analytics alignment (finalize-first outputs)
Description:
Align analytics outputs to contract-driven finalize and deterministic ordering.
Pattern:
```python
from codeintel.build.analytics.utilities.finalize import finalize_analytics_result

result = finalize_analytics_result("analytics.graph_metrics", table)
good = result.good
```
Target files:
- `src/codeintel/build/analytics/utilities/finalize.py`
- `src/codeintel/build/analytics/utilities/datasets.py`
- `src/codeintel/build/analytics/graphs/*.py`
- `src/codeintel/build/analytics/functions/*.py`
- `src/codeintel/core/schemas/table_registry.py`
Implementation checklist:
- [ ] Route analytics outputs through finalize helpers.
- [ ] Persist errors/alignment/stats artifacts alongside analytics tables.
- [ ] Add finalize_policy coverage for analytics schema entries.

### Scope 11 - Observability artifacts and repro extraction
Description:
Standardize run manifests, scan telemetry, provenance propagation, and bounded
sampling for diagnostics and validation paths.
Pattern:
```python
from codeintel.core.columnar.run_manifest import RunManifestOptions, write_run_manifest

write_run_manifest(
    output_dir,
    options=RunManifestOptions(
        determinism=ctx.runtime_profile.determinism,
        ordering=plan.ordering,
        scan_telemetry=telemetry,
        profile_name=ctx.runtime_profile.name,
    ),
)
```
Target files:
- `src/codeintel/core/columnar/run_manifest.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/core/columnar/iter.py`
- `src/codeintel/build/graphs/validation/runner.py`
- `src/codeintel/build/hamilton/diagnostics.py`
- `tools/arrowdsl/repro_extract.py`
Implementation checklist:
- [ ] Emit run manifests for validation and diagnostics runs.
- [ ] Add bounded sampling helpers for diagnostics outputs.
- [ ] Provide repro extraction using provenance fields.

### Scope 12 - Exports and materializers (finalize before write)
Description:
Ensure all export and cache write paths finalize outputs and apply canonical
ordering when determinism is required.
Pattern:
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table

finalized = finalize_table(table, spec=FinalizeSpec(table_key=table_key, mode="strict"))
writer.write_table(table_key, finalized.good)
```
Target files:
- `src/codeintel/build/exports/common.py`
- `src/codeintel/build/exports/engine.py`
- `src/codeintel/build/exports/validation.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/hamilton/materializers/arrow_parquet_cache.py`
Implementation checklist:
- [ ] Finalize outputs prior to export and cache writes.
- [ ] Enforce canonical ordering for deterministic artifacts.
- [ ] Keep writers streaming where feasible.

### Scope 13 - Causal audits (anti-joins and deterministic samples)
Description:
Replace row-wise audits with plan-first anti-joins and stable sampling.
Pattern:
```python
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan
from codeintel.core.columnar.kernels import stable_sort_table

missing = Plan.table(edges).hash_join(
    right=Plan.table(nodes),
    spec=HashJoinSpec(left_keys=("dst_id",), right_keys=("cpg_node_id",), how="left anti"),
)
sample = stable_sort_table(missing.to_table(), sort_keys=[("dst_id", "ascending")]).slice(0, 50)
```
Target files:
- `src/codeintel/build/causal_analysis/cpg_edge_integrity.py`
- `src/codeintel/build/causal_analysis/cpg_symbol_destination_audit.py`
Implementation checklist:
- [ ] Replace row-wise audits with anti-join plans.
- [ ] Use stable sort and bounded sampling for reports.
- [ ] Emit counts as structured outputs instead of ad hoc prints.

## Sequencing Recommendation
1) Runtime profiles + QuerySpec control plane (Scopes 01-02).
2) Finalize gate + kernel lane consolidation (Scopes 04-05).
3) Build graph pipelines + CPG2 assembly (Scopes 06-07).
4) Ingestion pipelines + ingestion compute (Scopes 08-09).
5) Analytics alignment + observability artifacts (Scopes 10-11).
6) Exports/materializers + causal audits (Scopes 12-13).

## Expected Outcome
After this plan, core/build/analytics/ingestion pipelines will:
- Use plan-first Acero execution with centralized QuerySpec scans.
- Keep row-count changes in kernel helpers (explode, sort, dedupe).
- Enforce determinism and structured artifacts at finalize boundaries.
- Emit run manifests and scan telemetry for reproducibility.
- Remain modular, high-throughput, and consistent across the repo.
