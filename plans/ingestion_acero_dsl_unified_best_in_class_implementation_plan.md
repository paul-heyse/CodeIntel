# Ingestion Acero/DSL Unified Best-in-Class Implementation Plan

## Goal
Rebuild ingestion to be plan-first (Acero/DSL), reader-first (streaming),
and finalize-first (determinism + artifacts) while preserving explicit
carve-outs for Python parsing steps.

## Guiding Principles
- Plan-first: all filtering/projection/join logic goes through the DSL.
- Reader-first: use `to_reader()` + `finalize_reader` as the default boundary.
- Kernel lane: row-count-changing transforms live in standardized helpers.
- Determinism is explicit: ordering metadata is declared and enforced at finalize.
- Carve-outs are intentional: parsing stays Python, but output enters Arrow ASAP.

## Scope 01 — Ingestion DSL Facade + Runtime Profiles
Description:
Create a thin ingestion-specific DSL facade that standardizes QuerySpec +
ExecutionContext usage and removes ad hoc scan/plan construction in ingestion.

Representative pattern:
```python
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.plan_ops import build_query_plan_for_context
from codeintel.ingestion.compute.base import build_ingest_query_spec

spec = build_ingest_query_spec(
    table_key="core.modules",
    columns=("repo", "commit", "path", "module", "language"),
    repo=repo,
    commit=commit,
)
plan = build_query_plan_for_context(dataset, spec=spec, ctx=ExecutionContext())
reader = plan.to_reader(use_threads=True)
```

Target files:
- `src/codeintel/ingestion/compute/plan_surface.py` (new)
- `src/codeintel/ingestion/compute/base.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/execution_context.py`
- `src/codeintel/core/columnar/queryspec.py`

Implementation checklist:
- [ ] Add a dedicated ingestion plan facade that accepts `table_key`, `scope`,
      and `ExecutionContext`.
- [ ] Route ingestion scans through QuerySpec + context defaults.
- [ ] Expose a minimal surface: `ingest_plan_for_table(...)`,
      `ingest_reader_for_table(...)`.

## Scope 02 — QuerySpec Scans + Plan-First Scoping
Description:
Replace any residual ad hoc scoping and scanning with QuerySpec plans,
including ingestion scoping by repo/commit and optional path filters.

Representative pattern:
```python
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.ingestion.compute.base import build_ingest_query_spec
from codeintel.build.tabular.plan_ops import Plan

scope = SnapshotScope.from_snapshot(env.snapshot)
spec = build_ingest_query_spec("core.modules", repo=scope.repo, commit=scope.commit)
plan = Plan.table(tabular_to_arrow_table(value))
predicate = spec.scan_filter_expression()
if predicate is not None:
    plan = plan.filter(predicate)
```

Target files:
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`
- `src/codeintel/build/hamilton/native/ingestion/pipelines.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/ingestion/compute/base.py`
- `src/codeintel/ingestion/compute/repo_scan.py`

Implementation checklist:
- [ ] Replace any remaining `tabular_to_scoped_table` with QuerySpec-based scoping.
- [ ] Ensure pushdown projection/predicate are built from QuerySpec only.
- [ ] Keep scoping behavior consistent for table inputs and dataset inputs.

## Scope 03 — Reader-First Finalize Gate
Description:
Enforce a single ingestion boundary: `finalize_reader(...)` over streaming
readers. Avoid `to_table()` or `reader_to_table` before finalize except in
explicit, narrow debug boundaries.

Representative pattern:
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_reader

final = finalize_reader(
    reader,
    spec=FinalizeSpec(table_key="core.syntax_nodes", mode="tolerant"),
)
persist_arrow_tables(storage, {"core.syntax_nodes": final.good}, scope=scope)
```

Target files:
- `src/codeintel/ingestion/compute/base.py`
- `src/codeintel/ingestion/compute/cst_extract.py`
- `src/codeintel/ingestion/compute/tree_sitter_index.py`
- `src/codeintel/ingestion/compute/ast_extract.py`
- `src/codeintel/ingestion/compute/symtable_extract.py`
- `src/codeintel/ingestion/compute/dis_extract.py`
- `src/codeintel/ingestion/compute/inspect_extract.py`

Implementation checklist:
- [ ] Replace `.to_table()` output boundaries with `.to_reader()` plus finalize.
- [ ] Disallow `reader_to_table` in ingestion unless explicitly required
      by downstream APIs.
- [ ] Ensure all ingestion outputs pass through finalize once, not multiple times.

## Scope 04 — Kernel Lane: List Explode + Aligned Lists
Description:
Standardize edge creation using kernel helpers (list_parent_indices + take),
including list-aligned validation and null list policies.

Representative pattern:
```python
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges_with_aligned_lists

spec = ExplodeSpec(
    src_col="src_id",
    dst_list_col="callee_ids",
    aligned_list_cols=("callsite_spans",),
    repeat_cols=("repo", "commit", "rel_path"),
    null_list_policy="error",
)
result = explode_edges_with_aligned_lists(table, spec=spec)
edges = result.good
```

Target files:
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/ingestion/compute/cst_extract.py`
- `src/codeintel/ingestion/compute/tree_sitter_index.py`
- `src/codeintel/core/columnar/explode_ops.py`

Implementation checklist:
- [ ] Replace Python row loops used for edge construction with explode helpers.
- [ ] Enforce list-aligned validation for per-edge attributes.
- [ ] Emit structured error tables for misaligned lists or null list policies.

## Scope 05 — Columnar Buffers and Batch Collectors Only
Description:
Use `ColumnarRowBuffer` / `ColumnarBatchCollector` as the only row assembly
mechanism. Remove `pa.Table.from_pylist` and large Python list builds.

Representative pattern:
```python
from codeintel.core.columnar.rows import columnar_batch_collector_for_table_key

collector = columnar_batch_collector_for_table_key("core.syntax_nodes", batch_size=4096)
collector.append({"repo": repo, "commit": commit, "node_id": node_id, "kind": kind})
reader = collector.to_reader()
```

Target files:
- `src/codeintel/ingestion/compute/ast_extract.py`
- `src/codeintel/ingestion/compute/cst_extract.py`
- `src/codeintel/ingestion/compute/tree_sitter_index.py`
- `src/codeintel/ingestion/compute/dis_extract.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`

Implementation checklist:
- [ ] Replace `pa.Table.from_pylist` with columnar buffers.
- [ ] Avoid building large Python lists before appending to buffers.
- [ ] Prefer `ColumnarBatchCollector` when row volume is large or streaming.

## Scope 06 — Ingestion Normalization Helpers
Description:
Centralize ingestion normalization (scoping, projection, contract alignment)
in shared helpers to avoid bespoke logic per module.

Representative pattern:
```python
from codeintel.build.hamilton.transforms.ingestion_normalize import scoped_table_for_ingest

scoped = scoped_table_for_ingest(
    value,
    table_key="core.modules",
    scope=scope,
    columns=("repo", "commit", "path", "module", "language"),
    require_scope_columns=True,
)
```

Target files:
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/file_line_index.py`
- `src/codeintel/ingestion/compute/base.py`

Implementation checklist:
- [ ] Ensure all ingestion scoping calls route through a single helper.
- [ ] Align projection defaults to the schema registry.
- [ ] Keep normalization behaviors consistent between table and dataset inputs.

## Scope 07 — Storage Adapters: Streaming Writes
Description:
Avoid materializing RecordBatchReaders before persistence by adding
streaming writer paths to storage adapters.

Representative pattern:
```python
def write_reader(self, table_key: str, reader: pa.RecordBatchReader, *, scope: str | None) -> None:
    relation = self._gateway.relation_from_arrow_reader(reader)
    self._backend.bulk_insert_relation(table_key, relation, scope=scope)
```

Target files:
- `src/codeintel/ingestion/adapters/duckdb_storage.py`
- `src/codeintel/ingestion/ports/storage.py`
- `src/codeintel/storage/gateway.py`
- `src/codeintel/storage/duckdb_policy_backend.py`

Implementation checklist:
- [ ] Add a reader-based insert path that avoids `reader_to_table`.
- [ ] Keep existing table-based writes as a fallback.
- [ ] Preserve finalize policies before persistence (strict mode).

## Scope 08 — Determinism + Provenance for Ingestion
Description:
Ensure ingestion output ordering and determinism tiers are explicit and
enforced at finalize; expose provenance columns for error artifacts.

Representative pattern:
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec

spec = FinalizeSpec(
    table_key="core.syntax_edges",
    mode="tolerant",
    order_by=(("repo", "ascending"), ("commit", "ascending"), ("src_id", "ascending")),
    emit_artifacts=True,
)
```

Target files:
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/columnar/ordering.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/ingestion/compute/base.py`

Implementation checklist:
- [ ] Require canonical sort keys for ingestion tables when determinism is canonical.
- [ ] Propagate provenance columns into error artifacts when enabled.
- [ ] Keep ordering transitions explicit in plan metadata.

## Scope 09 — Extraction Targets Orchestration
Description:
Make extraction targets return readers and rely on shared finalize/persist
steps rather than bespoke materialization logic per target.

Representative pattern:
```python
readers = {
    "core.ast_nodes": collectors.ast_nodes.to_reader(),
    "core.ast_metrics": collectors.metrics.to_reader(),
}
finalized, warnings = finalize_arrow_readers(readers)
persist_arrow_tables(storage, finalized, scope=scope)
```

Target files:
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- `src/codeintel/ingestion/compute/ast_extract.py`
- `src/codeintel/ingestion/compute/cst_extract.py`
- `src/codeintel/ingestion/compute/symtable_extract.py`
- `src/codeintel/ingestion/compute/inspect_extract.py`
- `src/codeintel/ingestion/compute/dis_extract.py`

Implementation checklist:
- [ ] Return readers from compute steps and finalize in the target layer.
- [ ] Centralize warning/telemetry propagation for all extraction targets.
- [ ] Keep compute steps pure and free of persistence logic.

## Scope 10 — Validation and Telemetry (Non-Pytest)
Description:
Add runtime validation/telemetry for ingestion runs without pytest by
emitting run manifests and scan telemetry.

Representative pattern:
```python
from codeintel.core.columnar.run_manifest import write_run_manifest, RunManifestOptions

write_run_manifest(
    output_dir,
    options=RunManifestOptions(
        profile_name=ctx.runtime_profile.name,
        determinism=ctx.resolve_determinism(),
        ordering=plan.ordering,
        scan_telemetry=telemetry,
    ),
)
```

Target files:
- `src/codeintel/core/columnar/run_manifest.py`
- `src/codeintel/build/hamilton/diagnostics.py`
- `src/codeintel/build/graphs/validation/runner.py`
- `src/codeintel/ingestion/compute/base.py`

Implementation checklist:
- [ ] Emit run manifests for ingestion pipelines and validation runs.
- [ ] Record scan telemetry for ingestion datasets.
- [ ] Attach finalize artifacts (alignment/stats/errors) to run metadata.

## Sequencing Recommendation
1) DSL facade + QuerySpec scoping (Scopes 01-02).
2) Reader-first finalize gate + columnar buffers (Scopes 03-05).
3) Kernel lane explode + normalization helpers (Scopes 04-06).
4) Storage streaming + determinism policies (Scopes 07-08).
5) Orchestration + telemetry (Scopes 09-10).

## Expected Outcome
After this plan, ingestion pipelines are:
- Fully plan-first and Acero/DSL driven.
- Streaming by default with explicit finalize boundaries.
- Deterministic when required and observable via structured artifacts.
- Modular and maintainable with a single ingestion DSL surface.
