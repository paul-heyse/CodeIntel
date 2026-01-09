# Build Shared Utilities Adoption Followup Plan (Ingestion + Scans + Joins)

## Purpose
Extend shared compute utilities adoption in `src/codeintel/build` based on the agreed best-in-class
choices and the identified gaps. This plan focuses on ingestion boundaries, deterministic join
behavior, nested payload normalization, and scan pushdown/telemetry.

## Best-in-class decisions (resolved)
- Finalize gates should cover ingestion outputs that become persisted datasets or shared inputs
  (syntax/scip/symtable), not only graph/analytics tables.
- SCIP external-symbol joins should produce deterministic ordering (e.g., repo/commit/symbol), even
  if logical ordering is not required, to stabilize diffs, caching, and tests.

---

## Scope Item 1: Plan/HashJoinSpec adoption + deterministic ordering for SCIP external-symbol joins

### Rationale
`scip.py` currently uses raw Acero declarations for distinct/anti joins, bypassing the
Plan/HashJoinSpec policy (no key validity filtering, explicit outputs, or deterministic ordering).

### Status
Completed.

### Completed changes
- Migrated external-symbol distinct + anti joins to `Plan`/`HashJoinSpec` with deterministic
  ordering via `Plan.order_by`.
- Added join-key prechecks via `finalize_table` (tolerant) for external-symbol inputs to route
  invalid keys into finalize error tables before joins.
- Centralized SCIP base output finalization through `finalize_ingest_table`.

### Pattern to deploy
```python
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan

join_keys = ["repo", "commit", "symbol"]

left = Plan.table(left_table).project({name: E.field(name) for name in join_keys})
right = Plan.table(right_table).project({name: E.field(name) for name in join_keys})

joined = left.hash_join(
    right=right,
    spec=HashJoinSpec(
        left_keys=join_keys,
        right_keys=join_keys,
        how="left anti",
        left_output=join_keys,
        right_output=[],
    ),
)

result = (
    joined.order_by(sort_keys=[(key, "ascending") for key in join_keys])
    .to_table(use_threads=True)
)
```

### Target files
- src/codeintel/build/hamilton/native/ingestion/scip.py

### Checklist
- Replace `_distinct_external_symbol_rows` with `Plan.table(...).aggregate(...)` and apply a
  deterministic sort on the output.
- Replace `_left_anti_external_symbols` with `Plan.table(...).hash_join(...)` using `HashJoinSpec`.
- Apply join-key validity prechecks (finalize tolerant) where table keys are known.
- Ensure explicit `left_output`/`right_output` to avoid payload bloat.
- Apply `Plan.order_by` for deterministic output ordering.

---

## Scope Item 2: Finalize gate adoption for ingestion outputs

### Rationale
Multiple ingestion outputs stop at `align_table_to_contract` + `dedupe_table_for_table`, so finalize
invariants, artifacts, and strict/tolerant policy are not enforced.

### Status
Completed.

### Finalize policy table (syntax/scip/symtable)

| Group | Table keys | Target name | Mode | Required non-null | Invariants | Dedupe | Artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Syntax (augment) | core.parse_manifest<br>core.syntax_nodes_augmented<br>core.syntax_edges_augmented<br>core.ts_nodes<br>core.ts_edges<br>core.ts_syntax_node_xref | syntax_augment | strict | Schema non-nullable columns | list_alignment_specs_for_table_key(table_key) | Enabled | emit_artifacts=True |
| Syntax (augment diagnostics) | core.ts_weld_coverage | syntax_augment | tolerant | Schema non-nullable columns | list_alignment_specs_for_table_key(table_key) | Enabled | emit_artifacts=True |
| Syntax (enrich) | core.syntax_defs_resolved<br>core.syntax_refs_resolved<br>core.syntax_calls_resolved<br>core.syntax_imports_resolved | syntax_enrich | strict | Schema non-nullable columns | list_alignment_specs_for_table_key(table_key) | Enabled | emit_artifacts=True |
| SCIP (core) | core.scip_symbols<br>core.scip_occurrences<br>core.scip_symbol_information<br>core.scip_symbol_relationships<br>core.scip_external_symbols<br>core.scip_module_state | scip | strict | Schema non-nullable columns | list_alignment_specs_for_table_key(table_key) | Enabled; prefer columns for scip_external_symbols | emit_artifacts=True |
| SCIP (metadata/diagnostics) | core.scip_diagnostics<br>core.scip_index_metadata | scip | tolerant | Schema non-nullable columns | list_alignment_specs_for_table_key(table_key) | Enabled | emit_artifacts=True |
| SCIP resolution | core.scip_symbol_goid_xref<br>core.scip_occurrence_span_xref<br>core.scip_occurrence_syntax_xref | scip_resolution | strict | Schema non-nullable columns | list_alignment_specs_for_table_key(table_key) | Enabled | emit_artifacts=True |
| Symtable | core.py_sym_scopes<br>core.py_sym_symbols<br>core.py_sym_scope_edges<br>core.py_sym_namespace_edges<br>core.py_sym_function_partitions<br>core.py_sym_bindings<br>core.py_sym_unresolved_bindings<br>core.py_sym_resolution_edges | symtable | strict | Schema non-nullable columns | list_alignment_specs_for_table_key(table_key) | Enabled | emit_artifacts=True |

Notes:
- Required non-null columns are derived from the `TableSchema` for each table key.
- Dedupe prefer columns for `core.scip_external_symbols`: `package_manager`, `package_name`, `package_version`.
- `list_alignment_specs_for_table_key` currently returns no specs for these tables; keep the hook for future list alignment rules.

### Completed changes
- Added a shared finalize policy helper (`finalize_ingest_table`) with schema-driven required
  non-null columns, list-alignment invariants, and per-table dedupe preferences.
- Replaced `align_table_to_contract` + `dedupe_table_for_table` with `finalize_ingest_table` across
  syntax/scip/symtable ingestion outputs, including symtable payload assembly and scip resolution
  outputs.

### Pattern to deploy
```python
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table

result = finalize_table(
    table,
    spec=FinalizeSpec(
        table_key=table_key,
        mode="strict",
        required_non_null=("repo", "commit"),
        invariants=(),
        emit_artifacts=True,
        target_name=target_name,
    ),
)

good = result.good
errors = result.errors
```

### Target files
- src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py
- src/codeintel/build/hamilton/native/ingestion/syntax_augment.py
- src/codeintel/build/hamilton/native/ingestion/extraction_targets.py
- src/codeintel/build/hamilton/native/ingestion/scip_resolution.py
- src/codeintel/build/hamilton/native/ingestion/scip.py
- src/codeintel/build/hamilton/transforms/ingestion_normalize.py

### Checklist
- Define a table-key driven finalize policy (strict vs tolerant) for syntax/scip/symtable outputs.
- Replace `align_table_to_contract` + `dedupe_table_for_table` with `finalize_table` for ingestion
  outputs; use `emit_artifacts=True` and surface artifacts where available.
- Use strict mode for canonical/shared inputs (syntax nodes/edges, scip symbol xref, symtable core
  outputs) and tolerant mode for high-variance or best-effort outputs.
- Route dedupe via `FinalizeSpec(dedupe=...)` instead of separate calls.
- Centralize finalize configuration in `ingestion_normalize.py` (or a new helper module) so
  ingestion paths share one policy surface.

---

## Scope Item 3: Adopt nested_ops for extras struct assembly + deep cast

### Rationale
`syntax_augment.py` builds struct payloads manually, which can drift as fields evolve. The
`nested_ops` helpers enforce typed extras and contract-aligned casts.

### Status
Completed.

### Completed changes
- Replaced manual struct assembly with `make_extras_struct` and contract-derived field typing for
  tree-sitter payloads and `extras`.
- Added a contract-driven deep cast for `core.syntax_nodes_augmented` to align nested extras.

### Pattern to deploy
```python
from codeintel.build.tabular.nested_ops import make_extras_struct, deep_cast_table_to_contract
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema

extras = make_extras_struct(
    table,
    fields={
        "ast_nodes": pa.list_(pa.struct([...]))
        "ts_nodes": pa.list_(pa.struct([...]))
    },
)

contract = arrow_contract_for_table_schema(table_schema)
casted = deep_cast_table_to_contract(table, contract)
```

### Target files
- src/codeintel/build/hamilton/native/ingestion/syntax_augment.py

### Checklist
- Replace manual `pa.StructArray.from_arrays(...)` payload assembly with `make_extras_struct`.
- Use `deep_cast_table_to_contract` after alignment to ensure nested types match the contract.
- Keep field ordering consistent with the contract schema to avoid downstream drift.

---

## Scope Item 4: Scan pushdown + telemetry adoption in build readers

### Rationale
Several build readers call `scan_parquet_dataset` or direct scanners without pushdown/telemetry
options, losing projection mapping and scan metrics.

### Status
Completed.

### Completed changes
- Swapped build export and post-run readers to `scan_parquet_dataset_with_telemetry` and logged
  telemetry payloads.
- Updated causal analysis scans to use reader-based scans with telemetry + normalization fallback.

### Pattern to deploy
```python
from codeintel.core.datasets.scanning import ParquetScanOptions, scan_parquet_dataset_with_telemetry

options = ParquetScanOptions(
    columns=("repo", "commit", "rel_path"),
    repo=repo,
    commit=commit,
)

reader, telemetry = scan_parquet_dataset_with_telemetry(
    dataset_root=dataset_root,
    table_key=table_key,
    snapshot_id=snapshot_id,
    options=options,
)
```

### Target files
- src/codeintel/build/exports/common.py
- src/codeintel/build/hamilton/post_run_quality_outputs.py
- src/codeintel/build/causal_analysis/scan_utils.py

### Checklist
- Replace `scan_parquet_dataset` with `scan_parquet_dataset_with_telemetry` where possible.
- Use `ParquetScanOptions` for columns/repo/commit filters and enable telemetry.
- Thread telemetry into logs or metadata at scan boundaries (reuse existing reporting patterns).
- Keep fallback behavior for missing snapshots.

---

## Scope Item 5: Finalize error stage coverage + join precheck surfacing + reader finalize adoption

### Rationale
Finalize error tables previously captured only required-non-null and invariants. Alignment and
dedupe failures were not represented, join precheck errors were not surfaced beyond the returned
`FinalizeResult`, derived join inputs were filtered by validity rather than routed through finalize,
and ingestion joins always materialized before finalize.

### Status
Partially completed.

### Completed changes
- Added `stage` and `key_fields` to finalize error rows, and populated alignment + dedupe error
  tables for schema alignment and duplicate primary keys.
- Added `finalize_join_keys` and replaced join-key filters with join prechecks in ingestion + CPG
  SCIP joins, with warning logs when rows are dropped.
- Added `finalize_reader` + `finalize_ingest_reader` and used it for the symtable unresolved
  bindings join to avoid an extra materialization before finalize.

### Remaining scope
- Persist join-precheck errors (or attach telemetry) rather than logging-only, so dropped rows are
  visible in artifacts alongside other finalize errors.
- Apply `finalize_ingest_reader` to other ingestion join pipelines where outputs are immediately
  finalized and not reused for further compute.

### Pattern to deploy
```python
from codeintel.build.tabular.finalize_ops import finalize_join_keys

result = finalize_join_keys(
    table,
    required_non_null=join_keys,
    key_fields=join_keys,
)
if result.errors.num_rows:
    LOG.warning(
        "Join key precheck dropped %d rows table=%s keys=%s",
        result.errors.num_rows,
        table_key or "derived",
        ",".join(join_keys),
    )
```

### Target files
- src/codeintel/build/tabular/finalize_ops.py
- src/codeintel/build/hamilton/transforms/ingestion_normalize.py
- src/codeintel/build/hamilton/native/ingestion/extraction_targets.py
- src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py
- src/codeintel/build/hamilton/native/ingestion/syntax_augment.py
- src/codeintel/build/hamilton/native/ingestion/scip_resolution.py
- src/codeintel/build/hamilton/native/ingestion/scip.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py

### Checklist
- Add alignment + dedupe error tables to finalize errors with per-row stages.
- Replace validity filters for derived join inputs with `finalize_join_keys`.
- Log join precheck errors with table name and join keys.
- Introduce reader-based finalize helpers and adopt them where joins feed directly into finalize.

---

## Execution order (recommended)
1. SCIP join migration + deterministic ordering (Scope Item 1).
2. Finalize gate adoption for ingestion outputs (Scope Item 2).
3. Nested extras struct normalization (Scope Item 3).
4. Scan pushdown + telemetry adoption (Scope Item 4).
5. Finalize error stage coverage + join precheck surfacing + reader finalize adoption (Scope Item 5).

## Success criteria
- SCIP external-symbol outputs are deterministic and filtered on valid keys.
- Ingestion outputs are finalized with consistent strict/tolerant policies and artifacts.
- Nested extras payloads are contract-aligned and version-stable.
- Finalize errors include schema alignment + dedupe stages, and join precheck drops are surfaced.
- Build readers use pushdown/telemetry consistently.

---

## Additional adoption (post-scope)

### Completed changes
- Finalize gate adoption for remaining ingestion outputs:
  - `core.modules`, `core.file_state`, `core.repo_map` now use `finalize_ingest_table` with updated
    row counts and shared dedupe preference for file_state.
  - `core.ts_*` ingestion outputs now finalize with strict/tolerant modes (tolerant for
    `core.ts_parse_errors` and `core.ts_changed_ranges`).
  - `core.file_line_index` now finalizes to enforce schema invariants and emit artifacts.
- Call wiring alignment now uses finalize gate for `graph.cpg_call_targets` and
  `graph.cpg_call_candidates`, emitting finalize artifacts and alignment reports.

### Target files
- src/codeintel/build/hamilton/transforms/ingestion_normalize.py
- src/codeintel/build/hamilton/native/ingestion/ingest_targets.py
- src/codeintel/build/hamilton/native/ingestion/tree_sitter.py
- src/codeintel/build/hamilton/native/ingestion/file_line_index.py
- src/codeintel/build/hamilton/native/graphs/call_wiring.py

---

## Best-in-class upgrade plan (compute_improvement_deepdive alignment)

### Phase 1: Non-breaking upgrades

#### 1) Error metadata in finalize outputs

Goal: add stage + key field context to finalize error rows without changing good outputs.

Status
Completed.

Completed changes
- Added `key_fields` to `FinalizeSpec` and threaded primary-key values into error tables.
- Added `stage` to finalize error rows; currently populated for invariant/required-non-null errors.
- Populated key fields in `finalize_ingest_table` and call-wiring finalize calls.

Remaining scope
- Extend stage coverage to schema-alignment and dedupe errors if/when those error sources are added.

Pattern to deploy
```python
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table

result = finalize_table(
    table,
    spec=FinalizeSpec(
        table_key=table_key,
        mode="strict",
        key_fields=("repo", "commit", "rel_path"),
        emit_artifacts=True,
        target_name=target_name,
    ),
)

errors = result.errors
```

Implementation sketch
```python
# finalize_ops.py (error table schema expansion)
def _error_columns(..., spec: ErrorSpec) -> dict[str, pa.Array | pa.ChunkedArray]:
    return {
        "row_id": pc.take(row_id, indices),
        "error_code": pa.array([spec.error_code] * count, type=pa.string()),
        "stage": pa.array([spec.stage] * count, type=pa.string()),
        "column": pa.array([spec.column] * count, type=pa.string()),
        "detail": pa.array([spec.detail] * count, type=pa.string()),
    }
```

Implementation tasks (completed)
- Add key_fields to FinalizeSpec and thread into ErrorSpec creation.
- Add stage to error table rows (currently populated for invariant errors).
- Extend finalize_ingest_table to populate key_fields from table schema primary keys.
- Update call_wiring finalize_table calls to pass key_fields where relevant.

Target files
- src/codeintel/build/tabular/finalize_ops.py
- src/codeintel/build/hamilton/transforms/ingestion_normalize.py
- src/codeintel/build/hamilton/native/graphs/call_wiring.py

Checklist
- Add key_fields to FinalizeSpec and propagate into error tables.
- Add stage to error tables ("schema", "invariant", "dedupe").
- Preserve current good-table outputs.

#### 2) Deterministic dedupe policy

Goal: guarantee stable dedupe by sorting on primary keys (and prefer columns) before
drop_duplicates.

Status
Completed.

Completed changes
- Sorted by primary keys (ascending) with prefer columns (descending) before dedupe in Arrow and
  Polars paths.
- Preserved Arrow fallback dedupe path when drop_duplicates is unavailable.

Pattern to deploy
```python
from codeintel.build.tabular.kernels import stable_sort_indices
from codeintel.build.schemas.service import get_schema_service

schema = get_schema_service().require_table_schema(table_key)
primary = list(schema.primary_key)
sort_keys = [(name, "ascending") for name in primary]
sorted_table = table.take(stable_sort_indices(table, sort_keys=sort_keys))
deduped = sorted_table.drop_duplicates(primary)
```

Implementation tasks (completed)
- Sort by primary keys (ascending) before drop_duplicates to stabilize order.
- Apply prefer_columns sort (descending) before primary-key sort when present.
- Keep compute fallback path for older Arrow builds without drop_duplicates.

Target files
- src/codeintel/build/tabular/dedupe_ops.py
- src/codeintel/build/tabular/finalize_ops.py

Checklist
- Sort by primary keys before dedupe.
- Apply prefer_columns ordering before primary-key sort when present.
- Maintain current fallback paths for Arrow versions without drop_duplicates.

#### 3) Explode error context enrichment

Goal: include scope keys on explode errors to make diagnostics actionable.

Status
Completed.

Completed changes
- Added `error_context_cols` for call-wiring explode edges so error rows include
  repo/commit/rel_path/call_id.

Pattern to deploy
```python
from codeintel.build.tabular.explode_ops import ExplodeSpec, explode_edges

exploded = explode_edges(
    table,
    spec=ExplodeSpec(
        src_col="call_id",
        dst_list_col="callee_ids",
        repeat_cols=("repo", "commit", "rel_path", "call_id"),
        aligned_list_cols=("callee_spans",),
        error_context_cols=("repo", "commit", "rel_path", "call_id"),
    ),
)
```

Implementation tasks (completed)
- Add error_context_cols for explode calls in call_wiring edges.
- Ensure error tables include scope keys (repo/commit/rel_path/call_id).
- Add regression checks for explode error row counts when misalignment occurs.

Target files
- src/codeintel/build/hamilton/native/graphs/call_wiring.py
- src/codeintel/build/tabular/explode_ops.py

Checklist
- Pass error_context_cols for edge explodes.
- Ensure explode errors include scope keys + row_id.

---

### Phase 2: Larger shifts (best-in-class compute)

#### 4) Join-key error routing before hash join

Goal: route invalid join-key rows into finalize errors rather than silently filtering.

Status
Completed (with fallback filtering for derived tables lacking a table key).

Completed changes
- Added pre-join finalize checks (tolerant) with required_non_null=join_keys where table keys are
  known.
- Retained join-key validity filters for derived tables that do not map to a contract table key.

Pattern to deploy
```python
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan

key_fields = ("repo", "commit", "rel_path")
spec = FinalizeSpec(
    table_key=table_key,
    mode="tolerant",
    required_non_null=key_fields,
    target_name=target_name,
    emit_artifacts=True,
)
left_result = finalize_table(left, spec=spec)
right_result = finalize_table(right, spec=spec)

joined = (
    Plan.table(left_result.good)
    .project({name: E.field(name) for name in join_keys})
    .hash_join(
        right=Plan.table(right_result.good).project({name: E.field(name) for name in join_keys}),
        spec=HashJoinSpec(
            left_keys=list(join_keys),
            right_keys=list(join_keys),
            how="left outer",
            left_output=list(join_keys),
            right_output=[],
        ),
    )
)
```

Implementation tasks (completed)
- Add pre-join finalize_table calls for left/right inputs with required_non_null=join_keys.
- Capture errors from pre-join finalization (log or emit artifacts as needed).
- Remove direct key validity filters once finalize handles invalid keys.

Remaining scope
- For derived tables without a contract table key, either introduce a synthetic contract or keep
  filtering until a schema is defined.

Target files
- src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py
- src/codeintel/build/hamilton/native/ingestion/syntax_augment.py
- src/codeintel/build/hamilton/native/ingestion/extraction_targets.py
- src/codeintel/build/hamilton/native/ingestion/scip_resolution.py
- src/codeintel/build/hamilton/native/ingestion/scip.py

Checklist
- Replace join-key filters with finalize strict/tolerant prechecks.
- Surface invalid join-key rows in finalize artifacts.
- Keep join outputs deterministic (stable sort/order_by).

#### 5) Plan-level ordering for deterministic output

Goal: move stable sorting into the Plan to avoid post-materialization sorts.

Status
Completed.

Completed changes
- Replaced post-materialization stable_sort_indices calls with `Plan.order_by` in ingestion join
  helpers (syntax_enrich, syntax_augment, extraction_targets, scip_resolution) and SCIP
  external-symbol joins.
- Applied plan-level ordering in call-wiring edge materialization before finalize.

Pattern to deploy
```python
from codeintel.build.tabular.plan_ops import Plan

ordered = (
    Plan.table(table)
    .order_by(sort_keys=[("repo", "ascending"), ("commit", "ascending")])
)
reader = ordered.to_reader(use_threads=True)
```

Implementation tasks (completed)
- Replace stable_sort_indices post-materialization with Plan.order_by where feasible.
- Ensure sort keys match table schema ordering and join keys.
- Keep deterministic ordering for outputs that feed dedupe/finalize.

Target files
- src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py
- src/codeintel/build/hamilton/native/ingestion/syntax_augment.py
- src/codeintel/build/hamilton/native/ingestion/extraction_targets.py
- src/codeintel/build/hamilton/native/ingestion/scip_resolution.py
- src/codeintel/build/hamilton/native/ingestion/scip.py
- src/codeintel/build/hamilton/native/graphs/call_wiring.py

Checklist
- Replace stable_sort_indices calls with Plan.order_by where feasible.
- Preserve deterministic ordering for join outputs and aggregations.

#### 6) Streaming finalize boundary

Goal: allow finalize at the reader boundary to keep streaming semantics until the gate.

Status
Partially completed.

Completed changes
- Added `finalize_reader` helper in finalize_ops and adopted it for call-wiring edge materialization.

Remaining scope
- Adopt `finalize_reader` in ingestion join pipelines where a reader boundary is available.
- Re-evaluate `materialize_plan(...).to_table()` call sites to keep streaming until finalize.

Pattern to deploy
```python
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_reader

result = finalize_reader(
    reader,
    spec=FinalizeSpec(
        table_key=table_key,
        mode="strict",
        target_name=target_name,
        emit_artifacts=True,
    ),
)
good = result.good
```

Implementation tasks (partially completed)
- Add finalize_reader helper that consumes RecordBatchReader and returns FinalizeResult.
- Keep finalize_reader behavior consistent with finalize_table (strict/tolerant, artifacts).
- Update join pipelines to materialize at finalize_reader boundary rather than to_table.

Target files
- src/codeintel/build/tabular/finalize_ops.py (new helper)
- src/codeintel/build/hamilton/native/graphs/call_wiring.py
- src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py
- src/codeintel/build/hamilton/native/ingestion/syntax_augment.py
- src/codeintel/build/hamilton/native/ingestion/extraction_targets.py
- src/codeintel/build/hamilton/native/ingestion/scip_resolution.py
- src/codeintel/build/hamilton/native/ingestion/scip.py

Checklist
- Add finalize_reader helper that materializes in a single boundary.
- Update join pipelines to use to_reader -> finalize_reader.
- Keep strict/tolerant behavior consistent with finalize_table.
