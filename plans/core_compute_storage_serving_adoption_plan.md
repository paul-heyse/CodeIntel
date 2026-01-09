# Core Compute Utilities Adoption Plan (Storage + Serving)

## Purpose
Adopt the core columnar compute utilities (Acero plans, kernels, finalize gate, and
schema alignment helpers) across storage and serving. The focus is design-stage
changes that improve correctness, consistency, and deterministic outputs.

## Goals
- Unify scan and filter pushdown behavior across storage/serving.
- Route output boundaries through finalize gates (alignment + invariants + dedupe).
- Standardize deterministic ordering and stable identifiers.
- Replace ad-hoc compute fallbacks with core compute helpers.

## Status update
- Scope Items 1–15 completed.
- Advanced follow-up items planned (Scope Items 16–23).

## Non-goals
- Full ingestion pipeline refactors.
- Rewriting business logic unrelated to columnar compute.

---

## Scope Item 1: Finalize gate adoption at serving/storage boundaries (Completed)

### Rationale
Multiple areas align reader/table schemas, then separately validate or dedupe.
Finalize gates provide a single contract boundary for alignment + invariants +
dedupe + error artifacts.

### Pattern to deploy
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table

finalized = finalize_table(
    table,
    spec=FinalizeSpec(
        table_key=table_key,
        mode="tolerant",
        required_non_null=("repo", "commit"),
        invariants=(),
        emit_artifacts=True,
    ),
)

good = finalized.good
errors = finalized.errors
```

### Target files
- src/codeintel/serving/semantic/kernel.py
- src/codeintel/storage/warehouse.py
- src/codeintel/storage/serving/snapshot_service.py
- src/codeintel/storage/repositories/base.py

### Detailed checklist
- Identify output boundaries that currently call `align_reader_to_contract`.
- For each boundary, materialize table (or implement reader-level finalize) and
  replace alignment-only with finalize gate.
- Decide `FinalizeSpec` per table: `mode`, `required_non_null`, and invariants.
- If errors are emitted, surface artifacts in logs/metrics or store alongside export.
- Ensure strict mode is used where failure should abort export or materialization.

---

## Scope Item 2: Unify parquet scan pushdown + telemetry (Completed)

### Rationale
Multiple scan entry points use `dataset.scanner()` directly. Centralize on
`DatasetScanOptions` and `scan_parquet_dataset_with_telemetry` to standardize
pushdown filters/projections and scan metrics.

### Pattern to deploy
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.datasets.scanning import ParquetScanOptions, scan_parquet_dataset_with_telemetry

options = ParquetScanOptions(
    columns=["repo", "commit", "rel_path"],
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
- src/codeintel/serving/semantic/duckdb_relation_builder.py
- src/codeintel/storage/queries/parquet.py
- src/codeintel/storage/serving/snapshot_service.py
- src/codeintel/storage/datasets/manifest_index.py

### Detailed checklist
- Replace direct `dataset.scanner(...)` uses with `DatasetScanOptions` or
  `scan_parquet_dataset_with_telemetry`.
- Route filters/projections through `ParquetScanOptions`.
- Enable `metrics_enabled` and record telemetry at plan boundaries.
- Ensure `configure_arrow_threading()` is called once per process at scan entry.

---

## Scope Item 3: Consolidate compute helpers in parquet queries (Completed)

### Rationale
`src/codeintel/storage/queries/parquet.py` re-implements count/duplicate/orphan
checks with manual fallbacks. Use core compute helpers and/or Acero plans for
consistent behavior and kernel fallbacks.

### Pattern to deploy
```python
from codeintel.core.columnar.compute import count_distinct, count_non_positive, orphan_ref_count
from codeintel.core.columnar.masks import filter_valid

values = table.column(column)
non_positive = count_non_positive(values)

distinct = count_distinct(values)

target = filter_valid(target_values)
orphan_count = orphan_ref_count(values, target, allow_null=True)
```

### Target files
- src/codeintel/storage/queries/parquet.py

### Detailed checklist
- Replace `_count_non_positive` and `_count_duplicates` with core compute helpers.
- Replace orphan count logic with `orphan_ref_count`.
- Keep fallback logic only where kernel unavailability is possible.
- Maintain existing behavior for null handling and return defaults.

---

## Scope Item 4: Deterministic ordering at output boundaries (Completed)

### Rationale
Serving exports and manifest reads should be deterministic. Use stable sort indices
or `Plan.order_by` with explicit sort keys.

### Pattern to deploy
```python
from codeintel.core.columnar.kernels import stable_sort_indices

sorted_table = table.take(
    stable_sort_indices(table, sort_keys=[("repo", "ascending"), ("commit", "ascending")])
)
```

### Target files
- src/codeintel/serving/semantic/kernel.py
- src/codeintel/storage/tracking/build_tracking.py
- src/codeintel/serving/semantic/duckdb_relation_builder.py

### Detailed checklist
- Identify export paths that currently rely on implicit ordering.
- Add explicit sort keys near export/materialization boundaries.
- Ensure sort keys are aligned with primary keys or contract ordering.
- Confirm ordering does not conflict with performance for very large exports.

---

## Scope Item 5: Dedupe policy at contract boundaries (Completed)

### Rationale
Duplicated rows appear in multiple flows (manifest table, exports). Centralize
via `dedupe_table_for_table` or finalize dedupe configuration.

### Pattern to deploy
```python
from codeintel.core.columnar.dedupe_ops import dedupe_table_for_table

clean = dedupe_table_for_table(
    table_key,
    table,
    prefer_columns=("computed_at",),
)
```

### Target files
- src/codeintel/storage/tracking/build_tracking.py
- src/codeintel/storage/warehouse.py
- src/codeintel/serving/semantic/kernel.py

### Detailed checklist
- Add dedupe as part of finalize gate where possible.
- For legacy paths, apply dedupe just before write or export.
- Verify primary keys are defined in schema registry for these tables.

---

## Scope Item 6: Deep casting + chunk consolidation (Completed)

### Rationale
Nested columns require stable type casting, and chunked arrays can degrade
performance in downstream kernels. Use `deep_cast_table_to_contract` and
`combine_table_chunks` at boundary points.

### Pattern to deploy
```python
from codeintel.core.columnar.compute_helpers import combine_table_chunks
from codeintel.core.columnar.nested_ops import deep_cast_table_to_contract

compact = combine_table_chunks(table)
casted = deep_cast_table_to_contract(compact, contract_schema)
```

### Target files
- src/codeintel/serving/semantic/kernel.py
- src/codeintel/storage/serving/snapshot_service.py
- src/codeintel/serving/semantic/engines/polars_engine.py

### Detailed checklist
- Apply `combine_table_chunks` before finalize and/or export.
- Apply deep cast after schema alignment and before export or storage.
- Validate list/struct/map types match contract schema post-cast.

---

## Scope Item 7: Plan-based scans where feasible (Acero DSL) (Completed)

### Rationale
Use `Plan` for scan -> project -> filter -> order chains to standardize compute
behavior and enable later Acero pushdown improvements.

### Pattern to deploy
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import Plan

plan = (
    Plan.scan(
        dataset,
        columns={"repo": E.field("repo"), "commit": E.field("commit")},
        filter_expr=E.field("repo") == E.scalar(repo),
    )
    .project({"repo": E.field("repo"), "commit": E.field("commit")})
    .filter(E.is_valid("repo"))
)

reader = plan.to_reader(use_threads=True)
```

### Target files
- src/codeintel/storage/queries/parquet.py
- src/codeintel/serving/semantic/duckdb_relation_builder.py

### Detailed checklist
- Identify scan-only paths with simple project/filter patterns.
- Replace `dataset.scanner(...)` with `Plan.scan(...).project(...).filter(...)`.
- Ensure `Plan` output is used as a `RecordBatchReader` where streaming is required.
- Keep fallbacks to current scan paths for unsupported kernels or data types.

---

## Scope Item 8: Arrow-aware finalize gate in ingestion storage port (Completed)

### Rationale
Ingestion compute steps write raw row sequences through the storage port without
finalize, dedupe, or deep casting. Introduce an Arrow-aware write path so the
same finalize + ordering guarantees apply at ingestion boundaries.

### Decision
Use `pa.Table` as the canonical Arrow write surface for ingestion boundaries.
Provide a convenience `write_reader(...)` overload that materializes a
`RecordBatchReader` to a `pa.Table` immediately, then applies finalize, dedupe,
deep cast, chunk consolidation, and stable ordering. This keeps a single,
explicit materialization point aligned with finalize gate requirements.

### Pattern to deploy
```python
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.compute_helpers import combine_table_chunks
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.kernels import stable_sort_indices
from codeintel.core.columnar.nested_ops import deep_cast_table_to_contract
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.service import get_schema_service

table_schema = get_schema_service().require_table_schema(table_key)
contract = arrow_contract_for_table_schema(table_schema=table_schema)
required_non_null = tuple(
    column.name for column in table_schema.columns if not column.nullable
)

table = reader_to_table(reader, schema=contract)
table = combine_table_chunks(table)
table = deep_cast_table_to_contract(table, contract)
finalized = finalize_table(
    table,
    spec=FinalizeSpec(
        table_key=table_key,
        mode="strict",
        required_non_null=required_non_null,
        invariants=(),
        emit_artifacts=True,
    ),
)
good = finalized.good
if table_schema.primary_key:
    sort_keys = [(key, "ascending") for key in table_schema.primary_key]
    good = good.take(stable_sort_indices(good, sort_keys=sort_keys))
```

### Target files
- src/codeintel/ingestion/ports/storage.py
- src/codeintel/ingestion/adapters/duckdb_storage.py
- tests/_helpers/fakes/storage.py

### Detailed checklist
- Extend `IngestStoragePort` with `write_table`/`write_reader` for Arrow payloads.
- Implement finalize + deep cast + dedupe + ordering in the adapter write path.
- Preserve existing `write_batch` for unit tests or list-based callers.
- Surface finalize artifacts in ingestion logs/telemetry when strict mode fails.

---

## Scope Item 9: Route ingestion compute outputs through Arrow writes (Completed)

### Rationale
Ingestion compute modules already assemble structured rows; routing through the
Arrow write path allows finalize gates to enforce invariants and ordering.

### Pattern to deploy
```python
from codeintel.core.columnar.rows import columnar_batch_collector_for_table_key
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE

collector = columnar_batch_collector_for_table_key(
    table_key,
    batch_size=DEFAULT_ARROW_BATCH_SIZE,
)
for row in rows:
    collector.append(row)

table = collector.to_table()
result = storage.write_table(table_key, table, scope=scope)
```

### Target files
- src/codeintel/ingestion/compute/ast_extract.py
- src/codeintel/ingestion/compute/cst_extract.py
- src/codeintel/ingestion/compute/dis_extract.py
- src/codeintel/ingestion/compute/docstrings_extract.py
- src/codeintel/ingestion/compute/inspect_extract.py
- src/codeintel/ingestion/compute/symtable_extract.py
- src/codeintel/ingestion/compute/tree_sitter_index.py
- src/codeintel/ingestion/compute/config_ingest.py
- src/codeintel/ingestion/compute/typing_ingest.py
- src/codeintel/ingestion/compute/tests_ingest.py
- src/codeintel/ingestion/compute/repo_scan.py

### Detailed checklist
- Replace tuple/list row assembly with `ColumnarBatchCollector` where possible.
- Call the new Arrow-aware storage write method for table outputs.
- Ensure compute steps pass `scope` for finalize error visibility.
- Keep batch sizing consistent with existing ingestion profiles.

---

## Scope Item 10: Finalize gate for core columnar row assembly (Completed)

### Rationale
`table_for_columnar_rows` aligns to the contract but does not run finalize or
dedupe. Adding finalize here centralizes invariants for all row-buffer clients.

### Pattern to deploy
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.schemas.service import get_schema_service

table_schema = get_schema_service().require_table_schema(table_key)
required_non_null = tuple(
    column.name for column in table_schema.columns if not column.nullable
)

finalized = finalize_table(
    aligned_table,
    spec=FinalizeSpec(
        table_key=table_key,
        mode="tolerant",
        required_non_null=required_non_null,
        invariants=(),
        emit_artifacts=True,
    ),
)
good = finalized.good
```

### Target files
- src/codeintel/core/columnar/rows.py

### Detailed checklist
- Add an optional finalize step to `table_for_columnar_rows`.
- Route dedupe and required non-null checks through the finalize spec.
- Expose finalize artifacts to callers that need diagnostics.
- Keep a no-finalize path for callers explicitly opting out.

---

## Scope Item 11: Dataset read boundary finalize + deep cast (Completed)

### Rationale
Dataset reads in storage repositories return raw Arrow tables without finalize,
deep casts, or chunk consolidation. Align read paths with serving/storage
boundaries to keep invariants consistent.

### Pattern to deploy
```python
from codeintel.core.columnar.compute_helpers import combine_table_chunks
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.nested_ops import deep_cast_table_to_contract
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.service import get_schema_service

table_schema = get_schema_service().require_table_schema(table_key)
contract = arrow_contract_for_table_schema(table_schema=table_schema)

table = combine_table_chunks(table)
table = deep_cast_table_to_contract(table, contract)
finalized = finalize_table(
    table,
    spec=FinalizeSpec(
        table_key=table_key,
        mode="tolerant",
        required_non_null=tuple(
            column.name for column in table_schema.columns if not column.nullable
        ),
        invariants=(),
        emit_artifacts=True,
    ),
)
```

### Target files
- src/codeintel/storage/repositories/datasets.py

### Detailed checklist
- Materialize readers to tables and consolidate chunks before returning.
- Apply deep cast to contract schema prior to finalize.
- Keep tolerant mode for read paths; only fail on invariants if configured.
- Reuse finalized table for dict/row conversions to avoid inconsistencies.

---

## Scope Item 12: Plan-based scans in core dataset utilities (Completed)

### Rationale
Core dataset scanning utilities still rely on scanner-only paths. Adding an
Acero plan path mirrors storage/serving behavior and keeps pushdown consistent.

### Pattern to deploy
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import Plan

plan = Plan.scan(
    dataset,
    columns=columns,
    filter_expr=filter_expression,
)
reader = plan.to_reader(use_threads=scan_options.use_threads)
```

### Target files
- src/codeintel/core/datasets/scanning.py
- src/codeintel/core/datasets/arrow_store.py

### Detailed checklist
- Add a `Plan.scan` path for simple project/filter scans.
- Preserve `DatasetScanOptions` behavior (batch size, threads, projection).
- Keep fallback to `build_scanner` when plan execution is unsupported.
- Thread telemetry for plan-based scans through existing logging hooks.

---

## Scope Item 13: Deep cast + chunk consolidation before finalize in warehouse writes (Completed)

### Rationale
Warehouse write paths finalize tables without chunk consolidation or deep casting.
Aligning to the standard nested-ops flow (combine chunks, deep cast, finalize) keeps
storage writes consistent with dataset read boundaries and avoids downstream type drift.

### Pattern to deploy
```python
from codeintel.core.columnar.compute_helpers import combine_table_chunks
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.nested_ops import deep_cast_table_to_contract

table = combine_table_chunks(table)
table = deep_cast_table_to_contract(table, contract_schema)
finalized = finalize_table(
    table,
    spec=FinalizeSpec(table_key=table_key, mode=_finalize_mode(validation_mode)),
)
good = finalized.good
```

### Target files
- src/codeintel/storage/warehouse.py

### Detailed checklist
- Identify the table-materialization branch in `_write_tabular`.
- Apply `combine_table_chunks` before finalize when the input is an Arrow table.
- Apply `deep_cast_table_to_contract` when a contract schema is available.
- Keep finalize mode aligned to the existing validation mode selection.
- Ensure schema metadata remains set after casting and before write.

---

## Scope Item 14: Deterministic ordering fallback for warehouse writes/exports (Completed)

### Rationale
Serving paths already use hash-based ordering fallbacks. Applying the same fallback
for warehouse writes and any remaining export paths ensures deterministic ordering
when no explicit sort keys are configured.

### Pattern to deploy
```python
from codeintel.core.columnar.kernels import hash_struct_ordinal, stable_sort_indices

ordinal = hash_struct_ordinal(
    table,
    columns=tuple(primary_key),
    modulus=2**31 - 1,
)
table_with = table.append_column("__ci_ordinal", ordinal)
indices = stable_sort_indices(table_with, sort_keys=[("__ci_ordinal", "ascending")])
ordered = table_with.take(indices).drop_columns(["__ci_ordinal"])
```

### Target files
- src/codeintel/storage/warehouse.py
- src/codeintel/storage/serving/snapshot_service.py

### Detailed checklist
- Resolve explicit sort keys from schema policy or primary key if configured.
- If sort keys are missing, add the hash-based ordinal ordering fallback.
- Apply ordering after finalize (and after deep cast) to keep stable results.
- Remove temporary ordinal columns before persisting or exporting.
- Keep behavior identical for callers already providing explicit ordering.

---

## Scope Item 15: Explode + list alignment ops for list-flattening paths (Completed)

### Rationale
List payload flattening should use the standardized explode utilities to keep
alignment validation and null handling consistent across storage and serving.

### Pattern to deploy
```python
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_list_struct

result = explode_list_struct(
    table,
    spec=ExplodeSpec(
        src_col="parent_id",
        dst_list_col="items",
        repeat_cols=("repo", "commit"),
        aligned_list_cols=(),
        null_list_policy="error",
        null_child_policy="drop",
        enforce_parent_valid=True,
    ),
)

good = result.good
errors = result.errors
```

### Target files
- src/codeintel/serving/semantic/kernel.py
- src/codeintel/storage/queries/parquet.py

### Detailed checklist
- Locate any list-flattening or list-indexing code paths in serving queries.
- Replace ad-hoc list handling with `explode_list_struct` or `explode_edges`.
- Ensure alignment errors are surfaced through finalize or validation hooks.
- Preserve existing null handling semantics by mapping to explode policies.
- Apply deterministic ordering after explode when results are exported.
- No serving list-flattening paths were found; parquet metrics now flatten list payloads.

---

## Execution order (recommended)
1. Finalize gate adoption (Scope Item 1) to establish consistent boundaries. (Done)
2. Scan/telemetry standardization (Scope Item 2) for pushdown consistency. (Done)
3. Parquet query compute consolidation (Scope Item 3). (Done)
4. Deterministic ordering + dedupe (Scope Items 4 and 5). (Done)
5. Deep cast + chunk consolidation (Scope Item 6). (Done)
6. Plan-based scans where feasible (Scope Item 7). (Done)
7. Arrow-aware ingestion writes (Scope Item 8). (Done)
8. Ingestion compute output routing (Scope Item 9). (Done)
9. Columnar row finalize gate (Scope Item 10). (Done)
10. Dataset read boundary finalize (Scope Item 11). (Done)
11. Core dataset plan-based scans (Scope Item 12). (Done)
12. Warehouse deep cast + chunk consolidation (Scope Item 13). (Done)
13. Deterministic ordering fallback for warehouse writes/exports (Scope Item 14). (Done)
14. Explode/list alignment ops for list-flattening paths (Scope Item 15). (Done)

## Success criteria
- Exports and serving results are deterministic and contract-aligned.
- Scan telemetry is consistent across serving and storage.
- Duplicate rows are removed per contract primary key policies.
- Complex schema types are cast consistently and validated at boundaries.

---

## Advanced follow-up plan (best-in-class)

## Scope Item 16: Fused Acero scan plan builder (Planned)

### Rationale
`ScanNodeOptions` pushdown does not guarantee final semantics. Build a shared
plan helper that always appends explicit `project` and `filter` nodes, and
exposes optional `order_by` for deterministic pipelines.

### Pattern to deploy
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import Plan

plan = Plan.scan(
    dataset,
    columns=projection,
    filter_expr=filter_expr,
    implicit_ordering=True,
    require_sequenced_output=True,
).project(projection).filter(filter_expr)
```

### Target files
- src/codeintel/core/columnar/plan_ops.py
- src/codeintel/core/datasets/scanning.py
- src/codeintel/core/datasets/arrow_store.py
- src/codeintel/storage/queries/parquet.py
- src/codeintel/serving/semantic/duckdb_relation_builder.py

### Detailed checklist
- Add a single plan builder that accepts projection + filter + ordering hints.
- Ensure plan output uses `to_reader()` for streaming call sites.
- Retain scanner fallback for unsupported kernels or data types.
- Normalize expression building through `expr_vocab.E`.

---

## Scope Item 17: Finalize artifact schema + multi-error reporting (Planned)

### Rationale
Finalize gates currently emit artifacts but do not standardize the error table
schema or multi-error reporting. A uniform artifact schema improves diagnostics,
metrics, and regression testing.

### Pattern to deploy
```python
error_rows = build_error_rows(
    table,
    bad_mask,
    error_code="NULL_REQUIRED_FIELD",
    stage="invariant",
    key_fields=("repo", "commit"),
)
stats = error_rows.group_by(["error_code"]).aggregate([("row_id", "count")])
```

### Target files
- src/codeintel/core/columnar/finalize_ops.py
- src/codeintel/core/validation/engine.py
- src/codeintel/storage/validation/columnar.py
- src/codeintel/serving/export/ndjson.py

### Detailed checklist
- Define a canonical error schema (`row_id`, `error_code`, `stage`, key fields).
- Emit one error row per invariant failure (multi-error mode).
- Add stats aggregation to finalize results for metrics/logging.
- Thread provenance fields (e.g., `__filename`) into error rows when available.

---

## Scope Item 18: Nested list/struct policies in finalize gates (Planned)

### Rationale
List semantics (null vs empty) and aligned list constraints must be enforced
before explode. Nested struct field requirements should also be validated
consistently.

### Pattern to deploy
```python
spec = FinalizeSpec(
    table_key=table_key,
    mode="tolerant",
    required_non_null=("repo", "commit"),
    list_alignments=("callee_ids", "callsite_spans"),
    required_struct_fields={"extras": ("parse_version",)},
)
```

### Target files
- src/codeintel/core/columnar/finalize_ops.py
- src/codeintel/core/validation/schema_constraints.py
- src/codeintel/core/columnar/explode_ops.py

### Detailed checklist
- Add list alignment checks using `list_value_length`.
- Enforce null-list policy (error vs empty) per table contract.
- Validate required struct fields via `struct_field`.
- Surface nested validation errors through finalize artifacts.

---

## Scope Item 19: Determinism tiers + dedupe tie-breakers (Planned)

### Rationale
Dedupe based on `hash_first` is order-dependent. Introduce deterministic tiers
and explicit tie-breaker policies for best-in-class reproducibility.

### Pattern to deploy
```python
sort_keys = [("repo", "ascending"), ("commit", "ascending"), ("node_id", "ascending")]
sorted_table = table.take(stable_sort_indices(table, sort_keys=sort_keys))
deduped = sorted_table.group_by(keys, use_threads=False).aggregate(agg_specs)
```

### Target files
- src/codeintel/core/columnar/dedupe_ops.py
- src/codeintel/storage/warehouse.py
- src/codeintel/serving/semantic/kernel.py
- src/codeintel/storage/tracking/build_tracking.py

### Detailed checklist
- Define deterministic tiers (strict vs best-effort) in dedupe helpers.
- Require explicit tie-breakers for order-dependent aggregations.
- Provide order-independent dedupe mode using `min/max` winner selection.
- Document default tie-breaker policy per table contract.

---

## Scope Item 20: Dataset scan control plane + provenance (Planned)

### Rationale
Unify dataset discovery and preflight telemetry. Emit provenance columns for
error analysis and stable debugging.

### Pattern to deploy
```python
options = DatasetScanOptions(
    columns={"__filename": E.field("__filename"), **projection},
    filter_expression=filter_expr,
    batch_readahead=16,
    fragment_readahead=4,
)
```

### Target files
- src/codeintel/core/columnar/streaming.py
- src/codeintel/core/datasets/scanning.py
- src/codeintel/core/datasets/scanner_ops.py
- src/codeintel/storage/datasets/manifest_index.py

### Detailed checklist
- Centralize dataset factory options (ignore prefixes, exclude invalid files).
- Emit `get_fragments` and `count_rows` telemetry before heavy scans.
- Add provenance projection (`__filename`, `__fragment_index`, `__batch_index`).
- Standardize scan settings defaults across storage/serving.

---

## Scope Item 21: Schema evolution policy + allowed promotions (Planned)

### Rationale
Nested schema evolution needs explicit, contract-first promotion rules to avoid
silent drift.

### Pattern to deploy
```python
unified = unify_schemas_with_contract_first(contract_schema, schemas)
casted = deep_cast_table_to_contract(table, unified)
```

### Target files
- src/codeintel/core/columnar/nested_ops.py
- src/codeintel/core/columnar/schema_ops.py
- src/codeintel/core/schemas/primitives.py
- src/codeintel/core/schemas/output_registry.py

### Detailed checklist
- Define allowed promotions (int widths, float widths, list child types).
- Enforce contract-first schema ordering and metadata precedence.
- Emit `NESTED_CAST_FAILED` artifacts for disallowed promotions.
- Add tests for nested schema drift scenarios.

---

## Scope Item 22: Expression/kernel vocabulary enforcement (Planned)

### Rationale
Prevent mixed usage of eager kernels vs expressions and ensure pushdown-friendly
expressions are consistently built.

### Pattern to deploy
```python
from codeintel.core.columnar.expr_vocab import E

filter_expr = E.and_(E.field("repo") == E.scalar(repo), E.is_valid("commit"))
```

### Target files
- src/codeintel/core/columnar/expr_vocab.py
- src/codeintel/core/columnar/kernels.py
- tools/guardrails.py

### Detailed checklist
- Add guardrails to forbid direct `pc.field/pc.scalar` in app code.
- Require `expr_vocab` usage for dataset pushdown expressions.
- Expand kernels module to cover remaining common transforms.
- Add lint checks for mixed eager/plan-time usage.

---

## Scope Item 23: Substrait/engine escape hatch boundary (Planned)

### Rationale
Define a single boundary for using alternate engines when Acero is insufficient,
while preserving finalize-gate guarantees.

### Pattern to deploy
```python
reader = run_external_plan(plan_spec)
finalized = finalize_table(reader_to_table(reader), spec=FinalizeSpec(...))
```

### Target files
- src/codeintel/core/columnar/streaming.py
- src/codeintel/core/columnar/plan_ops.py
- src/codeintel/serving/semantic/duckdb_relation_builder.py

### Detailed checklist
- Define a plan interface for external execution (Substrait/DuckDB/DataFusion).
- Ensure all escape-hatch outputs pass through finalize gates.
- Track engine provenance in telemetry and error artifacts.
- Keep a single, audited registry of approved external plan runners.

---

## Execution order (advanced, recommended)
1. Fused Acero scan plan builder (Scope Item 16).
2. Finalize artifact schema + multi-error reporting (Scope Item 17).
3. Nested list/struct policies in finalize gates (Scope Item 18).
4. Determinism tiers + dedupe tie-breakers (Scope Item 19).
5. Dataset scan control plane + provenance (Scope Item 20).
6. Schema evolution policy + allowed promotions (Scope Item 21).
7. Expression/kernel vocabulary enforcement (Scope Item 22).
8. Substrait/engine escape hatch boundary (Scope Item 23).
