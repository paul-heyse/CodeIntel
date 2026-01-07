# Build Helper/Context Unification Follow‑Up Plan

This plan addresses remaining consistency gaps in helper adoption across
`src/codeintel/build`, focusing on snapshot filtering, row decoding, and
compute‑filter ergonomics. Each scope item includes goals, target files,
and representative code patterns.

## Scope Item 1: Safe filtering with expressions + fallback masks

Goal: eliminate ad‑hoc try/except around `table.filter(expr)` while preserving
the expression‑first, mask‑fallback semantics for in‑memory filtering.

Status: Completed (safe_filter_expr and usage updates applied across CPG planes,
CDG, and ingestion filters).

Targets:
- `src/codeintel/build/tabular/compute_helpers.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`
- `src/codeintel/build/hamilton/native/graphs/cdg.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`

Plan:
1. Add a `safe_filter_expr` helper that accepts a `pa.Table` and a
   `ComputeExpression`, plus an optional fallback mask builder.
2. Replace local try/except blocks with `safe_filter_expr`, passing the
   existing mask fallback logic where needed.
3. Keep `safe_filter` for mask‑only paths to avoid behavioral change.

Pattern:
```python
def safe_filter_expr(
    table: pa.Table,
    expr: ComputeExpression,
    *,
    fallback_mask: Callable[[pa.Table], pa.Array | pa.ChunkedArray] | None = None,
) -> pa.Table:
    try:
        return table.filter(expr)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        if fallback_mask is None:
            return table
        return safe_filter(table, fallback_mask(table))
```

## Scope Item 2: Compute masks on RecordBatch + unified RecordBatch filtering

Goal: standardize mask logic across `pa.Table` and `pa.RecordBatch` so
string_view/scalar normalization and Kleene semantics are applied everywhere.

Status: Completed (RecordBatch filtering now uses safe_filter_batch; table masks
use safe_filter consistently).

Targets:
- `src/codeintel/build/tabular/compute_helpers.py`
- `src/codeintel/build/hamilton/native/ingestion/pipelines.py`
- `src/codeintel/build/hamilton/transforms/tabular_steps.py`

Plan:
1. Add `safe_filter_batch` (or broaden `safe_filter`) to handle RecordBatch.
2. Replace `batch.filter(mask)` sites with the helper to keep error handling
   and mask normalization consistent.

Pattern:
```python
def safe_filter_batch(
    batch: pa.RecordBatch,
    mask: pa.Array | pa.ChunkedArray,
) -> pa.RecordBatch:
    try:
        return batch.filter(mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return batch
```

## Scope Item 3: Snapshot filtering via FilterExprContext/SnapshotScope

Goal: replace Python‑level row filtering with shared snapshot filtering logic.

Status: Completed (FilterExprContext applied in compute + subsystems modules).

Targets:
- `src/codeintel/build/analytics/compute/dependencies/compute.py`
- `src/codeintel/build/analytics/compute/data_models/usage.py`
- `src/codeintel/build/analytics/data_models/core.py`
- `src/codeintel/build/analytics/subsystems/affinity.py`
- `src/codeintel/build/analytics/subsystems/materialize.py`

Plan:
1. Replace `_rows_for_snapshot` implementations with `FilterExprContext` or
   `SnapshotScope.filter_arrow_table`.
2. Apply filtering to the Arrow table before iterating rows.

Pattern:
```python
def _rows_for_snapshot(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
) -> list[dict[str, object]]:
    context = FilterExprContext(repo=repo, commit=commit)
    filtered = context.apply(frame)
    return list(iter_rows(filtered))
```

## Scope Item 4: Adopt tabular_to_scoped_table for core.goids filtering

Goal: reuse scoped conversion helper instead of manual filtering for GOID‑driven
function analytics.

Status: Completed (core.goids now scoped via SnapshotScope + tabular_to_scoped_table).

Targets:
- `src/codeintel/build/analytics/functions/metrics.py`

Plan:
1. Replace `tabular_to_arrow_table(...).select(...)` + row filtering with
   `tabular_to_scoped_table`.
2. Keep required column validation but operate on the scoped table.

Pattern:
```python
scope = SnapshotScope(repo=snapshot.repo, commit=snapshot.commit)
goids_table = tabular_to_scoped_table(
    goids_input,
    columns=sorted(required),
    scope=scope,
    require_scope_columns=True,
)
```

## Scope Item 5: RowDecoder adoption for JSON/list payloads

Goal: centralize JSON/list decoding logic and maintain consistent parsing rules.

Status: Completed (RowDecoder applied in dependency compute, data model usage,
and subsystem affinity).

Targets:
- `src/codeintel/build/analytics/compute/dependencies/compute.py`
- `src/codeintel/build/analytics/compute/data_models/usage.py`
- `src/codeintel/build/analytics/subsystems/affinity.py`

Plan:
1. Replace bespoke `json.loads` helpers with `RowDecoder`.
2. Keep existing parsing semantics for list/scalar values.

Pattern:
```python
decoder = RowDecoder(columns=("reference_modules",))
for row in iter_rows(frame):
    decoded = decoder.decode(row)
    modules = decoded.get("reference_modules")
```

## Scope Item 6: DatasetMetadataContext in deprecated nx engine (conditional)

Goal: align metadata access with dataset metadata helpers, if the legacy engine
remains active.

Status: Pending (approved to implement).

Targets:
- `src/codeintel/build/graphs/engine/nx_engine.py`

Plan:
1. Replace `ds.dataset(...).schema` reads with `DatasetMetadataContext` to
   reuse `_metadata`/`_common_metadata` and avoid scanning the dataset.
2. If the nx engine is fully retired, skip this change.

Pattern:
```python
metadata_ctx = DatasetMetadataContext(dataset_root=snapshot_dir, table_key=table_key)
schema = metadata_ctx.read_schema()
metadata = metadata_from_schema(schema) if schema is not None else {}
```
