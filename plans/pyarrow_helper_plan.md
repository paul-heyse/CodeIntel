# PyArrow Helper Remediation and Core Reuse Plan

## Context and decisions

- Primary runtime consumers rely on Parquet outputs, so JSON typing differences are not
  blocking today. We will **defer JSONL serialization changes** and revisit once JSON
  outputs are a runtime requirement.
- Join validation should fail when join keys include NULLs, even if upstream normally
  filters them. We will add a fast, compute-native NULL check in `_ensure_unique_keys`.

## Scope 1: Enforce NULL key detection in `_ensure_unique_keys`

Status: Completed

### Plan

- Add a vectorized NULL detection step before uniqueness checks.
- Use `pyarrow.compute` kernels (`is_null`, `or_kleene`, `any`) to avoid Python loops.
- Keep the check localized to `_ensure_unique_keys` so all join validations benefit.
- Emit a clear error message when NULLs are present (optionally include count).

### Code pattern

```python
null_mask = None
for key in keys:
    key_mask = pc.is_null(table[key])
    null_mask = key_mask if null_mask is None else pc.or_kleene(null_mask, key_mask)
if null_mask is not None:
    any_null = scalar_from_compute("any", [null_mask])
    if any_null:
        null_count = scalar_from_compute("sum", [pc.cast(null_mask, pa.int64())])
        count_info = f" (rows={null_count})" if isinstance(null_count, int) else ""
        msg = f"Join validation failed for {label}: NULL keys detected{count_info}"
        raise ValueError(msg)
```

### Target files

- `src/codeintel/build/tabular/arrow_ops.py`
- `tests/build/test_arrow_ops.py`

## Scope 2: Preserve schema metadata in dictionary encoding

Status: Completed

### Plan

- Preserve field metadata and nullability when dictionary-encoding columns.
- Use `Field.with_type(...)` instead of re-creating fields from scratch.
- Preserve schema-level metadata when re-assembling the encoded table.

### Code pattern

```python
field = table.schema.field(name)
if encoded is None:
    fields.append(field)
    arrays.append(column)
else:
    fields.append(field.with_type(encoded.type))
    arrays.append(encoded)

return pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=table.schema.metadata))
```

### Target files

- `src/codeintel/build/exports/writers.py`
- `tests/build/exports/test_writers.py`

## Scope 3: Preserve row counts when dropping all columns

Status: Completed

### Plan

- If all columns are dropped, return a zero-column table that preserves `num_rows`.
- Prefer `table.select([])` to keep row count stable without reconstructing arrays.
- Add regression tests that assert row counts are preserved.

### Code pattern

```python
if not existing:
    return table.select([])
```

### Target files

- `src/codeintel/build/tabular/table_ops.py`
- `tests/build/test_table_ops.py`

## Scope 4: Guard `take_by_key` against missing keys

Status: Completed

### Plan

- Introduce a `missing_policy` parameter to control behavior (`"error"`, `"null"`).
- Use `index_in` results to compute a missing mask (`< 0`).
- Keep `TakeOptions(boundscheck=True)` to avoid unsafe out-of-bounds behavior.
- For `"null"`, replace missing values with typed nulls via `pc.if_else`.

### Code pattern

```python
indices = index_in(keys, value_set=key_set)
missing_mask = pc.less(indices, pc.scalar(0))
missing_any = scalar_from_compute("any", [missing_mask])
if missing_policy == "error" and missing_any:
    raise ValueError("take_by_key missing keys")

safe_indices = pc.if_else(missing_mask, pc.scalar(0), indices)
selected = take_array(ensure_array(values), safe_indices)
if missing_policy == "null":
    nulls = pa.nulls(len(indices), type=values.type)
    return pc.if_else(missing_mask, nulls, selected)
return selected
```

### Target files

- `src/codeintel/build/tabular/array_ops.py`
- `tests/build/test_array_ops.py`

## Scope 5: JSONL fast-path consistency (deferred)

Status: Deferred

### Plan

- No code changes in this iteration; JSONL is currently not a runtime contract.
- Document the divergence between Arrow JSON and `coerce_export_row` serialization.
- Revisit once JSON outputs are used in production or for contract testing.

### Code pattern (current)

```python
if record_type is None and json_writer_available():
    write_json_streaming(writer_reader, output_path)
    return counting_iter.rows
```

### Target files

- `src/codeintel/build/exports/writers.py`
- `docs/architecture/pyarrow_helper_plan.md`

## Scope 6: Migrate conversion helpers to core

Status: Completed

### Plan

- Create `src/codeintel/core/columnar/conversion.py` with shared helpers:
  `table_to_reader`, `reader_to_table`, `tabular_to_arrow_reader`,
  `tabular_to_arrow_table`, `arrow_reader_to_lazyframe`, `table_to_lazyframe`,
  `record_batch_reader_from_iterable`, `table_to_frame`, `tabular_to_frame`.
- Move or share GOID column coercion so all consumers get consistent decimal casts.
- Re-export from `src/codeintel/build/tabular/conversion.py` to keep build imports stable.
- Update all call sites that currently hand-roll conversions or reader wrappers.

### Code pattern

```python
from codeintel.core.columnar.conversion import table_to_reader, tabular_to_arrow_reader

reader = table_to_reader(table, batch_size=DEFAULT_ARROW_BATCH_SIZE)
reader = tabular_to_arrow_reader(value)
```

### Target files

- `src/codeintel/core/columnar/conversion.py` (new)
- `src/codeintel/build/tabular/conversion.py`
- `src/codeintel/storage/warehouse.py`
- `src/codeintel/storage/tracking/build_tracking.py`
- `src/codeintel/core/columnar/schema_alignment.py`
- `src/codeintel/core/columnar/stream.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/core/datasets/arrow_store.py`
- `src/codeintel/cli/core/columnar.py`
- `src/codeintel/serving/semantic/engines/polars_engine.py`
- `src/codeintel/core/validation/engine.py`
- `src/codeintel/core/query_results.py`
- `src/codeintel/core/schemas/type_mappings.py`

## Scope 7: Merge compute helpers and masks into core

Status: Completed

### Plan

- Fold `codeintel.build.tabular.compute_helpers` into `codeintel.core.columnar.compute_helpers`.
- Fold `codeintel.build.tabular.compute_masks` into `codeintel.core.columnar.masks` with
  string/binary view normalization and expression helpers.
- Standardize on these helpers anywhere we currently call `pc.call_function` directly
  for common kernels.

### Code pattern

```python
from codeintel.core.columnar.compute_helpers import safe_filter, scalar_from_compute
from codeintel.core.columnar.masks import non_empty_string_mask

mask = non_empty_string_mask(table["name"])
filtered = safe_filter(table, mask)
max_value = scalar_from_compute("max", [table["score"]])
```

### Target files

- `src/codeintel/core/columnar/compute_helpers.py`
- `src/codeintel/core/columnar/masks.py`
- `src/codeintel/core/columnar/set_ops.py`
- `src/codeintel/storage/queries/parquet.py`
- `src/codeintel/core/validation/schema_constraints.py`
- `src/codeintel/build/tabular/compute_helpers.py`
- `src/codeintel/build/tabular/compute_masks.py`

## Scope 8: Centralize Parquet scan helpers

Status: Partially complete (core helper added + primary call sites updated)

### Plan

- Move `ParquetScanOptions`, `scan_parquet_dataset`, and `scan_parquet_table` out of
  build-only helpers into a shared core storage module.
- Standardize scanning options to reuse `DatasetScanOptions` and `build_scanner`.
- Keep filtering (`repo`, `commit`) and column projection consistent with dataset
  scanning used elsewhere.

### Code pattern

```python
from codeintel.core.datasets.scanning import scan_parquet_dataset

reader = scan_parquet_dataset(
    dataset_root=root,
    table_key=table_key,
    snapshot_id=snapshot_id,
    options=ParquetScanOptions(columns=columns, repo=repo, commit=commit),
)
```

### Target files

- `src/codeintel/core/datasets/scanning.py` (new or extended)
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/exports/common.py`
- `src/codeintel/storage/queries/parquet.py`
- `src/codeintel/storage/datasets/scanning.py`
- `src/codeintel/build/scopes/snapshot.py` (follow-up if adopting shared helper there)
- `src/codeintel/storage/datasets/manifest_index.py` (follow-up if adopting shared helper there)

## Scope 9: Writers/IPC helper convergence (deferred)

Status: Deferred

### Plan

- Defer until JSON output is a production contract.
- When activated, unify `write_jsonl_reader` / `write_arrow_reader` with
  `serving/export/ndjson.py` and `core/columnar/ipc_ops.py` to avoid drift.

### Code pattern (future)

```python
from codeintel.core.columnar.ipc_ops import write_ipc_stream
from codeintel.serving.export.ndjson import iter_ndjson_bytes_from_reader
```

### Target files

- `src/codeintel/build/exports/writers.py`
- `src/codeintel/serving/export/ndjson.py`
- `src/codeintel/core/columnar/ipc_ops.py`

## Validation and rollout

- Run quality gates: `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Add focused tests per scope item under `tests/build/` and `tests/core/` as listed.
- Run segmented pytest by impacted directories after each scope item.
