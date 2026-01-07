# PyArrow/Polars Pylist Elimination Plan

## Goals
- Eliminate `to_pylist()` usage in `src/codeintel/build` by replacing row-wise Python loops with Arrow compute or Polars lazy execution.
- Keep runtime fully columnar and streaming where possible (especially for JSON export).
- Add reusable helpers in `src/codeintel/build/tabular/arrow_ops.py` to standardize Arrow/Polars interop and compute idioms.

## Scope
- All `to_pylist()` usages under `src/codeintel/build` (currently ~100+ sites).
- JSON export paths in `src/codeintel/build/exports/writers.py` and any JSONL exporters.
- New reusable compute utilities in `src/codeintel/build/tabular/arrow_ops.py` (plus minimal changes in `compute_masks.py` if needed).

## Guiding Principles
1. **Arrow compute first**: Use `pyarrow.compute` kernels (`pc.*`) for filtering, mapping, fill, and aggregation.
2. **Polars fallback only when Arrow lacks kernels**: e.g., list aggregation on `struct` types or other unsupported kernels.
3. **Keep data as Arrow**: avoid converting to Python lists or dicts; accept Arrow/ChunkedArray at boundaries.
4. **Streaming outputs**: JSON export should be RecordBatchReader-driven, not table-to-pylist.

---

## Proposed Utilities in `arrow_ops.py`

### 1) Array normalization
```python
import pyarrow as pa
import pyarrow.compute as pc

def ensure_array(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    return values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
```

### 2) Arrow compute wrappers (Pyright-safe)
Use `pc.call_function` when `pc.<name>` is missing in type stubs.

```python
def index_in(values: pa.Array | pa.ChunkedArray, value_set: pa.Array | pa.ChunkedArray) -> pa.Array:
    values_arr = ensure_array(values)
    value_set_arr = ensure_array(value_set)
    return pc.call_function("index_in", [values_arr], options=pc.SetLookupOptions(value_set=value_set_arr))
```

### 3) Take by key (vectorized mapping)
```python
def take_by_key(
    keys: pa.Array | pa.ChunkedArray,
    key_set: pa.Array | pa.ChunkedArray,
    values: pa.Array | pa.ChunkedArray,
) -> pa.Array:
    indices = index_in(keys, key_set)
    return pc.take(ensure_array(values), indices)
```

### 4) List aggregation with Polars fallback
Arrow does not always support `list` aggregation on `struct` types. Use Polars when needed.

```python
import polars as pl

def group_list_or_polars(table: pa.Table, keys: list[str], value_col: str) -> pa.Table:
    try:
        return table.group_by(keys).aggregate([(value_col, "list")])
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
        frame = pl.from_arrow(table).lazy()
        result = frame.group_by(keys).agg(pl.col(value_col).list()).collect(streaming=True)
        return result.to_arrow()
```

### 5) Streaming JSON export
Prefer `pyarrow.json.write_json` from a `RecordBatchReader`.

```python
import pyarrow as pa
import pyarrow.json as paj


def write_json_streaming(reader: pa.RecordBatchReader, output_path: str) -> None:
    paj.write_json(reader, output_path)
```

When you have a table:
```python
def table_to_reader(table: pa.Table, *, max_chunksize: int = 65536) -> pa.RecordBatchReader:
    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=max_chunksize))
```

### 6) Schema-safe string/binary view normalization
```python
def normalize_string_view(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return pc.cast(values, pa.string()) if pa.types.is_string_view(values.type) else values
```

---

## Pylist Replacement Patterns

### A) Row loops -> Arrow compute
**Before**
```python
rows = table.to_pylist()
filtered = [r for r in rows if r["kind"] == "function"]
```

**After**
```python
mask = pc.equal(table["kind"], pa.scalar("function"))
filtered = table.filter(mask)
```

### B) Lookup maps -> `index_in` + `take`
**Before**
```python
mapping = {r["id"]: r["value"] for r in lookup.to_pylist()}
values = [mapping.get(row["id"]) for row in base.to_pylist()]
```

**After**
```python
values = take_by_key(base["id"], lookup["id"], lookup["value"])
```

### C) Group-by list aggregation on struct -> Polars fallback
```python
payloads = group_list_or_polars(payload_table, ["syntax_node_id"], "payload")
```

### D) JSON export via streaming
```python
reader = table_to_reader(table)
write_json_streaming(reader, output_path)
```

---

## File-by-file helper mapping

This section lists each file currently using `to_pylist()` and the Arrow/Polars helper
functions to apply when replacing those row-wise loops.

- `src/codeintel/build/analytics/cfg_dfg/cfg_core.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `compute_masks.is_valid_mask`,
  `arrow_ops.index_in`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/cfg_dfg/dfg_core.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`: `compute_masks.is_in_mask`,
  `compute_masks.is_valid_mask`, `arrow_ops.index_in`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/compute/data_models/usage.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `compute_masks.non_empty_string_mask`,
  `arrow_ops.take_by_key`, `arrow_ops.group_list_or_polars`.
- `src/codeintel/build/analytics/compute/dependencies/compute.py`: `compute_masks.equal_mask`,
  `compute_masks.is_valid_mask`, `compute_masks.is_in_mask`,
  `arrow_ops.group_list_or_polars`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/compute/functions/goids.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `arrow_ops.index_in`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/data_models/core.py`: `compute_masks.is_in_mask`,
  `compute_masks.non_empty_string_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/entrypoints/core.py`: `compute_masks.non_empty_string_mask`,
  `compute_masks.is_valid_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/functions/function_contracts.py`: `compute_masks.equal_mask`,
  `compute_masks.is_in_mask`, `arrow_ops.take_by_key`, `arrow_ops.group_list_or_polars`.
- `src/codeintel/build/analytics/functions/function_effects.py`: `compute_masks.equal_mask`,
  `compute_masks.is_valid_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/functions/metrics.py`: `compute_masks.equal_mask`,
  `compute_masks.is_in_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/graphs/config_data_flow.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `arrow_ops.take_by_key`, `arrow_ops.group_list_or_polars`.
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `arrow_ops.group_list_or_polars`.
- `src/codeintel/build/analytics/semantic_roles/core.py`: `compute_masks.is_in_mask`,
  `compute_masks.non_empty_string_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/subsystems/affinity.py`: `compute_masks.equal_mask`,
  `compute_masks.is_valid_mask`, `arrow_ops.group_list_or_polars`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/subsystems/cache.py`: `compute_masks.is_in_mask`,
  `compute_masks.is_valid_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/subsystems/materialize.py`: `compute_masks.is_valid_mask`,
  `arrow_ops.table_to_reader`, `arrow_ops.write_json_streaming`.
- `src/codeintel/build/analytics/utilities/catalogs.py`: `compute_masks.is_in_mask`,
  `arrow_ops.take_by_key`.
- `src/codeintel/build/analytics/utilities/datasets.py`: `compute_masks.is_valid_mask`,
  `arrow_ops.table_to_reader`.
- `src/codeintel/build/exports/writers.py`: `arrow_ops.table_to_reader`,
  `arrow_ops.write_json_streaming`, `arrow_ops.normalize_string_view_array`.
- `src/codeintel/build/graphs/assembly/readers.py`: `arrow_ops.table_to_reader`,
  `arrow_ops.normalize_string_view_array`.
- `src/codeintel/build/graphs/validation/checks/anomaly.py`: `compute_masks.equal_mask`,
  `compute_masks.is_valid_mask`, `compute_masks.is_in_mask`.
- `src/codeintel/build/graphs/validation/checks/database.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `compute_masks.is_valid_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/analytics/config_graphs.py`: `compute_masks.is_in_mask`,
  `arrow_ops.group_list_or_polars`.
- `src/codeintel/build/hamilton/native/analytics/data_models.py`: `compute_masks.is_in_mask`,
  `compute_masks.non_empty_string_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/analytics/entrypoints.py`:
  `compute_masks.non_empty_string_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/analytics/function_ast_features.py`:
  `compute_masks.is_in_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`: `compute_masks.is_in_mask`,
  `arrow_ops.group_list_or_polars`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/analytics/py_cpg_quality_report.py`:
  `compute_masks.is_valid_mask`, `compute_masks.equal_mask`.
- `src/codeintel/build/hamilton/native/analytics/semantic_roles.py`: `compute_masks.is_in_mask`,
  `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/analytics/subsystem_agreement.py`:
  `compute_masks.is_in_mask`, `arrow_ops.group_list_or_polars`.
- `src/codeintel/build/hamilton/native/analytics/subsystem_metrics.py`:
  `compute_masks.is_in_mask`, `arrow_ops.group_list_or_polars`.
- `src/codeintel/build/hamilton/native/analytics/tables_dependencies.py`:
  `compute_masks.is_in_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/graphs/call_graph.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `arrow_ops.index_in`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/graphs/cdg.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`: `compute_masks.is_in_mask`,
  `compute_masks.is_valid_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `arrow_ops.group_list_or_polars`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`: `compute_masks.is_in_mask`,
  `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`:
  `compute_masks.is_valid_mask`, `arrow_ops.group_list_or_polars`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/graphs/goids.py`: `compute_masks.is_in_mask`,
  `compute_masks.equal_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/graphs/import_graph.py`: `compute_masks.is_in_mask`,
  `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/graphs/symbol_use.py`: `compute_masks.is_in_mask`,
  `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/ingestion/file_line_index.py`:
  `compute_masks.non_empty_string_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/ingestion/scip.py`: `compute_masks.is_in_mask`,
  `compute_masks.is_valid_mask`, `arrow_ops.group_list_or_polars`.
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`:
  `compute_masks.is_in_mask`, `compute_masks.equal_mask`, `arrow_ops.take_by_key`.
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`:
  `arrow_ops.index_in`, `arrow_ops.take_by_key`, `arrow_ops.group_list_or_polars`,
  `arrow_ops.normalize_string_view_array`.
- `src/codeintel/build/tabular/arrow_ops.py`: `arrow_ops.index_in`,
  `compute_masks.is_in_mask` for internal dedupe replacements.

---

## Implementation Phases

### Phase 1: Utilities + JSON streaming export
- Add `ensure_array`, `index_in`, `take_by_key`, `group_list_or_polars`, `table_to_reader`, `write_json_streaming`, and normalization helpers to `arrow_ops.py`.
- Update JSON/JSONL exports to use streaming writers and RecordBatchReader.
- Add or reuse compute masks where needed.

**Checklist**
- [ ] `arrow_ops.py` updated with new helpers and docstrings.
- [ ] `exports/writers.py` uses `pyarrow.json.write_json` + `RecordBatchReader`.
- [ ] Zero `to_pylist()` usage in JSON export paths.

### Phase 2: Hamilton native analytics/graphs (highest impact)
- Replace loops in `src/codeintel/build/hamilton/native/*` with Arrow compute.
- Use `index_in`/`take_by_key` for join-like mappings.
- Use `group_list_or_polars` for list aggregation on structs.

**Checklist**
- [ ] No `to_pylist()` in `src/codeintel/build/hamilton/native/**`.
- [ ] Any list aggregation for struct types uses Polars fallback.

### Phase 3: Analytics + Validation + Graph checks
- Convert `analytics/*` and `graphs/validation/*` row loops to Arrow compute.
- Use `pc.value_counts`, `pc.unique`, `pc.filter`, `pc.fill_null` as appropriate.

**Checklist**
- [ ] No `to_pylist()` in `analytics/*`.
- [ ] No `to_pylist()` in `graphs/validation/*`.

### Phase 4: Final sweep + guardrails
- Replace any remaining `to_pylist()` in `src/codeintel/build`.
- Add a lint/CI check (or a simple assertion script) to prevent future reintroduction.

**Checklist**
- [ ] `rg -n "to_pylist" src/codeintel/build` returns 0 hits.
- [ ] Build run completes with same outputs as before.

---

## Acceptance Criteria
- `to_pylist()` is fully removed from `src/codeintel/build`.
- JSON export uses streaming `pyarrow.json.write_json` and does not load full tables into Python.
- No regressions in build output (datasets/diagnostics remain consistent).

## Risks & Mitigations
- **Arrow kernel gaps (list-of-struct groupby)**: Use Polars lazy fallback via `group_list_or_polars`.
- **ChunkedArray handling**: Normalize with `ensure_array` and string/binary view casting.
- **Ordering sensitivity**: When switching to Polars groupby, use `maintain_order=True` only if required.

## Notes on Best-Practice Usage
- Use `pc.call_function` for kernels missing in stubs.
- Use `RecordBatchReader` + `pyarrow.json.write_json` for streaming.
- For large scans, prefer `table.to_batches(max_chunksize=...)` to control memory.
