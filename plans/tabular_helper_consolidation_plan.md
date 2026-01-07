# Tabular Helper Consolidation Plan

## Goals
- Consolidate duplicated tabular helpers across `src/codeintel/build` into a small, coherent API surface.
- Normalize Arrow/Polars behavior (string/binary views, record batch readers, dedupe) to reduce drift.
- Keep Arrow-first design while providing stable, well-named helpers for call sites.

## Scope
- `src/codeintel/build/tabular/*` and direct call sites under `src/codeintel/build`.
- Shared Arrow utilities in `src/codeintel/build/graphs/assembly/*`.
- Selected core helpers in `src/codeintel/core/columnar/*` when they are natural sources of truth.

## Guiding Principles
1. **One canonical helper per concern**: keep a single implementation and re-export for compatibility.
2. **Arrow-first, Polars fallback**: when Arrow lacks a kernel, fall back to Polars explicitly.
3. **Stable imports**: preserve public imports through re-exports with deprecation notes.
4. **Explicit behavior**: encode empty-iterable behavior, batch sizing, and schema alignment as parameters.
5. **Compute-friendly normalization**: normalize chunk layout and dictionaries before heavy kernels.
6. **Expression-first filtering**: prefer compute expressions for filters and projections.

---

## Proposed Module Layout (Target State)
- `src/codeintel/build/tabular/array_ops.py`
  - Array-level helpers: `ensure_array`, `normalize_string_view_array`,
    `normalize_binary_view_array`, `value_set_array`, `index_in`, `take_by_key`.
- `src/codeintel/build/tabular/compute_masks.py`
  - Mask helpers only, powered by `array_ops`.
- `src/codeintel/build/tabular/compute_helpers.py`
  - Safe compute wrappers: `safe_filter`, `scalar_from_compute`, `call_compute`.
- `src/codeintel/build/tabular/conversion.py`
  - Canonical conversion API: `tabular_to_*`, `table_to_reader`, `record_batch_reader_from_iterable`,
    `relation_to_arrow_reader`, `relation_to_polars_lazy`.
- `src/codeintel/build/tabular/table_ops.py`
  - Table ops: `select_table_columns`, `ensure_table_columns`, `drop_table_columns`,
    `rename_table_columns`, `table_rows`/`to_records`, `empty_table_for_columns`.
- `src/codeintel/build/tabular/arrow_ops.py`
  - Arrow-specific joins, dedupe, concat, scanning, JSON streaming.
  - Re-export array ops and conversions for stable imports.
- `src/codeintel/core/columnar/type_normalization.py` (new)
  - Shared string/binary view normalization for arrays, tables, and schemas.
- `src/codeintel/core/columnar/compute_config.py` (new)
  - Shared compute options and normalization utilities (sort, cast, take, scalar aggregates).

---

## Performance Enhancements via PyArrow Compute

### 1) Normalize tables before joins/sorts/aggregations
Unify dictionaries and compact chunks to reduce kernel overhead and avoid inconsistent dictionaries.

```python
# src/codeintel/build/tabular/arrow_ops.py
import pyarrow as pa

def normalize_table_for_compute(table: pa.Table) -> pa.Table:
    normalized = table.unify_dictionaries()
    return normalized.combine_chunks()
```

Usage:
```python
left = normalize_table_for_compute(left)
right = normalize_table_for_compute(right)
```

### 2) Expression-first filtering and projection
Use compute expressions for `Table.filter` and dataset scans to enable pushdown and avoid Python masks.

```python
import pyarrow.compute as pc
import pyarrow.dataset as ds

expr = (pc.field("kind") == pc.scalar("function")) & pc.is_valid(pc.field("goid_h128"))
filtered = table.filter(expr)

scan_expr = (ds.field("repo") == "my_repo") & (ds.field("commit") == "abcd")
scanner = dataset.scanner(filter=scan_expr, columns=["repo", "commit", "goid_h128"])
```

### 3) Join residual filters with Arrow
Use `filter_expression` to apply post-match predicates inside Arrow joins.

```python
import pyarrow.compute as pc

joined = left.join(
    right,
    keys=["repo", "commit"],
    join_type="left outer",
    filter_expression=(pc.field("status") == pc.scalar("active")),
    use_threads=True,
)
```

### 4) Sort-by-expression without extra columns
Sort by a computed key using `sort_indices` and `take`.

```python
import pyarrow.compute as pc

indices = pc.sort_indices(
    table,
    options=pc.SortOptions(
        sort_keys=[(pc.field("score") / pc.scalar(100), "descending")],
        null_placement="at_end",
    ),
)
sorted_table = table.take(indices)
```

### 5) Standardize compute options
Centralize options objects to remove ad-hoc kernel configuration and align behavior.

```python
# src/codeintel/core/columnar/compute_config.py
import pyarrow as pa
import pyarrow.compute as pc

DEFAULT_CAST_OPTIONS = pc.CastOptions.safe(target_type=pa.string())
DEFAULT_SCALAR_AGG_OPTIONS = pc.ScalarAggregateOptions(skip_nulls=True)
DEFAULT_SORT_OPTIONS = pc.SortOptions(sort_keys=[("repo", "ascending")])
DEFAULT_TAKE_OPTIONS = pc.TakeOptions(boundscheck=True)
```

### 6) Scanner tuning for dataset reads
Expose scan options for readahead and batch sizing where datasets are used.

```python
import pyarrow.dataset as ds

scanner = dataset.scanner(
    columns=["repo", "commit", "goid_h128"],
    use_threads=True,
    fragment_readahead=8,
    batch_readahead=32,
)
```

---

## Consolidation Targets and Implementation Notes

### 1) Array ops and masks
**Problem:** `ensure_array`, `value_set_array`, `index_in*`, and string view normalization exist in
multiple places with subtle differences.

**Plan:**
- Create `array_ops.py` and move the canonical implementations there.
- Update `compute_masks.py`, `arrow_ops.py`, and
  `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py` to import from `array_ops`.
- Keep backward compatibility by re-exporting array helpers from `arrow_ops.py`.

**New helper patterns:**
```python
# src/codeintel/build/tabular/array_ops.py
import pyarrow as pa
import pyarrow.compute as pc

def ensure_array(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    return values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values

def normalize_string_view_array(
    values: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    target_type = string_view_cast_type(values.type)
    if target_type == values.type:
        return values
    try:
        return pc.cast(values, target_type, safe=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return values

def value_set_array(
    value_set: list[object] | pa.Array | pa.ChunkedArray,
    *,
    like: pa.Array | pa.ChunkedArray | None = None,
) -> pa.Array:
    if isinstance(value_set, (pa.Array, pa.ChunkedArray)):
        resolved = ensure_array(value_set)
    else:
        resolved = pa.array(list(value_set))
    if like is not None and pa.types.is_string_view(like.type):
        try:
            resolved = pc.cast(resolved, pa.string())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
            return resolved
    return resolved
```

**Usage:**
```python
from codeintel.build.tabular.array_ops import normalize_string_view_array, value_set_array

def is_in_mask(values: pa.Array | pa.ChunkedArray, *, value_set: list[object]) -> pa.Array:
    normalized = normalize_string_view_array(values)
    resolved = value_set_array(value_set, like=normalized)
    options = pc.SetLookupOptions(value_set=resolved)
    return pc.call_function("is_in", [normalized], options=options)
```

---

### 2) Conversion and reader utilities
**Problem:** `table_to_reader`, `relation_to_arrow_reader`, and
`_record_batch_reader_from_iterable` are duplicated across modules with different behaviors.

**Plan:**
- Make `conversion.py` the canonical location for these helpers.
- Add `table_to_reader(..., batch_size=DEFAULT_ARROW_BATCH_SIZE)` and a
  `record_batch_reader_from_iterable(..., empty_policy="none")` helper.
- Update `arrow_ops.py` to import and re-export `table_to_reader` from `conversion.py`.
- Update `duckdb_relation.py` to use conversion helpers or only provide relation-specific utilities.
- Update `arrow_dataset_saver.py` to use the shared iterable reader helper.

**New helper patterns:**
```python
# src/codeintel/build/tabular/conversion.py
def table_to_reader(
    table: pa.Table,
    *,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
) -> pa.RecordBatchReader:
    batches = table.to_batches(max_chunksize=batch_size)
    return pa.RecordBatchReader.from_batches(table.schema, batches)

def record_batch_reader_from_iterable(
    batches: Iterable[pa.RecordBatch],
    *,
    empty_policy: Literal["none", "error"] = "none",
) -> pa.RecordBatchReader | None:
    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration:
        if empty_policy == "error":
            msg = "RecordBatch iterable is empty"
            raise ValueError(msg)
        return None
    if not isinstance(first, pa.RecordBatch):
        msg = f"Unsupported tabular input type: {type(first).__name__}"
        raise TypeError(msg)
    def _iter_batches() -> Iterable[pa.RecordBatch]:
        yield first
        for batch in iterator:
            if not isinstance(batch, pa.RecordBatch):
                msg = "RecordBatch iterable contains non-RecordBatch values"
                raise TypeError(msg)
            yield batch
    return pa.RecordBatchReader.from_batches(first.schema, _iter_batches())
```

**Usage:**
```python
from codeintel.build.tabular.conversion import table_to_reader, record_batch_reader_from_iterable

reader = table_to_reader(table, batch_size=DEFAULT_ARROW_BATCH_SIZE)
reader_from_batches = record_batch_reader_from_iterable(batches, empty_policy="error")
```

---

### 3) Table ops consolidation
**Problem:** `select_table_columns`, `ensure_table_columns`, `drop_table_columns`,
`rename_table_columns`, and row conversion live in multiple places (`graphs/assembly`, `frames`).

**Plan:**
- Create `table_ops.py` as the canonical table ops module.
- Update `graphs/assembly/readers.py` to re-export from `table_ops.py`.
- Replace local helpers in `syntax_enrich.py` with `table_ops.ensure_table_columns`.

**New helper patterns:**
```python
# src/codeintel/build/tabular/table_ops.py
def select_table_columns(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    if not columns:
        return table
    present = [column for column in columns if column in table.column_names]
    if not present:
        return empty_table_for_columns(columns)
    return table.select(present)

def empty_table_for_columns(columns: Sequence[str]) -> pa.Table:
    arrays = [pa.array([], type=pa.null()) for _ in columns]
    return pa.Table.from_arrays(arrays, names=list(columns))
```

**Usage:**
```python
from codeintel.build.tabular.table_ops import ensure_table_columns, select_table_columns

selected = select_table_columns(table, ["repo", "commit", "rel_path"])
normalized = ensure_table_columns(selected, ["repo", "commit", "rel_path", "language"])
```

---

### 4) String and binary view normalization
**Problem:** String view normalization is implemented multiple times
(`arrow_ops`, `compute_masks`, `view_outputs`, `schema_alignment`).

**Plan:**
- Add `codeintel.core.columnar.type_normalization` with:
  - `string_view_cast_type`
  - `normalize_string_view_array`
  - `normalize_string_view_table`
  - `normalize_string_view_schema`
  - `binary_view_cast_type` and equivalents
- Update callers to use the shared functions.

**New helper patterns:**
```python
# src/codeintel/core/columnar/type_normalization.py
def normalize_string_view_table(table: pa.Table) -> pa.Table:
    schema = normalize_string_view_schema(table.schema)
    if schema == table.schema:
        return table
    return table.cast(schema)
```

**Usage:**
```python
from codeintel.core.columnar.type_normalization import normalize_string_view_table

table = normalize_string_view_table(table)
```

---

### 5) Constant columns and empty tables
**Problem:** `constant_array` is duplicated with simpler implementations in ingestion utilities.

**Plan:**
- Keep `compute_columns.constant_array` as canonical.
- Replace ad-hoc constant array logic in `syntax_augment.py` with `constant_array`.
- Distinguish schema-free empties (`empty_table_for_columns`) from schema-driven
  `empty_table_for_table`.

**New helper patterns:**
```python
from codeintel.build.tabular.compute_columns import constant_array

table = table.append_column("producer", constant_array("tree_sitter", table.num_rows))
```

---

### 6) Dedupe normalization across Arrow and Polars
**Problem:** Dedupe logic is split across Arrow and LazyFrame modules with different semantics.

**Plan:**
- Add `dedupe_tabular` in `arrow_ops.py` (or a new `dedupe.py`) that accepts
  `InferableTabularInput` and dispatches to Arrow or Polars.
- Update `frames.dedupe_frame_for_table` to delegate to the new helper.
- Keep existing `dedupe_table_for_table` for Arrow-specific use.

**New helper patterns:**
```python
def dedupe_tabular(
    table_key: str,
    value: InferableTabularInput,
    *,
    prefer_columns: Sequence[str] | None = None,
) -> pa.Table:
    table = tabular_to_arrow_table(value)
    return dedupe_table_for_table(table_key, table, prefer_columns=prefer_columns)
```

---

### 7) Arrow join helper upgrades
Add `filter_expression`, `use_threads`, and normalization hooks to the join helper.

```python
def arrow_join_tables(
    left: pa.Table,
    right: pa.Table,
    *,
    spec: ArrowJoinSpec,
    filter_expression: pc.Expression | None = None,
) -> pa.Table:
    left = normalize_table_for_compute(left)
    right = normalize_table_for_compute(right)
    return left.join(
        right,
        keys=tuple(keys),
        right_keys=tuple(right_keys) if right_keys is not None else None,
        join_type=join_type,
        left_suffix=spec.left_suffix,
        right_suffix=right_suffix,
        coalesce_keys=spec.coalesce_keys,
        filter_expression=filter_expression,
        use_threads=True,
    )
```

---

### 7) Wrapper module cleanup
**Problem:** `graphs/assembly/*` and `hamilton/native/ingestion/frame_utils.py`
wrap or duplicate tabular utilities, making ownership unclear.

**Plan:**
- Convert `graphs/assembly/readers.py` into a pure re-export module for
  `tabular/table_ops.py` and `tabular/conversion.py`.
- Keep `frame_utils.py` as a compatibility layer but internally forward to
  `tabular/frames.py` or `core.columnar.rows` (with deprecation comment).

---

## Migration Map (Old -> New)
- `compute_masks.ensure_array` -> `tabular.array_ops.ensure_array`
- `compute_masks.value_set_array` -> `tabular.array_ops.value_set_array`
- `compute_masks.index_in_values` -> `tabular.array_ops.index_in`
- `arrow_ops.table_to_reader` -> `tabular.conversion.table_to_reader`
- `conversion.relation_to_arrow_reader` -> `duckdb_relation.relation_to_arrow_reader` (or re-export)
- `syntax_augment._constant_array` -> `compute_columns.constant_array`
- `graphs.assembly.select_table_columns` -> `tabular.table_ops.select_table_columns`
- `frames.to_records` + `graphs.assembly.table_rows` -> `tabular.table_ops.to_records`

---

## Call-site Migration Checklist (Compute Hotspots)
### src/codeintel/build/hamilton/native/ingestion/syntax_augment.py
- [ ] `_ensure_array`: replace with `tabular.array_ops.ensure_array` and remove the local helper.
- [ ] `_constant_array`: replace with `compute_columns.constant_array` and remove the local helper.
- [ ] `_producer_table`: call `normalize_table_for_compute` before `group_by`.
- [ ] `_xref_exact`: normalize join inputs before `arrow_join_tables`.
- [ ] `_unmatched_ts_nodes`: normalize join inputs before `arrow_join_tables`.
- [ ] `_xref_fuzzy`: normalize join inputs; evaluate replacing Python row loops with `index_in`.
- [ ] `_group_payloads_by_syntax_node`: replace manual index loops with `group_list_or_polars` if viable.
- [ ] `_ts_payloads_by_syntax_node`: normalize join inputs and use `array_ops.ensure_array`.
- [ ] `_augment_syntax_nodes`: use `array_ops.ensure_array` for payload arrays.
- [ ] `_weld_coverage_table`: normalize tables before `group_by` and joins.
- [ ] `_column_or_null`: replace `_ensure_array` usage with `array_ops.ensure_array`.
- [ ] `_ts_nodes_to_syntax_nodes`: replace `_constant_array` usage.
- [ ] `_ts_edges_to_syntax_edges`: replace `_constant_array` usage.

### src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py
- [ ] `cpg2_nodes__cfg_blocks`: normalize join inputs before `arrow_join_tables`.
- [ ] `cpg2_edges__cfg_edges`: normalize outputs before downstream joins.
- [ ] `cpg2_edges__dfg_edges`: normalize outputs before downstream joins.
- [ ] `cpg2_edges__cdg_edges`: normalize outputs before downstream joins.
- [ ] `_cfg_block_lookup`: normalize join inputs before `arrow_join_tables`.
- [ ] `_join_block_lookup`: normalize join inputs before `arrow_join_tables`.
- [ ] `_join_block_anchors`: normalize join inputs before `arrow_join_tables`.
- [ ] `_filter_valid_edges`: prefer expression-first filters; keep `safe_filter` fallback.
- [ ] `_filter_valid_nodes`: prefer expression-first filters; keep `safe_filter` fallback.

### src/codeintel/build/hamilton/native/graphs/cpg2/planes/goids.py
- [ ] `cpg2_nodes__goids`: normalize join inputs before `arrow_join_tables`.
- [ ] `_filter_valid_nodes`: prefer expression-first filters; keep `safe_filter` fallback.

### src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py
- [ ] `cpg2_nodes__import_modules`: normalize join inputs before `arrow_join_tables`.
- [ ] `cpg2_edges__call_graph_edges`: normalize join inputs before `arrow_join_tables`.
- [ ] `cpg2_edges__import_graph_edges`: normalize join inputs before `arrow_join_tables`.
- [ ] `_filter_valid_edges`: prefer expression-first filters; keep `safe_filter` fallback.
- [ ] `_filter_valid_nodes`: prefer expression-first filters; keep `safe_filter` fallback.

### src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py
- [ ] `cpg2_nodes__scip_symbols`: normalize join inputs before `arrow_join_tables`.
- [ ] `_filter_valid_edges`: prefer expression-first filters; keep `safe_filter` fallback.
- [ ] `_filter_valid_nodes`: prefer expression-first filters; keep `safe_filter` fallback.

### src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py
- [ ] `cpg2_edges__scip_symbol_relationships`: normalize join inputs before `arrow_join_tables`.
- [ ] `cpg2_edges__scip_symbol_goid_xref`: normalize join inputs before `arrow_join_tables`.
- [ ] `_filter_valid_edges`: prefer expression-first filters; keep `safe_filter` fallback.

### src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py
- [ ] `cpg2_nodes__syntax_nodes`: normalize join inputs before `arrow_join_tables`.
- [ ] `cpg2_edges__syntax_edges`: normalize join inputs before `arrow_join_tables`.

### src/codeintel/build/hamilton/native/graphs/cdg.py
- [ ] `_prefilter_cdg_blocks`: prefer expression-first filters; keep `safe_filter` fallback.
- [ ] `_prefilter_cdg_edges`: prefer expression-first filters; keep `safe_filter` fallback.

### src/codeintel/build/hamilton/native/graphs/cfg_dfg.py
- [ ] `_collect_ast_function_keys`: prefer expression-first filters; keep `safe_filter` fallback.

### src/codeintel/build/analytics/cfg_dfg/helpers.py
- [ ] `load_function_metadata`: filter by repo/commit/kind via expressions before iterating.

### src/codeintel/build/analytics/cfg_dfg/cfg_core.py
- [ ] `load_cfg_blocks`: apply expression-first filters before `iter_rows` loops.

### src/codeintel/build/analytics/cfg_dfg/dfg_core.py
- [ ] `load_dfg_edges`: apply expression-first filters before `iter_rows` loops.

### src/codeintel/build/graphs/engine/views.py
- [ ] `load_call_graph`: use `conversion.table_to_reader` and normalize inputs before iteration.
- [ ] `load_import_graph`: use `conversion.table_to_reader` and normalize inputs before iteration.
- [ ] `load_symbol_module_graph`: use `conversion.table_to_reader` and normalize inputs before iteration.
- [ ] `load_symbol_function_graph`: use `conversion.table_to_reader` and normalize inputs before iteration.

### src/codeintel/build/exports/writers.py
- [ ] `write_jsonl_reader`: keep streaming path (`write_json_streaming`) as the default.
- [ ] `write_jsonl_records`: avoid materializing row lists; prefer batch streaming.
- [ ] `write_json_array`: keep batch streaming and avoid `to_pylist`.
- [ ] `_iter_json_rows_from_batch`: keep row-wise iteration isolated to batch scope.

### src/codeintel/build/exports/engine.py
- [ ] `export_jsonl_for_table`: ensure reader stays streaming and avoids table materialization.
- [ ] `export_jsonl_for_table_from_snapshot`: ensure reader stays streaming and avoids table materialization.

### src/codeintel/build/exports/jsonl.py
- [ ] `export_repo_map_json`: use JSONL streaming for large outputs and avoid table-to-list conversion.

### src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py
- [ ] `_record_batch_reader_from_iterable`: replace with `conversion.record_batch_reader_from_iterable`.
- [ ] `_normalize_tabular_data`: delegate reader/table conversion to `conversion` helpers.

### src/codeintel/build/tabular/compute_masks.py
- [ ] `ensure_array`: remove local helper and import from `array_ops`.
- [ ] `value_set_array`: remove local helper and import from `array_ops`.
- [ ] `index_in_values`: replace with `array_ops.index_in`.
- [ ] `equal_mask`: add expression-based variant for `Table.filter`.
- [ ] `not_equal_mask`: add expression-based variant for `Table.filter`.
- [ ] `is_in_mask`: add expression-based variant for `Table.filter`.
- [ ] `is_valid_mask`: add expression-based variant for `Table.filter`.
- [ ] `non_empty_string_mask`: add expression-based variant for `Table.filter`.

### src/codeintel/build/hamilton/native/views/view_outputs.py
- [ ] `_coerce_string_view`: replace with `type_normalization.normalize_string_view_table`.
- [ ] `_execute_view_query`: normalize inputs before DuckDB registration.

---

## Implementation Steps (Phased)
1. **Introduce `array_ops.py` and `type_normalization.py`.**
2. **Update `compute_masks.py` and `arrow_ops.py` to import from `array_ops`.**
3. **Normalize conversion utilities in `conversion.py`; remove duplicates.**
4. **Introduce `table_ops.py` and re-export from `graphs/assembly/readers.py`.**
5. **Update call sites (`syntax_enrich`, `syntax_augment`, `view_outputs`, etc.).**
6. **Consolidate dedupe helpers and update `frames.py`.**
7. **Add targeted tests or smoke checks for the new helpers.**

---

## Quality Gates (Per AOP)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Focused `uv run pytest -q` for affected directories and then segmented by major directory.

---

## Proposed Code Patterns Summary
```python
# Array ops
from codeintel.build.tabular.array_ops import ensure_array, index_in, take_by_key

# Conversion
from codeintel.build.tabular.conversion import table_to_reader, tabular_to_arrow_table

# Table ops
from codeintel.build.tabular.table_ops import select_table_columns, ensure_table_columns

# String view normalization
from codeintel.core.columnar.type_normalization import normalize_string_view_table

# Dedupe
from codeintel.build.tabular.arrow_ops import dedupe_table_for_table
```
