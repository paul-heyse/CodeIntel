# PyArrow Core Utilities + Storage `to_pylist` Replacement Plan

## Goals
- Remove remaining `to_pylist()` usage in non-build code paths, especially `src/codeintel/storage/**`.
- Centralize Arrow iteration + compute helpers in `src/codeintel/core/columnar`.
- Keep Arrow-first behavior with explicit, reusable helpers and minimal Python fallbacks.
- Preserve existing semantics (null handling, counts, masks, filters).

## Scope
- `src/codeintel/storage/queries/parquet.py`
- `src/codeintel/storage/tracking/build_tracking.py`
- New/updated helpers under `src/codeintel/core/columnar/*`
- Optional storage-specific guardrail in `tools/guardrails.py`

## Guiding Principles (Aligned)
1. **One canonical helper per concern**: core helpers are the single source of truth.
2. **Arrow-first, explicit fallback**: compute kernels first, then iterator fallback if needed.
3. **Stable imports**: storage depends only on `src/codeintel/core/**` helpers.
4. **Explicit behavior**: null semantics and empty-table behavior are parameters, not implicit.
5. **Compute-friendly normalization**: normalize chunk layout for compute kernels when needed.
6. **Expression-first filtering**: prefer `pc.*` expressions for masks and filters.

---

## Proposed Module Layout (Target State)
- `src/codeintel/core/columnar/iter.py` (new)
  - `iter_array_values`, `iter_rows`, `iter_batches`
- `src/codeintel/core/columnar/compute.py` (new)
  - `count_true`, `count_non_positive`, `count_distinct`, `orphan_ref_count`
- `src/codeintel/core/columnar/__init__.py`
  - Re-export new iter + compute helpers

> Note: Build can re-export these via `codeintel.build.tabular.arrow_ops` for compatibility, but
> storage should import directly from core.

---

## Core Helper Patterns (Snippets)

### 1) Iteration helpers (core)
```python
# src/codeintel/core/columnar/iter.py
from collections.abc import Iterator, Sequence
import pyarrow as pa


def iter_array_values(values: pa.Array | pa.ChunkedArray) -> Iterator[object]:
    if isinstance(values, pa.ChunkedArray):
        for chunk in values.iterchunks():
            for item in chunk:
                yield item.as_py()
        return
    for item in values:
        yield item.as_py()


def iter_rows(
    table_or_batch: pa.Table | pa.RecordBatch,
    columns: Sequence[str] | None = None,
) -> Iterator[dict[str, object]]:
    if isinstance(table_or_batch, pa.Table):
        column_names = list(columns) if columns is not None else list(table_or_batch.column_names)
        if not column_names:
            return
        selected = table_or_batch.select(column_names)
        for batch in selected.to_batches():
            yield from iter_rows(batch, column_names)
        return
    batch = table_or_batch
    column_names = list(columns) if columns is not None else list(batch.schema.names)
    if not column_names:
        return
    arrays = [batch.column(column_name) for column_name in column_names]
    for row_index in range(batch.num_rows):
        yield {
            column_name: arrays[idx][row_index].as_py()
            for idx, column_name in enumerate(column_names)
        }
```

### 2) Compute helpers (core)
```python
# src/codeintel/core/columnar/compute.py
import pyarrow as pa
import pyarrow.compute as pc


def count_true(mask: pa.Array | pa.ChunkedArray) -> int:
    value = pc.sum(mask)
    return int(value.as_py() or 0)


def count_non_positive(values: pa.Array | pa.ChunkedArray) -> int:
    mask = pc.less_equal(values, pa.scalar(0))
    return count_true(mask)


def count_distinct(values: pa.Array | pa.ChunkedArray) -> int:
    value = pc.count_distinct(values)
    return int(value.as_py() or 0)


def orphan_ref_count(
    source: pa.Array | pa.ChunkedArray,
    target: pa.Array | pa.ChunkedArray,
    *,
    allow_null: bool,
) -> int:
    in_target = pc.is_in(source, value_set=target)
    missing = pc.invert(in_target)
    if allow_null:
        return count_true(pc.or_kleene(missing, pc.is_null(source)))
    return count_true(missing)
```

---

## Storage Refactors (Aligned with Compute + Iteration)

### A) `src/codeintel/storage/queries/parquet.py`
- **Replace** Python loops with compute helpers.

**Non-positive count**
```python
count = count_non_positive(values)
```
Fallback if kernel fails: `iter_array_values(values)` + `_is_non_positive`.

**Duplicate count**
```python
non_null = pc.drop_null(values)
distinct = count_distinct(non_null)
count = int(pc.count(non_null).as_py() or 0)
duplicates = count - distinct
```
Fallback: `iter_array_values` + `set`/`repr`.

**Orphan refs**
```python
count = orphan_ref_count(
    source_values,
    target_values,
    allow_null=fk.allow_null,
)
```

### B) `src/codeintel/storage/tracking/build_tracking.py`
Replace list-based masks with compute expressions:

**Match mask**
```python
mask = pc.and_kleene(
    pc.equal(table["target"], pa.scalar(target)),
    pc.and_kleene(
        pc.equal(table["repo"], pa.scalar(repo)),
        pc.equal(table["commit"], pa.scalar(commit)),
    ),
)
```

**Invert mask**
```python
mask = pc.invert(mask)
```

---

## Guardrail (Optional, but aligned)
Add a storage-only guardrail to `tools/guardrails.py`:
- name: `storage_to_pylist`
- pattern: `\.to_pylist\(`
- include_prefixes: `("src/codeintel/storage/",)`
- allow_prefixes: `("tests/", "docs/", "plans/")`

This complements the existing build guardrail and keeps hot paths clean.

---

## Phased Implementation (Aligned)

### Phase 1 — Core helpers
- [ ] Add `iter.py` in `src/codeintel/core/columnar`.
- [ ] Add `compute.py` in `src/codeintel/core/columnar`.
- [ ] Re-export helpers in `src/codeintel/core/columnar/__init__.py`.

### Phase 2 — Storage queries
- [ ] Update `storage/queries/parquet.py` to use compute helpers.
- [ ] Use iterator fallback only when kernels are unavailable.

### Phase 3 — Build tracking
- [ ] Replace list-built masks in `storage/tracking/build_tracking.py`.
- [ ] Ensure all masks use `pc.*` expressions.

### Phase 4 — Guardrail (Optional)
- [ ] Add `storage_to_pylist` guardrail rule.
- [ ] Ensure it appears in `tools/quality_report.py` results.

---

## Acceptance Criteria
- `rg -n "to_pylist" src/codeintel/storage` returns 0 hits.
- Storage queries return identical counts (non-positive, duplicates, orphan refs).
- Guardrail passes and shows in the quality report (if enabled).

## Risks & Mitigations
- **Kernel gaps**: fallback to iterator-based logic with `iter_array_values`.
- **Null semantics**: use `pc.is_null`, `pc.drop_null`, and `pc.or_kleene` explicitly.
- **Chunked arrays**: operate on chunked arrays directly; normalize only when required.

## Validation Steps
- `uv run ruff check --fix`
- `uv run pyright`
- `uv run pyrefly check`
- (Optional) targeted storage query tests
