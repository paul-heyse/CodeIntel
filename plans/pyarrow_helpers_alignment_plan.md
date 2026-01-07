# PyArrow Helpers Alignment Plan

## Goals
- Standardize PyArrow compute/mask usage across build pipelines.
- Centralize error handling and type normalization for filters and scalars.
- Replace ad-hoc Arrow patterns with shared helpers to reduce divergence.

## Design Decisions (Approved)
- Extend `safe_filter` to accept `pyarrow.compute.Expression` and handle all Arrow exceptions
  inside the helper; use explicit try/except only when a different fallback is required.
- Standardize ingestion/transforms on `compute_masks` for all boolean mask construction,
  even when operating on `RecordBatch` inputs.

## Scope Items and Code Patterns

### 1) `call_wiring` equality masks (Medium)
**Files**: `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
**Why**: Normalize string_view/scalar comparisons and avoid dropping valid rows.

**Pattern**
```python
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import equal_mask

kind_mask = equal_mask(cfg_blocks.column("kind"), pa.scalar("entry"))
filtered = safe_filter(cfg_blocks, kind_mask)
```

### 2) Validity masks via `compute_masks` (Low)
**Files**:
- `src/codeintel/build/hamilton/native/ingestion/pipelines.py`
- `src/codeintel/build/hamilton/transforms/tabular_steps.py`

**Why**: Use canonical helpers for `is_valid`/`and_kleene` and keep mask semantics consistent.

**Pattern**
```python
from codeintel.build.tabular.compute_masks import and_kleene, is_valid_mask

mask = is_valid_mask(batch.column(index))
mask = mask if current is None else and_kleene(current, mask)
```

### 3) Mask reductions via `scalar_from_compute` (Low)
**Files**:
- `src/codeintel/build/hamilton/native/graphs/cpg2/assemble.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`

**Why**: Centralize compute error handling and scalar conversion.

**Pattern**
```python
from codeintel.build.tabular.compute_helpers import scalar_from_compute

total = scalar_from_compute("sum", [mask])
return int(total or 0)
```

### 4) Constant columns via `constant_array` / `append_constant_columns` (Low)
**Files**:
- `src/codeintel/build/analytics/utilities/datasets.py`
- `src/codeintel/build/analytics/subsystems/cache.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`

**Why**: Avoid ad-hoc `pa.array([value] * num_rows)` and handle empty tables cleanly.

**Pattern**
```python
from codeintel.build.tabular.compute_columns import constant_array

created = constant_array(created_at, table.num_rows)
table = table.append_column("created_at", created)
```

### 5) Consolidate compute error helpers (Low)
**Files**:
- `src/codeintel/build/tabular/dedupe_ops.py`
- `src/codeintel/build/schemas/observations.py`

**Why**: Reduce duplicate helper logic and keep Arrow compute failures consistent.

**Pattern**
```python
from codeintel.build.tabular.compute_helpers import scalar_from_compute

result = scalar_from_compute("min", [values])
```

### 6) Expression-aware `safe_filter` (Low, consolidation)
**Files**:
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`

**Why**: Replace repeated try/except around `table.filter(expr)` with one helper.

**Pattern**
```python
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import is_valid_expr

expr = is_valid_expr("cpg_node_id")
filtered = safe_filter(table, expr)
```

## Implementation Steps
1. **Upgrade `safe_filter`** in `compute_helpers` to accept `pc.Expression` in addition to
   boolean arrays/ChunkedArrays, keeping a single error-handling path.
2. **Normalize mask usage** in ingestion/transforms (`pipelines.py`, `tabular_steps.py`) by
   replacing raw `pc.call_function("is_valid"/"and_kleene")` with `compute_masks` helpers.
3. **Replace equality masks** in `call_wiring` with `equal_mask` + `safe_filter`.
4. **Replace scalar reductions** with `scalar_from_compute` where `pc.call_function("sum")`
   is used for boolean masks.
5. **Replace constant column construction** with `constant_array` or `append_constant_columns`.
6. **Consolidate compute helpers** in `dedupe_ops` and `observations` to remove duplicate
   compute error handling.
7. **Update expression filters** to use `safe_filter` for `pc.Expression` inputs.

## Validation
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Run targeted `pytest` subsets for ingestion/graph build paths touched by the changes.

