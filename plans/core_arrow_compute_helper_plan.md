# Core Arrow Compute Helper Consolidation Plan

## Goals
- Consolidate Arrow/compute utilities into `src/codeintel/core/columnar` for reuse by storage, serving,
  and build.
- Standardize mask construction, compute kernels, and normalization to eliminate duplicated logic.
- Preserve Arrow-first behavior with explicit, safe fallbacks and consistent options.

## Scope (Initial)
- New helpers in `src/codeintel/core/columnar/*` for compute, masks, and normalization.
- Update call sites in:
  - `src/codeintel/storage/queries/parquet.py`
  - `src/codeintel/storage/tracking/build_tracking.py`
  - `src/codeintel/storage/datasets/arrow_store.py`
  - `src/codeintel/core/datasets/arrow_store.py`
  - `src/codeintel/core/validation/schema_constraints.py`
- Build tabular helpers remain in `src/codeintel/build/tabular/*`, with selective re-exports
  from core where appropriate.

## Guiding Principles
1. **Core-first**: storage/serving depend only on core helpers, not build tabular helpers.
2. **Arrow-first**: kernel-based compute and expression filters before Python fallbacks.
3. **Explicit options**: shared compute options are centralized and reused.
4. **Stable semantics**: null handling is explicit and consistent across all helpers.
5. **Minimal public surface**: only well-defined helpers are exported in `core/columnar/__init__.py`.

---

## Target Helper Modules (Core)

### 1) `src/codeintel/core/columnar/compute_helpers.py` (new)
Purpose: consistent, typed wrappers around `pc.call_function` and scalars.

```python
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

def call_compute(
    name: str,
    args: list[object],
    *,
    options: pc.FunctionOptions | None = None,
) -> object | None:
    try:
        return pc.call_function(name, args, options=options)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return None


def require_array(result: object | None, *, name: str) -> pa.Array | pa.ChunkedArray:
    if isinstance(result, (pa.Array, pa.ChunkedArray)):
        return result
    msg = f"Arrow compute {name} did not return an array."
    raise TypeError(msg)


def require_scalar(result: object | None, *, name: str) -> pa.Scalar:
    if isinstance(result, pa.Scalar):
        return result
    msg = f"Arrow compute {name} did not return a scalar."
    raise TypeError(msg)
```

### 2) `src/codeintel/core/columnar/masks.py` (new)
Purpose: stable helpers for boolean masks and null handling.

```python
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import call_compute, require_array


def fill_null_false(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    filled = call_compute("fill_null", [mask, pa.scalar(value=False)])
    return require_array(filled, name="fill_null")


def invert_mask(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    inverted = call_compute("invert", [mask])
    return require_array(inverted, name="invert")


def and_mask(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    combined = call_compute("and_kleene", [left, right])
    return require_array(combined, name="and_kleene")
```

### 3) `src/codeintel/core/columnar/compute_config.py` (new)
Purpose: shared compute options and casting defaults.

```python
import pyarrow as pa
import pyarrow.compute as pc

DEFAULT_SCALAR_AGG = pc.ScalarAggregateOptions(skip_nulls=True)
DEFAULT_SCALAR_AGG_ALLOW_NULL = pc.ScalarAggregateOptions(skip_nulls=False)
DEFAULT_CAST_SAFE = pc.CastOptions.safe(target_type=pa.string())
DEFAULT_TAKE = pc.TakeOptions(boundscheck=True)
```

### 4) `src/codeintel/core/columnar/normalization.py` (new)
Purpose: canonical normalization for arrays/tables before compute kernels.

```python
from __future__ import annotations

import pyarrow as pa

def normalize_array(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    if isinstance(values, pa.ChunkedArray):
        if values.num_chunks == 0:
            return pa.array([], type=values.type)
        return values.combine_chunks()
    return values


def normalize_table(table: pa.Table) -> pa.Table:
    return table.unify_dictionaries().combine_chunks()
```

### 5) `src/codeintel/core/columnar/set_ops.py` (new)
Purpose: canonical set membership helpers.

```python
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import call_compute, require_array
from codeintel.core.columnar.normalization import normalize_array

def value_set(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    return normalize_array(values)


def is_in_mask(
    values: pa.Array | pa.ChunkedArray,
    *,
    target: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    options = pc.SetLookupOptions(value_set=value_set(target))
    result = call_compute("is_in", [values], options=options)
    return require_array(result, name="is_in")
```

### 6) `src/codeintel/core/columnar/__init__.py` (update)
Re-export the stable surface: `call_compute`, `require_array`, `require_scalar`,
`fill_null_false`, `invert_mask`, `and_mask`, `normalize_array`, `normalize_table`,
`is_in_mask`, `value_set`.

---

## Implementation Phases

### Phase 1 — Core helpers + re-exports
- [ ] Add `compute_helpers.py`, `masks.py`, `compute_config.py`, `normalization.py`, `set_ops.py`.
- [ ] Update `src/codeintel/core/columnar/__init__.py` to re-export helpers.
- [ ] Ensure docstrings follow NumPy style and type hints are complete.

### Phase 2 — Storage call sites
- [ ] `src/codeintel/storage/tracking/build_tracking.py`
  - Replace local `_compute_array`/mask logic with core `call_compute`, `fill_null_false`,
    `invert_mask`, `and_mask`, `normalize_array`.
- [ ] `src/codeintel/storage/queries/parquet.py`
  - Replace local `_compute_*` wrappers and mask utilities with core helpers.
  - Keep explicit fallback to iterator helpers for unsupported kernels.

### Phase 3 — Dataset and validation helpers
- [ ] `src/codeintel/storage/datasets/arrow_store.py`
  - Normalize tables with `normalize_table` prior to write or heavy compute.
  - Use shared compute options for aggregate stats.
- [ ] `src/codeintel/core/datasets/arrow_store.py`
  - Align scanning and stats to shared config/options.
- [ ] `src/codeintel/core/validation/schema_constraints.py`
  - Centralize `pc.cast` usage via shared helpers/options.

### Phase 4 — Guardrails and cleanup
- [ ] Add a `core_compute_helpers` guardrail if we see new direct `pc.*` usage in storage.
- [ ] Ensure `tools/quality_report.py` captures guardrail results.
- [ ] Remove any redundant local helper duplicates after migration.

---

## Migration Patterns (Representative)

### A) Replace `pc.*` chains with mask helpers
```python
# before
mask = pc.and_kleene(pc.equal(a, pa.scalar("x")), pc.equal(b, pa.scalar("y")))
mask = pc.fill_null(mask, pa.scalar(value=False))

# after
from codeintel.core.columnar.masks import and_mask, fill_null_false
from codeintel.core.columnar.compute_helpers import call_compute, require_array

left = require_array(call_compute("equal", [a, pa.scalar("x")]), name="equal")
right = require_array(call_compute("equal", [b, pa.scalar("y")]), name="equal")
mask = fill_null_false(and_mask(left, right))
```

### B) Normalize once before compute-heavy ops
```python
from codeintel.core.columnar.normalization import normalize_table

table = normalize_table(table)
```

### C) Shared compute options
```python
from codeintel.core.columnar.compute_config import DEFAULT_SCALAR_AGG
from codeintel.core.columnar.compute_helpers import call_compute, require_scalar

result = require_scalar(call_compute("count", [values], options=DEFAULT_SCALAR_AGG), name="count")
count = int(result.as_py() or 0)
```

---

## Acceptance Criteria
- Storage and core modules no longer re-implement local compute/mask wrappers.
- `pc.call_function` usage in storage is centralized in core helpers.
- `rg -n "to_pylist" src/codeintel/storage` remains at 0 hits.
- `uv run ruff check --fix`, `uv run pyright --warnings --pythonversion=3.13`,
  `uv run pyrefly check` pass cleanly.

## Risks & Mitigations
- **Kernel gaps**: keep iterator fallbacks where Arrow kernels are unavailable.
- **Type drift**: enforce core helper signatures with strict return checks.
- **Overreach**: keep build tabular helpers intact and only migrate storage/core.
