# Core Arrow Consolidation Expansion Plan

## Goals
- Consolidate duplicated Arrow functionality into `src/codeintel/core` for storage, serving, and build.
- Maximize use of PyArrow native capabilities (datasets, compute, schema evolution, IPC, Acero).
- Reduce drift by enforcing a single helper per concern with well-defined semantics.

## Scope
- New/expanded helpers under `src/codeintel/core/columnar/*` and `src/codeintel/core/datasets/*`.
- Refactor call sites across storage, serving, and build to use core helpers.
- Keep build tabular helpers as compatibility shims where needed (re-exports only).

## Guiding Principles
1. **Core-first**: storage/serving import only `codeintel.core` helpers.
2. **Arrow-native**: use `pyarrow.compute`, `pyarrow.dataset`, `pyarrow.ipc`, and Acero before Python fallbacks.
3. **Schema evolution friendly**: use `pa.unify_schemas` + `Table.cast` to align outputs.
4. **Predictable performance**: normalize dictionaries/chunks prior to heavy kernels.
5. **Explicit options**: shared compute/scanner options are centralized.

---

## Proposed Core Helpers

### 1) Schema + concat helpers (`src/codeintel/core/columnar/schema_ops.py`)
Consolidate schema unify + concat logic currently split between build and core.

```python
from __future__ import annotations

import pyarrow as pa

def unify_schemas(
    schemas: list[pa.Schema],
    *,
    promote_options: str | None = "permissive",
) -> pa.Schema:
    if promote_options:
        return pa.unify_schemas(schemas, promote_options=promote_options)
    return pa.unify_schemas(schemas)


def concat_tables_unified(
    tables: list[pa.Table],
    *,
    promote_options: str | None = "permissive",
) -> pa.Table:
    schemas = [table.schema for table in tables]
    unified = unify_schemas(schemas, promote_options=promote_options)
    aligned = [table.cast(unified) for table in tables]
    return pa.concat_tables(aligned, promote=True)
```

**Refactor call sites**
- `src/codeintel/build/tabular/arrow_ops.py`: re-export `concat_tables_unified`.
- `src/codeintel/build/hamilton/native/ingestion/syntax_*` and CPG assemblers: use core helper.
- `src/codeintel/build/hamilton/native/graphs/cpg2/assemble.py`: replace local concat usage.
- `src/codeintel/build/hamilton/native/graphs/pdg.py`: replace build helper import.

---

### 2) Table/array normalization (`src/codeintel/core/columnar/normalization.py`)
Expand normalization into compute-friendly patterns and reuse across serving/build.

```python
from __future__ import annotations

import pyarrow as pa

def normalize_table_for_compute(table: pa.Table) -> pa.Table:
    table = table.unify_dictionaries()
    return table.combine_chunks()


def normalize_array_for_compute(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    if isinstance(values, pa.ChunkedArray):
        return values.combine_chunks()
    return values
```

**Refactor call sites**
- `src/codeintel/serving/semantic/engines/polars_engine.py`
- `src/codeintel/storage/datasets/arrow_store.py`
- `src/codeintel/core/datasets/arrow_store.py`
- `src/codeintel/build/tabular/array_ops.py` (re-export only).
- `src/codeintel/build/tabular/arrow_ops.py` (normalize before joins/sorts).

---

### 3) Dataset scanning utilities (`src/codeintel/core/datasets/scanner_ops.py`)
Centralize dataset scanner defaults and predicates.

```python
from __future__ import annotations

import pyarrow as pa
import pyarrow.dataset as ds

def build_scanner(
    dataset: ds.Dataset,
    *,
    columns: list[str] | None = None,
    filter_expr: ds.Expression | None = None,
    use_threads: bool = True,
    fragment_readahead: int | None = None,
    batch_readahead: int | None = None,
) -> ds.Scanner:
    return dataset.scanner(
        columns=columns,
        filter=filter_expr,
        use_threads=use_threads,
        fragment_readahead=fragment_readahead,
        batch_readahead=batch_readahead,
    )
```

**Refactor call sites**
- `src/codeintel/storage/datasets/scanning.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/serving/semantic/engines/*`
- `src/codeintel/core/columnar/streaming.py` (if any dataset scanner wiring remains).

---

### 4) IPC streaming helpers (`src/codeintel/core/columnar/ipc_ops.py`)
Provide canonical Arrow IPC streaming paths.

```python
from __future__ import annotations

import pyarrow as pa
import pyarrow.ipc as ipc

def write_ipc_stream(
    batches: pa.RecordBatchReader,
    *,
    sink: pa.NativeFile,
) -> None:
    with ipc.new_stream(sink, batches.schema) as writer:
        for batch in batches:
            writer.write_batch(batch)


def read_ipc_stream(source: pa.NativeFile) -> pa.RecordBatchReader:
    return ipc.open_stream(source)
```

**Refactor call sites**
- `src/codeintel/core/exports/arrow_ipc.py`
- Any ad-hoc IPC writers in serving/build.
- `src/codeintel/serving/semantic/kernel.py` (if IPC is used for payload emission).

---

### 5) Mask + set helpers expansion (`src/codeintel/core/columnar/masks.py`, `set_ops.py`)
Extend mask and membership ops for reuse in storage/serving.

```python
from codeintel.core.columnar.compute_helpers import call_compute, require_array

def is_valid_mask(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return require_array(call_compute("is_valid", [values]), name="is_valid")


def filter_valid(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    mask = is_valid_mask(values)
    return require_array(call_compute("filter", [values, mask]), name="filter")
```

**Refactor call sites**
- `src/codeintel/storage/queries/parquet.py` (validity filter, orphan counts)
- `src/codeintel/core/validation/schema_constraints.py`
- `src/codeintel/storage/tracking/build_tracking.py` (mask construction).

---

### 6) Group-by helpers (`src/codeintel/core/columnar/groupby.py`)
Standardize `Table.group_by(...).aggregate(...)` usage with shared options.

```python
from __future__ import annotations

import pyarrow as pa

def group_by_aggregate(
    table: pa.Table,
    *,
    keys: list[str],
    aggregations: list[tuple[str, str]],
) -> pa.Table:
    return table.group_by(keys).aggregate(aggregations)
```

**Refactor call sites**
- Replace ad-hoc grouped aggregations in build/analytics modules.
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/graph_metrics_ext.py`
- `src/codeintel/build/analytics/semantic_roles/core.py`

---

### 7) Acero exec plan helpers (optional)
Add a small helper to compose filter → project → aggregate pipelines.

```python
from __future__ import annotations

import pyarrow as pa
import pyarrow.acero as acero

def build_exec_plan(
    table: pa.Table,
    *,
    filter_expr: pa.compute.Expression | None,
    projections: list[str],
    aggregations: list[tuple[str, str]],
    keys: list[str],
) -> pa.Table:
    decl = acero.Declaration.from_table(table)
    if filter_expr is not None:
        decl = acero.Declaration("filter", [decl], filter=filter_expr)
    decl = acero.Declaration("project", [decl], expressions=projections)
    decl = acero.Declaration(
        "aggregate",
        [decl],
        keys=keys,
        aggregates=aggregations,
    )
    return decl.to_table()
```

**Potential call sites**
- `src/codeintel/build/analytics/*` where filter → project → aggregate is chained.
- `src/codeintel/serving/semantic/engines/*` for repeated scan/aggregate pipelines.

---

## Phased Implementation

### Phase 1 — Core helpers
- [ ] Add `schema_ops.py`, `scanner_ops.py`, `ipc_ops.py`, `groupby.py`.
- [ ] Extend `normalization.py`, `masks.py`, `set_ops.py`.
- [ ] Re-export in `src/codeintel/core/columnar/__init__.py`.

### Phase 2 — Storage + serving refactors
- [ ] Replace schema concat logic in build/serving with `core.columnar.schema_ops`.
- [ ] Replace dataset scanning boilerplate with `core.datasets.scanner_ops`.
- [ ] Move IPC streaming to `core.columnar.ipc_ops`.
- [ ] Replace validity filters and set membership with expanded core mask helpers.

### Phase 3 — Optional performance upgrades
- [ ] Introduce Acero exec plan usage in hotspots.
- [ ] Add runtime tuning helper (thread counts, memory pools) if beneficial.

---

## Acceptance Criteria
- All schema unification and table concatenation in build/serving uses core helpers.
- Dataset scanning configuration is centralized.
- IPC streaming uses a single core helper.
- No duplicate mask/set helpers across modules.
- `uv run ruff check --fix`, `uv run pyright --warnings --pythonversion=3.13`, and
  `uv run pyrefly check` pass cleanly.

---

## Risks & Mitigations
- **Arrow kernel gaps**: retain fallback to iterator-based logic as needed.
- **Type drift**: enforce explicit cast + schema unify helpers.
- **Acero API volatility**: keep it optional and behind a helper to swap if needed.
