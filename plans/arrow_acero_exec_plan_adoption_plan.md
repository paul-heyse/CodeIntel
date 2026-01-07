# Arrow Acero Exec-Plan Adoption Plan

## Goals
- Adopt PyArrow Acero exec plans for **filter → project → aggregate** paths where it is safe and beneficial.
- Centralize the Acero usage behind helpers with **safe fallbacks** to existing Arrow group-by logic.
- Make the Acero path **opt-in via config**, with clear diagnostics and no behavior regressions.

## Non-Goals
- Replace all Arrow compute usage with Acero.
- Change ordering semantics for list aggregations or other order-sensitive outputs.
- Add multi-core execution logic (Acero is about kernel pipeline efficiency, not threading policy).

## Candidate Pipelines (from current usage)
These are the clearest pipelines for optional Acero adoption (simple filter/project/aggregate, no ordering constraints), including dedupe where it materially reduces compute cost:
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
  - `_producer_table`: `group_by(rel_path) -> min(producer)`
  - `_weld_coverage_table._count_by`: `group_by(keys) -> count(node_id)`
- `src/codeintel/build/tabular/dedupe_ops.py`
  - `_dedupe_table_via_compute`: `group_by(keys) -> min(row_index)`
- `src/codeintel/build/tabular/arrow_ops.py`
  - `_ensure_unique_keys`: `group_by(keys) -> count(key)`

Not Acero targets (keep existing path):
- `group_list_or_polars(..., maintain_order=True)` in `syntax_augment.py` (order-sensitive list aggregation).
- DuckDB relation filters/aggregates in storage/serving (not Arrow tables).

---

## Phase 1 — Core helper expansion (Acero wrapper + guardrails)

### 1.1 Add a safe Acero wrapper
**Target file**: `src/codeintel/core/columnar/acero_ops.py`

Add helper(s) that try Acero and fall back cleanly if unavailable or unsupported.

```python
from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.normalization import normalize_table_for_compute

try:
    from pyarrow import acero
except ImportError:  # pragma: no cover
    acero = None


def acero_available() -> bool:
    return acero is not None


def group_by_aggregate_exec_plan(
    table: pa.Table,
    *,
    keys: Sequence[str],
    aggregations: Sequence[tuple[str, str]],
    filter_expr: pc.Expression | None = None,
    projections: Sequence[str] | None = None,
    prefer_acero: bool = True,
) -> pa.Table:
    normalized = normalize_table_for_compute(table)
    if not prefer_acero or acero is None:
        return normalized.group_by(list(keys)).aggregate(list(aggregations))
    if projections is None:
        needed = list(keys) + [col for col, _ in aggregations]
        projections = list(dict.fromkeys(needed))
    decl = acero.Declaration.from_table(normalized)
    if filter_expr is not None:
        decl = acero.Declaration("filter", [decl], filter=filter_expr)
    decl = acero.Declaration(
        "project",
        [decl],
        expressions=[pc.field(name) for name in projections],
    )
    decl = acero.Declaration(
        "aggregate",
        [decl],
        keys=list(keys),
        aggregates=list(aggregations),
    )
    try:
        return decl.to_table()
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError, TypeError):
        return normalized.group_by(list(keys)).aggregate(list(aggregations))
```

### 1.2 Optional config toggle
**Target file**: `config/codeintel.build.toml`

Add a toggle so the Acero path can be enabled/disabled without code changes.

```toml
[arrow.acero]
exec_plan_enabled = true
```

**Target file**: `src/codeintel/build/hamilton/native/options/` (new or existing options module)
- Add a small options struct or config loader that surfaces `exec_plan_enabled` to pipelines.

---

## Phase 2 — Syntax augmentation aggregations

### 2.1 `_producer_table` (min aggregation)
**Target file**: `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`

Replace direct `group_by().aggregate()` with the Acero helper when enabled.

```python
from codeintel.core.columnar.acero_ops import group_by_aggregate_exec_plan

# inside _producer_table
aggregated = group_by_aggregate_exec_plan(
    normalize_table_for_compute(selected),
    keys=["rel_path"],
    aggregations=[("producer", "min")],
    prefer_acero=options.exec_plan_enabled,
)
```

### 2.2 `_weld_coverage_table._count_by` (count aggregation)
**Target file**: `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`

```python
aggregated = group_by_aggregate_exec_plan(
    normalize_table_for_compute(table),
    keys=key_cols,
    aggregations=[(count_col, "count")],
    prefer_acero=options.exec_plan_enabled,
)
```

---

## Phase 3 — Deduplication compute path

### 3.1 `_dedupe_table_via_compute` (min aggregation)
**Target file**: `src/codeintel/build/tabular/dedupe_ops.py`

Replace the manual group-by call with the Acero helper (safe fallback preserved).

```python
from codeintel.core.columnar.acero_ops import group_by_aggregate_exec_plan

# inside _dedupe_table_via_compute
indexed = table.append_column(row_index_name, row_index)
aggregated = group_by_aggregate_exec_plan(
    indexed,
    keys=list(key_columns),
    aggregations=[(row_index_name, "min")],
    prefer_acero=options.exec_plan_enabled,
)
```

### 3.2 Optional: dedupe as a reusable helper
**Target file**: `src/codeintel/core/columnar/acero_ops.py`

If we want to reduce duplication further, add a small wrapper that specializes the Acero aggregate for the dedupe pattern. This keeps `_dedupe_table_via_compute` minimal and ensures any future callers use the same normalized pattern.

```python
def dedupe_index_exec_plan(
    table: pa.Table,
    *,
    keys: Sequence[str],
    row_index_name: str,
    prefer_acero: bool,
) -> pa.Table:
    return group_by_aggregate_exec_plan(
        table,
        keys=keys,
        aggregations=[(row_index_name, "min")],
        prefer_acero=prefer_acero,
    )
```

---

## Phase 4 — Central helper usage in join validation

### 4.1 `_ensure_unique_keys` count aggregation
**Target file**: `src/codeintel/build/tabular/arrow_ops.py`

Swap group-by count with Acero helper, keeping normalization and fallback in place.

```python
from codeintel.core.columnar.acero_ops import group_by_aggregate_exec_plan

# inside _ensure_unique_keys
aggregated = group_by_aggregate_exec_plan(
    _group_by_table_keys(table, keys),
    keys=list(keys),
    aggregations=[(count_source, "count")],
    prefer_acero=options.exec_plan_enabled,
)
```

---

## Diagnostics + Validation

### Add lightweight telemetry for Acero usage
**Target files**:
- `src/codeintel/build/hamilton/hooks/telemetry_hook.py` (optional)
- `build/diagnostics/hamilton_event_stream.jsonl`

Emit a simple field like `acero_used=true/false` on nodes that use exec-plan helpers.

### Acceptance criteria
- Output row counts match prior baseline for:
  - `core.ts_syntax_node_xref`, `core.ts_weld_coverage`, `core.syntax_nodes_augmented`
  - Deduped tables where primary keys are applied
- No regressions when `exec_plan_enabled=false`
- When `exec_plan_enabled=true`, Acero path is used and does **not** raise errors

---

## File Targets Summary

### New/Updated Helpers
- `src/codeintel/core/columnar/acero_ops.py` (new wrapper + fallback)
- `src/codeintel/build/hamilton/native/options/` (config plumbing for exec_plan_enabled)
- `config/codeintel.build.toml` (add `[arrow.acero]` toggle)

### Pipeline Call Sites
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/tabular/dedupe_ops.py`
- `src/codeintel/build/tabular/arrow_ops.py`

---

## Notes
- Acero should be treated as **optional acceleration**; behavior must match the existing compute path.
- Avoid Acero for list aggregations when `maintain_order=True`.
- Keep all output schemas and column names identical to current implementations.
