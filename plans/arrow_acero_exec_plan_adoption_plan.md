# Arrow Acero Exec-Plan Adoption Plan

## Goals
- Adopt PyArrow Acero exec plans for **filter → project → aggregate** paths where it is safe and beneficial.
- Centralize the Acero usage behind helpers with **safe fallbacks** to existing Arrow group-by logic.
- Build on existing helpers (`acero_ops`, `groupby`) to avoid parallel APIs and keep outputs stable.
- Add optional streaming execution (`to_reader`) for large intermediates when feasible.
- Make the Acero path **opt-in via config**, with clear diagnostics and no behavior regressions.

## Non-Goals
- Replace all Arrow compute usage with Acero.
- Change ordering semantics for list aggregations or other order-sensitive outputs.
- Add multi-core execution logic (Acero is about kernel pipeline efficiency, not threading policy).
- Convert join pipelines to Acero hash joins in this iteration (leave as future expansion).

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

## Phase 1 — Core helper expansion (extend existing helpers + guardrails)

### 1.1 Extend `acero_ops` with a safe exec-plan group-by helper
**Target file**: `src/codeintel/core/columnar/acero_ops.py`

Add helper(s) that try Acero and fall back cleanly if unavailable or unsupported. Reuse
the existing `build_exec_plan` and add a dedicated group-by helper that uses Acero
`*NodeOptions` so output names are stable and options can be set explicitly.

```python
from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_config import DEFAULT_SCALAR_AGG
from codeintel.core.columnar.normalization import normalize_table_for_compute

try:
    import pyarrow.acero as acero
except ImportError:  # pragma: no cover
    acero = None

def group_by_aggregate_exec_plan(
    table: pa.Table,
    *,
    keys: Sequence[str],
    aggregations: Sequence[tuple[str, str, str]],
    filter_expr: pc.Expression | None = None,
    projections: Sequence[str] | None = None,
    prefer_acero: bool = True,
    return_reader: bool = False,
) -> pa.Table | pa.RecordBatchReader:
    normalized = normalize_table_for_compute(table)
    base_aggs = [(col, func) for col, func, _ in aggregations]
    if not prefer_acero or acero is None:
        grouped = normalized.group_by(list(keys)).aggregate(base_aggs)
        return _rename_grouped_outputs(grouped, aggregations)
    proj_cols = list(
        dict.fromkeys(
            list(projections or []) + list(keys) + [col for col, _, _ in aggregations]
        )
    )
    decls = [
        acero.Declaration(
            "table_source",
            acero.TableSourceNodeOptions(normalized),
        ),
    ]
    if filter_expr is not None:
        decls.append(acero.Declaration("filter", acero.FilterNodeOptions(filter_expr)))
    decls.append(
        acero.Declaration(
            "project",
            acero.ProjectNodeOptions(
                expressions=[pc.field(name) for name in proj_cols],
                names=proj_cols,
            ),
        )
    )
    agg_specs = [
        (pc.field(col), func, DEFAULT_SCALAR_AGG, out_name)
        for col, func, out_name in aggregations
    ]
    decls.append(
        acero.Declaration(
            "aggregate",
            acero.AggregateNodeOptions(
                aggregates=agg_specs,
                keys=[pc.field(name) for name in keys],
            ),
        )
    )
    plan = acero.Declaration.from_sequence(decls)
    try:
        return plan.to_reader() if return_reader else plan.to_table()
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError, TypeError):
        grouped = normalized.group_by(list(keys)).aggregate(base_aggs)
        return _rename_grouped_outputs(grouped, aggregations)


def _rename_grouped_outputs(
    grouped: pa.Table,
    aggregations: Sequence[tuple[str, str, str]],
) -> pa.Table:
    rename: dict[str, str] = {}
    for col, func, out_name in aggregations:
        default_name = f"{col}_{func}"
        if default_name in grouped.column_names and out_name != default_name:
            rename[default_name] = out_name
    if not rename:
        return grouped
    return grouped.rename_columns(
        [rename.get(name, name) for name in grouped.column_names]
    )
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

### 1.3 Optional streaming result mode
Where downstream accepts a `RecordBatchReader`, use `return_reader=True` to avoid
materializing large intermediate tables.

### 1.4 Optional dataset source integration
When inputs come from datasets/scanners, add a helper that starts the plan with a
`scan` node (projection + filter pushdown) instead of materializing a table.

```python
from collections.abc import Sequence

import pyarrow.acero as acero
import pyarrow.compute as pc
import pyarrow.dataset as ds

def build_scan_exec_plan(
    dataset: ds.Dataset,
    *,
    scan_filter: pc.Expression | None,
    scan_columns: Sequence[str] | None,
) -> acero.Declaration:
    scan_opts = acero.ScanNodeOptions(
        dataset,
        filter=scan_filter,
        columns=list(scan_columns) if scan_columns else None,
        use_threads=True,
    )
    return acero.Declaration("scan", scan_opts)
```

---

## Phase 2 — Syntax augmentation aggregations

### 2.1 `_producer_table` (min aggregation)
**Target file**: `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`

Replace direct `group_by().aggregate()` with the Acero helper when enabled.

```python
from codeintel.core.columnar.acero_ops import group_by_aggregate_exec_plan

# inside _producer_table
aggregated = group_by_aggregate_exec_plan(
    selected,
    keys=["rel_path"],
    aggregations=[("producer", "min", "producer")],
    prefer_acero=options.exec_plan_enabled,
)
```

### 2.2 `_weld_coverage_table._count_by` (count aggregation)
**Target file**: `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`

```python
aggregated = group_by_aggregate_exec_plan(
    table,
    keys=key_cols,
    aggregations=[(count_col, "count", name)],
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
    aggregations=[(row_index_name, "min", row_index_name)],
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
        aggregations=[(row_index_name, "min", row_index_name)],
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
    aggregations=[(count_source, "count", f"{count_source}_count")],
    prefer_acero=options.exec_plan_enabled,
)
```

---

## Phase 5 — Future expansion: HashJoin pipelines
Acero supports `HashJoinNodeOptions` (including residual `filter_expression`). Once the
core aggregate paths are stable, we can migrate selected join pipelines to Acero to
fuse join → filter → aggregate in one plan. Keep this gated behind a separate config
flag and validate row counts + null-handling equivalence.

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
- `src/codeintel/core/columnar/acero_ops.py` (extend existing helper + fallback)
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
- Acero projections must be **scalar expressions**; use `pc.field` + `pc.scalar`.
