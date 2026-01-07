# Empty Dataset Resolution Plan

This document defines the implementation scope to eliminate empty outputs for:
- `graph.cdg_edges`
- `graph.cpg_edges_calls`
- `graph.cpg_edges_ret_to_call`
- `analytics.config_data_flow`
- `analytics.config_graph_metrics_*`

Each scope item below includes representative code patterns and a file target list.

## 1. GOID normalization in row-level graph builders (CDG)

Goal:
- Ensure `function_goid_h128` values from Arrow (DECIMAL) are normalized before
  row-level grouping so CDG rows are not silently dropped.

Root cause:
- `graph.cfg_blocks` and `graph.cfg_edges` store DECIMAL(38,0) GOIDs; row iteration
  currently uses an int-only coercer and discards Decimal values.

Plan:
1. Introduce a local GOID normalizer that wraps `normalize_decimal_id`.
2. Use the normalizer in `_cdg_edges_for_function` input grouping and any other
   row-level functions that read GOIDs from Arrow rows.
3. Add a lightweight diagnostic counter for dropped rows due to missing GOIDs.

Representative code pattern:
```python
from __future__ import annotations

from codeintel.core.data_models.ids import normalize_decimal_id


def _coerce_goid(value: object) -> int | None:
    return normalize_decimal_id(value)


def _group_rows_by_goid(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        goid = _coerce_goid(row.get("function_goid_h128"))
        if goid is None:
            continue
        grouped.setdefault(goid, []).append(row)
    return grouped
```

File targets:
- `src/codeintel/build/hamilton/native/graphs/cdg.py`
- `src/codeintel/core/data_models/ids.py` (if a shared helper is added)

## 2. Tiered SCIP -> GOID resolution (CALLS + RET_TO_CALL)

Goal:
- Increase `core.scip_symbol_goid_xref` coverage so `callee_goid_h128` and
  block IDs resolve, enabling `graph.cpg_edges_calls` and
  `graph.cpg_edges_ret_to_call` to materialize.

Root cause:
- Current join uses `(rel_path, start_line, end_line)` only, which yields very
  low exact matches in the current dataset.

Plan:
1. Add a definition anchor table (e.g., `core.definition_spans`) that records
   byte and line spans for GOID definitions.
2. Update SCIP resolution to perform a tiered join:
   - byte span join (primary)
   - line/col join (secondary)
   - line-only join with confidence downgrade (fallback)
3. Add `match_kind` and `match_confidence` columns to
   `core.scip_symbol_goid_xref` for auditability.
4. Emit a diagnostics summary for match coverage by tier.

Representative code pattern:
```python
from __future__ import annotations

import pyarrow as pa

from codeintel.build.tabular.arrow_ops import ArrowJoinSpec, arrow_join_tables


def _resolve_symbol_goids(
    occurrences: pa.Table,
    def_spans: pa.Table,
) -> pa.Table:
    byte_join = arrow_join_tables(
        occurrences,
        def_spans,
        spec=ArrowJoinSpec(
            on=["rel_path", "start_byte", "end_byte"],
            how="left",
            validate="m:1",
        ),
    )
    missing = byte_join.filter(pa.compute.is_null(byte_join["goid_h128"]))
    line_join = arrow_join_tables(
        missing,
        def_spans,
        spec=ArrowJoinSpec(
            on=["rel_path", "start_line", "end_line"],
            how="left",
            validate="m:1",
        ),
    )
    resolved = pa.concat_tables([byte_join, line_join], promote=True)
    return resolved
```

File targets:
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/graphs/goids.py` (definition anchors)
- `src/codeintel/core/schemas/output_registry.py` (schema updates)
- `src/codeintel/core/schemas/row_models.py` (if row model updates are needed)
- `docs/architecture/contract_system.md` (contract update notes)

## 3. Config reference extraction stage (config_data_flow + config_graph_metrics)

Goal:
- Populate `reference_paths` and `reference_modules` so config flow and config
  graph metrics have non-empty inputs.

Root cause:
- `config_ingest` currently writes `"[]"` for `reference_paths` and
  `reference_modules`, so downstream analysis has no reference data.

Plan:
1. Add a new analytics compute stage that scans syntax/AST for config key usage
   and emits `analytics.config_references`.
2. Join `analytics.config_references` into `analytics.config_values` or pass it
   directly into config graph/flow computations.
3. Ensure reference columns are stored as msgpack payloads (list of strings).

Representative code pattern:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from codeintel.core.serialization.payload import encode_payload


@dataclass(frozen=True, slots=True)
class ConfigReferenceRow:
    repo: str
    commit: str
    config_path: str
    key: str
    reference_paths: list[str]
    reference_modules: list[str]
    reference_count: int
    created_at: datetime


def build_config_reference_rows(
    repo: str,
    commit: str,
    references: dict[tuple[str, str], set[str]],
) -> list[dict[str, object]]:
    now = datetime.now(tz=UTC)
    rows: list[dict[str, object]] = []
    for (config_path, key), paths in references.items():
        rows.append(
            {
                "repo": repo,
                "commit": commit,
                "config_path": config_path,
                "key": key,
                "reference_paths": encode_payload(sorted(paths)),
                "reference_modules": encode_payload(_modules_from_paths(paths)),
                "reference_count": len(paths),
                "created_at": now,
            }
        )
    return rows
```

File targets:
- `src/codeintel/build/analytics/graphs/config_references.py` (new)
- `src/codeintel/build/hamilton/native/analytics/config_graphs.py`
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/core/schemas/output_registry.py` (new table)
- `src/codeintel/core/schemas/row_models.py` (new model)

## 4. Empty dataset guardrails + diagnostics

Goal:
- Make empty critical datasets explicit with actionable diagnostics.

Plan:
1. Add a guardrail report that checks required tables for `row_count > 0`.
2. Write `build/diagnostics/empty_dataset_report.json` with per-table status,
   counts, and the likely upstream dependency chain.
3. Wire the report into the build finalization sequence and surface warnings
   in CLI output when any required table is empty.

Representative code pattern:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EmptyDatasetFinding:
    table_key: str
    row_count: int
    status: str


REQUIRED_TABLES = {
    "graph.cdg_edges": 1,
    "graph.cpg_edges_calls": 1,
    "graph.cpg_edges_ret_to_call": 1,
    "analytics.config_data_flow": 1,
    "analytics.config_graph_metrics_keys": 1,
    "analytics.config_graph_metrics_modules": 1,
}


def write_empty_dataset_report(report_path: Path, findings: list[EmptyDatasetFinding]) -> None:
    payload = {"findings": [finding.__dict__ for finding in findings]}
    report_path.write_text(json.dumps(payload, indent=2))
```

File targets:
- `src/codeintel/build/diagnostics/empty_dataset.py` (new)
- `src/codeintel/cli/handlers/build.py` (finalization hook)
- `docs/architecture/hamilton-dag-metadata.md` (optional documentation)

## Validation and acceptance criteria

Acceptance checks:
- `graph.cdg_edges` has `row_count > 0` for scoped builds with CFG blocks.
- `core.scip_symbol_goid_xref` has a non-zero `goid_h128` coverage rate.
- `graph.cpg_edges_calls` and `graph.cpg_edges_ret_to_call` both have rows.
- `analytics.config_data_flow` and `analytics.config_graph_metrics_*` have rows
  when config usage exists in scoped code.
- `build/diagnostics/empty_dataset_report.json` reports all required tables as
  `ok` for a normal build run.
