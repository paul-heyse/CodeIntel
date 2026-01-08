# Empty Dataset Resolution Plan

This document defines the implementation scope to eliminate empty outputs for:
- `graph.cdg_edges`
- `graph.cpg_edges_calls`
- `graph.cpg_edges_ret_to_call`
- `analytics.config_data_flow`
- `analytics.config_graph_metrics_*`

It also adds a build-facing guardrail dataset (`build.empty_dataset_issues`) plus a
standalone JSON report derived from that dataset for fast diagnostics.

Each scope item below includes representative code patterns and a file target list.

## 1. GOID normalization in row-level graph builders (CDG)

Status: Completed

Goal:
- Ensure `function_goid_h128` values from Arrow (DECIMAL) are normalized before
  row-level grouping so CDG rows are not silently dropped.

Root cause:
- `graph.cfg_blocks` and `graph.cfg_edges` store DECIMAL(38,0) GOIDs; row iteration
  currently uses an int-only coercer and discards Decimal values.

Implementation:
1. Use `normalize_decimal_id` to accept DECIMAL, int, or string values.
2. Apply the normalizer anywhere CDG grouping reads GOIDs from row dicts.
3. Log missing-GOID counts for blocks/edges during CDG assembly.

Representative code pattern:
```python
from __future__ import annotations

from collections import Counter

from codeintel.core.data_models.ids import normalize_decimal_id


def _coerce_goid(value: object) -> int | None:
    return normalize_decimal_id(value)


def _group_rows_by_goid(
    rows: list[dict[str, object]],
) -> tuple[dict[int, list[dict[str, object]]], Counter[str]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    diagnostics: Counter[str] = Counter()
    for row in rows:
        goid = _coerce_goid(row.get("function_goid_h128"))
        if goid is None:
            diagnostics["missing_function_goid"] += 1
            continue
        grouped.setdefault(goid, []).append(row)
    return grouped, diagnostics
```

File targets:
- `src/codeintel/build/hamilton/native/graphs/cdg.py` (implemented)

## 2. Tiered SCIP -> GOID resolution (CALLS + RET_TO_CALL)

Status: Completed

Goal:
- Increase `core.scip_symbol_goid_xref` coverage so `callee_goid_h128` and
  block IDs resolve, enabling `graph.cpg_edges_calls` and
  `graph.cpg_edges_ret_to_call` to materialize.

Root cause:
- Current join uses `core.goids` line spans only. Byte/column spans from
  `core.syntax_defs_resolved` are not used, so exact matches are scarce.

Implementation:
1. Derive `definition_anchors` from `core.syntax_defs_resolved` with non-null
   `goid_h128`, keeping byte and line/col spans.
2. Update SCIP resolution to perform a tiered join:
   - byte span join (primary)
   - line/col join (secondary)
   - line-only join (fallback, existing behavior)
3. Extend `match_kind` values (`byte_span`, `line_col`, `line_span`, `line_start`)
   and log a coverage summary by tier.

Representative code pattern:
```python
from __future__ import annotations

import pyarrow as pa

from codeintel.build.tabular.arrow_ops import (
    ArrowJoinSpec,
    arrow_join_tables,
    build_join_options,
)
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import invert_mask, is_valid_mask


def _definition_anchors(defs_resolved: pa.Table) -> pa.Table:
    anchors = defs_resolved.select(
        [
            "repo",
            "commit",
            "rel_path",
            "goid_h128",
            "start_line",
            "start_col",
            "end_line",
            "end_col",
            "start_byte",
            "end_byte",
        ]
    )
    return safe_filter(anchors, is_valid_mask(anchors["goid_h128"]))


def _byte_span_matches(
    defs: pa.Table,
    anchors: pa.Table,
) -> tuple[pa.Table, pa.Table]:
    spec = ArrowJoinSpec(
        on=["rel_path", "start_byte", "end_byte"],
        how="left",
        validate="m:1",
    )
    joined = arrow_join_tables(
        defs,
        anchors,
        spec=spec,
        options=build_join_options(defs, anchors),
    )
    matched = safe_filter(joined, is_valid_mask(joined["goid_h128"]))
    missing = safe_filter(joined, invert_mask(is_valid_mask(joined["goid_h128"])))
    return matched, missing
```

File targets:
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py` (implemented)

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

## 4. Empty dataset guardrails + diagnostics

Goal:
- Make empty critical datasets explicit and persist guardrail findings.

Plan:
1. Add a `build.empty_dataset_issues` table schema to persist per-table status.
2. Persist the dataset during build finalization using dataset manifests
   (`load_dataset_manifest`) to avoid rescanning data.
3. Include `status`, `row_count`, and a short dependency chain derived from the
   target DAG (target -> immediate deps) for actionable diagnostics.
4. Emit `build/diagnostics/empty_dataset_issues.json` by loading the dataset and
   serializing its rows, so the report is derived from `build.empty_dataset_issues`.

Representative code pattern:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.dataset as ds

from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.datasets.arrow_store import (
    ArrowDatasetWriteOptions,
    write_dataset,
)
from codeintel.core.datasets.manifests import load_dataset_manifest

EMPTY_DATASET_ISSUES_TABLE_KEY = "build.empty_dataset_issues"


@dataclass(frozen=True, slots=True)
class EmptyDatasetIssue:
    run_id: str
    repo: str
    commit: str
    table_key: str
    required_min_rows: int
    row_count: int
    status: str
    dependency_chain: list[str]
    recorded_at: datetime


def _issue_rows(
    required: dict[str, int],
    *,
    dataset_root: Path,
    snapshot_id: str,
    run_id: str,
    repo: str,
    commit: str,
    dependency_map: dict[str, list[str]],
) -> list[dict[str, object]]:
    recorded_at = datetime.now(tz=UTC)
    rows: list[dict[str, object]] = []
    for table_key, min_rows in required.items():
        manifest = load_dataset_manifest(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
        row_count = 0 if manifest is None or manifest.row_count is None else manifest.row_count
        status = "empty" if row_count < min_rows else "ok"
        rows.append(
            {
                "run_id": run_id,
                "repo": repo,
                "commit": commit,
                "table_key": table_key,
                "required_min_rows": min_rows,
                "row_count": row_count,
                "status": status,
                "dependency_chain": dependency_map.get(table_key, []),
                "recorded_at": recorded_at,
            }
        )
    return rows


def persist_empty_dataset_issues(
    *,
    dataset_root: Path,
    snapshot_id: str,
    issues: list[dict[str, object]],
) -> None:
    if issues:
        table, _ = table_for_rows(EMPTY_DATASET_ISSUES_TABLE_KEY, issues)
    else:
        table = empty_table_for_table(EMPTY_DATASET_ISSUES_TABLE_KEY)
    options = ArrowDatasetWriteOptions(partition_columns=("repo", "commit"))
    write_dataset(
        dataset_root=dataset_root,
        table_key=EMPTY_DATASET_ISSUES_TABLE_KEY,
        snapshot_id=snapshot_id,
        data=table,
        options=options,
    )


def write_empty_dataset_report(path: Path, dataset: ds.Dataset) -> None:
    findings = [row for row in iter_rows(dataset.to_table())]
    payload = {"findings": findings}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

File targets:
- `src/codeintel/core/schemas/table_registry.py` (new `build.empty_dataset_issues`)
- `src/codeintel/build/hamilton/empty_dataset_issues.py` (new)
- `src/codeintel/build/hamilton/executor.py` (persist dataset during finalization)
- `src/codeintel/build/hamilton/diagnostics.py` (write JSON report)

## Validation and acceptance criteria

Acceptance checks:
- `graph.cdg_edges` has `row_count > 0` for scoped builds with CFG blocks.
- `core.scip_symbol_goid_xref` shows coverage for `byte_span` or `line_span`
  match kinds and non-zero resolved GOIDs.
- `graph.cpg_edges_calls` and `graph.cpg_edges_ret_to_call` both have rows.
- `analytics.config_data_flow` and `analytics.config_graph_metrics_*` have rows
  when config usage exists in scoped code.
- `build.empty_dataset_issues` is written for build runs and includes per-table
  status rows for the required table list.
- `build/diagnostics/empty_dataset_issues.json` is emitted and mirrors the
  dataset rows.
