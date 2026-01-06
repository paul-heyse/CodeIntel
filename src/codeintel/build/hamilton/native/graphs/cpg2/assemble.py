"""CPG assembly helpers and diagnostics emission."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa

from codeintel.build.graphs.assembly import ensure_table_columns
from codeintel.build.tabular.arrow_ops import (
    align_table_to_contract,
    concat_tables_unified,
    dedupe_table_for_table,
)
from codeintel.build.hamilton.diagnostics import diagnostics_dir
from codeintel.build.hamilton.env import BuildEnv
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.schemas.row_models import columns_for_table_key

LOG = logging.getLogger(__name__)

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"


def emit_cpg_diagnostics(
    env: BuildEnv,
    *,
    plane_row_counts: Mapping[str, object] | None = None,
    anchor_resolution: Mapping[str, object] | None = None,
    join_drop_rates: Mapping[str, object] | None = None,
    contract_mismatches: Mapping[str, object] | None = None,
) -> None:
    """Emit CPG diagnostics under build/diagnostics without blocking execution."""
    try:
        diag_dir = diagnostics_dir(env.paths.build_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)
        _merge_json(diag_dir / "cpg_plane_row_counts.json", plane_row_counts)
        _merge_json(diag_dir / "cpg_anchor_resolution.json", anchor_resolution)
        _merge_json(diag_dir / "cpg_join_drop_rates.json", join_drop_rates)
        _merge_json(diag_dir / "cpg_contract_mismatches.json", contract_mismatches)
    except (OSError, ValueError, TypeError) as exc:
        LOG.warning("build.cpg.diagnostics_failed error=%s", exc)


def assemble_cpg_nodes(tables: Sequence[pa.Table]) -> pa.Table:
    """Assemble CPG nodes from per-plane tables.

    Returns
    -------
    pyarrow.Table
        Contract-aligned CPG nodes table.
    """
    tables = [table for table in tables if table.num_rows > 0]
    if not tables:
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    combined = concat_tables_unified(tables)
    combined = _ensure_contract_columns(CPG_NODES_TABLE_KEY, combined)
    combined = dedupe_table_for_table(CPG_NODES_TABLE_KEY, combined)
    return align_table_to_contract(CPG_NODES_TABLE_KEY, combined, extras_policy=None)


def assemble_cpg_edges(tables: Sequence[pa.Table]) -> pa.Table:
    """Assemble CPG edges from per-plane tables.

    Returns
    -------
    pyarrow.Table
        Contract-aligned CPG edges table.
    """
    tables = [table for table in tables if table.num_rows > 0]
    if not tables:
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    combined = concat_tables_unified(tables)
    combined = _ensure_contract_columns(CPG_EDGES_TABLE_KEY, combined)
    combined = dedupe_table_for_table(CPG_EDGES_TABLE_KEY, combined)
    return align_table_to_contract(CPG_EDGES_TABLE_KEY, combined, extras_policy=None)


def _merge_json(path: Path, payload: Mapping[str, object] | None) -> None:
    if payload is None:
        return
    if not payload:
        return
    existing = _read_json(path)
    merged = _merge_mapping(existing, payload)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            return parsed
        return {}
    except (OSError, ValueError, TypeError) as exc:
        LOG.warning("build.cpg.diagnostics_read_failed path=%s error=%s", path, exc)
        return {}


def _merge_mapping(
    left: Mapping[str, object] | None,
    right: Mapping[str, object],
) -> dict[str, object]:
    merged: dict[str, object] = {}
    if isinstance(left, Mapping):
        merged.update(left)
    for key, value in right.items():
        existing_value = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing_value, Mapping):
            merged[key] = _merge_mapping(existing_value, value)
        else:
            merged[key] = value
    return merged


def _ensure_contract_columns(table_key: str, table: pa.Table) -> pa.Table:
    columns = columns_for_table_key(table_key)
    if columns is None:
        return table
    return ensure_table_columns(table, columns)


__all__ = [
    "CPG_EDGES_TABLE_KEY",
    "CPG_NODES_TABLE_KEY",
    "assemble_cpg_edges",
    "assemble_cpg_nodes",
    "emit_cpg_diagnostics",
]
