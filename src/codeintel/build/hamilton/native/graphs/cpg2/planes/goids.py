"""GOID plane CPG nodes."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    build_anchor_map,
    canonicalize_for_table,
    identity_keys,
)
from codeintel.build.tabular.arrow_ops import ArrowJoinSpec, arrow_join_tables
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import is_valid_mask
from codeintel.core.columnar.rows import empty_table_for_table

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
GOIDS_TABLE_KEY = "core.goids"


@dataclass(frozen=True)
class GoidNodeDiagnostics:
    """Diagnostics for GOID CPG node emission."""

    total_rows: int
    resolved_rows: int
    dropped_rows: int


def cpg_nodes_from_goids(
    goids: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG nodes from GOID rows.

    Returns
    -------
    pyarrow.Table
        CPG node table for GOIDs.
    """
    required = {"goid_h128", "repo", "commit", "rel_path"}
    if not required.issubset(set(goids.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    normalized = canonicalize_for_table(goids, table_key=GOIDS_TABLE_KEY)
    anchors = build_anchor_map(
        normalized,
        table_key=GOIDS_TABLE_KEY,
        pk_columns=identity_keys(GOIDS_TABLE_KEY),
        include_source_pk_json=True,
    )
    joined = arrow_join_tables(
        normalized,
        anchors,
        spec=ArrowJoinSpec(on=["goid_h128"], how="left"),
    )
    joined = append_constant_columns(
        joined,
        {
            "node_kind": "GOID",
            "source_table_key": GOIDS_TABLE_KEY,
            "start_byte": None,
            "end_byte": None,
            "extras_json": None,
        },
    )
    selected = joined.select(
        [
            "repo",
            "commit",
            "cpg_node_id",
            "node_kind",
            "source_table_key",
            "source_pk_json",
            "rel_path",
            "start_byte",
            "end_byte",
            "extras_json",
        ]
    )
    filtered = _filter_valid_nodes(selected)
    if diagnostics is not None:
        diagnostics["goids"] = GoidNodeDiagnostics(
            total_rows=selected.num_rows,
            resolved_rows=filtered.num_rows,
            dropped_rows=selected.num_rows - filtered.num_rows,
        )
    return filtered


def _filter_valid_nodes(table: pa.Table) -> pa.Table:
    mask = is_valid_mask(table.column("cpg_node_id"))
    return safe_filter(table, mask)


__all__ = ["GoidNodeDiagnostics", "cpg_nodes_from_goids"]
