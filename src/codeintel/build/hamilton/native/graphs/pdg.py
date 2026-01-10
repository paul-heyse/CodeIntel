"""Program dependence graph sources for graph targets."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.graphs.assembly import tabular_to_table
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.columnar.schema_ops import concat_tables_unified

PDG_EDGES_TABLE_KEY = "graph.pdg_edges"
PDG_TARGET_NAME = "pdg"
PDG_SORT_KEYS: tuple[SortKey, ...] = (
    ("repo", "ascending"),
    ("commit", "ascending"),
    ("function_goid_h128", "ascending"),
    ("src_block_id", "ascending"),
    ("dst_block_id", "ascending"),
    ("edge_kind", "ascending"),
    ("src_var", "ascending"),
    ("dst_var", "ascending"),
    ("via_phi", "ascending"),
    ("use_kind", "ascending"),
    ("via_succ_block_id", "ascending"),
    ("via_edge_kind", "ascending"),
)


def _dfg_edges_table(dfg_edges: pa.Table) -> pa.Table:
    if dfg_edges.num_rows == 0:
        return dfg_edges
    return append_constant_columns(
        dfg_edges,
        {
            "edge_kind": "DFG",
            "via_succ_block_id": None,
            "via_edge_kind": None,
        },
    )


def _cdg_edges_table(cdg_edges: pa.Table) -> pa.Table:
    if cdg_edges.num_rows == 0:
        return cdg_edges
    return append_constant_columns(
        cdg_edges,
        {
            "edge_kind": "CDG",
            "src_var": None,
            "dst_var": None,
            "via_phi": None,
            "use_kind": None,
        },
    )


def pdg_edges(
    q__graph__dfg_edges: InferableTabularInput,
    q__graph__cdg_edges: InferableTabularInput,
) -> InferableTabularInput:
    """Build program dependence edges from DFG and CDG inputs.

    Returns
    -------
    InferableTabularInput
        Arrow table for graph.pdg_edges.
    """
    dfg_edges = tabular_to_table(q__graph__dfg_edges)
    cdg_edges = tabular_to_table(q__graph__cdg_edges)
    dfg_table = _dfg_edges_table(dfg_edges)
    cdg_table = _cdg_edges_table(cdg_edges)
    tables = [table for table in (dfg_table, cdg_table) if table.num_rows > 0]
    if not tables:
        return empty_table_for_table(PDG_EDGES_TABLE_KEY)
    combined = concat_tables_unified(tables)
    return Plan.table(combined).order_by(sort_keys=list(PDG_SORT_KEYS))


__all__ = [
    "PDG_EDGES_TABLE_KEY",
    "pdg_edges",
]
