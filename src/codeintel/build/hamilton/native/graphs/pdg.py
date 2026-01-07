"""Program dependence graph sources for graph targets."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.graphs.assembly import (
    align_table_to_contract,
    empty_reader,
    tabular_to_table,
)
from codeintel.build.tabular.arrow_ops import dedupe_table_for_table, emit_alignment_report
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.schema_ops import concat_tables_unified

PDG_EDGES_TABLE_KEY = "graph.pdg_edges"
PDG_TARGET_NAME = "pdg"


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
        return empty_reader(PDG_EDGES_TABLE_KEY)
    combined = concat_tables_unified(tables)
    deduped = dedupe_table_for_table(PDG_EDGES_TABLE_KEY, combined)
    return align_table_to_contract(
        PDG_EDGES_TABLE_KEY,
        deduped,
        target_name=PDG_TARGET_NAME,
        reporter=emit_alignment_report,
    )


__all__ = [
    "PDG_EDGES_TABLE_KEY",
    "pdg_edges",
]
