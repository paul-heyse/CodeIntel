"""Program dependence graph sources for graph targets."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.tabular.arrow_ops import dedupe_table_for_table
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_reader_for_table

PDG_EDGES_TABLE_KEY = "graph.pdg_edges"


def _dfg_edges_table(dfg_edges: pa.Table) -> pa.Table:
    if dfg_edges.num_rows == 0:
        return dfg_edges
    table = dfg_edges.append_column("edge_kind", pa.array(["DFG"] * dfg_edges.num_rows))
    table = table.append_column("via_succ_block_id", pa.nulls(dfg_edges.num_rows))
    return table.append_column("via_edge_kind", pa.nulls(dfg_edges.num_rows))


def _cdg_edges_table(cdg_edges: pa.Table) -> pa.Table:
    if cdg_edges.num_rows == 0:
        return cdg_edges
    table = cdg_edges.append_column("edge_kind", pa.array(["CDG"] * cdg_edges.num_rows))
    table = table.append_column("src_var", pa.nulls(cdg_edges.num_rows))
    table = table.append_column("dst_var", pa.nulls(cdg_edges.num_rows))
    table = table.append_column("via_phi", pa.nulls(cdg_edges.num_rows))
    return table.append_column("use_kind", pa.nulls(cdg_edges.num_rows))


def pdg_edges(
    q__graph__dfg_edges: InferableTabularInput,
    q__graph__cdg_edges: InferableTabularInput,
) -> InferableTabularInput:
    """Build program dependence edges from DFG and CDG inputs.

    Returns
    -------
    InferableTabularInput
        Arrow reader for graph.pdg_edges.
    """
    dfg_edges = tabular_to_arrow_table(q__graph__dfg_edges)
    cdg_edges = tabular_to_arrow_table(q__graph__cdg_edges)
    dfg_table = _dfg_edges_table(dfg_edges)
    cdg_table = _cdg_edges_table(cdg_edges)
    tables = [table for table in (dfg_table, cdg_table) if table.num_rows > 0]
    if not tables:
        return empty_reader_for_table(PDG_EDGES_TABLE_KEY)
    combined = pa.concat_tables(tables, promote=True)
    deduped = dedupe_table_for_table(PDG_EDGES_TABLE_KEY, combined)
    return pa.RecordBatchReader.from_batches(deduped.schema, deduped.to_batches())


__all__ = [
    "PDG_EDGES_TABLE_KEY",
    "pdg_edges",
]
