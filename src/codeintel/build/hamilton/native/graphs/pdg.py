"""Program dependence graph sources for graph targets."""

from __future__ import annotations

import polars as pl

from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.frames import dedupe_frame_for_table, empty_frame_for_table
from codeintel.build.tabular.types import InferableTabularInput

PDG_EDGES_TABLE_KEY = "graph.pdg_edges"


def _dfg_edges_frame(dfg_edges: pl.LazyFrame) -> pl.LazyFrame:
    return dfg_edges.with_columns(
        pl.lit("DFG").alias("edge_kind"),
        pl.lit(None).alias("via_succ_block_id"),
        pl.lit(None).alias("via_edge_kind"),
    ).select(
        [
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "edge_kind",
            "src_var",
            "dst_var",
            "via_phi",
            "use_kind",
            "via_succ_block_id",
            "via_edge_kind",
        ]
    )


def _cdg_edges_frame(cdg_edges: pl.LazyFrame) -> pl.LazyFrame:
    return cdg_edges.with_columns(
        pl.lit("CDG").alias("edge_kind"),
        pl.lit(None).alias("src_var"),
        pl.lit(None).alias("dst_var"),
        pl.lit(None).alias("via_phi"),
        pl.lit(None).alias("use_kind"),
    ).select(
        [
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "edge_kind",
            "src_var",
            "dst_var",
            "via_phi",
            "use_kind",
            "via_succ_block_id",
            "via_edge_kind",
        ]
    )


def pdg_edges(
    q__graph__dfg_edges: InferableTabularInput,
    q__graph__cdg_edges: InferableTabularInput,
) -> pl.LazyFrame:
    """Build program dependence edges from DFG and CDG inputs.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for graph.pdg_edges.
    """
    dfg_edges = tabular_to_lazyframe(q__graph__dfg_edges)
    cdg_edges = tabular_to_lazyframe(q__graph__cdg_edges)

    dfg_frame = _dfg_edges_frame(dfg_edges)
    cdg_frame = _cdg_edges_frame(cdg_edges)

    combined = pl.concat([dfg_frame, cdg_frame], how="vertical_relaxed")
    if not combined.columns:
        return empty_frame_for_table(PDG_EDGES_TABLE_KEY)
    combined = dedupe_frame_for_table(combined, table_key=PDG_EDGES_TABLE_KEY)
    return combined.select(
        [
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "edge_kind",
            "src_var",
            "dst_var",
            "via_phi",
            "use_kind",
            "via_succ_block_id",
            "via_edge_kind",
        ]
    )


__all__ = [
    "PDG_EDGES_TABLE_KEY",
    "pdg_edges",
]
