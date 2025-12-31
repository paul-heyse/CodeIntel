"""CFG/DFG relation sources for graph targets."""

from __future__ import annotations

import polars as pl

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_frame_for_table
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_lazyframe
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput, TabularFrame

CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
CFG_EDGES_TABLE_KEY = "graph.cfg_edges"
DFG_EDGES_TABLE_KEY = "graph.dfg_edges"


def cfg_blocks_compute(q__core__goids: InferableTabularInput) -> TabularFrame:
    """Build placeholder CFG blocks from core.goids.

    Parameters
    ----------
    q__core__goids
        Relation for ``core.goids``.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for computed CFG blocks.
    """
    frame = tabular_to_lazyframe(q__core__goids)
    start_line = pl.coalesce([pl.col("start_line"), pl.lit(0)]).cast(pl.Int64)
    end_line = pl.coalesce([pl.col("end_line"), pl.col("start_line"), pl.lit(0)]).cast(pl.Int64)
    frame = frame.with_columns(
        pl.col("goid_h128").alias("function_goid_h128"),
        pl.lit(0).cast(pl.Int64).alias("block_idx"),
        pl.concat_str([pl.col("goid_h128").cast(pl.Utf8), pl.lit(":0")]).alias("block_id"),
        pl.lit("entry").alias("label"),
        pl.col("rel_path").alias("file_path"),
        start_line.alias("start_line"),
        end_line.alias("end_line"),
        pl.lit("entry").alias("kind"),
        pl.lit("[]").alias("stmts_json"),
        pl.lit(0).cast(pl.Int64).alias("in_degree"),
        pl.lit(0).cast(pl.Int64).alias("out_degree"),
    )
    return frame.select(
        [
            "function_goid_h128",
            "block_idx",
            "block_id",
            "label",
            "file_path",
            "start_line",
            "end_line",
            "kind",
            "stmts_json",
            "in_degree",
            "out_degree",
        ]
    )


def cfg_edges_compute(cfg_blocks: TabularFrame) -> TabularFrame:
    """Build placeholder CFG edges from CFG blocks.

    Parameters
    ----------
    cfg_blocks
        Relation for computed CFG blocks.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for computed CFG edges.
    """
    return cfg_blocks.select(
        [
            "function_goid_h128",
            pl.col("block_id").alias("src_block_id"),
            pl.col("block_id").alias("dst_block_id"),
            pl.lit("self").alias("edge_kind"),
        ]
    )


def dfg_edges_compute(cfg_blocks: TabularFrame) -> TabularFrame:
    """Build placeholder DFG edges from CFG blocks.

    Parameters
    ----------
    cfg_blocks
        Relation for computed CFG blocks.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for computed DFG edges.
    """
    frame = cfg_blocks.select(
        [
            "function_goid_h128",
            pl.col("block_id").alias("src_block_id"),
            pl.col("block_id").alias("dst_block_id"),
        ]
    )
    frame = frame.with_columns(
        pl.lit(None).cast(pl.Utf8).alias("src_var"),
        pl.lit(None).cast(pl.Utf8).alias("dst_var"),
        pl.lit("self").alias("edge_kind"),
        pl.lit(value=False).cast(pl.Boolean).alias("via_phi"),
        pl.lit(None).cast(pl.Utf8).alias("use_kind"),
    )
    return frame.select(
        [
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "src_var",
            "dst_var",
            "edge_kind",
            "via_phi",
            "use_kind",
        ]
    )


def cfg_blocks_existing(env: BuildEnv) -> TabularFrame:
    """Load CFG blocks from the dataset snapshot.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for existing CFG blocks.
    """
    return load_snapshot_lazyframe(
        env=env,
        table_key=CFG_BLOCKS_TABLE_KEY,
        snapshot_id=env.commit,
    )


def cfg_edges_existing(env: BuildEnv) -> TabularFrame:
    """Load CFG edges from the dataset snapshot.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for existing CFG edges.
    """
    return load_snapshot_lazyframe(
        env=env,
        table_key=CFG_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def dfg_edges_existing(env: BuildEnv) -> TabularFrame:
    """Load DFG edges from the dataset snapshot.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for existing DFG edges.
    """
    return load_snapshot_lazyframe(
        env=env,
        table_key=DFG_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def cfg_blocks_empty(env: BuildEnv) -> TabularFrame:
    """Return an empty frame for CFG blocks.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame for CFG blocks.
    """
    _ = env
    return empty_frame_for_table(CFG_BLOCKS_TABLE_KEY)


def cfg_edges_empty(env: BuildEnv) -> TabularFrame:
    """Return an empty frame for CFG edges.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame for CFG edges.
    """
    _ = env
    return empty_frame_for_table(CFG_EDGES_TABLE_KEY)


def dfg_edges_empty(env: BuildEnv) -> TabularFrame:
    """Return an empty frame for DFG edges.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame for DFG edges.
    """
    _ = env
    return empty_frame_for_table(DFG_EDGES_TABLE_KEY)


__all__ = [
    "CFG_BLOCKS_TABLE_KEY",
    "CFG_EDGES_TABLE_KEY",
    "DFG_EDGES_TABLE_KEY",
    "cfg_blocks_compute",
    "cfg_blocks_empty",
    "cfg_blocks_existing",
    "cfg_edges_compute",
    "cfg_edges_empty",
    "cfg_edges_existing",
    "dfg_edges_compute",
    "dfg_edges_empty",
    "dfg_edges_existing",
]
