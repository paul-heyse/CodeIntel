"""CFG/DFG relation sources for graph targets."""

from __future__ import annotations

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_frame_for_table
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_lazyframe
from codeintel.build.tabular.types import TabularFrame

CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
CFG_EDGES_TABLE_KEY = "graph.cfg_edges"
DFG_EDGES_TABLE_KEY = "graph.dfg_edges"


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
    "cfg_blocks_empty",
    "cfg_blocks_existing",
    "cfg_edges_empty",
    "cfg_edges_existing",
    "dfg_edges_empty",
    "dfg_edges_existing",
]
