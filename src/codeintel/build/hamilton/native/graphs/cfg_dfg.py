"""CFG/DFG relation sources for graph targets."""

from __future__ import annotations

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_relation_for_table
from codeintel.storage.gateway import DuckDBRelation

CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
CFG_EDGES_TABLE_KEY = "graph.cfg_edges"
DFG_EDGES_TABLE_KEY = "graph.dfg_edges"


def cfg_blocks_existing(env: BuildEnv) -> DuckDBRelation:
    """Load CFG blocks from the existing table.

    Returns
    -------
    DuckDBRelation
        Relation for existing CFG blocks.
    """
    return env.gateway.relation_from_table_key(CFG_BLOCKS_TABLE_KEY)


def cfg_edges_existing(env: BuildEnv) -> DuckDBRelation:
    """Load CFG edges from the existing table.

    Returns
    -------
    DuckDBRelation
        Relation for existing CFG edges.
    """
    return env.gateway.relation_from_table_key(CFG_EDGES_TABLE_KEY)


def dfg_edges_existing(env: BuildEnv) -> DuckDBRelation:
    """Load DFG edges from the existing table.

    Returns
    -------
    DuckDBRelation
        Relation for existing DFG edges.
    """
    return env.gateway.relation_from_table_key(DFG_EDGES_TABLE_KEY)


def cfg_blocks_empty(env: BuildEnv) -> DuckDBRelation:
    """Return an empty relation for CFG blocks.

    Returns
    -------
    DuckDBRelation
        Empty relation for CFG blocks.
    """
    return empty_relation_for_table(env.gateway.con, CFG_BLOCKS_TABLE_KEY)


def cfg_edges_empty(env: BuildEnv) -> DuckDBRelation:
    """Return an empty relation for CFG edges.

    Returns
    -------
    DuckDBRelation
        Empty relation for CFG edges.
    """
    return empty_relation_for_table(env.gateway.con, CFG_EDGES_TABLE_KEY)


def dfg_edges_empty(env: BuildEnv) -> DuckDBRelation:
    """Return an empty relation for DFG edges.

    Returns
    -------
    DuckDBRelation
        Empty relation for DFG edges.
    """
    return empty_relation_for_table(env.gateway.con, DFG_EDGES_TABLE_KEY)


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
