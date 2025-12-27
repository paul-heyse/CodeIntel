"""Call graph relation sources for graph targets."""

from __future__ import annotations

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_relation_for_table
from codeintel.storage.gateway import DuckDBRelation

CALL_GRAPH_NODES_TABLE_KEY = "graph.call_graph_nodes"
CALL_GRAPH_EDGES_TABLE_KEY = "graph.call_graph_edges"


def call_graph_nodes_existing(env: BuildEnv) -> DuckDBRelation:
    """Load call graph nodes from the existing table.

    Returns
    -------
    DuckDBRelation
        Relation for existing call graph nodes.
    """
    return env.gateway.relation_from_table_key(CALL_GRAPH_NODES_TABLE_KEY)


def call_graph_edges_existing(env: BuildEnv) -> DuckDBRelation:
    """Load call graph edges from the existing table.

    Returns
    -------
    DuckDBRelation
        Relation for existing call graph edges.
    """
    return env.gateway.relation_from_table_key(CALL_GRAPH_EDGES_TABLE_KEY)


def call_graph_nodes_empty(env: BuildEnv) -> DuckDBRelation:
    """Return an empty relation for call graph nodes.

    Returns
    -------
    DuckDBRelation
        Empty relation for call graph nodes.
    """
    return empty_relation_for_table(env.gateway.con, CALL_GRAPH_NODES_TABLE_KEY)


def call_graph_edges_empty(env: BuildEnv) -> DuckDBRelation:
    """Return an empty relation for call graph edges.

    Returns
    -------
    DuckDBRelation
        Empty relation for call graph edges.
    """
    return empty_relation_for_table(env.gateway.con, CALL_GRAPH_EDGES_TABLE_KEY)


__all__ = [
    "CALL_GRAPH_EDGES_TABLE_KEY",
    "CALL_GRAPH_NODES_TABLE_KEY",
    "call_graph_edges_empty",
    "call_graph_edges_existing",
    "call_graph_nodes_empty",
    "call_graph_nodes_existing",
]
