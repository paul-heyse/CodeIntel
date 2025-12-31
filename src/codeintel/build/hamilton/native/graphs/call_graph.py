"""Call graph relation sources for graph targets."""

from __future__ import annotations

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_frame_for_table
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_lazyframe
from codeintel.build.tabular.types import TabularFrame

CALL_GRAPH_NODES_TABLE_KEY = "graph.call_graph_nodes"
CALL_GRAPH_EDGES_TABLE_KEY = "graph.call_graph_edges"


def call_graph_nodes_existing(env: BuildEnv) -> TabularFrame:
    """Load call graph nodes from the dataset snapshot.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for existing call graph nodes.
    """
    return load_snapshot_lazyframe(
        env=env,
        table_key=CALL_GRAPH_NODES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def call_graph_edges_existing(env: BuildEnv) -> TabularFrame:
    """Load call graph edges from the dataset snapshot.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for existing call graph edges.
    """
    return load_snapshot_lazyframe(
        env=env,
        table_key=CALL_GRAPH_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def call_graph_nodes_empty(env: BuildEnv) -> TabularFrame:
    """Return an empty frame for call graph nodes.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame for call graph nodes.
    """
    _ = env
    return empty_frame_for_table(CALL_GRAPH_NODES_TABLE_KEY)


def call_graph_edges_empty(env: BuildEnv) -> TabularFrame:
    """Return an empty frame for call graph edges.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame for call graph edges.
    """
    _ = env
    return empty_frame_for_table(CALL_GRAPH_EDGES_TABLE_KEY)


__all__ = [
    "CALL_GRAPH_EDGES_TABLE_KEY",
    "CALL_GRAPH_NODES_TABLE_KEY",
    "call_graph_edges_empty",
    "call_graph_edges_existing",
    "call_graph_nodes_empty",
    "call_graph_nodes_existing",
]
