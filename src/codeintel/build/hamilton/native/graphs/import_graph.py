"""Import graph relation sources for graph targets."""

from __future__ import annotations

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_frame_for_table
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_lazyframe
from codeintel.build.tabular.types import TabularFrame

IMPORT_MODULES_TABLE_KEY = "graph.import_modules"
IMPORT_GRAPH_EDGES_TABLE_KEY = "graph.import_graph_edges"


def import_modules_existing(env: BuildEnv) -> TabularFrame:
    """Load import modules from the dataset snapshot.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for existing import modules.
    """
    return load_snapshot_lazyframe(
        env=env,
        table_key=IMPORT_MODULES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def import_graph_edges_existing(env: BuildEnv) -> TabularFrame:
    """Load import graph edges from the dataset snapshot.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for existing import graph edges.
    """
    return load_snapshot_lazyframe(
        env=env,
        table_key=IMPORT_GRAPH_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def import_modules_empty(env: BuildEnv) -> TabularFrame:
    """Return an empty frame for import modules.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame for import modules.
    """
    _ = env
    return empty_frame_for_table(IMPORT_MODULES_TABLE_KEY)


def import_graph_edges_empty(env: BuildEnv) -> TabularFrame:
    """Return an empty frame for import graph edges.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame for import graph edges.
    """
    _ = env
    return empty_frame_for_table(IMPORT_GRAPH_EDGES_TABLE_KEY)


__all__ = [
    "IMPORT_GRAPH_EDGES_TABLE_KEY",
    "IMPORT_MODULES_TABLE_KEY",
    "import_graph_edges_empty",
    "import_graph_edges_existing",
    "import_modules_empty",
    "import_modules_existing",
]
