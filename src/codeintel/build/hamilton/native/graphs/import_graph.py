"""Import graph relation sources for graph targets."""

from __future__ import annotations

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_relation_for_table
from codeintel.storage.gateway import DuckDBRelation

IMPORT_MODULES_TABLE_KEY = "graph.import_modules"
IMPORT_GRAPH_EDGES_TABLE_KEY = "graph.import_graph_edges"


def import_modules_existing(env: BuildEnv) -> DuckDBRelation:
    """Load import modules from the existing table.

    Returns
    -------
    DuckDBRelation
        Relation for existing import modules.
    """
    return env.gateway.relation_from_table_key(IMPORT_MODULES_TABLE_KEY)


def import_graph_edges_existing(env: BuildEnv) -> DuckDBRelation:
    """Load import graph edges from the existing table.

    Returns
    -------
    DuckDBRelation
        Relation for existing import graph edges.
    """
    return env.gateway.relation_from_table_key(IMPORT_GRAPH_EDGES_TABLE_KEY)


def import_modules_empty(env: BuildEnv) -> DuckDBRelation:
    """Return an empty relation for import modules.

    Returns
    -------
    DuckDBRelation
        Empty relation for import modules.
    """
    return empty_relation_for_table(env.gateway.con, IMPORT_MODULES_TABLE_KEY)


def import_graph_edges_empty(env: BuildEnv) -> DuckDBRelation:
    """Return an empty relation for import graph edges.

    Returns
    -------
    DuckDBRelation
        Empty relation for import graph edges.
    """
    return empty_relation_for_table(env.gateway.con, IMPORT_GRAPH_EDGES_TABLE_KEY)


__all__ = [
    "IMPORT_GRAPH_EDGES_TABLE_KEY",
    "IMPORT_MODULES_TABLE_KEY",
    "import_graph_edges_empty",
    "import_graph_edges_existing",
    "import_modules_empty",
    "import_modules_existing",
]
