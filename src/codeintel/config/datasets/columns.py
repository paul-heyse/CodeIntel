"""Column and SQL helpers decoupled from legacy SQL builder APIs."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Final, TypeVar

from codeintel.config.datasets.contracts import get_table_schemas

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_Column = TypeVar("_Column", bound=str)


@lru_cache(maxsize=1)
def load_columns_by_table() -> dict[str, list[str]]:
    """Return registry column lists keyed by table.

    Returns
    -------
    dict[str, list[str]]
        Mapping of table key to column names.
    """
    table_schemas = get_table_schemas()
    return {
        table_key: [col.name for col in schema.columns]
        for table_key, schema in table_schemas.items()
    }


def serialize_row(row: Mapping[_Column, object], columns: Sequence[_Column]) -> tuple[object, ...]:
    """Serialize a mapping using a stable column sequence.

    Returns
    -------
    tuple[object, ...]
        Row values ordered according to the provided columns.
    """
    return tuple(row[column] for column in columns)


AST_NODES_DELETE: Final[str] = (
    "DELETE FROM core.ast_nodes "
    "WHERE path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
)
AST_METRICS_DELETE: Final[str] = (
    "DELETE FROM core.ast_metrics "
    "WHERE rel_path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
)
CST_NODES_DELETE: Final[str] = (
    "DELETE FROM core.cst_nodes "
    "WHERE rel_path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
)
CFG_BLOCKS_DELETE: Final[str] = (
    "DELETE FROM graph.cfg_blocks "
    "WHERE function_goid_h128 IN (SELECT goid_h128 FROM core.goids WHERE repo = ? AND commit = ?)"
)
CFG_EDGES_DELETE: Final[str] = (
    "DELETE FROM graph.cfg_edges "
    "WHERE function_goid_h128 IN (SELECT goid_h128 FROM core.goids WHERE repo = ? AND commit = ?)"
)
DFG_EDGES_DELETE: Final[str] = (
    "DELETE FROM graph.dfg_edges "
    "WHERE function_goid_h128 IN (SELECT goid_h128 FROM core.goids WHERE repo = ? AND commit = ?)"
)
CALL_GRAPH_NODES_DELETE: Final[str] = (
    "DELETE FROM graph.call_graph_nodes "
    "WHERE goid_h128 IN (SELECT goid_h128 FROM core.goids WHERE repo = ? AND commit = ?)"
)
GOID_CROSSWALK_UPDATE_SCIP: Final[str] = (
    "UPDATE core.goid_crosswalk "
    "SET scip_symbol = ? "
    "WHERE repo = ? AND commit = ? AND goid_h128 = ? AND scip_symbol IS NULL"
)
CALL_GRAPH_EDGES_DELETE: Final[str] = (
    "DELETE FROM graph.call_graph_edges "
    "WHERE caller_goid_h128 IN (SELECT goid_h128 FROM core.goids WHERE repo = ? AND commit = ?)"
)
SYMBOL_USE_DELETE: Final[str] = (
    "DELETE FROM core.symbol_use "
    "WHERE repo = ? AND commit = ? AND start_line >= ? AND end_line <= ?"
)
FILE_STATE_DELETE: Final[str] = "DELETE FROM core.file_state WHERE repo = ? AND commit = ?"
TAGS_INDEX_DELETE: Final[str] = "DELETE FROM graph.tags_index WHERE repo = ? AND commit = ?"
TEST_CATALOG_UPDATE_GOIDS: Final[str] = (
    "UPDATE analytics.test_catalog "
    "SET function_goid_h128 = ?, urn = ? "
    "WHERE test_id = ? AND rel_path = ? AND repo = ? AND commit = ?"
)


__all__ = [
    "AST_METRICS_DELETE",
    "AST_NODES_DELETE",
    "CALL_GRAPH_EDGES_DELETE",
    "CALL_GRAPH_NODES_DELETE",
    "CFG_BLOCKS_DELETE",
    "CFG_EDGES_DELETE",
    "CST_NODES_DELETE",
    "DFG_EDGES_DELETE",
    "FILE_STATE_DELETE",
    "GOID_CROSSWALK_UPDATE_SCIP",
    "SYMBOL_USE_DELETE",
    "TAGS_INDEX_DELETE",
    "TEST_CATALOG_UPDATE_GOIDS",
    "load_columns_by_table",
    "serialize_row",
]
