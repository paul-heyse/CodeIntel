"""SQL generation helpers for DuckDB INSERT and DELETE statements.

This module provides:
- INSERT SQL generation from TableSchema definitions
- DELETE SQL generation with repo/commit filters
- Column name accessors for tables
- Row serialization helpers
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Final, TypeVar

from codeintel.config.datasets.contracts import get_table_schemas
from codeintel.storage.sql.primitives import QueryBuilder, SafeColumn, SafeTable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_Column = TypeVar("_Column", bound=str)


def build_insert_sql(table_key: str) -> str:
    """Generate an INSERT SQL statement from the TableSchema.

    Parameters
    ----------
    table_key
        Fully qualified DuckDB table identifier, e.g. "analytics.function_metrics".

    Returns
    -------
    str
        INSERT INTO statement with placeholders.

    Raises
    ------
    ValueError
        If no schema is defined for the given table key.
    """
    table_schemas = get_table_schemas()
    schema = table_schemas.get(table_key)
    if schema is None:
        message = f"No schema defined for table key: {table_key}"
        raise ValueError(message)
    col_names = [col.name for col in schema.columns]

    safe_cols = [SafeColumn(name) for name in col_names]
    return QueryBuilder.insert(SafeTable(table_key), safe_cols)


def build_delete_sql(table_key: str) -> str | None:
    """Generate a DELETE SQL statement from the TableSchema.

    Parameters
    ----------
    table_key
        Fully qualified DuckDB table identifier.

    Returns
    -------
    str | None
        DELETE FROM statement with placeholders, or None if not applicable.
    """
    table_schemas = get_table_schemas()
    schema = table_schemas.get(table_key)
    if schema is None:
        return None
    col_names = [col.name for col in schema.columns]
    if "repo" in col_names and "commit" in col_names:
        return QueryBuilder.delete_repo_commit(SafeTable(table_key))
    return None


def build_insert_sql_by_table() -> dict[str, str]:
    """Generate INSERT SQL statements for all non-view tables.

    Returns
    -------
    dict[str, str]
        Mapping from table key to INSERT SQL statement.
    """
    result: dict[str, str] = {}
    table_schemas = get_table_schemas()
    for table_key in table_schemas:
        if table_key.startswith("docs."):
            continue
        result[table_key] = build_insert_sql(table_key)
    return result


def build_delete_sql_by_table() -> dict[str, str]:
    """Generate DELETE SQL statements for all tables with repo+commit columns.

    Returns
    -------
    dict[str, str]
        Mapping from table key to DELETE SQL statement.
    """
    result: dict[str, str] = {}
    table_schemas = get_table_schemas()
    for table_key in table_schemas:
        if table_key.startswith("docs."):
            continue
        sql = build_delete_sql(table_key)
        if sql is not None:
            result[table_key] = sql
    return result


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
    "WHERE path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)"
)


FILE_STATE_DELETE: Final[str] = (
    "DELETE FROM core.file_state WHERE repo = ? AND rel_path = ? AND language = ?"
)


TAGS_INDEX_DELETE: Final[str] = "DELETE FROM analytics.tags_index"
SYMBOL_USE_DELETE: Final[str] = "DELETE FROM graph.symbol_use_edges"
CALL_GRAPH_NODES_DELETE: Final[str] = "DELETE FROM graph.call_graph_nodes"
CFG_BLOCKS_DELETE: Final[str] = "DELETE FROM graph.cfg_blocks"
CFG_EDGES_DELETE: Final[str] = "DELETE FROM graph.cfg_edges"
DFG_EDGES_DELETE: Final[str] = "DELETE FROM graph.dfg_edges"


TEST_CATALOG_UPDATE_GOIDS: Final[str] = (
    "UPDATE analytics.test_catalog "
    "SET test_goid_h128 = ?, urn = ? "
    "WHERE test_id = ? AND rel_path = ? AND repo = ? AND commit = ?"
)
GOID_CROSSWALK_UPDATE_SCIP: Final[str] = (
    "UPDATE core.goid_crosswalk SET scip_symbol = ? WHERE goid = ? AND repo = ? AND commit = ?"
)


def load_columns_by_table() -> dict[str, list[str]]:
    """Return column-name lists for all tables tracked in TABLE_SCHEMAS.

    This is the no-DB-connection alternative to load_registry_columns().

    Returns
    -------
    dict[str, list[str]]
        Mapping of table key -> ordered column names.
    """
    table_schemas = get_table_schemas()
    return {
        table_key: [col.name for col in schema.columns]
        for table_key, schema in table_schemas.items()
    }


def get_table_columns(table_key: str) -> list[str]:
    """Return ordered column names for a specific table.

    Parameters
    ----------
    table_key
        Fully qualified DuckDB table identifier, e.g. "analytics.function_metrics".

    Returns
    -------
    list[str]
        Column names in definition order.

    Raises
    ------
    KeyError
        If no schema is defined for the given table key.
    """
    table_schemas = get_table_schemas()
    schema = table_schemas.get(table_key)
    if schema is None:
        message = f"No schema defined for table key: {table_key}"
        raise KeyError(message)
    return [col.name for col in schema.columns]


def get_contract_columns(table_key: str) -> tuple[str, ...]:
    """Retrieve column names from the TableSchema for a given table key.

    Parameters
    ----------
    table_key
        Fully qualified DuckDB table identifier, e.g. "analytics.function_metrics".

    Returns
    -------
    tuple[str, ...]
        Column names in schema definition order.

    Raises
    ------
    ValueError
        If no schema is defined for the given table key.
    """
    table_schemas = get_table_schemas()
    schema = table_schemas.get(table_key)
    if schema is None:
        message = f"No schema defined for table key: {table_key}"
        raise ValueError(message)
    return tuple(col.name for col in schema.columns)


def serialize_row(row: Mapping[_Column, object], columns: Sequence[_Column]) -> tuple[object, ...]:
    """Serialize a mapping using a stable column sequence.

    Parameters
    ----------
    row
        Row data as a mapping from column name to value.
    columns
        Ordered sequence of column names.

    Returns
    -------
    tuple[object, ...]
        Values ordered according to ``columns``.
    """
    return tuple(row[column] for column in columns)


def get_insert_sql_by_table() -> dict[str, str]:
    """Return the INSERT_SQL_BY_TABLE dictionary.

    Returns
    -------
    dict[str, str]
        Mapping from table key to INSERT SQL.
    """
    return _insert_sql_cache()


def get_delete_sql_by_table() -> dict[str, str]:
    """Return the DELETE_SQL_BY_TABLE dictionary.

    Returns
    -------
    dict[str, str]
        Mapping from table key to DELETE SQL.
    """
    return _delete_sql_cache()


@lru_cache(maxsize=1)
def _insert_sql_cache() -> dict[str, str]:
    return build_insert_sql_by_table()


@lru_cache(maxsize=1)
def _delete_sql_cache() -> dict[str, str]:
    return build_delete_sql_by_table()


def __getattr__(name: str) -> object:
    """Lazy attribute access for INSERT_SQL_BY_TABLE and DELETE_SQL_BY_TABLE.

    Parameters
    ----------
    name
        The attribute name to access.

    Returns
    -------
    object
        The requested attribute value.

    Raises
    ------
    AttributeError
        If the attribute is not found.
    """
    if name == "INSERT_SQL_BY_TABLE":
        return get_insert_sql_by_table()
    if name == "DELETE_SQL_BY_TABLE":
        return get_delete_sql_by_table()
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "AST_METRICS_DELETE",
    "AST_NODES_DELETE",
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
    "build_delete_sql",
    "build_delete_sql_by_table",
    "build_insert_sql",
    "build_insert_sql_by_table",
    "get_contract_columns",
    "get_delete_sql_by_table",
    "get_insert_sql_by_table",
    "get_table_columns",
    "load_columns_by_table",
    "serialize_row",
]
