"""Bridge helpers to consume registry-derived contracts for ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from codeintel.storage.registry_contracts import (
    ColumnDef,
    build_registry_contracts,
    render_create_table_from_catalog,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import DuckDBConnection


def load_registry_columns(
    con: DuckDBConnection,
) -> Mapping[str, list[str]]:
    """
    Return column-name lists for all tables tracked in the registry.

    Parameters
    ----------
    con :
        Active DuckDB connection.

    Returns
    -------
    Mapping[str, list[str]]
        Mapping of table key -> ordered column names.
    """
    contracts: Mapping[str, list[ColumnDef]] = build_registry_contracts(con)
    return {table_key: [col.name for col in cols] for table_key, cols in contracts.items()}


def render_registry_create_table(con: DuckDBConnection, table_key: str) -> str:
    """
    Render a CREATE TABLE statement from catalog metadata for the given table.

    Parameters
    ----------
    con :
        Active DuckDB connection.
    table_key : str
        Fully qualified table name (schema.table).

    Returns
    -------
    str
        Deterministic CREATE TABLE DDL reflecting the current catalog types.
    """
    return render_create_table_from_catalog(con, table_key)
