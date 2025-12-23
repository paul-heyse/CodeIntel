"""Test helpers for building minimal OutputContract objects."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.contracts import OutputContract
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.helpers.table_key import split_table_key


def table_schema_for_key(table_key: str) -> TableSchema:
    """Return a minimal TableSchema for tests using the given table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema
        TableSchema with a single non-nullable id column.
    """
    schema, name = split_table_key(table_key)
    return TableSchema(
        schema=schema,
        name=name,
        columns=[Column("id", "VARCHAR", nullable=False)],
    )


def contract_for_keys(table_keys: Iterable[str]) -> OutputContract:
    """Return an OutputContract with minimal schemas for the provided table keys.

    Parameters
    ----------
    table_keys
        Table keys to include in the contract.

    Returns
    -------
    OutputContract
        OutputContract containing minimal table schemas.
    """
    tables = tuple(table_schema_for_key(key) for key in table_keys)
    return OutputContract(tables=tables)


if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from codeintel.storage.gateway import StorageGateway


@dataclass
class ContractCtx:
    """Context for contract-focused tests."""

    gateway: StorageGateway
    repo: str
    commit: str


def count_rows(
    con: DuckDBPyConnection,
    sql: str,
    parameters: Sequence[object],
) -> int:
    """Return the row count for a query.

    Parameters
    ----------
    con
        DuckDB connection to execute against.
    sql
        SQL query returning a single count.
    parameters
        Parameters for the query.

    Returns
    -------
    int
        Count of rows returned by the query.
    """
    result = con.execute(sql, parameters).fetchone()
    if result is None:
        return 0
    return int(result[0])
