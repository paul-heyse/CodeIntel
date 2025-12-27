"""Test helpers for building minimal OutputContract objects."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.contracts import OutputContract, TableOutputDescriptor


def table_output_for_key(table_key: str) -> TableOutputDescriptor:
    """Return a minimal TableOutputDescriptor for the given table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableOutputDescriptor
        Output descriptor for the table key.
    """
    return TableOutputDescriptor(table_key=table_key)


def contract_for_keys(table_keys: Iterable[str]) -> OutputContract:
    """Return an OutputContract with table outputs for the provided table keys.

    Parameters
    ----------
    table_keys
        Table keys to include in the contract.

    Returns
    -------
    OutputContract
        OutputContract containing table output descriptors.
    """
    tables = tuple(table_output_for_key(key) for key in table_keys)
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
