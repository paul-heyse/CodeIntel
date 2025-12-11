"""Shared guardrails for profile writers.

This module provides utilities for writing profile data with schema validation
and bulk insertion support.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.config.datasets import load_columns_by_table
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.gateway import DuckDBConnection
from codeintel.storage.sql import PreparedStatements

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway

SerializeRow = Callable[[Mapping[str, object]], tuple[object, ...]]


@dataclass(frozen=True)
class WriterContext:
    """Dependencies and schema contract for a profile writer."""

    table_key: str
    columns: Sequence[str]
    serialize_row: SerializeRow
    repo: str
    commit: str
    delete_sql: str
    ensure_schema_fn: Callable[[DuckDBConnection, str], None]
    prepared_statements_fn: Callable[[DuckDBConnection, str], PreparedStatements]


def write_rows_with_registry_guard(
    gateway: StorageGateway,
    *,
    rows: Iterable[Mapping[str, object]],
    context: WriterContext,
    delete_on_empty: bool = True,
) -> int:
    """Ensure schema alignment and perform delete/insert for a profile table.

    Returns
    -------
    int
        Number of inserted rows.

    Raises
    ------
    RuntimeError
        If columns from TABLE_SCHEMAS diverge from serializer constants.
    """
    rows_list = list(rows)
    if not rows_list and not delete_on_empty:
        return 0

    con = gateway.con
    ensure_schema_fn = context.ensure_schema_fn
    ensure_schema_fn(con, context.table_key)
    registry_cols = load_columns_by_table().get(context.table_key)
    if registry_cols is None or tuple(registry_cols) != tuple(context.columns):
        message = f"Columns for {context.table_key} differ from serializer constants."
        raise RuntimeError(message)

    # Delete existing data using policy backend
    backend = DuckDBPolicyBackend(gateway)
    backend.delete_for_snapshot(context.table_key, repo=context.repo, commit=context.commit)

    if not rows_list:
        return 0

    # Write rows using Ibis
    tuples = [context.serialize_row(row) for row in rows_list]
    gateway.ibis.write(context.table_key, tuples, columns=list(context.columns))
    return len(tuples)


@dataclass(frozen=True)
class PolicyWriterConfig:
    """Configuration for policy-backend-based row writing."""

    table_key: str
    columns: Sequence[str]
    serialize_row: SerializeRow
    repo: str
    commit: str


def write_rows_via_policy_backend(
    gateway: StorageGateway,
    *,
    rows: Iterable[Mapping[str, object]],
    config: PolicyWriterConfig,
) -> int:
    """Write rows using DuckDBPolicyBackend for bulk insert.

    This function provides a cleaner API that uses the centralized policy
    backend for SQL generation, replacing direct executemany calls.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    rows
        Iterable of row dictionaries to insert.
    config
        Writer configuration including table key, columns, and serializer.

    Returns
    -------
    int
        Number of rows inserted.
    """
    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend  # noqa: PLC0415

    rows_list = list(rows)
    if not rows_list:
        return 0

    backend = DuckDBPolicyBackend(gateway)

    # Delete existing rows for this snapshot
    backend.delete_for_snapshot(config.table_key, repo=config.repo, commit=config.commit)

    # Serialize and insert
    tuples = [config.serialize_row(row) for row in rows_list]
    return backend.bulk_insert(config.table_key, tuples, columns=list(config.columns))
