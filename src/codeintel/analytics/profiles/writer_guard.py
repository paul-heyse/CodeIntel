"""Shared guardrails for profile writers.

This module provides utilities for writing profile data with schema validation
and bulk insertion support.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.config.datasets import load_columns_by_table
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.ibis_types import and_predicates, ibis_bool

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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
    ensure_schema_fn: Callable[[StorageGateway, str], None]


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

    ensure_schema_fn = context.ensure_schema_fn
    ensure_schema_fn(gateway, context.table_key)
    registry_cols = load_columns_by_table().get(context.table_key)
    if registry_cols is None or tuple(registry_cols) != tuple(context.columns):
        message = f"Columns for {context.table_key} differ from serializer constants."
        raise RuntimeError(message)

    table = gateway.ibis.table(context.table_key)

    where = and_predicates(
        ibis_bool(table.repo == context.repo),
        ibis_bool(table.commit == context.commit),
    )
    gateway.ibis.delete(context.table_key, where=where)

    if not rows_list:
        return 0

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
    rows_list = list(rows)
    if not rows_list:
        return 0

    backend = DuckDBPolicyBackend(gateway)

    backend.delete_for_snapshot(config.table_key, repo=config.repo, commit=config.commit)

    tuples = [config.serialize_row(row) for row in rows_list]
    return backend.bulk_insert(config.table_key, tuples, columns=list(config.columns))


def create_profile_writer(
    table_key: str,
    columns: Sequence[str],
    serialize_row: SerializeRow,
) -> Callable[[StorageGateway, Iterable[Mapping[str, object]]], int]:
    """Create a profile writer function for the specified table.

    This factory produces standardized profile writer functions that handle
    registry alignment checks and delete-before-insert semantics.

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g. "analytics.function_profile").
    columns
        Column names in insertion order.
    serialize_row
        Function to convert row dict to tuple for insertion.

    Returns
    -------
    Callable[[StorageGateway, Iterable[Mapping[str, object]]], int]
        Writer function that takes gateway and rows, returning row count.

    Examples
    --------
    >>> write_function_profile = create_profile_writer(
    ...     "analytics.function_profile",
    ...     FUNCTION_PROFILE_COLUMNS,
    ...     function_profile_row_to_tuple,
    ... )
    >>> count = write_function_profile(gateway, rows)
    """

    def writer(gateway: StorageGateway, rows: Iterable[Mapping[str, object]]) -> int:
        """Write profile rows to the configured table.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rows
            Iterable of row dictionaries to insert.

        Returns
        -------
        int
            Number of rows inserted.
        """
        rows_list = list(rows)
        if not rows_list:
            return 0

        repo = str(rows_list[0]["repo"])
        commit = str(rows_list[0]["commit"])
        context = WriterContext(
            table_key=table_key,
            columns=columns,
            serialize_row=serialize_row,
            repo=repo,
            commit=commit,
            ensure_schema_fn=lambda _gateway, _table: None,
        )
        return write_rows_with_registry_guard(
            gateway,
            rows=rows_list,
            context=context,
            delete_on_empty=False,
        )

    return writer
