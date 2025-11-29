"""Shared guardrails for profile writers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from codeintel.storage.gateway import DuckDBConnection
from codeintel.storage.sql_helpers import PreparedStatements

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
    load_registry_columns_fn: Callable[[DuckDBConnection], Mapping[str, Sequence[str]]]
    prepared_statements_fn: Callable[[DuckDBConnection, str], PreparedStatements]


def write_rows_with_registry_guard(
    con: DuckDBConnection,
    *,
    rows: Iterable[Mapping[str, object]],
    context: WriterContext,
    delete_on_empty: bool = True,
) -> int:
    """
    Ensure registry alignment and perform delete/insert for a profile table.

    Returns
    -------
    int
        Number of inserted rows.

    Raises
    ------
    RuntimeError
        If registry columns diverge from serializer constants.
    """
    rows_list = list(rows)
    if not rows_list and not delete_on_empty:
        return 0

    ensure_schema_fn = context.ensure_schema_fn
    ensure_schema_fn(con, context.table_key)
    registry_cols = context.load_registry_columns_fn(con).get(context.table_key)
    if registry_cols is None or tuple(registry_cols) != tuple(context.columns):
        message = f"Registry columns for {context.table_key} differ from serializer constants."
        raise RuntimeError(message)

    stmt = context.prepared_statements_fn(con, context.table_key)
    con.execute(context.delete_sql, [context.repo, context.commit])
    if not rows_list:
        return 0

    tuples = [context.serialize_row(row) for row in rows_list]
    con.executemany(stmt.insert_sql, tuples)
    return len(tuples)
