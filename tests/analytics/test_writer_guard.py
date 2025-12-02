"""Contract tests for writer_guard utilities."""

from __future__ import annotations

from typing import cast

import pytest

from codeintel.analytics.profiles.writer_guard import WriterContext, write_rows_with_registry_guard
from codeintel.config.datasets import TABLE_SCHEMAS
from codeintel.storage.gateway import DuckDBConnection
from codeintel.storage.sql_helpers import PreparedStatements


class _FakeCon:
    def __init__(self) -> None:
        self.executed: list[tuple[str, list[object] | None]] = []
        self.executemany_calls: list[tuple[str, list[list[object]]]] = []

    def execute(self, sql: str, params: list[object] | None = None) -> _FakeCon:
        self.executed.append((sql, params))
        return self

    def executemany(self, sql: str, params_list: list[list[object]]) -> None:
        self.executemany_calls.append((sql, params_list))


# Get a real table and its columns for testing
_TEST_TABLE = "analytics.coverage_lines"
_TEST_SCHEMA = TABLE_SCHEMAS[_TEST_TABLE]
_TEST_COLUMNS = tuple(col.name for col in _TEST_SCHEMA.columns)


def _ctx(table: str = _TEST_TABLE, *, repo: str = "r", commit: str = "c") -> WriterContext:
    schema = TABLE_SCHEMAS.get(table)
    if schema is None:
        msg = f"Table {table} not found in TABLE_SCHEMAS."
        raise ValueError(msg)
    columns = tuple(col.name for col in schema.columns)
    delete_sql = f"DELETE FROM {table} WHERE repo = ? AND commit = ?"  # noqa: S608
    insert_sql = (
        f"INSERT INTO {table} ("  # noqa: S608
        + ", ".join(f'"{c}"' for c in columns)
        + ") VALUES ("
        + ", ".join("?" for _ in columns)
        + ")"
    )
    return WriterContext(
        table_key=table,
        columns=columns,
        serialize_row=lambda row: tuple(row.get(c) for c in columns),
        repo=repo,
        commit=commit,
        delete_sql=delete_sql,
        ensure_schema_fn=lambda _con, _table: None,
        prepared_statements_fn=lambda _con, _table: PreparedStatements(insert_sql=insert_sql),
    )


def test_writer_guard_happy_path_executes_delete_and_insert() -> None:
    """Delete then insert when columns match TABLE_SCHEMAS and rows are present."""
    fake_con = _FakeCon()
    # Create a row with all required columns
    row_data = {col: f"val_{i}" for i, col in enumerate(_TEST_COLUMNS)}
    row_data["repo"] = "r"
    row_data["commit"] = "c"
    rows = [row_data]

    inserted = write_rows_with_registry_guard(
        cast("DuckDBConnection", fake_con), rows=rows, context=_ctx()
    )

    if inserted != len(rows):
        pytest.fail("Inserted count mismatch for writer_guard happy path.")
    if not fake_con.executed:
        pytest.fail("Delete call not issued as expected.")
    if not fake_con.executemany_calls:
        pytest.fail("Insert call not issued as expected.")


@pytest.mark.parametrize(
    ("delete_on_empty", "expected_calls"),
    [
        (True, 1),
        (False, 0),
    ],
)
def test_writer_guard_empty_rows(*, delete_on_empty: bool, expected_calls: int) -> None:
    """Delete on empty rows only if configured."""
    fake_con = _FakeCon()

    inserted = write_rows_with_registry_guard(
        cast("DuckDBConnection", fake_con),
        rows=[],
        context=_ctx(),
        delete_on_empty=delete_on_empty,
    )

    if inserted != 0:
        pytest.fail("Writer should report zero inserts for empty rows.")
    if len(fake_con.executed) != expected_calls:
        pytest.fail("Delete call count mismatch for empty rows.")
    if fake_con.executemany_calls:
        pytest.fail("Insert should not be called for empty rows.")


def test_writer_guard_columns_mismatch_raises() -> None:
    """Context columns that don't match TABLE_SCHEMAS should raise."""
    fake_con = _FakeCon()
    base_ctx = _ctx()
    # Create a context with wrong columns (fewer than expected)
    mismatch_ctx = WriterContext(
        table_key=base_ctx.table_key,
        columns=("repo", "commit"),  # Missing columns
        serialize_row=lambda row: (row["repo"], row["commit"]),
        repo=base_ctx.repo,
        commit=base_ctx.commit,
        delete_sql=base_ctx.delete_sql,
        ensure_schema_fn=base_ctx.ensure_schema_fn,
        prepared_statements_fn=base_ctx.prepared_statements_fn,
    )

    with pytest.raises(RuntimeError):
        write_rows_with_registry_guard(
            cast("DuckDBConnection", fake_con),
            rows=[{"repo": "r", "commit": "c"}],
            context=mismatch_ctx,
        )


def test_writer_guard_columns_order_drift_raises() -> None:
    """Drifted column order should raise."""
    fake_con = _FakeCon()
    base_ctx = _ctx()
    # Create a context with swapped column order
    swapped_columns = (_TEST_COLUMNS[1], _TEST_COLUMNS[0], *_TEST_COLUMNS[2:])
    drift_ctx = WriterContext(
        table_key=base_ctx.table_key,
        columns=swapped_columns,
        serialize_row=base_ctx.serialize_row,
        repo=base_ctx.repo,
        commit=base_ctx.commit,
        delete_sql=base_ctx.delete_sql,
        ensure_schema_fn=base_ctx.ensure_schema_fn,
        prepared_statements_fn=base_ctx.prepared_statements_fn,
    )

    with pytest.raises(RuntimeError):
        write_rows_with_registry_guard(
            cast("DuckDBConnection", fake_con),
            rows=[{"repo": "r", "commit": "c"}],
            context=drift_ctx,
        )


def test_writer_guard_nonexistent_table_raises() -> None:
    """Table not in TABLE_SCHEMAS should raise."""
    fake_con = _FakeCon()
    nonexistent_ctx = WriterContext(
        table_key="nonexistent.table",
        columns=("repo", "commit", "value"),
        serialize_row=lambda row: (row["repo"], row["commit"], row["value"]),
        repo="r",
        commit="c",
        delete_sql="DELETE FROM nonexistent.table WHERE repo = ? AND commit = ?",
        ensure_schema_fn=lambda _con, _table: None,
        prepared_statements_fn=lambda _con, _table: PreparedStatements(
            insert_sql="INSERT INTO nonexistent.table VALUES (?, ?, ?)"
        ),
    )

    with pytest.raises(RuntimeError):
        write_rows_with_registry_guard(
            cast("DuckDBConnection", fake_con),
            rows=[{"repo": "r", "commit": "c", "value": 1}],
            context=nonexistent_ctx,
        )
