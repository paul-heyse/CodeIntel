"""Contract tests for writer_guard utilities."""

from __future__ import annotations

from typing import cast

import pytest

from codeintel.analytics.profiles.writer_guard import WriterContext, write_rows_with_registry_guard
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


def _ctx(
    table: str = "analytics.test_profile", *, repo: str = "r", commit: str = "c"
) -> WriterContext:
    delete_lookup = {
        "analytics.test_profile": "DELETE FROM analytics.test_profile WHERE repo = ? AND commit = ?",
        "analytics.custom_profile": "DELETE FROM analytics.custom_profile WHERE repo = ? AND commit = ?",
    }
    insert_lookup = {
        "analytics.test_profile": "INSERT INTO analytics.test_profile VALUES (?, ?, ?)",
        "analytics.custom_profile": "INSERT INTO analytics.custom_profile VALUES (?, ?, ?)",
    }
    if table not in delete_lookup:
        msg = "Unsupported table_key for writer_guard test context."
        raise ValueError(msg)
    return WriterContext(
        table_key=table,
        columns=("repo", "commit", "value"),
        serialize_row=lambda row: (row["repo"], row["commit"], row["value"]),
        repo=repo,
        commit=commit,
        delete_sql=delete_lookup[table],
        ensure_schema_fn=lambda _con, _table: None,
        load_registry_columns_fn=lambda _con: {table: ("repo", "commit", "value")},
        prepared_statements_fn=lambda _con, _table: PreparedStatements(
            insert_sql=insert_lookup[table]
        ),
    )


def test_writer_guard_happy_path_executes_delete_and_insert() -> None:
    """Delete then insert when registry matches and rows are present."""
    fake_con = _FakeCon()
    rows = [{"repo": "r", "commit": "c", "value": 1}, {"repo": "r", "commit": "c", "value": 2}]

    inserted = write_rows_with_registry_guard(
        cast("DuckDBConnection", fake_con), rows=rows, context=_ctx()
    )

    expected_delete = (
        "DELETE FROM analytics.test_profile WHERE repo = ? AND commit = ?",
        ["r", "c"],
    )
    expected_insert = (
        "INSERT INTO analytics.test_profile VALUES (?, ?, ?)",
        [("r", "c", 1), ("r", "c", 2)],
    )
    if inserted != len(rows):
        pytest.fail("Inserted count mismatch for writer_guard happy path.")
    if fake_con.executed != [expected_delete]:
        pytest.fail("Delete call not issued as expected.")
    if fake_con.executemany_calls != [expected_insert]:
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


def test_writer_guard_registry_mismatch_raises() -> None:
    """Missing registry entry should raise."""
    fake_con = _FakeCon()
    base_ctx = _ctx()
    mismatch_ctx = WriterContext(
        table_key=base_ctx.table_key,
        columns=("repo", "commit", "value"),
        serialize_row=base_ctx.serialize_row,
        repo=base_ctx.repo,
        commit=base_ctx.commit,
        delete_sql=base_ctx.delete_sql,
        ensure_schema_fn=base_ctx.ensure_schema_fn,
        load_registry_columns_fn=lambda _con: {},  # missing entry
        prepared_statements_fn=base_ctx.prepared_statements_fn,
    )

    with pytest.raises(RuntimeError):
        write_rows_with_registry_guard(
            cast("DuckDBConnection", fake_con),
            rows=[{"repo": "r", "commit": "c", "value": 1}],
            context=mismatch_ctx,
        )


def test_writer_guard_registry_columns_drift_raises() -> None:
    """Drifted registry columns should raise."""
    fake_con = _FakeCon()
    base_ctx = _ctx()
    drift_ctx = WriterContext(
        table_key=base_ctx.table_key,
        columns=("repo", "commit", "value"),
        serialize_row=base_ctx.serialize_row,
        repo=base_ctx.repo,
        commit=base_ctx.commit,
        delete_sql=base_ctx.delete_sql,
        ensure_schema_fn=base_ctx.ensure_schema_fn,
        load_registry_columns_fn=lambda _con: {base_ctx.table_key: ("repo", "commit", "other")},
        prepared_statements_fn=base_ctx.prepared_statements_fn,
    )

    with pytest.raises(RuntimeError):
        write_rows_with_registry_guard(
            cast("DuckDBConnection", fake_con),
            rows=[{"repo": "r", "commit": "c", "value": 1}],
            context=drift_ctx,
        )


def test_writer_guard_respects_custom_delete_sql() -> None:
    """Delete SQL in context is honored."""
    fake_con = _FakeCon()
    ctx = _ctx(table="analytics.custom_profile")

    write_rows_with_registry_guard(
        cast("DuckDBConnection", fake_con),
        rows=[{"repo": "r", "commit": "c", "value": 1}],
        context=ctx,
    )

    if not fake_con.executed or not fake_con.executed[0][0].startswith(
        "DELETE FROM analytics.custom_profile"
    ):
        pytest.fail("Custom delete SQL was not used.")
