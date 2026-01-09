"""Helpers for deterministic table snapshots in tests."""

from __future__ import annotations

import difflib
import json
from typing import TYPE_CHECKING

from codeintel.core.columnar.conversion import tabular_to_arrow_table
from codeintel.core.hashing.short import sha256_short
from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    import duckdb


def snapshot_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    *,
    order_by: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    hash_rows: bool = False,
) -> list[tuple[object, ...]]:
    """Snapshot a table with deterministic ordering.

    Returns
    -------
    list[tuple[object, ...]]
        Rows from the table in deterministic order.
    """
    if order_by is None:
        rows = con.execute("PRAGMA table_info(?)", [table]).fetchall()
        order_by = [str(row[1]) for row in rows] if rows else []

    relation = con.table(table)
    if columns is not None:
        relation = relation.select(*columns)
    if order_by:
        ordered = ", ".join(order_by)
        relation = relation.order(ordered)
    table_data = tabular_to_arrow_table(relation)
    column_names = table_data.column_names
    rows = [tuple(row[name] for name in column_names) for row in table_data.to_pylist()]
    if hash_rows:
        return [(_hash_row(row),) for row in rows]
    return rows


def snapshot_tables(
    con: duckdb.DuckDBPyConnection,
    tables: Iterable[str],
    *,
    order_by: Mapping[str, Sequence[str]] | None = None,
    columns: Mapping[str, Sequence[str]] | None = None,
    hash_rows: bool = False,
) -> dict[str, list[tuple[object, ...]]]:
    """Snapshot multiple tables into a mapping.

    Returns
    -------
    dict[str, list[tuple[object, ...]]]
        Mapping of table names to their ordered rows.
    """
    return {
        table: snapshot_table(
            con,
            table,
            order_by=(order_by or {}).get(table),
            columns=(columns or {}).get(table),
            hash_rows=hash_rows,
        )
        for table in tables
    }


def write_table_snapshot(path: Path, snapshot: Mapping[str, list[tuple[object, ...]]]) -> None:
    """Write table snapshots to disk as JSON."""
    payload = {table: [list(row) for row in rows] for table, rows in snapshot.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_table_snapshot(path: Path) -> dict[str, list[list[object]]]:
    """Load a JSON table snapshot from disk.

    Returns
    -------
    dict[str, list[list[object]]]
        Snapshot data as loaded from JSON.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def assert_snapshot_equal(
    actual: Mapping[str, list[tuple[object, ...]]],
    expected: Mapping[str, list[list[object]]],
) -> None:
    """Assert that two table snapshots match."""
    normalized = {table: [list(row) for row in rows] for table, rows in actual.items()}
    expect_equal(normalized, dict(expected), label="table_snapshot")


def diff_table_snapshot(
    actual: Mapping[str, list[tuple[object, ...]]],
    expected: Mapping[str, list[list[object]]],
    *,
    fromfile: str = "expected",
    tofile: str = "actual",
    context_lines: int = 3,
) -> str:
    """Return a unified diff between expected and actual snapshots.

    Returns
    -------
    str
        Unified diff text, or "(no diff)" when identical.
    """
    normalized = {table: [list(row) for row in rows] for table, rows in actual.items()}
    exp_lines = json.dumps(dict(expected), indent=2, sort_keys=True).splitlines()
    act_lines = json.dumps(normalized, indent=2, sort_keys=True).splitlines()
    diff = difflib.unified_diff(
        exp_lines,
        act_lines,
        fromfile=fromfile,
        tofile=tofile,
        n=context_lines,
    )
    lines = list(diff)
    if not lines:
        return "(no diff)"
    return "\n".join(lines)


def _hash_row(row: tuple[object, ...]) -> str:
    """Return a stable hash for a row tuple.

    Returns
    -------
    str
        Short hash digest of the row contents.
    """
    payload = "|".join(str(value) for value in row)
    return sha256_short(payload, length=16, used_for_security=False)


__all__ = [
    "assert_snapshot_equal",
    "diff_table_snapshot",
    "load_table_snapshot",
    "snapshot_table",
    "snapshot_tables",
    "write_table_snapshot",
]
