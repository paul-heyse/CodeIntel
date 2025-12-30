"""View-specific assertion helpers for storage tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests._helpers.sql import count_table_rows, validate_identifier

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.storage.gateway import StorageGateway


def assert_view_invariants(
    gateway: StorageGateway,
    table_key: str,
    required_columns: Iterable[str],
    *,
    repo: str | None = None,
    commit: str | None = None,
    min_rows: int | None = None,
) -> None:
    """Assert minimal invariants for inferred docs views.

    Parameters
    ----------
    gateway
        Storage gateway with database connection.
    table_key
        Fully qualified view name (schema.table).
    required_columns
        Column names that must appear in the view output.
    repo
        Optional repo identifier to validate when repo/commit columns exist.
    commit
        Optional commit identifier to validate when repo/commit columns exist.
    min_rows
        Optional minimum row count expected from the view.

    Raises
    ------
    AssertionError
        If any invariant fails.
    """
    safe_table = validate_identifier(table_key, kind="table")
    cursor = gateway.con.execute(f"SELECT * FROM {safe_table} LIMIT 0")
    description = cursor.description or []
    columns = [str(col[0]) for col in description]
    if not columns:
        message = f"Expected columns for view {table_key}, got none"
        raise AssertionError(message)

    columns_lower = {col.lower() for col in columns}
    required_lower = {str(col).lower() for col in required_columns}
    missing = sorted(required_lower - columns_lower)
    if missing:
        message = f"Missing required columns in {table_key}: {missing}"
        raise AssertionError(message)

    if repo is not None and commit is not None and {"repo", "commit"} <= columns_lower:
        mismatch = gateway.con.execute(
            f"""
            SELECT 1
            FROM {safe_table}
            WHERE repo != ? OR commit != ?
            LIMIT 1
            """,
            [repo, commit],
        ).fetchone()
        if mismatch is not None:
            message = f"Unexpected repo/commit rows in {table_key}"
            raise AssertionError(message)

    if min_rows is not None:
        count = count_table_rows(gateway.con, safe_table)
        if count < min_rows:
            message = f"Expected at least {min_rows} rows in {table_key}, got {count}"
            raise AssertionError(message)


__all__ = ["assert_view_invariants"]
