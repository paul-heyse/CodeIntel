"""Ensure DuckDB schemas are applied and aligned with TABLE_SCHEMAS."""

from __future__ import annotations

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schemas import assert_schema_alignment


def test_apply_and_validate_schema_alignment(fresh_gateway: StorageGateway) -> None:
    """
    Schema application should create expected columns (incl. decorator spans).

    Raises
    ------
    AssertionError
        If schema drift is detected or decorator columns are missing.
    """
    con = fresh_gateway.con
    issues = assert_schema_alignment(con, strict=False)
    if issues:
        message = f"Schema drift detected: {issues}"
        raise AssertionError(message)

    cols = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'core' AND table_name = 'ast_nodes'
        ORDER BY ordinal_position
        """
    ).fetchall()
    col_names = [row[0] for row in cols]
    expected_columns = {"decorator_start_line", "decorator_end_line"}
    if not expected_columns.issubset(set(col_names)):
        message = f"Missing decorator columns in core.ast_nodes: {col_names}"
        raise AssertionError(message)
