"""Ensure dynamic prepared statements align with registry columns."""

from __future__ import annotations

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql.builder import prepared_statements_dynamic
from tests._helpers.macros import assert_ingest_macros_registered


def test_dynamic_prepared_statements_match_registry(macro_gateway: StorageGateway) -> None:
    """
    Dynamic prepared statements should use the registry column order.

    Raises
    ------
    AssertionError
        If placeholders and registry column counts diverge.
    """
    con = macro_gateway.con
    stmts = prepared_statements_dynamic(con, "analytics.function_metrics")
    registry_cols = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'analytics' AND table_name = 'function_metrics'
        ORDER BY ordinal_position
        """
    ).fetchall()
    col_count = len(registry_cols)
    placeholder_count = stmts.insert_sql.count("?")
    if placeholder_count != col_count:
        message = f"Placeholder count {placeholder_count} != registry cols {col_count}"
        raise AssertionError(message)


def test_ingest_macros_registered(macro_gateway: StorageGateway) -> None:
    """Macro-backed ingest tables must have their macros present after bootstrap."""
    assert_ingest_macros_registered(macro_gateway.con)
