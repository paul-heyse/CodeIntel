"""Ensure dynamic prepared statements align with registry columns."""

from __future__ import annotations

import pytest

from codeintel.config.schemas.sql_builder import prepared_statements_dynamic
from codeintel.ingestion.ingest_service import INGEST_MACRO_TABLES, macro_exists
from codeintel.storage.gateway import StorageGateway


def test_dynamic_prepared_statements_match_registry(fresh_gateway: StorageGateway) -> None:
    """
    Dynamic prepared statements should use the registry column order.

    Raises
    ------
    AssertionError
        If placeholders and registry column counts diverge.
    """
    con = fresh_gateway.con
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


def test_ingest_macros_registered(fresh_gateway: StorageGateway) -> None:
    """Macro-backed ingest tables must have their macros present after bootstrap."""
    con = fresh_gateway.con
    for table_key in sorted(INGEST_MACRO_TABLES):
        macro_name = f"metadata.ingest_{table_key.split('.', maxsplit=1)[1]}"
        if not macro_exists(con, macro_name):
            pytest.fail(f"Missing ingest macro {macro_name}")
