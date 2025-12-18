"""Tests for contract-driven metadata DDL bootstrap."""

from __future__ import annotations

import duckdb

from codeintel.storage.metadata.bootstrap import apply_metadata_ddl
from codeintel.storage.metadata.schema import METADATA_TABLES
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def test_apply_metadata_ddl_is_idempotent() -> None:
    """apply_metadata_ddl can be safely applied multiple times."""
    con = duckdb.connect(":memory:")
    try:
        apply_metadata_ddl(con)
        apply_metadata_ddl(con)

        expected_names = {table.name for table in METADATA_TABLES}
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'metadata'"
        ).fetchall()
        actual_names = {str(row[0]) for row in rows}

        expect_true(expected_names.issubset(actual_names), message="all metadata tables exist")
        expect_equal(len(expected_names), len(actual_names), label="metadata table count")
    finally:
        con.close()
