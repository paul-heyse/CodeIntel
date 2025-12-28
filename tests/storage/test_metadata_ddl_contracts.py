"""Tests for contract-driven metadata DDL bootstrap."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pytest

from codeintel.storage.constants import META_CATALOG_NAME
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.metadata.ddl import apply_metadata_ddl
from codeintel.storage.metadata.meta_catalog import attach_meta_database
from codeintel.storage.metadata.schema import METADATA_TABLES
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def test_apply_metadata_ddl_is_idempotent() -> None:
    """apply_metadata_ddl can be safely applied multiple times."""
    con = duckdb.connect(":memory:")
    try:
        config = StorageConfig(
            db_path=Path(":memory:"),
            read_only=False,
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
        )
        attach_meta_database(con, config=config)
        apply_metadata_ddl(con, catalog=META_CATALOG_NAME)
        apply_metadata_ddl(con, catalog=META_CATALOG_NAME)

        expected_names = {table.name for table in METADATA_TABLES}
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'metadata' AND table_catalog = ? "
            "AND table_type = 'BASE TABLE'",
            [META_CATALOG_NAME],
        ).fetchall()
        actual_names = {str(row[0]) for row in rows}

        expect_true(expected_names.issubset(actual_names), message="all metadata tables exist")
        expect_equal(len(expected_names), len(actual_names), label="metadata table count")
    finally:
        con.close()


def test_attach_meta_database_warns_on_missing_read_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing meta catalogs should warn and skip attach in read-only mode."""
    con = duckdb.connect(":memory:")
    try:
        config = StorageConfig(
            db_path=tmp_path / "primary.duckdb",
            meta_db_path=tmp_path / "meta.duckdb",
            read_only=True,
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
        )
        with caplog.at_level(logging.WARNING):
            attach_meta_database(con, config=config)

        warning_messages = [record.message for record in caplog.records]
        expect_true(
            any("Meta database not found for read-only attach" in msg for msg in warning_messages),
            message="missing_meta_warning",
        )
        rows = con.execute("PRAGMA database_list").fetchall()
        attached = any(row[1] == META_CATALOG_NAME for row in rows)
        expect_true(not attached, message="meta_not_attached")
    finally:
        con.close()
