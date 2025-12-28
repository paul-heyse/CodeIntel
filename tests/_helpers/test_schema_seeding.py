"""Tests for shared schema seeding helpers."""

from __future__ import annotations

import duckdb

from codeintel.storage.constants import META_CATALOG_NAME, SCHEMAS
from tests._helpers.assertions.expectation_assertions import expect_true
from tests._helpers.schemas import ensure_production_schemas


def _list_schemas(
    con: duckdb.DuckDBPyConnection,
    *,
    catalog: str | None = None,
) -> set[str]:
    table_ref = "information_schema.schemata"
    if catalog is None:
        rows = con.execute(f"SELECT schema_name FROM {table_ref}").fetchall()
    else:
        rows = con.execute(
            f"SELECT schema_name FROM {table_ref} WHERE catalog_name = ?",
            [catalog],
        ).fetchall()
    return {str(row[0]) for row in rows}


def test_ensure_production_schemas_is_idempotent() -> None:
    """Production schema seeding should be safe to call multiple times."""
    con = duckdb.connect()
    try:
        ensure_production_schemas(con)
        ensure_production_schemas(con)
        schemas = _list_schemas(con)
        meta_schemas = _list_schemas(con, catalog=META_CATALOG_NAME)
        for schema_name in SCHEMAS:
            expect_true(
                schema_name in schemas,
                message=f"Expected schema {schema_name!r} to exist",
            )
        expect_true("metadata" in meta_schemas, message="Expected metadata schema to exist")
    finally:
        con.close()
