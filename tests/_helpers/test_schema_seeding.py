"""Tests for shared schema seeding helpers."""

from __future__ import annotations

import duckdb

from codeintel.storage.constants import META_CATALOG_NAME, SCHEMAS
from codeintel.storage.helpers.table_key import fully_qualified_table_ref
from tests._helpers.assertions.expectation_assertions import expect_true
from tests._helpers.schemas import ensure_production_schemas


def _list_schemas(
    con: duckdb.DuckDBPyConnection,
    *,
    catalog: str | None = None,
) -> set[str]:
    if catalog is None:
        table_ref = "information_schema.schemata"
    else:
        table_ref = fully_qualified_table_ref("information_schema.schemata", catalog=catalog)
    rows = con.execute(f"SELECT schema_name FROM {table_ref}").fetchall()
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
