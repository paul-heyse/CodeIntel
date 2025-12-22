"""Row serialization checks for ingestion helpers."""

from __future__ import annotations

from codeintel.core.schemas import MappingSchemaProvider, SchemaService, clear_schema_service
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.core.schemas.service import set_schema_service
from codeintel.core.schemas.table_registry import TABLE_SCHEMAS
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_ingestion_row_serializer_matches_schema_order() -> None:
    """Row serializer should follow schema column ordering."""
    service = SchemaService(table_provider=MappingSchemaProvider(TABLE_SCHEMAS))
    set_schema_service(service)
    try:
        table_key = "core.modules"
        schema = TABLE_SCHEMAS[table_key]
        serializer = row_serializer_for_table_key(table_key)

        row = {
            "module": "pkg.module",
            "path": "pkg/module.py",
            "repo": "demo/repo",
            "commit": "deadbeef",
            "language": "python",
            "tags": [],
            "owners": [],
        }
        expected = tuple(row[column.name] for column in schema.columns)

        expect_equal(serializer(row), expected)
    finally:
        clear_schema_service()
