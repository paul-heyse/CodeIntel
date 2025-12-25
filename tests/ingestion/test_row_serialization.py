"""Row serialization checks for ingestion helpers."""

from __future__ import annotations

from codeintel.build.schemas import get_schema_provider
from codeintel.core.schemas import SchemaService, clear_schema_service
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.core.schemas.service import set_schema_service
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_ingestion_row_serializer_matches_schema_order() -> None:
    """Row serializer should follow schema column ordering."""
    schema_provider = get_schema_provider()
    set_schema_service(SchemaService(table_provider=schema_provider))
    try:
        table_key = "core.modules"
        schema = schema_provider.require_table_schema(table_key)
        serializer = row_serializer_for_table_key(table_key)

        row = {
            "module": "pkg.module",
            "path": "pkg/module.py",
            "repo": "demo/repo",
            "commit": "deadbeef",
            "language": "python",
            "tags": [],
            "owners": [],
            "row_hash": "rowhash",
        }
        expected = tuple(row[column.name] for column in schema.columns)

        expect_equal(serializer(row), expected)
    finally:
        clear_schema_service()
