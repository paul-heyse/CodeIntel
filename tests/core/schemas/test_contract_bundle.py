"""Tests for ContractBundle aggregation."""

from __future__ import annotations

import msgspec

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.core.schemas.service import SchemaService


def _sample_schema() -> TableSchema:
    """Return a small schema for ContractBundle tests.

    Returns
    -------
    TableSchema
        Sample schema used by the tests.
    """
    return TableSchema(
        schema="core",
        name="bundle_demo",
        columns=[
            Column("id", "BIGINT", nullable=False),
            Column("name", "VARCHAR"),
        ],
        primary_key=("id",),
    )


def test_schema_service_bundle_has_row_struct_helpers() -> None:
    """Ensure contract bundles expose row struct helpers."""
    schema = _sample_schema()
    provider = MappingSchemaProvider({schema.table_key: schema})
    service = SchemaService(table_provider=provider)
    bundle = service.get_bundle(schema.table_key)

    assert bundle.table_schema is not None
    assert bundle.arrow_schema is not None
    assert bundle.row_binding is not None
    assert bundle.row_struct is not None
    assert issubclass(bundle.row_struct, msgspec.Struct)
    assert bundle.row_struct_builder is not None
    assert callable(bundle.row_struct_builder)
    assert bundle.row_struct_serializer is not None
    assert callable(bundle.row_struct_serializer)
