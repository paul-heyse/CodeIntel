"""Helpers for configuring SchemaService in tests."""

from __future__ import annotations

from codeintel.core.schemas import MappingSchemaProvider, SchemaService
from codeintel.core.schemas.service import get_schema_service, set_schema_service
from codeintel.storage.contracts.schema_provider import get_schema_provider


def ensure_schema_service() -> SchemaService:
    """Ensure the global SchemaService is initialized.

    Returns
    -------
    SchemaService
        Configured schema service instance.
    """
    try:
        return get_schema_service()
    except RuntimeError:
        provider = get_schema_provider()
        schemas = {schema.table_key: schema for schema in provider.iter_table_schemas()}
        service = SchemaService(table_provider=MappingSchemaProvider(schemas))
        set_schema_service(service)
        return service


__all__ = ["ensure_schema_service"]
