"""Helpers for configuring SchemaService in tests."""

from __future__ import annotations

from codeintel.build.catalogs.canonical import load_contract_catalog
from codeintel.core.schemas import MappingSchemaProvider, SchemaService
from codeintel.core.schemas.service import get_schema_service, set_schema_service
from codeintel.storage.contracts.catalog_state import get_contract_catalog
from codeintel.storage.contracts.provider import set_contract_catalog
from codeintel.storage.contracts.schema_provider import (
    clear_schema_provider_cache,
    get_schema_provider,
)


def ensure_storage_contract_catalog() -> None:
    """Ensure the storage contract catalog is loaded for schema access."""
    if get_contract_catalog() is not None:
        return
    contracts = load_contract_catalog()
    set_contract_catalog(contracts)
    clear_schema_provider_cache()


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
        ensure_storage_contract_catalog()
        provider = get_schema_provider()
        schemas = {schema.table_key: schema for schema in provider.iter_table_schemas()}
        service = SchemaService(table_provider=MappingSchemaProvider(schemas))
        set_schema_service(service)
        return service


__all__ = ["ensure_schema_service", "ensure_storage_contract_catalog"]
