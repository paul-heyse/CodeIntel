"""Helpers for configuring SchemaService in tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.schemas import (
    ContractResolutionMode,
    ContractResolutionSettings,
    iter_contracts,
)
from codeintel.core.schemas import MappingSchemaProvider, SchemaService
from codeintel.core.schemas.service import get_schema_service, set_schema_service
from codeintel.storage.constants import META_CATALOG_NAME
from codeintel.storage.contracts.catalog_state import get_contract_catalog
from codeintel.storage.contracts.provider import set_contract_catalog
from codeintel.storage.contracts.schema_provider import (
    clear_schema_provider_cache,
    get_schema_provider,
)
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.metadata.ddl import apply_metadata_ddl
from codeintel.storage.metadata.meta_catalog import attach_meta_database
from codeintel.storage.schema import ensure_schemas_preserve

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


def ensure_storage_contract_catalog() -> None:
    """Ensure the storage contract catalog is loaded for schema access."""
    if get_contract_catalog() is not None:
        return
    settings = ContractResolutionSettings(mode=ContractResolutionMode.FULL)
    contracts = {contract.table_key: contract for contract in iter_contracts(settings=settings)}
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


def ensure_production_schemas(con: DuckDBPyConnection) -> None:
    """Ensure production schemas, tables, and metadata exist in a test database.

    Parameters
    ----------
    con
        DuckDB connection to seed with production schemas.
    """
    ensure_storage_contract_catalog()
    ensure_schemas_preserve(con)
    config = StorageConfig(
        db_path=_resolve_db_path(con),
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
    attach_meta_database(con, config=config)
    apply_metadata_ddl(con, catalog=META_CATALOG_NAME)


def _resolve_db_path(con: DuckDBPyConnection) -> Path:
    rows = con.execute("PRAGMA database_list").fetchall()
    for row in rows:
        if str(row[1]) == "main":
            db_path = row[2]
            if db_path:
                return Path(str(db_path))
            break
    for row in rows:
        db_path = row[2]
        if db_path:
            return Path(str(db_path))
    return Path(":memory:")


__all__ = [
    "ensure_production_schemas",
    "ensure_schema_service",
    "ensure_storage_contract_catalog",
]
