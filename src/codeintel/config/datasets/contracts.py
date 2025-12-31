"""Dataset schema and row binding utilities.

This module provides access to canonical table schemas and schema-generated
row bindings. For dataset contracts, use the build-owned providers:

- Table schemas: `codeintel.build.schemas.get_schema_provider()`
- Row bindings: `codeintel.build.schemas.get_row_binding()`
- Dataset contracts: `codeintel.build.schemas.get_contract_for_table_key()`
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.row_models import GeneratedRowBinding, row_binding_for_table_schema
from codeintel.core.schemas.service import get_schema_service
from codeintel.core.schemas.table_registry import TABLE_SCHEMAS

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema


def get_table_schemas() -> dict[str, TableSchema]:
    """Return canonical table schemas.

    Returns
    -------
    dict[str, TableSchema]
        Mapping from table_key to canonical TableSchema.
    """
    return TABLE_SCHEMAS


@lru_cache(maxsize=1)
def get_row_bindings() -> dict[str, GeneratedRowBinding]:
    """Return schema-generated row bindings for available table schemas.

    Returns
    -------
    dict[str, GeneratedRowBinding]
        Mapping from table_key to a schema-generated row binding. When a
        SchemaService is configured, DAG-first schemas are used.
    """
    bindings: dict[str, GeneratedRowBinding] = {}
    try:
        service = get_schema_service()
    except RuntimeError:
        service = None
    if service is None:
        for schema in TABLE_SCHEMAS.values():
            bindings[schema.table_key] = row_binding_for_table_schema(schema)
        return bindings
    for schema in service.iter_table_schemas():
        binding = service.get_row_binding(schema.table_key)
        if binding is not None:
            bindings[schema.table_key] = binding
    return bindings


__all__ = [
    "DatasetContract",
    "get_composite_schemas",
    "get_row_bindings",
    "get_table_schemas",
]
