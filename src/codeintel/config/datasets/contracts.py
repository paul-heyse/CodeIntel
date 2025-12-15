"""Dataset schema and row binding utilities.

This module provides access to declared table schemas and schema-generated
row bindings. For dataset contracts, use the build-owned providers:

- Table schemas: `codeintel.build.schemas.get_schema_provider()`
- Row bindings: `codeintel.build.schemas.get_row_binding()`
- Dataset contracts: `codeintel.build.schemas.get_contract_for_table_key()`
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.schemas.declared_schemas import TABLE_SCHEMAS
from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.core.schemas.contract_primitives import DatasetContract, RowBinding
from codeintel.core.schemas.row_models import row_binding_for_table_schema

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema


def get_table_schemas() -> dict[str, TableSchema]:
    """Return declared table schemas.

    Returns
    -------
    dict[str, TableSchema]
        Mapping from table_key to declared TableSchema.
    """
    return TABLE_SCHEMAS


@lru_cache(maxsize=1)
def get_row_bindings() -> dict[str, RowBinding]:
    """Return schema-generated row bindings for all declared table schemas.

    Returns
    -------
    dict[str, RowBinding]
        Mapping from table_key to a schema-generated RowBinding.
    """
    bindings: dict[str, RowBinding] = {}
    for table_key, schema in TABLE_SCHEMAS.items():
        generated = row_binding_for_table_schema(table_schema=schema)
        bindings[table_key] = RowBinding(
            row_type=generated.row_model,
            to_tuple=generated.serializer,
        )
    return bindings


__all__ = [
    "DatasetContract",
    "RowBinding",
    "get_composite_schemas",
    "get_row_bindings",
    "get_table_schemas",
]
