"""Declared schema provider.

This provider exposes the current "declared" table schemas (from dataset
contracts) through the core `SchemaProvider` interface. It is the initial
implementation used before Hamilton-native schema inference is enabled.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.schemas.declared_registry import iter_declared_schemas
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider

if TYPE_CHECKING:
    from codeintel.core.schemas.provider import SchemaProvider


@lru_cache
def declared_schema_provider() -> SchemaProvider:
    """Return a SchemaProvider backed by declared dataset table schemas.

    Returns
    -------
    SchemaProvider
        Schema provider exposing table schemas from declared dataset contracts.
    """
    return MappingSchemaProvider(_as_mapping())


@lru_cache(maxsize=1)
def _as_mapping() -> dict[str, TableSchema]:
    """Return declared schemas as a mapping.

    Returns
    -------
    dict[str, TableSchema]
        Mapping of table_key to declared TableSchema.
    """
    return {schema.table_key: schema for schema in iter_declared_schemas()}


__all__ = ["declared_schema_provider"]
