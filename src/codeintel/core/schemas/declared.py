"""Declared schema access helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from types import MappingProxyType

from codeintel.config.datasets.contracts import get_table_schemas
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider, SchemaProvider


@lru_cache(maxsize=1)
def _registry() -> Mapping[str, TableSchema]:
    """Return the declared schema registry.

    Returns
    -------
    Mapping[str, TableSchema]
        Read-only mapping of table_key to TableSchema.
    """
    return MappingProxyType(get_table_schemas())


def get_declared_schema(table_key: str) -> TableSchema | None:
    """Return the declared schema for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema | None
        Declared schema when available, otherwise None.
    """
    return _registry().get(table_key)


def iter_declared_schemas() -> Iterable[TableSchema]:
    """Iterate all declared TableSchema definitions.

    Returns
    -------
    Iterable[TableSchema]
        Declared schema definitions.
    """
    return _registry().values()


@lru_cache(maxsize=1)
def declared_schema_provider() -> SchemaProvider:
    """Return a SchemaProvider backed by declared dataset table schemas.

    Returns
    -------
    SchemaProvider
        Schema provider exposing table schemas from declared dataset contracts.
    """
    return MappingSchemaProvider(dict(_registry()))


__all__ = [
    "declared_schema_provider",
    "get_declared_schema",
    "iter_declared_schemas",
]
