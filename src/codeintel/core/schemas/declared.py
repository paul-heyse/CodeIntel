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


def source_declared_schema_provider(*, exclude_table_keys: Iterable[str]) -> SchemaProvider:
    """Return a SchemaProvider that filters out excluded table keys.

    Parameters
    ----------
    exclude_table_keys
        Table keys to exclude from the provider (e.g., DAG-produced outputs).

    Returns
    -------
    SchemaProvider
        Schema provider exposing only source table schemas.
    """
    return _source_declared_schema_provider(frozenset(exclude_table_keys))


@lru_cache(maxsize=4)
def _source_declared_schema_provider(exclude_table_keys: frozenset[str]) -> SchemaProvider:
    filtered = {
        table_key: schema
        for table_key, schema in _registry().items()
        if table_key not in exclude_table_keys
    }
    return MappingSchemaProvider(filtered)


__all__ = [
    "declared_schema_provider",
    "get_declared_schema",
    "iter_declared_schemas",
    "source_declared_schema_provider",
]
