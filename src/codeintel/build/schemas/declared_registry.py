"""Declared schema registry for build-time lookups.

This module centralizes access to statically declared TableSchema definitions
without exposing the raw TABLE_SCHEMAS mapping. It provides a small API for
schema lookups that is safe to import from target spec helpers and other
build modules without introducing target-system cycles.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from types import MappingProxyType

from codeintel.config.datasets.declared_schemas import TABLE_SCHEMAS
from codeintel.core.schemas.primitives import TableSchema


@lru_cache(maxsize=1)
def _registry() -> Mapping[str, TableSchema]:
    """Return the declared schema registry.

    Returns
    -------
    Mapping[str, TableSchema]
        Read-only mapping of table_key to TableSchema.
    """
    return MappingProxyType(TABLE_SCHEMAS)


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


__all__ = [
    "get_declared_schema",
    "iter_declared_schemas",
]
