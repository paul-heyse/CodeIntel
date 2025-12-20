"""Canonical schema provider registry.

This module provides the single entry point for schema resolution throughout
the codebase. All schema access should go through `get_schema_provider()` to
enable transparent schema inference and future schema sources.

The registry uses a unified schema provider with a three-tier fallback chain:

1. Hamilton-native inference for q__-driven Ibis compute nodes
2. Target-declared schemas from OutputContract.tables
3. Raw declared schemas for source tables

Examples
--------
>>> from codeintel.build.schemas import get_schema_provider
>>> provider = get_schema_provider()
>>> schema = provider.require_table_schema("analytics.function_metrics")
>>> schema.table_key
'analytics.function_metrics'
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.schemas.service import clear_schema_service_cache, get_schema_service

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider


def get_schema_provider() -> SchemaProvider:
    """Return the canonical schema provider for the current build context.

    This is the single entry point for all schema resolution. Returns the
    unified provider with a fallback chain:

    1. Hamilton-native inference (for Ibis compute nodes)
    2. Target-declared schemas (from OutputContract.tables)
    3. Raw declared schemas (for source tables)

    Returns
    -------
    SchemaProvider
        The active unified schema provider.

    Examples
    --------
    >>> provider = get_schema_provider()
    >>> schema = provider.get_table_schema("analytics.function_metrics")
    >>> schema is not None
    True
    """
    return get_schema_service().table_provider


def require_table_schema(table_key: str) -> TableSchema:
    """Require a table schema by key, raising if not found.

    Convenience function delegating to the canonical schema provider.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema
        The table schema for the given key.

    Examples
    --------
    >>> schema = require_table_schema("analytics.function_metrics")
    >>> schema.table_key
    'analytics.function_metrics'
    """
    return get_schema_service().require_table_schema(table_key)


def iter_table_schemas() -> Iterable[TableSchema]:
    """Iterate all known table schemas.

    Convenience function delegating to the canonical schema provider.

    Returns
    -------
    Iterable[TableSchema]
        All table schemas known to the provider.

    Examples
    --------
    >>> schemas = list(iter_table_schemas())
    >>> len(schemas) > 0
    True
    """
    return get_schema_service().iter_table_schemas()


def clear_schema_provider_cache() -> None:
    """Clear the schema provider cache.

    Clears both the registry cache and the underlying unified provider cache.
    Useful for testing when schema definitions may change between tests.
    """
    clear_schema_service_cache()


__all__ = [
    "clear_schema_provider_cache",
    "get_schema_provider",
    "iter_table_schemas",
    "require_table_schema",
]
