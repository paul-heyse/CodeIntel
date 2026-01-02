"""Canonical schema provider registry.

This module provides the single entry point for schema resolution throughout
the codebase. All schema access should go through `get_schema_provider()` to
enable transparent schema inference and future schema sources.

The registry uses a unified schema provider with a three-tier fallback chain:

1. Hamilton-native inference for q__-driven Ibis compute nodes
2. Declared schema overrides for non-inferable DAG outputs
3. Raw declared schemas for source tables

Examples
--------
>>> from codeintel.build.schemas import get_schema_provider
>>> provider = get_schema_provider()
>>> schema = provider.require_table_schema("core.modules")
>>> schema.table_key
'core.modules'
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from codeintel.core.imports.lazy import lazy_getattr
from codeintel.core.schemas import SchemaService

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider


def _schema_service() -> SchemaService:
    get_service = cast(
        "Callable[[], SchemaService]",
        lazy_getattr("codeintel.build.schemas.service", "get_schema_service"),
    )
    return get_service()


def _clear_schema_service_cache() -> None:
    clear_cache = cast(
        "Callable[[], None]",
        lazy_getattr("codeintel.build.schemas.service", "clear_schema_service_cache"),
    )
    clear_cache()


def get_schema_provider() -> SchemaProvider:
    """Return the canonical schema provider for the current build context.

    This is the single entry point for all schema resolution. Returns the
    unified provider with a fallback chain:

    1. Hamilton-native inference (for Ibis compute nodes)
    2. Declared schema overrides for non-inferable DAG outputs
    3. Raw declared schemas (for source tables)

    Returns
    -------
    SchemaProvider
        The active unified schema provider.

    Examples
    --------
    >>> provider = get_schema_provider()
    >>> schema = provider.get_table_schema("core.modules")
    >>> schema is not None
    True
    """
    return _schema_service().table_provider


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
    >>> schema = require_table_schema("core.modules")
    >>> schema.table_key
    'core.modules'
    """
    return _schema_service().require_table_schema(table_key)


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
    return _schema_service().iter_table_schemas()


def clear_schema_provider_cache() -> None:
    """Clear the schema provider cache.

    Clears both the registry cache and the underlying unified provider cache.
    Useful for testing when schema definitions may change between tests.
    """
    _clear_schema_service_cache()


__all__ = [
    "clear_schema_provider_cache",
    "get_schema_provider",
    "iter_table_schemas",
    "require_table_schema",
]
