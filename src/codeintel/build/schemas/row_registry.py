"""Row binding registry for schema-generated row bindings.

This module provides the canonical entry point for row binding resolution
with schema-generated equivalents that include provenance metadata.

Examples
--------
>>> from codeintel.build.schemas.row_registry import get_row_binding
>>> binding = get_row_binding("analytics.function_metrics")
>>> binding.table_key
'analytics.function_metrics'
>>> binding.row_model  # Generated frozen dataclass
<class 'Analytics__function_metrics__Row'>
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.schemas.service import get_schema_service

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.schemas.row_models import GeneratedRowBinding


@lru_cache(maxsize=256)
def get_row_binding(table_key: str) -> GeneratedRowBinding:
    """Return a schema-generated row binding for a table key.

    This function provides the canonical way to obtain row bindings. It
    resolves the table schema via the schema provider and generates a
    complete binding with row model, serializer, and provenance metadata.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    GeneratedRowBinding
        Generated binding with row model, serializer, table_key, and
        schema_hash for cache invalidation.

    Examples
    --------
    >>> binding = get_row_binding("core.modules")
    >>> binding.table_key
    'core.modules'
    >>> len(binding.schema_hash)
    64

    Notes
    -----
    Raises ``KeyError`` (via ``require_table_schema``) if no schema is
    found for the table key.
    """
    service = get_schema_service()
    return service.require_row_binding(table_key)


@lru_cache(maxsize=256)
def column_names_for_table_key(table_key: str) -> tuple[str, ...]:
    """Return column names for a table key from the schema provider.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    tuple[str, ...]
        Column names in schema order, or empty tuple if the table is unknown.
    """
    service = get_schema_service()
    schema = service.get_table_schema(table_key)
    if schema is None:
        return ()
    return tuple(schema.column_names())


def iter_row_bindings() -> Iterable[GeneratedRowBinding]:
    """Iterate all available row bindings.

    Yields bindings for all table schemas known to the schema provider.
    Each binding includes the row model, serializer, and provenance metadata.

    Yields
    ------
    GeneratedRowBinding
        Generated binding for each known table schema.

    Examples
    --------
    >>> bindings = list(iter_row_bindings())
    >>> len(bindings) > 0
    True
    >>> all(b.table_key for b in bindings)
    True
    """
    service = get_schema_service()
    for table_schema in service.iter_table_schemas():
        binding = service.get_row_binding(table_schema.table_key)
        if binding is not None:
            yield binding


def clear_row_binding_cache() -> None:
    """Clear the row binding cache.

    Clears the LRU cache used by ``get_row_binding()``. Useful in testing
    scenarios where schema definitions may change between tests.

    Examples
    --------
    >>> binding1 = get_row_binding("core.modules")
    >>> clear_row_binding_cache()
    >>> binding2 = get_row_binding("core.modules")
    >>> # bindings are equivalent but regenerated
    """
    get_row_binding.cache_clear()
    column_names_for_table_key.cache_clear()
    try:
        get_schema_service().clear_caches()
    except RuntimeError:
        return


__all__ = [
    "clear_row_binding_cache",
    "column_names_for_table_key",
    "get_row_binding",
    "iter_row_bindings",
]
