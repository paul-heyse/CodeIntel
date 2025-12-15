"""Row binding registry for schema-generated row bindings.

This module provides the canonical entry point for row binding resolution,
replacing the legacy ``get_row_bindings()`` function with schema-generated
equivalents that include provenance metadata.

Examples
--------
>>> from codeintel.build.schemas.row_registry import get_row_binding
>>> binding = get_row_binding("analytics.function_metrics")
>>> binding.table_key
'analytics.function_metrics'
>>> binding.row_type  # Generated frozen dataclass
<class 'Analytics__function_metrics__Row'>
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.schemas.registry import get_schema_provider
from codeintel.core.schemas.row_models import (
    GeneratedRowBinding,
    row_binding_for_table_schema,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


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
    provider = get_schema_provider()
    table_schema = provider.require_table_schema(table_key)
    return row_binding_for_table_schema(table_schema=table_schema)


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
    provider = get_schema_provider()
    for table_schema in provider.iter_table_schemas():
        yield row_binding_for_table_schema(table_schema=table_schema)


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


__all__ = [
    "clear_row_binding_cache",
    "get_row_binding",
    "iter_row_bindings",
]
