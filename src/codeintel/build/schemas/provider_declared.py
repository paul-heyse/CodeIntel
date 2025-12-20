"""Declared schema provider.

This provider exposes source-only declared schemas for build-time use while
keeping the full declared registry available for storage bootstrap and legacy
call sites.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.core.schemas.declared import (
    declared_schema_provider as core_declared_schema_provider,
)
from codeintel.core.schemas.declared import (
    source_declared_schema_provider,
)

if TYPE_CHECKING:
    from codeintel.core.schemas.provider import SchemaProvider


@lru_cache(maxsize=1)
def declared_schema_provider() -> SchemaProvider:
    """Return a source-only declared schema provider for build usage.

    Returns
    -------
    SchemaProvider
        Provider exposing only source table schemas (excluding DAG outputs).
    """
    service = get_target_metadata_service()
    return source_declared_schema_provider(exclude_table_keys=service.system.all_table_keys)


def full_declared_schema_provider() -> SchemaProvider:
    """Return the full declared schema provider (including outputs).

    Returns
    -------
    SchemaProvider
        Provider exposing the full declared schema registry.
    """
    return core_declared_schema_provider()


__all__ = [
    "declared_schema_provider",
    "full_declared_schema_provider",
]
