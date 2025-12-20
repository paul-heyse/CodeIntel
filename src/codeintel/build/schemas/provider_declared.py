"""Declared schema provider.

This provider exposes source-only declared schemas for build-time use while
keeping the full declared registry available for storage bootstrap and legacy
call sites.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.target_inventory import get_output_inventory
from codeintel.build.target_metadata import OutputInventory
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
    return declared_schema_provider_for_inventory(get_output_inventory())


def declared_schema_provider_for_inventory(inventory: OutputInventory) -> SchemaProvider:
    """Return a source-only declared schema provider for a given inventory.

    Parameters
    ----------
    inventory
        Output inventory used to exclude DAG-produced table keys.

    Returns
    -------
    SchemaProvider
        Provider exposing only source table schemas.
    """
    return source_declared_schema_provider(exclude_table_keys=inventory.all_dataset_keys)


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
    "declared_schema_provider_for_inventory",
    "full_declared_schema_provider",
]
