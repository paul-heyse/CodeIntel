"""Declared schema provider.

This provider exposes source-only declared schemas for build-time use.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.target_metadata import OutputInventory, get_target_metadata_service
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
    inventory = get_target_metadata_service().outputs
    return declared_schema_provider_for_inventory(inventory)


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


__all__ = [
    "declared_schema_provider",
    "declared_schema_provider_for_inventory",
]
