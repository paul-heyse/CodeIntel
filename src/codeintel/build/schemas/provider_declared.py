"""Declared schema provider.

This provider exposes the current "declared" table schemas (from dataset
contracts) through the core `SchemaProvider` interface. It is the initial
implementation used before Hamilton-native schema inference is enabled.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.config.datasets.declared_schemas import TABLE_SCHEMAS
from codeintel.core.schemas.provider import MappingSchemaProvider

if TYPE_CHECKING:
    from codeintel.core.schemas.provider import SchemaProvider


@lru_cache
def declared_schema_provider() -> SchemaProvider:
    """Return a SchemaProvider backed by declared dataset table schemas.

    Returns
    -------
    SchemaProvider
        Schema provider exposing table schemas from declared dataset contracts.
    """
    return MappingSchemaProvider(TABLE_SCHEMAS)


__all__ = ["declared_schema_provider"]
