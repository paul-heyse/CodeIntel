"""Declared schema provider.

This provider exposes the current "declared" table schemas (from dataset
contracts) through the core `SchemaProvider` interface. It is the initial
implementation used before Hamilton-native schema inference is enabled.
"""

from __future__ import annotations

from codeintel.core.schemas.declared import declared_schema_provider

__all__ = ["declared_schema_provider"]
