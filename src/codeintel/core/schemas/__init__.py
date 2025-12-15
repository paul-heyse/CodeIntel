"""Core schema primitives and interfaces.

This package defines the canonical schema representation used across the
codebase (build, storage, and tooling). Higher-level layers can implement
`SchemaProvider` to supply schemas from different authorities (declared,
Hamilton-inferred, compiled manifests, etc.).
"""

from __future__ import annotations

from codeintel.core.schemas.hashing import canonical_type, schema_hash
from codeintel.core.schemas.primitives import Column, ColumnType, Index, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider, SchemaProvider
from codeintel.core.schemas.serde import table_schema_from_json_obj

__all__ = [
    "Column",
    "ColumnType",
    "Index",
    "MappingSchemaProvider",
    "SchemaProvider",
    "TableSchema",
    "canonical_type",
    "schema_hash",
    "table_schema_from_json_obj",
]
