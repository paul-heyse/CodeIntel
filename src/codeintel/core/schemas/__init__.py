"""Core schema primitives and interfaces.

This package defines the canonical schema representation used across the
codebase (build, storage, and tooling). Higher-level layers can implement
`SchemaProvider` to supply schemas from different authorities (declared,
Hamilton-inferred, compiled manifests, etc.).
"""

from __future__ import annotations

from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema
from codeintel.core.schemas.authority import (
    SchemaAuthority,
    SchemaDerivation,
    SchemaSelection,
)
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.hashing import canonical_type, schema_hash
from codeintel.core.schemas.json_schema_gen import json_schema_from_table_schema
from codeintel.core.schemas.pandera_gen import pandera_schema_from_table_schema
from codeintel.core.schemas.primitives import Column, ColumnType, Index, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider, SchemaProvider
from codeintel.core.schemas.row_models import (
    GeneratedRowBinding,
    row_binding_for_table_schema,
)
from codeintel.core.schemas.serde import table_schema_from_json_obj
from codeintel.core.schemas.service import (
    DatasetSchemaLike,
    DatasetSchemaProvider,
    SchemaRecord,
    SchemaService,
    clear_schema_service,
    get_schema_service,
    set_schema_service,
)

__all__ = [
    "Column",
    "ColumnType",
    "DatasetContract",
    "DatasetSchemaLike",
    "DatasetSchemaProvider",
    "GeneratedRowBinding",
    "Index",
    "MappingSchemaProvider",
    "SchemaAuthority",
    "SchemaDerivation",
    "SchemaProvider",
    "SchemaRecord",
    "SchemaSelection",
    "SchemaService",
    "TableSchema",
    "arrow_schema_from_table_schema",
    "canonical_type",
    "clear_schema_service",
    "get_schema_service",
    "json_schema_from_table_schema",
    "pandera_schema_from_table_schema",
    "row_binding_for_table_schema",
    "schema_hash",
    "set_schema_service",
    "table_schema_from_json_obj",
]
