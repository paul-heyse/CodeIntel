"""Core schema primitives and interfaces.

This package defines the canonical schema representation used across the
codebase (build, storage, and tooling). Higher-level layers can implement
`SchemaProvider` to supply schemas from different authorities (declared,
Hamilton-inferred, compiled manifests, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.core.imports.lazy import make_lazy_getattr
from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema
from codeintel.core.schemas.arrow_polars import (
    table_schema_from_arrow_schema,
    table_schema_from_polars_dataframe,
    table_schema_from_polars_lazyframe,
    table_schema_from_polars_schema,
)
from codeintel.core.schemas.authority import (
    SchemaAuthority,
    SchemaDerivation,
    SchemaSelection,
)
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.hashing import canonical_type, schema_hash
from codeintel.core.schemas.json_schema_gen import json_schema_from_table_schema
from codeintel.core.schemas.primitives import Column, ColumnType, Index, TableSchema
from codeintel.core.schemas.provider import (
    FallbackSchemaProvider,
    MappingSchemaProvider,
    SchemaProvider,
)
from codeintel.core.schemas.serde import table_schema_from_json_obj

if TYPE_CHECKING:
    from codeintel.core.schemas.row_models import (
        GeneratedRowBinding,
        columns_for_table_key,
        row_binding_for_table_key,
        row_binding_for_table_schema,
        row_model_for_table_key,
        row_struct_builder_for_table_schema,
        row_struct_for_table_key,
        row_struct_for_table_schema,
        row_struct_serializer_for_table_schema,
    )
    from codeintel.core.schemas.service import (
        ArrowSchemaProvider,
        ContractBundle,
        DatasetSchemaLike,
        DatasetSchemaProvider,
        SchemaRecord,
        SchemaService,
        clear_schema_service,
        get_schema_service,
        set_schema_service,
    )

__all__ = [
    "ArrowSchemaProvider",
    "Column",
    "ColumnType",
    "ContractBundle",
    "DatasetContract",
    "DatasetSchemaLike",
    "DatasetSchemaProvider",
    "FallbackSchemaProvider",
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
    "columns_for_table_key",
    "get_schema_service",
    "json_schema_from_table_schema",
    "row_binding_for_table_key",
    "row_binding_for_table_schema",
    "row_model_for_table_key",
    "row_struct_builder_for_table_schema",
    "row_struct_for_table_key",
    "row_struct_for_table_schema",
    "row_struct_serializer_for_table_schema",
    "schema_hash",
    "set_schema_service",
    "table_schema_from_arrow_schema",
    "table_schema_from_json_obj",
    "table_schema_from_polars_dataframe",
    "table_schema_from_polars_lazyframe",
    "table_schema_from_polars_schema",
]

_LAZY_ATTRS = {
    "ArrowSchemaProvider": ("codeintel.core.schemas.service", "ArrowSchemaProvider"),
    "ContractBundle": ("codeintel.core.schemas.service", "ContractBundle"),
    "DatasetSchemaLike": ("codeintel.core.schemas.service", "DatasetSchemaLike"),
    "DatasetSchemaProvider": ("codeintel.core.schemas.service", "DatasetSchemaProvider"),
    "SchemaRecord": ("codeintel.core.schemas.service", "SchemaRecord"),
    "SchemaService": ("codeintel.core.schemas.service", "SchemaService"),
    "clear_schema_service": ("codeintel.core.schemas.service", "clear_schema_service"),
    "get_schema_service": ("codeintel.core.schemas.service", "get_schema_service"),
    "set_schema_service": ("codeintel.core.schemas.service", "set_schema_service"),
    "GeneratedRowBinding": ("codeintel.core.schemas.row_models", "GeneratedRowBinding"),
    "columns_for_table_key": ("codeintel.core.schemas.row_models", "columns_for_table_key"),
    "row_binding_for_table_key": (
        "codeintel.core.schemas.row_models",
        "row_binding_for_table_key",
    ),
    "row_binding_for_table_schema": (
        "codeintel.core.schemas.row_models",
        "row_binding_for_table_schema",
    ),
    "row_model_for_table_key": ("codeintel.core.schemas.row_models", "row_model_for_table_key"),
    "row_struct_builder_for_table_schema": (
        "codeintel.core.schemas.row_models",
        "row_struct_builder_for_table_schema",
    ),
    "row_struct_for_table_key": ("codeintel.core.schemas.row_models", "row_struct_for_table_key"),
    "row_struct_for_table_schema": (
        "codeintel.core.schemas.row_models",
        "row_struct_for_table_schema",
    ),
    "row_struct_serializer_for_table_schema": (
        "codeintel.core.schemas.row_models",
        "row_struct_serializer_for_table_schema",
    ),
}

__getattr__ = make_lazy_getattr(_LAZY_ATTRS, __name__, cache_in_globals=globals())
