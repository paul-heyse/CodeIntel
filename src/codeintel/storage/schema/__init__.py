"""Schema management utilities for DuckDB and JSON Schema.

This package provides utilities for managing database schemas:

- schema.ddl: DuckDB DDL management via DuckDBPolicyBackend
- schema.json_schema: JSON Schema generation from TypedDict row models

All DDL is now generated from dataset contracts via the policy backend.
"""

from __future__ import annotations

from codeintel.storage.schema.ddl import (
    SCHEMAS,
    apply_all_schemas,
    assert_schema_alignment,
    create_schemas,
    ensure_schemas_preserve,
)
from codeintel.storage.schema.json_schema import (
    build_validator,
    generate_export_schemas,
    json_schema_from_typeddict,
    validate_row_with_schema,
)

__all__ = [
    "SCHEMAS",
    "apply_all_schemas",
    "assert_schema_alignment",
    "build_validator",
    "create_schemas",
    "ensure_schemas_preserve",
    "generate_export_schemas",
    "json_schema_from_typeddict",
    "validate_row_with_schema",
]
