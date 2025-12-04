"""Schema management utilities for DuckDB and JSON Schema.

This package provides utilities for managing database schemas:

- schema.ddl: DuckDB CREATE TABLE/INDEX DDL generation
- schema.json_schema: JSON Schema generation from TypedDict row models
"""

from __future__ import annotations

from codeintel.storage.schema.ddl import (
    INDEX_DDL,
    SCHEMAS,
    TABLE_DDL,
    TABLE_DDL_IF_NOT_EXISTS,
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
    "INDEX_DDL",
    "SCHEMAS",
    "TABLE_DDL",
    "TABLE_DDL_IF_NOT_EXISTS",
    "apply_all_schemas",
    "assert_schema_alignment",
    "build_validator",
    "create_schemas",
    "ensure_schemas_preserve",
    "generate_export_schemas",
    "json_schema_from_typeddict",
    "validate_row_with_schema",
]
