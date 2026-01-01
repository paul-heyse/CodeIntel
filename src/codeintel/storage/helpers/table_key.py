"""Table key utilities for schema-qualified name handling."""

from __future__ import annotations

from codeintel.core.table_key import (
    ParsedTableKey,
    TableKey,
    TableKeyValidationError,
    fully_qualified_table_ref,
    is_valid_table_key,
    parse_table_key,
    split_table_key,
    split_table_key_or_default,
    table_name_from_key,
    try_parse_table_key,
    validate_table_key,
)

__all__ = [
    "ParsedTableKey",
    "TableKey",
    "TableKeyValidationError",
    "fully_qualified_table_ref",
    "is_valid_table_key",
    "parse_table_key",
    "split_table_key",
    "split_table_key_or_default",
    "table_name_from_key",
    "try_parse_table_key",
    "validate_table_key",
]
