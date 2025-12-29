"""Compatibility wrapper for JSON serialization helpers."""

from __future__ import annotations

from codeintel.core.schemas.contracts import (
    column_from_json_obj,
    column_to_json_obj,
    index_from_json_obj,
    index_to_json_obj,
    table_schema_from_json_obj,
    table_schema_to_json_obj,
)

__all__ = [
    "column_from_json_obj",
    "column_to_json_obj",
    "index_from_json_obj",
    "index_to_json_obj",
    "table_schema_from_json_obj",
    "table_schema_to_json_obj",
]
