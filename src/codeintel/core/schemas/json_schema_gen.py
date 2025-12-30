"""JSON Schema generation from TableSchema definitions.

This module provides utilities for converting schema definitions to JSON Schema
(draft 2020-12) format:

- ``json_schema_from_table_schema``: Convert TableSchema primitives to JSON Schema

This module is intentionally independent of build-time registries to avoid
import cycles during bootstrap. Callers that need registry access should
use the configured ``SchemaService`` or storage validation helpers.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from codeintel.core.schemas.primitives import column_type_base

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import ColumnType, TableSchema


def _json_schema_type_for_column_type(col_type: ColumnType) -> dict[str, Any]:
    """Map a ColumnType literal to JSON Schema type definition.

    Parameters
    ----------
    col_type
        DuckDB column type literal.

    Returns
    -------
    dict[str, Any]
        JSON Schema type definition with type and optional format.
    """
    mapping: dict[str, dict[str, Any]] = {
        "BOOLEAN": {"type": "boolean"},
        "INTEGER": {"type": "integer"},
        "BIGINT": {"type": "integer"},
        "DOUBLE": {"type": "number"},
        "DECIMAL": {"type": "number"},
        "VARCHAR": {"type": "string"},
        "JSON": {},  # Any valid JSON value
        "TIMESTAMP": {"type": "string", "format": "date-time"},
        "TIMESTAMPTZ": {"type": "string", "format": "date-time"},
        "STRUCT": {"type": "object"},
        "MAP": {"type": "object"},
        "LIST": {"type": "array"},
        "UNION": {},
    }
    normalized = str(col_type).strip()
    base = column_type_base(normalized)
    if base == "DECIMAL":
        compact = normalized.upper().replace(" ", "")
        match = re.match(r"^DECIMAL\((\d+),(\d+)\)$", compact)
        if match is not None and int(match.group(2)) == 0:
            return {"type": "integer"}
    return mapping.get(base, {"type": "string"})


def json_schema_from_table_schema(
    table_schema: TableSchema,
    *,
    schema_id: str | None = None,
    include_description: bool = True,
) -> dict[str, Any]:
    """Generate JSON Schema 2020-12 from a TableSchema.

    Parameters
    ----------
    table_schema
        Source TableSchema to convert.
    schema_id
        Optional ``$id`` URI for the generated schema.
    include_description
        When True, include column and table descriptions in the schema.

    Returns
    -------
    dict[str, Any]
        JSON Schema dictionary compatible with draft 2020-12.

    Examples
    --------
    >>> from codeintel.core.schemas.primitives import Column, TableSchema
    >>> ts = TableSchema(
    ...     schema="analytics",
    ...     name="example",
    ...     columns=[
    ...         Column(name="id", type="INTEGER", nullable=False),
    ...         Column(name="name", type="VARCHAR", nullable=True),
    ...     ],
    ... )
    >>> js = json_schema_from_table_schema(ts)
    >>> js["$schema"]
    'https://json-schema.org/draft/2020-12/schema'
    >>> js["properties"]["id"]["type"]
    'integer'
    >>> js["properties"]["name"]["type"]
    ['string', 'null']
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for col in table_schema.columns:
        base_schema = _json_schema_type_for_column_type(col.type)
        field_schema: dict[str, Any] = {}

        if col.nullable:
            if "type" in base_schema:
                field_schema["type"] = [base_schema["type"], "null"]
            else:
                field_schema = base_schema.copy()
        else:
            field_schema = base_schema.copy()
            required.append(col.name)

        if "format" in base_schema:
            field_schema["format"] = base_schema["format"]

        if include_description and col.description:
            field_schema["description"] = col.description

        properties[col.name] = field_schema

    result: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }

    if schema_id:
        result["$id"] = schema_id

    result["title"] = table_schema.table_key

    if include_description and table_schema.description:
        result["description"] = table_schema.description

    if required:
        result["required"] = required

    return result


__all__ = ["json_schema_from_table_schema"]
