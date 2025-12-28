"""JSON Schema generation from schema definitions.

This module provides utilities for converting schema definitions to JSON Schema
(draft 2020-12) format:

- ``json_schema_from_table_schema``: Convert TableSchema primitives to JSON Schema
- ``pandera_to_json_schema``: Convert Pandera DataFrameSchema to JSON Schema

This module is intentionally independent of build-time registries to avoid
import cycles during bootstrap. Callers that need registry access should
use the configured ``SchemaService`` or storage validation helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pandera.pandas import Column, DataFrameSchema

    from codeintel.core.schemas.primitives import ColumnType, TableSchema


def _json_type_for_dtype(dtype: object) -> tuple[str, str | None]:
    """Map a Pandera/pandas dtype to JSON Schema type and format.

    Parameters
    ----------
    dtype
        Pandera column dtype to map.

    Returns
    -------
    tuple[str, str | None]
        JSON Schema type string and optional format string.
    """
    dtype_str = str(dtype).lower()
    if "bool" in dtype_str:
        return "boolean", None
    if "int" in dtype_str:
        return "integer", None
    if "float" in dtype_str or "double" in dtype_str:
        return "number", None
    if "datetime" in dtype_str:
        return "string", "date-time"
    return "string", None


def _extract_column_constraints(column: Column) -> dict[str, Any]:
    """Extract simple numeric constraints from Pandera checks.

    Parameters
    ----------
    column
        Pandera Column to extract constraints from.

    Returns
    -------
    dict[str, Any]
        Dictionary of JSON Schema constraints (minimum, maximum, etc.).
    """
    constraints: dict[str, Any] = {}
    checks = column.checks
    if checks is None:
        return constraints

    for check in checks:
        check_str = str(check)
        if ">= 0" in check_str or "(s >= 0)" in check_str:
            constraints["minimum"] = 0
        elif ">= 1" in check_str or "(s >= 1)" in check_str:
            constraints["minimum"] = 1
        elif "<= 1" in check_str and ">= 0" in check_str:
            constraints["minimum"] = 0
            constraints["maximum"] = 1

    return constraints


def _column_metadata_type(column: Column) -> str | None:
    metadata = getattr(column, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get("codeintel_column_type")
    return None


def _json_value_types(*, nullable: bool) -> list[str]:
    types = ["object", "array", "string", "number", "boolean"]
    if nullable:
        types.append("null")
    return types


def _build_field_schema(column: Column, *, include_constraints: bool) -> dict[str, Any]:
    if _column_metadata_type(column) == "JSON":
        return {"type": _json_value_types(nullable=column.nullable)}

    json_type, fmt = _json_type_for_dtype(column.dtype)
    types = [json_type]
    if column.nullable:
        types.append("null")
    field_schema: dict[str, Any] = {"type": types}
    if fmt is not None:
        field_schema["format"] = fmt
    if include_constraints:
        field_schema.update(_extract_column_constraints(column))
    return field_schema


def _build_json_schema_properties(
    df_schema: DataFrameSchema,
    *,
    include_constraints: bool,
) -> tuple[dict[str, Any], list[str]]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, column in df_schema.columns.items():
        properties[name] = _build_field_schema(
            column,
            include_constraints=include_constraints,
        )
        if not column.nullable:
            required.append(name)
    return properties, required


def _apply_schema_metadata(
    schema: dict[str, Any],
    *,
    include_metadata: bool,
    schema_name: str | None,
) -> None:
    if include_metadata and schema_name:
        schema["title"] = schema_name


def pandera_to_json_schema(
    df_schema: DataFrameSchema,
    *,
    include_constraints: bool = True,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """Convert a Pandera DataFrameSchema to a JSON Schema draft 2020-12 mapping.

    Parameters
    ----------
    df_schema
        Pandera schema to convert.
    include_constraints
        When True, include simple check-derived constraints (e.g., minimum).
    include_metadata
        When True, propagate the schema name into the JSON Schema title.

    Returns
    -------
    dict[str, Any]
        JSON Schema dictionary compatible with draft 2020-12.

    Examples
    --------
    >>> from pandera import DataFrameSchema, Column
    >>> schema = DataFrameSchema({"id": Column(int), "name": Column(str, nullable=True)})
    >>> json_schema = pandera_to_json_schema(schema)
    >>> json_schema["$schema"]
    'https://json-schema.org/draft/2020-12/schema'
    """
    properties, required = _build_json_schema_properties(
        df_schema,
        include_constraints=include_constraints,
    )
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    _apply_schema_metadata(
        schema,
        include_metadata=include_metadata,
        schema_name=df_schema.name,
    )

    return schema


# ---------------------------------------------------------------------------
# TableSchema -> JSON Schema conversion
# ---------------------------------------------------------------------------


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
        "DECIMAL(38,0)": {"type": "integer"},
        "DOUBLE": {"type": "number"},
        "DECIMAL": {"type": "number"},
        "VARCHAR": {"type": "string"},
        "JSON": {},  # Any valid JSON value
        "TIMESTAMP": {"type": "string", "format": "date-time"},
        "TIMESTAMPTZ": {"type": "string", "format": "date-time"},
    }
    return mapping.get(col_type, {"type": "string"})


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

        # Handle nullable columns using array type syntax
        if col.nullable:
            if "type" in base_schema:
                field_schema["type"] = [base_schema["type"], "null"]
            else:
                # JSON type has no specific type (accepts any JSON value)
                field_schema = base_schema.copy()
        else:
            field_schema = base_schema.copy()
            required.append(col.name)

        # Add format if present
        if "format" in base_schema:
            field_schema["format"] = base_schema["format"]

        # Add description if enabled and present
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


__all__ = ["json_schema_from_table_schema", "pandera_to_json_schema"]
