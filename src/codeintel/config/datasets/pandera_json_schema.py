"""Pandera to JSON Schema conversion helpers.

This module is intentionally independent of the dataset schema registry to
avoid import cycles during bootstrap. Callers that need registry access
should depend on ``codeintel.build.hamilton.contracts.schemas.validation`` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pandera import Column, DataFrameSchema


def _json_type_for_dtype(dtype: object) -> tuple[str, str | None]:
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
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, column in df_schema.columns.items():
        json_type, fmt = _json_type_for_dtype(column.dtype)
        types: list[str] = [json_type]
        if column.nullable:
            types.append("null")
        field_schema: dict[str, Any] = {"type": types}
        if fmt is not None:
            field_schema["format"] = fmt

        if include_constraints:
            constraints = _extract_column_constraints(column)
            field_schema.update(constraints)

        properties[name] = field_schema
        if not column.nullable:
            required.append(name)

    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    if include_metadata and df_schema.name:
        schema["title"] = df_schema.name

    return schema


__all__ = ["pandera_to_json_schema"]
