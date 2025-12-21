"""JSON serialization helpers for core schema primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast, get_args

from codeintel.core.schemas.primitives import Column, ColumnType, Index, TableSchema


def column_to_json_obj(column: Column) -> dict[str, object]:
    """Serialize a Column into a JSON object.

    Returns
    -------
    dict[str, object]
        JSON-serializable representation of the column.
    """
    payload: dict[str, object] = {
        "name": column.name,
        "type": column.type,
        "nullable": column.nullable,
    }
    if column.description is not None:
        payload["description"] = column.description
    return payload


def index_to_json_obj(index: Index) -> dict[str, object]:
    """Serialize an Index into a JSON object.

    Returns
    -------
    dict[str, object]
        JSON-serializable representation of the index.
    """
    payload: dict[str, object] = {
        "name": index.name,
        "columns": list(index.columns),
        "unique": index.unique,
    }
    return payload


def table_schema_to_json_obj(schema: TableSchema) -> dict[str, object]:
    """Serialize a TableSchema into a JSON object.

    Returns
    -------
    dict[str, object]
        JSON-serializable representation of the table schema.
    """
    payload: dict[str, object] = {
        "schema": schema.schema,
        "name": schema.name,
        "columns": [column_to_json_obj(col) for col in schema.columns],
        "primary_key": list(schema.primary_key),
        "indexes": [index_to_json_obj(idx) for idx in schema.indexes],
    }
    if schema.description is not None:
        payload["description"] = schema.description
    return payload


def _require_str(value: object, *, field: str) -> str:
    if isinstance(value, str) and value:
        return value
    msg = f"Expected non-empty string for {field}"
    raise TypeError(msg)


def _require_bool(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    msg = f"Expected bool for {field}"
    raise TypeError(msg)


def _parse_column_type(value: object, *, field: str) -> ColumnType:
    type_str = _require_str(value, field=field).strip()
    allowed = get_args(ColumnType)
    if type_str not in allowed:
        msg = f"Unsupported column type: {type_str}"
        raise ValueError(msg)
    return cast("ColumnType", type_str)


def column_from_json_obj(obj: Mapping[str, object]) -> Column:
    """Parse a Column from a JSON object.

    Parameters
    ----------
    obj
        JSON object representing a Column.

    Returns
    -------
    Column
        Parsed Column instance.
    """
    name = _require_str(obj.get("name"), field="name")
    col_type = _parse_column_type(obj.get("type"), field="type")
    nullable_obj = obj.get("nullable", True)
    nullable = _require_bool(nullable_obj, field="nullable")
    description_obj = obj.get("description")
    description = description_obj if isinstance(description_obj, str) else None
    return Column(name=name, type=col_type, nullable=nullable, description=description)


def index_from_json_obj(obj: Mapping[str, object]) -> Index:
    """Parse an Index from a JSON object.

    Parameters
    ----------
    obj
        JSON object representing an Index.

    Returns
    -------
    Index
        Parsed Index instance.

    Raises
    ------
    TypeError
        If required fields are missing or of the wrong type.
    """
    name = _require_str(obj.get("name"), field="name")
    columns_obj = obj.get("columns")
    if not isinstance(columns_obj, list) or not columns_obj:
        msg = "Expected non-empty list for columns"
        raise TypeError(msg)
    columns = [_require_str(item, field="columns[]") for item in columns_obj]
    unique_obj = obj.get("unique", False)
    unique = _require_bool(unique_obj, field="unique")
    return Index(name=name, columns=tuple(columns), unique=unique)


def table_schema_from_json_obj(obj: Mapping[str, object]) -> TableSchema:
    """Parse a TableSchema from a JSON object.

    Parameters
    ----------
    obj
        JSON object representing a TableSchema.

    Returns
    -------
    TableSchema
        Parsed TableSchema instance.

    Raises
    ------
    TypeError
        If required fields are missing or of the wrong type.
    """
    schema = _require_str(obj.get("schema"), field="schema")
    name = _require_str(obj.get("name"), field="name")

    columns_obj = obj.get("columns")
    if not isinstance(columns_obj, list) or not columns_obj:
        msg = "Expected non-empty list for columns"
        raise TypeError(msg)
    columns = [column_from_json_obj(item) for item in columns_obj if isinstance(item, Mapping)]
    if len(columns) != len(columns_obj):
        msg = "Invalid column object in columns"
        raise TypeError(msg)

    primary_key_obj = obj.get("primary_key", [])
    if not isinstance(primary_key_obj, list):
        msg = "Expected list for primary_key"
        raise TypeError(msg)
    primary_key = tuple(_require_str(item, field="primary_key[]") for item in primary_key_obj)

    indexes_obj = obj.get("indexes", [])
    if not isinstance(indexes_obj, list):
        msg = "Expected list for indexes"
        raise TypeError(msg)
    indexes = tuple(index_from_json_obj(item) for item in indexes_obj if isinstance(item, Mapping))
    if len(indexes) != len(indexes_obj):
        msg = "Invalid index object in indexes"
        raise TypeError(msg)

    description_obj = obj.get("description")
    description = description_obj if isinstance(description_obj, str) else None

    return TableSchema(
        schema=schema,
        name=name,
        columns=columns,
        primary_key=primary_key,
        indexes=indexes,
        description=description,
    )


__all__ = [
    "column_from_json_obj",
    "column_to_json_obj",
    "index_from_json_obj",
    "index_to_json_obj",
    "table_schema_from_json_obj",
    "table_schema_to_json_obj",
]
