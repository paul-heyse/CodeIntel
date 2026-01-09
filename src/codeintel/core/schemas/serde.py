"""JSON serialization helpers for core schema primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast, get_args

from codeintel.core.schemas.primitives import (
    Column,
    ColumnType,
    Index,
    ReplaceScope,
    TableSchema,
    TableWritePolicy,
    WriteMode,
    normalize_column_type,
)


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
    if schema.write_policy is not None:
        payload["write_policy"] = _write_policy_to_json_obj(schema.write_policy)
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
    try:
        return normalize_column_type(type_str)
    except ValueError as exc:
        msg = f"Unsupported column type: {type_str}"
        raise ValueError(msg) from exc


def _write_policy_to_json_obj(policy: TableWritePolicy) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": policy.mode,
        "replace_scope": policy.replace_scope,
        "conflict_columns": list(policy.conflict_columns or ()),
        "update_columns": list(policy.update_columns or ()),
        "hash_column": policy.hash_column,
        "use_staging": policy.use_staging,
    }
    if policy.stable_sort_keys is not None:
        payload["stable_sort_keys"] = list(policy.stable_sort_keys)
    return payload


def _parse_write_mode(value: object, *, field: str) -> WriteMode:
    mode = _require_str(value, field=field)
    if mode not in cast("tuple[str, ...]", get_args(WriteMode)):
        msg = f"Unsupported write mode: {mode}"
        raise ValueError(msg)
    return cast("WriteMode", mode)


def _parse_replace_scope(value: object, *, field: str) -> ReplaceScope:
    scope = _require_str(value, field=field)
    if scope not in cast("tuple[str, ...]", get_args(ReplaceScope)):
        msg = f"Unsupported replace scope: {scope}"
        raise ValueError(msg)
    return cast("ReplaceScope", scope)


def _parse_write_policy(value: object) -> TableWritePolicy:
    if not isinstance(value, Mapping):
        msg = "Expected object for write_policy"
        raise TypeError(msg)
    conflict_obj = value.get("conflict_columns", [])
    if not isinstance(conflict_obj, list):
        msg = "Expected list for write_policy.conflict_columns"
        raise TypeError(msg)
    update_obj = value.get("update_columns", [])
    if not isinstance(update_obj, list):
        msg = "Expected list for write_policy.update_columns"
        raise TypeError(msg)
    conflict_columns = tuple(
        _require_str(item, field="conflict_columns[]") for item in conflict_obj
    )
    update_columns = tuple(_require_str(item, field="update_columns[]") for item in update_obj)
    hash_value = value.get("hash_column")
    hash_column = hash_value if isinstance(hash_value, str) else None
    sort_obj = value.get("stable_sort_keys")
    if sort_obj is None:
        stable_sort_keys = None
    else:
        if not isinstance(sort_obj, list):
            msg = "Expected list for write_policy.stable_sort_keys"
            raise TypeError(msg)
        stable_sort_keys = tuple(
            _require_str(item, field="stable_sort_keys[]") for item in sort_obj
        )
    return TableWritePolicy(
        mode=_parse_write_mode(value.get("mode"), field="write_policy.mode"),
        replace_scope=_parse_replace_scope(value.get("replace_scope"), field="write_policy.scope"),
        conflict_columns=conflict_columns or None,
        update_columns=update_columns or None,
        hash_column=hash_column,
        use_staging=_require_bool(
            value.get("use_staging", False),
            field="write_policy.use_staging",
        ),
        stable_sort_keys=stable_sort_keys,
    )


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
    write_policy_obj = obj.get("write_policy")
    write_policy = _parse_write_policy(write_policy_obj) if write_policy_obj is not None else None

    return TableSchema(
        schema=schema,
        name=name,
        columns=columns,
        primary_key=primary_key,
        indexes=indexes,
        description=description,
        write_policy=write_policy,
    )


__all__ = [
    "column_from_json_obj",
    "column_to_json_obj",
    "index_from_json_obj",
    "index_to_json_obj",
    "table_schema_from_json_obj",
    "table_schema_to_json_obj",
]
