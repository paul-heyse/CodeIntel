"""Lightweight JSON Schema generation from TypedDict row models."""

from __future__ import annotations

import typing
from collections.abc import Mapping
from typing import get_args, get_origin

import jsonschema

TYPE_MAP: dict[type[object], dict[str, str]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


def _schema_for_type(annotation: object) -> dict[str, object]:
    schema: dict[str, object]
    origin = get_origin(annotation)
    if origin is None:
        if not isinstance(annotation, type):
            schema = {"type": "string"}
        else:
            schema = dict(TYPE_MAP.get(annotation, {"type": "string"}))
    elif origin in {list, tuple}:
        args = get_args(annotation)
        item_schema = _schema_for_type(args[0]) if args else {"type": "string"}
        schema = {"type": "array", "items": item_schema}
    elif origin is dict:
        schema = {"type": "object"}
    elif origin is typing.Literal:
        literals = get_args(annotation)
        schema = {"enum": list(literals)}
    elif origin is typing.Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        schemas = [_schema_for_type(arg) for arg in args]
        schema = schemas[0] if len(schemas) == 1 else {"anyOf": schemas}
    else:
        schema = {"type": "string"}
    return schema


def json_schema_from_typeddict(td_type: type[object]) -> dict[str, object]:
    """
    Generate a Draft 2020-12 JSON Schema from a TypedDict definition.

    Returns
    -------
    dict[str, object]
        JSON Schema mapping derived from the typed annotations.
    """
    annotations = typing.get_type_hints(td_type, include_extras=True)
    properties: dict[str, object] = {}
    required: list[str] = []
    for field, annotation in annotations.items():
        properties[field] = _schema_for_type(annotation)
        if get_origin(annotation) is not typing.Union or type(None) not in get_args(annotation):
            required.append(field)
    schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": sorted(required),
        "additionalProperties": False,
    }
    return schema


def validate_row_with_schema(row: Mapping[str, object], schema: Mapping[str, object]) -> None:
    """Validate a single row mapping against a generated JSON Schema."""
    validator = jsonschema.Draft202012Validator(schema)  # type: ignore[arg-type]
    validator.validate(dict(row))
