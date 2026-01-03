"""JSON Schema 2020-12 generation from typed models.

Generate JSON Schema from msgspec Structs (preferred) and dataclasses,
ensuring the schema is always in sync with the model.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Literal, Union, get_args, get_origin, get_type_hints

import msgspec


def generate_schema(model_class: type) -> dict[str, object]:
    """Generate JSON Schema 2020-12 from a dataclass.

    Parameters
    ----------
    model_class
        Dataclass to generate schema from.

    Returns
    -------
    dict[str, object]
        JSON Schema 2020-12 document.
    """
    if issubclass(model_class, msgspec.Struct):
        schema = msgspec.json.schema(model_class)
        return _strip_private_properties(schema)
    schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": getattr(model_class, "SCHEMA_ID", f"urn:codeintel:{model_class.__name__}"),
        "title": getattr(model_class, "SCHEMA_TITLE", model_class.__name__),
        "description": _extract_description(model_class),
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    hints = get_type_hints(model_class)
    props: dict[str, object] = {}

    for f in fields(model_class):
        if f.name.startswith("_"):
            continue

        field_type = hints.get(f.name, f.type)
        prop = _type_to_schema(field_type, f)
        props[f.name] = prop

    schema["properties"] = props
    return schema


def _strip_private_properties(schema: dict[str, object]) -> dict[str, object]:
    properties = schema.get("properties")
    if isinstance(properties, dict):
        private_keys = [key for key in properties if key.startswith("_")]
        for key in private_keys:
            properties.pop(key, None)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [key for key in required if key not in private_keys]
    defs = schema.get("$defs")
    if isinstance(defs, dict):
        for value in defs.values():
            if isinstance(value, dict):
                _strip_private_properties(value)
    return schema


def _extract_description(cls: type) -> str:
    """Extract first line of docstring as description.

    Parameters
    ----------
    cls
        Class to extract docstring from.

    Returns
    -------
    str
        First line of docstring or empty string.
    """
    if cls.__doc__:
        return cls.__doc__.split("\n")[0].strip()
    return ""


def _type_to_schema(type_hint: type, field_info: object) -> dict[str, object]:
    """Convert a Python type hint to JSON Schema property.

    Parameters
    ----------
    type_hint
        Python type annotation.
    field_info
        Dataclass field info (may be None).

    Returns
    -------
    dict[str, object]
        JSON Schema property definition.
    """
    result = _try_union_schema(type_hint, field_info)
    if result is not None:
        return result

    result = _try_literal_schema(type_hint, field_info)
    if result is not None:
        return result

    result = _try_dataclass_schema(type_hint)
    if result is not None:
        return result

    result = _try_collection_schema(type_hint)
    if result is not None:
        return result

    result = _try_primitive_schema(type_hint, field_info)
    if result is not None:
        return result

    return {"type": "string"}


def _try_union_schema(type_hint: type, field_info: object) -> dict[str, object] | None:
    """Try to create schema for Union types.

    Parameters
    ----------
    type_hint
        Python type annotation.
    field_info
        Dataclass field info.

    Returns
    -------
    dict[str, object] | None
        Schema if type is Union, None otherwise.
    """
    origin = get_origin(type_hint)
    if origin is not Union:
        return None

    args = get_args(type_hint)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1:
        return _type_to_schema(non_none[0], field_info)
    return {"oneOf": [_type_to_schema(a, field_info) for a in non_none]}


def _try_literal_schema(type_hint: type, field_info: object) -> dict[str, object] | None:
    """Try to create schema for Literal types.

    Parameters
    ----------
    type_hint
        Python type annotation.
    field_info
        Dataclass field info.

    Returns
    -------
    dict[str, object] | None
        Schema if type is Literal, None otherwise.
    """
    origin = get_origin(type_hint)
    if origin is not Literal:
        return None

    args = get_args(type_hint)
    return {
        "type": "string",
        "enum": list(args),
        "default": _get_default(field_info),
    }


def _try_dataclass_schema(type_hint: type) -> dict[str, object] | None:
    """Try to create schema for dataclass types.

    Parameters
    ----------
    type_hint
        Python type annotation.

    Returns
    -------
    dict[str, object] | None
        Schema if type is dataclass, None otherwise.
    """
    if is_dataclass(type_hint):
        return _dataclass_to_schema(type_hint)
    return None


def _try_collection_schema(type_hint: type) -> dict[str, object] | None:
    """Try to create schema for collection types.

    Parameters
    ----------
    type_hint
        Python type annotation.

    Returns
    -------
    dict[str, object] | None
        Schema if type is list/tuple, None otherwise.
    """
    origin = get_origin(type_hint)
    if origin not in {list, tuple}:
        return None

    args = get_args(type_hint)
    item_type = args[0] if args else str
    return {
        "type": "array",
        "items": _type_to_schema(item_type, None),
    }


def _try_primitive_schema(type_hint: type, field_info: object) -> dict[str, object] | None:
    """Try to create schema for primitive types.

    Parameters
    ----------
    type_hint
        Python type annotation.
    field_info
        Dataclass field info.

    Returns
    -------
    dict[str, object] | None
        Schema if type is primitive, None otherwise.
    """
    if type_hint is Path:
        return {"type": "string", "format": "path"}

    type_map: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    if type_hint not in type_map:
        return None

    prop: dict[str, object] = {"type": type_map[type_hint]}
    default = _get_default(field_info)
    if default is not None:
        prop["default"] = default
    return prop


def _dataclass_to_schema(cls: type) -> dict[str, object]:
    """Convert a dataclass to JSON Schema object definition.

    Parameters
    ----------
    cls
        Dataclass type.

    Returns
    -------
    dict[str, object]
        JSON Schema object definition.
    """
    schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    if cls.__doc__:
        schema["description"] = _extract_description(cls)

    hints = get_type_hints(cls)
    props: dict[str, object] = {}

    for f in fields(cls):
        if f.name.startswith("_"):
            continue
        field_type = hints.get(f.name, f.type)
        prop = _type_to_schema(field_type, f)
        props[f.name] = prop

    schema["properties"] = props
    return schema


def _get_default(field_info: object) -> str | int | float | bool | None:
    """Extract default value from dataclass field.

    Parameters
    ----------
    field_info
        Dataclass field info.

    Returns
    -------
    str | int | float | bool | None
        Default value or None.
    """
    if field_info is None:
        return None

    default = getattr(field_info, "default", MISSING)
    if default is not MISSING and isinstance(default, (str, int, float, bool)):
        return default
    return None


def export_schema(model_class: type, path: Path) -> None:
    """Export JSON Schema to file.

    Parameters
    ----------
    model_class
        Dataclass to generate schema from.
    path
        Output file path.
    """
    schema = generate_schema(model_class)
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


__all__ = [
    "export_schema",
    "generate_schema",
]
