"""Lightweight JSON Schema generation from TypedDict row models."""

from __future__ import annotations

import json
import types
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import get_args, get_origin

import jsonschema
from jsonschema.protocols import Validator

from codeintel.config.dataset_contract import DATASET_CONTRACTS
from codeintel.storage.datasets import DatasetRegistry

TYPE_MAP: dict[type[object], dict[str, object]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    object: {"type": ["object", "array", "string", "number", "boolean", "null"]},
}


def _append_null_type(base_schema: dict[str, object]) -> dict[str, object]:
    if "type" in base_schema and isinstance(base_schema["type"], str):
        return {**base_schema, "type": [base_schema["type"], "null"]}
    if "type" in base_schema and isinstance(base_schema["type"], list):
        types = [t for t in base_schema["type"] if isinstance(t, str)]
        if "null" not in types:
            types.append("null")
        return {**base_schema, "type": types}
    any_of = base_schema.get("anyOf")
    if isinstance(any_of, list):
        merged_any_of: list[object] = list(any_of)
        merged_any_of.append({"type": "null"})
        return {**base_schema, "anyOf": merged_any_of}
    return {"anyOf": [base_schema, {"type": "null"}]}


def _schema_for_union(args: tuple[object, ...]) -> dict[str, object]:
    allow_null = type(None) in args
    non_null = tuple(arg for arg in args if arg is not type(None))
    schemas = [_schema_for_type(arg) for arg in non_null] or [{"type": "null"}]
    base_schema: dict[str, object]
    base_schema = dict(schemas[0]) if len(schemas) == 1 else {"anyOf": [dict(s) for s in schemas]}
    if allow_null:
        return _append_null_type(base_schema)
    return base_schema


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
    elif origin in {typing.Union, types.UnionType}:
        schema = _schema_for_union(get_args(annotation))
    else:
        schema = {"type": "string"}
    return schema


def json_schema_from_typeddict(
    td_type: type[object],
    *,
    additional_properties: bool = True,
    schema_id: str | None = None,
    title: str | None = None,
) -> dict[str, object]:
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
        "$id": schema_id,
        "title": title,
        "type": "object",
        "properties": properties,
        "required": sorted(required),
        "additionalProperties": additional_properties,
    }
    return {k: v for k, v in schema.items() if v is not None}


type JsonValue = str | int | float | bool | Mapping[str, JsonValue] | Sequence[JsonValue] | None


def _coerce_json_value(value: object) -> JsonValue:
    """
    Convert a Python object into a JSON-serializable value for schema validation.

    Raises
    ------
    TypeError
        If the value cannot be coerced into a JSON-compatible type.

    Returns
    -------
    JsonValue
        JSON-serializable value matching schema expectations.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _coerce_json_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_coerce_json_value(v) for v in value]
    message = f"Unsupported value for JSON Schema validation: {value!r}"
    raise TypeError(message)


def validate_row_with_schema(row: Mapping[str, object], schema: Mapping[str, object]) -> None:
    """Validate a single row mapping against a generated JSON Schema."""
    validator = build_validator(schema)
    json_row: dict[str, JsonValue] = {k: _coerce_json_value(v) for k, v in row.items()}
    validator.validate(json_row)


def build_validator(schema: Mapping[str, object]) -> Validator:
    """
    Construct a Draft 2020-12 JSON Schema validator with typed schema input.

    Parameters
    ----------
    schema
        JSON Schema mapping to validate against.

    Returns
    -------
    Validator
        Validator instance bound to the provided schema.
    """
    return jsonschema.Draft202012Validator(dict(schema))


def generate_export_schemas(
    registry: DatasetRegistry,
    *,
    output_dir: Path,
    include_datasets: set[str] | None = None,
) -> list[Path]:
    """
    Generate export JSON Schemas for datasets with row bindings.

    Parameters
    ----------
    registry
        DatasetRegistry or mapping exposing ``by_name`` of datasets with row_binding.
    output_dir
        Directory to write schema files into (created if missing).
    include_datasets
        Optional subset of dataset names to emit; defaults to all with row bindings.

    Returns
    -------
    list[Path]
        Paths of written schema files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    datasets = getattr(registry, "by_name", {})
    for name, ds in sorted(datasets.items()):
        if include_datasets is not None and name not in include_datasets:
            continue
        contract = DATASET_CONTRACTS.get(name)
        if contract is None or contract.json_schema_id is None:
            continue
        row_binding = getattr(ds, "row_binding", None)
        if row_binding is None or row_binding.row_type is None:
            continue
        schema_id = f"https://schemas.codeintel.dev/export/{contract.json_schema_id}.json"
        schema = json_schema_from_typeddict(
            row_binding.row_type,
            additional_properties=True,
            schema_id=schema_id,
            title=f"{name} export",
        )
        path = output_dir / f"{contract.json_schema_id}.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
        written.append(path)
    return written
