"""Serialization helpers for DatasetContract payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from codeintel.config.datasets.primitives import CompositeSchema
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.serde import (
    column_from_json_obj,
    column_to_json_obj,
    table_schema_from_json_obj,
    table_schema_to_json_obj,
)


def _serialize_columns(columns: tuple[Column, ...]) -> list[dict[str, object]]:
    return [column_to_json_obj(col) for col in columns]


def _serialize_composite(schema: CompositeSchema) -> dict[str, object]:
    return {
        "composed_of": list(schema.composed_of),
        "shared_fragments": [_serialize_columns(fragment) for fragment in schema.shared_fragments],
        "additional_columns": _serialize_columns(schema.additional_columns),
        "column_mappings": dict(schema.column_mappings),
        "excluded_columns": list(schema.excluded_columns),
    }


def _parse_columns(value: object) -> tuple[Column, ...]:
    if not isinstance(value, list):
        msg = "Expected list for columns"
        raise TypeError(msg)
    columns = []
    for item in value:
        if not isinstance(item, Mapping):
            msg = "Invalid column object"
            raise TypeError(msg)
        columns.append(column_from_json_obj(item))
    return tuple(columns)


def _parse_composite(value: object) -> CompositeSchema:
    if not isinstance(value, Mapping):
        msg = "Expected object for composite schema"
        raise TypeError(msg)
    composed_of = value.get("composed_of", [])
    if not isinstance(composed_of, list):
        msg = "Expected list for composed_of"
        raise TypeError(msg)
    shared_fragments_raw = value.get("shared_fragments", [])
    if not isinstance(shared_fragments_raw, list):
        msg = "Expected list for shared_fragments"
        raise TypeError(msg)
    shared_fragments = tuple(_parse_columns(fragment) for fragment in shared_fragments_raw)
    additional_columns = _parse_columns(value.get("additional_columns", []))
    column_mappings_raw = value.get("column_mappings", {})
    if not isinstance(column_mappings_raw, Mapping):
        msg = "Expected object for column_mappings"
        raise TypeError(msg)
    column_mappings = {
        str(key): str(val) for key, val in column_mappings_raw.items() if isinstance(val, str)
    }
    excluded_raw = value.get("excluded_columns", [])
    if not isinstance(excluded_raw, list):
        msg = "Expected list for excluded_columns"
        raise TypeError(msg)
    excluded = frozenset(str(item) for item in excluded_raw if isinstance(item, str))
    return CompositeSchema(
        composed_of=tuple(str(item) for item in composed_of if isinstance(item, str)),
        shared_fragments=shared_fragments,
        additional_columns=additional_columns,
        column_mappings=column_mappings,
        excluded_columns=excluded,
    )


def contract_to_json_obj(contract: DatasetContract) -> dict[str, object]:
    """Serialize a DatasetContract into a JSON object.

    Returns
    -------
    dict[str, object]
        JSON-serializable representation of the dataset contract.
    """
    payload: dict[str, object] = {
        "table_key": contract.table_key,
        "name": contract.name,
        "json_schema_id": contract.json_schema_id,
        "jsonl_filename": contract.jsonl_filename,
        "parquet_filename": contract.parquet_filename,
        "is_view": contract.is_view,
        "owner_package": contract.owner_package,
        "tags": sorted(contract.tags),
        "description": contract.description,
        "family": contract.family,
        "owner": contract.owner,
        "freshness_sla": contract.freshness_sla,
        "retention_policy": contract.retention_policy,
        "stable_id": contract.stable_id,
        "schema_version": contract.schema_version,
        "upstream_dependencies": list(contract.upstream_dependencies),
        "validation_profile": contract.validation_profile,
    }
    payload["schema"] = table_schema_to_json_obj(contract.schema) if contract.schema else None
    payload["composition"] = (
        _serialize_composite(contract.composition) if contract.composition else None
    )
    return payload


def _parse_table_schema(value: object) -> TableSchema | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        msg = "Expected object for schema"
        raise TypeError(msg)
    return table_schema_from_json_obj(value)


def contract_from_json_obj(obj: Mapping[str, object]) -> DatasetContract:
    """Parse a DatasetContract from a JSON object.

    Returns
    -------
    DatasetContract
        Parsed dataset contract instance.

    Raises
    ------
    TypeError
        If required fields are missing or of the wrong type.
    """
    table_key = obj.get("table_key")
    name = obj.get("name")
    if not isinstance(table_key, str) or not isinstance(name, str):
        msg = "DatasetContract requires table_key and name"
        raise TypeError(msg)
    tags_raw = obj.get("tags", [])
    tags = frozenset(tags_raw) if isinstance(tags_raw, list) else frozenset()
    upstream_raw = obj.get("upstream_dependencies", [])
    upstream = (
        tuple(item for item in upstream_raw if isinstance(item, str))
        if isinstance(upstream_raw, list)
        else ()
    )
    validation_profile_raw = obj.get("validation_profile", "strict")
    if validation_profile_raw == "lenient":
        validation_profile: Literal["strict", "lenient"] = "lenient"
    else:
        validation_profile = "strict"
    owner_package_raw = obj.get("owner_package")
    owner_package = (
        cast("Literal['core','analytics','graphs','qa','docs']", owner_package_raw)
        if isinstance(owner_package_raw, str)
        else None
    )
    composition_obj = obj.get("composition")
    composition = _parse_composite(composition_obj) if composition_obj is not None else None
    return DatasetContract(
        table_key=table_key,
        name=name,
        schema=_parse_table_schema(obj.get("schema")),
        row_binding=None,
        json_schema_id=_as_optional_str(obj.get("json_schema_id")),
        jsonl_filename=_as_optional_str(obj.get("jsonl_filename")),
        parquet_filename=_as_optional_str(obj.get("parquet_filename")),
        is_view=_as_bool(obj.get("is_view"), default=False),
        owner_package=owner_package,
        tags=tags,
        description=_as_optional_str(obj.get("description")),
        family=_as_optional_str(obj.get("family")),
        owner=_as_optional_str(obj.get("owner")),
        freshness_sla=_as_optional_str(obj.get("freshness_sla")),
        retention_policy=_as_optional_str(obj.get("retention_policy")),
        stable_id=_as_optional_str(obj.get("stable_id")),
        schema_version=_as_optional_str(obj.get("schema_version")),
        upstream_dependencies=upstream,
        validation_profile=validation_profile,
        composition=composition,
    )


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


__all__ = [
    "contract_from_json_obj",
    "contract_to_json_obj",
]
