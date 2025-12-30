"""Iceberg schema conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
from pyiceberg.io.pyarrow import pyarrow_to_schema, schema_to_pyarrow
from pyiceberg.table.name_mapping import MappedField, NameMapping

from codeintel.core.hashing.fingerprint import stable_hash
from codeintel.core.schemas.contracts import (
    ArrowSchemaMetadata,
    arrow_schema_from_fields,
    arrow_schema_from_table_schema,
)
from codeintel.core.schemas.primitives import TableSchema

if TYPE_CHECKING:
    from pyiceberg.schema import Schema

_FIELD_ID_MODULUS = 2_000_000_000


@dataclass(frozen=True, slots=True)
class IcebergSchemaBundle:
    """Derived Iceberg schema bundle with name mapping."""

    schema: Schema
    name_mapping: NameMapping


def table_schema_to_iceberg_schema(
    table_schema: TableSchema,
    *,
    metadata: ArrowSchemaMetadata | None = None,
) -> IcebergSchemaBundle:
    """Convert a TableSchema into an Iceberg schema with stable field IDs.

    Returns
    -------
    IcebergSchemaBundle
        Iceberg schema and name mapping bundle.
    """
    arrow_schema = arrow_schema_from_table_schema(table_schema=table_schema, metadata=metadata)
    name_mapping = name_mapping_from_arrow_schema(arrow_schema, table_key=table_schema.table_key)
    iceberg_schema = pyarrow_to_schema(arrow_schema, name_mapping=name_mapping)
    return IcebergSchemaBundle(schema=iceberg_schema, name_mapping=name_mapping)


def iceberg_schema_to_arrow_schema(
    iceberg_schema: Schema,
    *,
    include_field_ids: bool = True,
    metadata: dict[bytes, bytes] | None = None,
) -> pa.Schema:
    """Convert an Iceberg schema into a PyArrow schema.

    Returns
    -------
    pyarrow.Schema
        Arrow schema converted from the Iceberg schema.
    """
    arrow_schema = schema_to_pyarrow(iceberg_schema, include_field_ids=include_field_ids)
    if metadata:
        return arrow_schema.with_metadata(metadata)
    return arrow_schema


def arrow_schema_with_iceberg_ids(
    table_schema: TableSchema,
    *,
    metadata: ArrowSchemaMetadata | None = None,
) -> pa.Schema:
    """Render an Arrow schema with Iceberg field IDs attached.

    Returns
    -------
    pyarrow.Schema
        Arrow schema that includes Iceberg field IDs.
    """
    base_schema = arrow_schema_from_table_schema(table_schema=table_schema, metadata=metadata)
    bundle = table_schema_to_iceberg_schema(table_schema, metadata=metadata)
    iceberg_schema = bundle.schema
    arrow_with_ids = schema_to_pyarrow(iceberg_schema, include_field_ids=True)
    return _merge_schema_metadata(base_schema, arrow_with_ids)


def name_mapping_from_arrow_schema(
    arrow_schema: pa.Schema,
    *,
    table_key: str,
) -> NameMapping:
    """Build an Iceberg NameMapping from a PyArrow schema.

    Returns
    -------
    pyiceberg.table.name_mapping.NameMapping
        Iceberg name mapping derived from the Arrow schema.
    """
    mapped_fields = [
        _mapped_field(field, table_key=table_key, path=(field.name,)) for field in arrow_schema
    ]
    return NameMapping(root=mapped_fields)


def iceberg_field_ids_for_table_schema(table_schema: TableSchema) -> dict[str, int]:
    """Return stable Iceberg field IDs for top-level columns.

    Returns
    -------
    dict[str, int]
        Mapping from column name to Iceberg field ID.
    """
    return {
        column.name: _stable_field_id(table_schema.table_key, (column.name,))
        for column in table_schema.columns
    }


def _mapped_field(field: pa.Field, *, table_key: str, path: tuple[str, ...]) -> MappedField:
    field_id = _stable_field_id(table_key, path)
    children = _mapped_children(field.type, table_key=table_key, path=path)
    return _mapped_field_from_parts(field_id, [field.name], children)


def _mapped_children(
    datatype: pa.DataType,
    *,
    table_key: str,
    path: tuple[str, ...],
) -> list[MappedField]:
    if pa.types.is_struct(datatype):
        struct = datatype
        return [
            _mapped_field(
                child,
                table_key=table_key,
                path=(*path, child.name),
            )
            for child in struct
        ]
    if pa.types.is_list(datatype) or pa.types.is_large_list(datatype):
        element_field = datatype.value_field
        element_path = (*path, "element")
        return [
            _mapped_field_from_parts(
                _stable_field_id(table_key, element_path),
                ["element"],
                _mapped_children(element_field.type, table_key=table_key, path=element_path),
            )
        ]
    if pa.types.is_map(datatype):
        map_type = datatype
        key_field = map_type.key_field
        value_field = map_type.item_field
        key_path = (*path, "key")
        value_path = (*path, "value")
        return [
            _mapped_field_from_parts(
                _stable_field_id(table_key, key_path),
                ["key"],
                _mapped_children(key_field.type, table_key=table_key, path=key_path),
            ),
            _mapped_field_from_parts(
                _stable_field_id(table_key, value_path),
                ["value"],
                _mapped_children(value_field.type, table_key=table_key, path=value_path),
            ),
        ]
    return []


def _stable_field_id(table_key: str, path: tuple[str, ...]) -> int:
    digest = stable_hash(table_key, ".".join(path))
    return _hash_to_int(digest)


def _hash_to_int(digest: str) -> int:
    raw = int(digest[:8], 16)
    return (raw % _FIELD_ID_MODULUS) + 1


def _mapped_field_from_parts(
    field_id: int,
    names: list[str],
    fields: list[MappedField],
) -> MappedField:
    return MappedField.model_validate(
        {
            "field-id": field_id,
            "names": names,
            "fields": fields,
        }
    )


def _merge_schema_metadata(base_schema: pa.Schema, with_ids: pa.Schema) -> pa.Schema:
    merged_fields: list[pa.Field] = []
    for base_field, id_field in zip(base_schema, with_ids, strict=True):
        merged_fields.append(_merge_field_metadata(base_field, id_field))
    merged_metadata = base_schema.metadata or with_ids.metadata
    return arrow_schema_from_fields(fields=merged_fields, metadata=merged_metadata)


def _merge_field_metadata(base_field: pa.Field, id_field: pa.Field) -> pa.Field:
    base_metadata = base_field.metadata or {}
    id_metadata = id_field.metadata or {}
    if not base_metadata:
        return id_field
    merged = dict(base_metadata)
    merged.update(id_metadata)
    return id_field.with_metadata(merged)


__all__ = [
    "IcebergSchemaBundle",
    "arrow_schema_with_iceberg_ids",
    "iceberg_field_ids_for_table_schema",
    "iceberg_schema_to_arrow_schema",
    "name_mapping_from_arrow_schema",
    "table_schema_to_iceberg_schema",
]
