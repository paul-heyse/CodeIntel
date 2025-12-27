"""PyArrow schema rendering for core TableSchema definitions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import Column, ColumnType, TableSchema

if TYPE_CHECKING:
    from collections.abc import Iterable

_DECIMAL_PATTERN = re.compile(r"^DECIMAL\\((\\d+),(\\d+)\\)$")
_ARROW_TYPE_MAP: dict[str, pa.DataType] = {
    "BOOLEAN": pa.bool_(),
    "INTEGER": pa.int32(),
    "BIGINT": pa.int64(),
    "DOUBLE": pa.float64(),
    "DECIMAL": pa.float64(),
    "VARCHAR": pa.string(),
    "JSON": pa.string(),
    "TIMESTAMP": pa.timestamp("us"),
    "TIMESTAMPTZ": pa.timestamp("us", tz="UTC"),
}


@dataclass(frozen=True, slots=True)
class ArrowSchemaProvenance:
    """Provenance metadata for schema rendering."""

    derivation_kind: str | None = None
    derivation_source: str | None = None
    inference_status: str | None = None
    inference_error: str | None = None

    def to_payload(self) -> dict[str, str]:
        """Return a JSON-serializable provenance payload.

        Returns
        -------
        dict[str, str]
            Provenance metadata mapping.
        """
        payload: dict[str, str] = {}
        if self.derivation_kind is not None:
            payload["derivation_kind"] = self.derivation_kind
        if self.derivation_source is not None:
            payload["derivation_source"] = self.derivation_source
        if self.inference_status is not None:
            payload["inference_status"] = self.inference_status
        if self.inference_error is not None:
            payload["inference_error"] = self.inference_error
        return payload


@dataclass(frozen=True, slots=True)
class ArrowSchemaMetadata:
    """Optional metadata inputs for Arrow schema rendering."""

    schema_hash: str | None = None
    schema_digest: str | None = None
    provenance: ArrowSchemaProvenance | None = None
    column_lineage: Mapping[str, Iterable[tuple[str, str]]] | None = None
    pii_by_column: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class _FieldMetadataContext:
    schema_hash_value: str
    schema_digest: str
    provenance_payload: Mapping[str, str]
    column_lineage: Mapping[str, Iterable[tuple[str, str]]] | None
    pii_by_column: Mapping[str, str] | None
    key_roles: Mapping[str, str]


def _arrow_decimal_type(normalized: str) -> pa.DataType:
    match = _DECIMAL_PATTERN.match(normalized)
    if match is None:
        return pa.decimal128(38, 0)
    precision = int(match.group(1))
    scale = int(match.group(2))
    return pa.decimal128(precision, scale)


def _arrow_type_for_column_type(column_type: ColumnType) -> pa.DataType:
    normalized = str(column_type).upper()
    if normalized.startswith("DECIMAL("):
        return _arrow_decimal_type(normalized)
    return _ARROW_TYPE_MAP.get(normalized, pa.string())


def _encode_metadata(metadata: Mapping[str, object]) -> dict[bytes, bytes] | None:
    encoded: dict[bytes, bytes] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str):
            raw = value
        else:
            raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
        encoded[key.encode("utf-8")] = raw.encode("utf-8")
    return encoded or None


def _key_roles(table_schema: TableSchema) -> dict[str, str]:
    roles: dict[str, str] = dict.fromkeys(table_schema.primary_key, "primary_key")
    for index in table_schema.indexes:
        if not index.unique:
            continue
        for column in index.columns:
            roles.setdefault(column, "unique_index")
    return roles


def _lineage_payload(lineage: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
    entries = sorted(lineage, key=lambda item: (item[0], item[1]))
    return [{"table_key": table_key, "column": column} for table_key, column in entries]


def _provenance_payload(provenance: ArrowSchemaProvenance | None) -> dict[str, str]:
    if provenance is None:
        return {}
    return provenance.to_payload()


def _field_metadata(
    column: Column,
    context: _FieldMetadataContext,
) -> dict[str, object]:
    field_metadata: dict[str, object] = {
        "codeintel.column_type": column.type,
        "codeintel.nullable": column.nullable,
        "codeintel.schema_hash": context.schema_hash_value,
        "codeintel.schema_digest": context.schema_digest,
    }
    if column.description is not None:
        field_metadata["codeintel.description"] = column.description
    key_role = context.key_roles.get(column.name)
    if key_role is not None:
        field_metadata["codeintel.key_role"] = key_role
    if context.pii_by_column is not None:
        pii_class = context.pii_by_column.get(column.name)
        if pii_class is not None:
            field_metadata["codeintel.pii_class"] = pii_class
    if context.provenance_payload:
        field_metadata["codeintel.provenance"] = dict(context.provenance_payload)
    if context.column_lineage is not None:
        lineage = context.column_lineage.get(column.name)
        if lineage:
            field_metadata["codeintel.lineage_edges"] = _lineage_payload(lineage)
    return field_metadata


def _schema_metadata(
    table_schema: TableSchema,
    schema_hash_value: str,
    schema_digest: str,
    provenance_payload: Mapping[str, str],
) -> dict[str, object]:
    schema_metadata: dict[str, object] = {
        "codeintel.table_key": table_schema.table_key,
        "codeintel.schema_hash": schema_hash_value,
        "codeintel.schema_digest": schema_digest,
        "codeintel.primary_key": list(table_schema.primary_key),
    }
    if table_schema.description is not None:
        schema_metadata["codeintel.description"] = table_schema.description
    if provenance_payload:
        schema_metadata["codeintel.provenance"] = dict(provenance_payload)
    return schema_metadata


def arrow_schema_from_table_schema(
    *,
    table_schema: TableSchema,
    metadata: ArrowSchemaMetadata | None = None,
) -> pa.Schema:
    """Render a PyArrow Schema with CodeIntel metadata.

    Parameters
    ----------
    table_schema
        Source TableSchema.
    metadata
        Optional rendering metadata (hashes, provenance, lineage, PII labels).

    Returns
    -------
    pa.Schema
        Rendered PyArrow schema with metadata attached.
    """
    resolved_metadata = metadata or ArrowSchemaMetadata()
    resolved_hash = resolved_metadata.schema_hash or schema_hash(table_schema)
    resolved_digest = resolved_metadata.schema_digest or fingerprint(table_schema.to_json_obj())
    provenance_payload = _provenance_payload(resolved_metadata.provenance)
    key_roles = _key_roles(table_schema)
    field_context = _FieldMetadataContext(
        schema_hash_value=resolved_hash,
        schema_digest=resolved_digest,
        provenance_payload=provenance_payload,
        column_lineage=resolved_metadata.column_lineage,
        pii_by_column=resolved_metadata.pii_by_column,
        key_roles=key_roles,
    )
    fields = [
        pa.field(
            column.name,
            _arrow_type_for_column_type(column.type),
            nullable=column.nullable,
            metadata=_encode_metadata(_field_metadata(column, field_context)),
        )
        for column in table_schema.columns
    ]

    schema_metadata = _schema_metadata(
        table_schema,
        schema_hash_value=resolved_hash,
        schema_digest=resolved_digest,
        provenance_payload=provenance_payload,
    )

    return pa.schema(fields, metadata=_encode_metadata(schema_metadata))


__all__ = [
    "ArrowSchemaMetadata",
    "ArrowSchemaProvenance",
    "arrow_schema_from_table_schema",
]
