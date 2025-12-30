"""Iceberg metadata cache refresh helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow as pa
from pyiceberg.types import ListType, MapType, NestedField, StructType

from codeintel.core.iceberg.schema import iceberg_schema_to_arrow_schema
from codeintel.core.schemas.contracts import encode_schema_ipc
from codeintel.core.time import utc_now
from codeintel.storage.metadata.meta_catalog import meta_table_ref

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pyiceberg.schema import Schema
    from pyiceberg.table import Table
    from pyiceberg.table.metadata import TableMetadata
    from pyiceberg.table.refs import SnapshotRef

    from codeintel.storage.gateway.protocol import MinimalGateway


SchemaFieldRow = dict[str, bool | int | str | None]


def refresh_iceberg_metadata_cache(
    *,
    gateway: MinimalGateway,
    table_key: str,
    table: Table,
    now: datetime | None = None,
) -> None:
    """Refresh the derived Iceberg metadata cache for a table.

    Parameters
    ----------
    gateway
        Storage gateway providing metadata catalog access.
    table_key
        Fully qualified table key (schema.table).
    table
        PyIceberg table to inspect.
    now
        Optional timestamp override for cache rows.
    """
    timestamp = now or utc_now()
    metadata = table.metadata
    _replace_rows(
        gateway=gateway,
        table_key=table_key,
        table_ref=meta_table_ref("metadata.iceberg_tables"),
        columns=_ICEBERG_TABLES_COLUMNS,
        rows=[_iceberg_table_row(table_key, table=table, metadata=metadata, timestamp=timestamp)],
    )
    _replace_rows(
        gateway=gateway,
        table_key=table_key,
        table_ref=meta_table_ref("metadata.iceberg_schemas"),
        columns=_ICEBERG_SCHEMAS_COLUMNS,
        rows=_iceberg_schema_rows(table_key, metadata=metadata, table=table),
    )
    _replace_rows(
        gateway=gateway,
        table_key=table_key,
        table_ref=meta_table_ref("metadata.iceberg_partition_specs"),
        columns=_ICEBERG_SPECS_COLUMNS,
        rows=_iceberg_partition_rows(table_key, metadata=metadata),
    )
    _replace_rows(
        gateway=gateway,
        table_key=table_key,
        table_ref=meta_table_ref("metadata.iceberg_sort_orders"),
        columns=_ICEBERG_SORT_COLUMNS,
        rows=_iceberg_sort_rows(table_key, metadata=metadata),
    )
    _replace_rows(
        gateway=gateway,
        table_key=table_key,
        table_ref=meta_table_ref("metadata.iceberg_snapshots"),
        columns=_ICEBERG_SNAPSHOT_COLUMNS,
        rows=_iceberg_snapshot_rows(table_key, metadata=metadata),
    )
    _replace_rows(
        gateway=gateway,
        table_key=table_key,
        table_ref=meta_table_ref("metadata.iceberg_arrow_schema"),
        columns=_ICEBERG_ARROW_COLUMNS,
        rows=_iceberg_arrow_schema_rows(table_key, metadata=metadata),
    )


_ICEBERG_TABLES_COLUMNS = (
    "table_key",
    "identifier",
    "location",
    "current_snapshot_id",
    "current_schema_id",
    "current_spec_id",
    "current_sort_order_id",
    "properties",
    "refs",
    "last_updated_at",
)

_ICEBERG_SCHEMAS_COLUMNS = (
    "table_key",
    "schema_id",
    "fields",
    "name_mapping_json",
    "schema_json",
)

_ICEBERG_SPECS_COLUMNS = (
    "table_key",
    "spec_id",
    "fields",
)

_ICEBERG_SORT_COLUMNS = (
    "table_key",
    "order_id",
    "fields",
)

_ICEBERG_SNAPSHOT_COLUMNS = (
    "table_key",
    "snapshot_id",
    "parent_snapshot_id",
    "committed_at",
    "operation",
    "summary",
    "manifest_list_path",
)

_ICEBERG_ARROW_COLUMNS = (
    "table_key",
    "schema_id",
    "arrow_schema_ipc",
    "arrow_schema_json",
)


def _replace_rows(
    *,
    gateway: MinimalGateway,
    table_key: str,
    table_ref: str,
    columns: Iterable[str],
    rows: list[tuple[object, ...]],
) -> None:
    con = gateway.con
    con.execute(f"DELETE FROM {table_ref} WHERE table_key = ?", [table_key])
    if not rows:
        return
    column_sql = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(rows[0]))
    sql = f"INSERT INTO {table_ref} ({column_sql}) VALUES ({placeholders})"
    con.executemany(sql, rows)


def _iceberg_table_row(
    table_key: str,
    *,
    table: Table,
    metadata: TableMetadata,
    timestamp: datetime,
) -> tuple[object, ...]:
    identifier = ".".join(table.name())
    return (
        table_key,
        identifier,
        table.location(),
        metadata.current_snapshot_id,
        metadata.current_schema_id,
        metadata.default_spec_id,
        metadata.default_sort_order_id,
        dict(metadata.properties),
        _refs_payload(metadata.refs or {}),
        timestamp,
    )


def _refs_payload(refs: Mapping[str, SnapshotRef]) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for name, ref in refs.items():
        payload[name] = {
            "snapshot_id": ref.snapshot_id,
            "ref_type": ref.snapshot_ref_type.value,
            "max_ref_age_ms": ref.max_ref_age_ms,
        }
    return payload


def _iceberg_schema_rows(
    table_key: str,
    *,
    metadata: TableMetadata,
    table: Table,
) -> list[tuple[object, ...]]:
    name_mapping = table.name_mapping()
    name_mapping_payload = (
        name_mapping.model_dump(by_alias=True, exclude_none=True) if name_mapping else None
    )
    return [
        (
            table_key,
            schema.schema_id,
            _flatten_schema_fields(schema),
            name_mapping_payload,
            schema.model_dump(by_alias=True, exclude_none=True),
        )
        for schema in metadata.schemas
    ]


def _iceberg_partition_rows(
    table_key: str,
    *,
    metadata: TableMetadata,
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for spec in metadata.partition_specs:
        fields = [
            {
                "field_id": field.field_id,
                "name": field.name,
                "transform": str(field.transform),
                "source_id": field.source_id,
            }
            for field in spec.fields
        ]
        rows.append((table_key, spec.spec_id, fields))
    return rows


def _iceberg_sort_rows(
    table_key: str,
    *,
    metadata: TableMetadata,
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for order in metadata.sort_orders:
        fields = [
            {
                "field_id": field.source_id,
                "transform": str(field.transform),
                "direction": field.direction.value,
                "null_order": field.null_order.value,
            }
            for field in order.fields
        ]
        rows.append((table_key, order.order_id, fields))
    return rows


def _iceberg_snapshot_rows(
    table_key: str,
    *,
    metadata: TableMetadata,
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for snapshot in metadata.snapshots or ():
        committed_at = datetime.fromtimestamp(snapshot.timestamp_ms / 1000, tz=UTC)
        summary = dict(snapshot.summary) if snapshot.summary is not None else None
        rows.append(
            (
                table_key,
                snapshot.snapshot_id,
                snapshot.parent_snapshot_id,
                committed_at,
                summary.get("operation") if summary is not None else None,
                summary,
                snapshot.manifest_list,
            )
        )
    return rows


def _iceberg_arrow_schema_rows(
    table_key: str,
    *,
    metadata: TableMetadata,
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for schema in metadata.schemas:
        arrow_schema = iceberg_schema_to_arrow_schema(schema, include_field_ids=True)
        rows.append(
            (
                table_key,
                schema.schema_id,
                encode_schema_ipc(arrow_schema),
                _arrow_schema_payload(arrow_schema),
            )
        )
    return rows


def _flatten_schema_fields(schema: Schema) -> list[SchemaFieldRow]:
    fields: list[SchemaFieldRow] = []
    for field in schema.fields:
        fields.extend(_flatten_field(field, parent_id=None))
    return fields


def _flatten_field(
    field: NestedField,
    *,
    parent_id: int | None,
) -> list[SchemaFieldRow]:
    iceberg_field = field
    fields: list[SchemaFieldRow] = [
        {
            "field_id": iceberg_field.field_id,
            "name": iceberg_field.name,
            "type": str(iceberg_field.field_type),
            "required": iceberg_field.required,
            "doc": iceberg_field.doc,
            "parent_id": parent_id,
        }
    ]
    field_type = iceberg_field.field_type
    if isinstance(field_type, StructType):
        for child in field_type.fields:
            fields.extend(_flatten_field(child, parent_id=iceberg_field.field_id))
    elif isinstance(field_type, ListType):
        element_field = NestedField(
            field_type.element_id,
            "element",
            field_type.element_type,
            required=field_type.element_required,
        )
        fields.extend(_flatten_field(element_field, parent_id=iceberg_field.field_id))
    elif isinstance(field_type, MapType):
        key_field = NestedField(
            field_type.key_id,
            "key",
            field_type.key_type,
            required=True,
        )
        value_field = NestedField(
            field_type.value_id,
            "value",
            field_type.value_type,
            required=field_type.value_required,
        )
        fields.extend(_flatten_field(key_field, parent_id=iceberg_field.field_id))
        fields.extend(_flatten_field(value_field, parent_id=iceberg_field.field_id))
    return fields


def _arrow_schema_payload(schema: pa.Schema) -> list[SchemaFieldRow]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]


__all__ = ["refresh_iceberg_metadata_cache"]
