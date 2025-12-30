"""Schema contract conversions and serialization helpers."""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast, get_args

import polars as pl
import pyarrow as pa
from sqlglot import exp

from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.json_schema_gen import json_schema_from_table_schema
from codeintel.core.schemas.primitives import (
    Column,
    ColumnType,
    Index,
    ReplaceScope,
    TableSchema,
    TableWritePolicy,
    WriteMode,
    column_type_base,
    normalize_column_type,
)
from codeintel.storage.helpers.table_key import split_table_key, validate_table_key

_DECIMAL_PATTERN = re.compile(r"^DECIMAL\\((\\d+),(\\d+)\\)$")
_DECIMAL_DEFAULT_PRECISION = 38
_DECIMAL_DEFAULT_SCALE = 0
_DECIMAL_PARAM_COUNT = 2
_DECIMAL_MAX_128_PRECISION = _DECIMAL_DEFAULT_PRECISION
_MAP_PARAM_COUNT = 2
_SQLGLOT_DIALECT = "duckdb"
ExtrasPolicy = Literal["retain", "reject", "drop"]
ARROW_SCHEMA_CONTRACT_VERSION = "v1"
DEFAULT_EXTRAS_POLICY: ExtrasPolicy = "reject"
DEFAULT_EXTRAS_COLUMN = "_ci_extras"
EXTRAS_POLICIES: frozenset[ExtrasPolicy] = frozenset({"retain", "reject", "drop"})
ARROW_SCHEMA_METADATA_KEYS: tuple[str, ...] = (
    "codeintel.table_key",
    "codeintel.schema_hash",
    "codeintel.schema_digest",
    "codeintel.primary_key",
    "codeintel.schema_contract_version",
    "codeintel.extras_policy",
    "codeintel.extras_column",
    "codeintel.extras_schema",
    "codeintel.description",
    "codeintel.provenance",
    "codeintel.iceberg_schema_id",
    "codeintel.iceberg_name_mapping_digest",
)
ARROW_FIELD_METADATA_KEYS: tuple[str, ...] = (
    "codeintel.column_type",
    "codeintel.nullable",
    "codeintel.schema_hash",
    "codeintel.schema_digest",
    "codeintel.description",
    "codeintel.key_role",
    "codeintel.pii_class",
    "codeintel.provenance",
    "codeintel.lineage_edges",
    "codeintel.iceberg_field_id",
)
_ARROW_TYPE_MAP: dict[str, pa.DataType] = {
    "BOOLEAN": pa.bool_(),
    "INTEGER": pa.int32(),
    "BIGINT": pa.int64(),
    "DOUBLE": pa.float64(),
    "VARCHAR": pa.string(),
    "JSON": pa.string(),
    "TIMESTAMP": pa.timestamp("us"),
    "TIMESTAMPTZ": pa.timestamp("us", tz="UTC"),
}


def _sqlglot_type(name: str) -> exp.DataType.Type | None:
    value = getattr(exp.DataType.Type, name, None)
    if isinstance(value, exp.DataType.Type):
        return value
    return None


def _sqlglot_types(*names: str) -> frozenset[exp.DataType.Type]:
    resolved: list[exp.DataType.Type] = []
    for name in names:
        value = _sqlglot_type(name)
        if value is not None:
            resolved.append(value)
    return frozenset(resolved)


_SQLGLOT_INTEGER_TYPES = _sqlglot_types(
    "INT",
    "INTEGER",
    "SMALLINT",
    "TINYINT",
    "MEDIUMINT",
    "UINT",
    "USMALLINT",
    "UTINYINT",
    "UMEDIUMINT",
)
_SQLGLOT_BIGINT_TYPES = _sqlglot_types(
    "BIGINT",
    "BIGSERIAL",
    "SERIAL",
    "SMALLSERIAL",
)
_SQLGLOT_BIGINT_DECIMAL_TYPES = _sqlglot_types(
    "UBIGINT",
    "INT128",
    "INT256",
    "UINT128",
    "UINT256",
)
_SQLGLOT_DECIMAL_TYPES = _sqlglot_types(
    "DECIMAL",
    "DECIMAL32",
    "DECIMAL64",
    "DECIMAL128",
    "DECIMAL256",
    "BIGDECIMAL",
    "BIGNUM",
    "UDECIMAL",
    "DECFLOAT",
)
_SQLGLOT_FLOAT_TYPES = _sqlglot_types(
    "FLOAT",
    "DOUBLE",
    "UDOUBLE",
    "REAL",
)
_SQLGLOT_STRING_TYPES = _sqlglot_types(
    "VARCHAR",
    "TEXT",
    "CHAR",
    "NCHAR",
    "NVARCHAR",
    "NAME",
    "LONGTEXT",
    "MEDIUMTEXT",
    "TINYTEXT",
    "FIXEDSTRING",
    "UUID",
)
_SQLGLOT_JSON_TYPES = _sqlglot_types(
    "JSON",
    "JSONB",
    "OBJECT",
    "VARIANT",
    "SUPER",
)
_SQLGLOT_BOOLEAN_TYPES = _sqlglot_types("BOOLEAN")
_SQLGLOT_TIMESTAMPTZ_TYPES = _sqlglot_types("TIMESTAMPTZ", "TIMESTAMPLTZ")
_SQLGLOT_TIMESTAMP_TYPES = _sqlglot_types(
    "TIMESTAMP",
    "TIMESTAMPNTZ",
    "TIMESTAMP_S",
    "TIMESTAMP_MS",
    "TIMESTAMP_NS",
    "DATETIME",
    "DATETIME2",
    "DATETIME64",
    "SMALLDATETIME",
    "TIME",
    "TIME_NS",
    "TIMETZ",
    "DATE",
    "DATE32",
)
_SQLGLOT_LIST_TYPES = _sqlglot_types("LIST", "ARRAY")
_SQLGLOT_MAP_TYPES = _sqlglot_types("MAP")
_SQLGLOT_STRUCT_TYPES = _sqlglot_types("STRUCT")
_SQLGLOT_UNION_TYPES = _sqlglot_types("UNION")

_STRUCT_FIELD_TYPE = cast("type[exp.Expression] | None", getattr(exp, "StructField", None))


def _is_struct_field(expr: exp.Expression) -> bool:
    if isinstance(_STRUCT_FIELD_TYPE, type):
        return isinstance(expr, _STRUCT_FIELD_TYPE)
    return expr.__class__.__name__ == "StructField"


@dataclass(frozen=True, slots=True)
class ArrowSchemaProvenance:
    """Provenance metadata for schema rendering."""

    derivation_kind: str | None = None
    derivation_source: str | None = None
    inference_status: str | None = None
    inference_error: str | None = None
    producer_target: str | None = None
    producer_module: str | None = None
    producer_version: str | None = None

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
        if self.producer_target is not None:
            payload["producer_target"] = self.producer_target
        if self.producer_module is not None:
            payload["producer_module"] = self.producer_module
        if self.producer_version is not None:
            payload["producer_version"] = self.producer_version
        return payload


@dataclass(frozen=True, slots=True)
class ArrowSchemaMetadata:
    """Optional metadata inputs for Arrow schema rendering."""

    schema_hash: str | None = None
    schema_digest: str | None = None
    provenance: ArrowSchemaProvenance | None = None
    column_lineage: Mapping[str, Iterable[tuple[str, str]]] | None = None
    pii_by_column: Mapping[str, str] | None = None
    contract_version: str | None = None
    extras_policy: ExtrasPolicy | None = None
    extras_column: str | None = None
    extras_schema: Mapping[str, str] | None = None
    iceberg_schema_id: int | None = None
    iceberg_name_mapping_digest: str | None = None
    iceberg_field_ids: Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True)
class _FieldMetadataContext:
    schema_hash_value: str
    schema_digest: str
    provenance_payload: Mapping[str, str]
    column_lineage: Mapping[str, Iterable[tuple[str, str]]] | None
    pii_by_column: Mapping[str, str] | None
    key_roles: Mapping[str, str]
    iceberg_field_ids: Mapping[str, int] | None


@dataclass(frozen=True, slots=True)
class _SchemaMetadataContext:
    table_schema: TableSchema
    schema_hash_value: str
    schema_digest: str
    provenance_payload: Mapping[str, str]
    contract_version: str
    extras_policy: ExtrasPolicy
    extras_column: str
    extras_schema: Mapping[str, str] | None
    iceberg_schema_id: int | None
    iceberg_name_mapping_digest: str | None


def _validate_extras_policy(value: ExtrasPolicy) -> None:
    if value not in EXTRAS_POLICIES:
        msg = f"Unsupported extras policy: {value!r}"
        raise ValueError(msg)


def _normalize_extras_schema(schema: Mapping[str, str] | None) -> Mapping[str, str] | None:
    if schema is None:
        return None
    normalized: dict[str, str] = {}
    for key, value in schema.items():
        if not isinstance(key, str):
            msg = f"extras schema key must be a string, got {type(key)}"
            raise TypeError(msg)
        if not isinstance(value, str):
            msg = f"extras schema value must be a string, got {type(value)}"
            raise TypeError(msg)
        normalized[key] = value
    return normalized


def _normalize_contract_metadata(
    metadata: ArrowSchemaMetadata | None,
) -> ArrowSchemaMetadata:
    resolved = metadata or ArrowSchemaMetadata()
    contract_version = resolved.contract_version or ARROW_SCHEMA_CONTRACT_VERSION
    extras_policy = resolved.extras_policy or DEFAULT_EXTRAS_POLICY
    _validate_extras_policy(extras_policy)
    extras_column = resolved.extras_column or DEFAULT_EXTRAS_COLUMN
    extras_schema = _normalize_extras_schema(resolved.extras_schema)
    return replace(
        resolved,
        contract_version=contract_version,
        extras_policy=extras_policy,
        extras_column=extras_column,
        extras_schema=extras_schema,
    )


def _arrow_decimal_type(normalized: str) -> pa.DataType:
    match = _DECIMAL_PATTERN.match(normalized)
    if match is None:
        return _decimal_for_precision_scale(
            _DECIMAL_DEFAULT_PRECISION,
            _DECIMAL_DEFAULT_SCALE,
        )
    precision = int(match.group(1))
    scale = int(match.group(2))
    return _decimal_for_precision_scale(precision, scale)


def _decimal_for_precision_scale(precision: int, scale: int) -> pa.DataType:
    if precision > _DECIMAL_MAX_128_PRECISION:
        return pa.decimal256(precision, scale)
    return pa.decimal128(precision, scale)


def _decimal_precision_scale(data_type: exp.DataType) -> tuple[int | None, int | None]:
    if len(data_type.expressions) < _DECIMAL_PARAM_COUNT:
        return None, None
    precision = _int_literal(data_type.expressions[0])
    scale = _int_literal(data_type.expressions[1])
    return precision, scale


def _int_literal(node: exp.Expression) -> int | None:
    if not isinstance(node, exp.DataTypeParam):
        return None
    literal = node.this
    if not isinstance(literal, exp.Literal) or literal.is_string:
        return None
    try:
        return int(literal.this)
    except (TypeError, ValueError):
        return None


def _arrow_list_type(data_type: exp.DataType) -> pa.DataType:
    if not data_type.expressions:
        msg = "ARRAY/ LIST types must specify the inner type"
        raise ValueError(msg)
    inner = data_type.expressions[0]
    if not isinstance(inner, exp.DataType):
        msg = "ARRAY/ LIST types must specify the inner type"
        raise TypeError(msg)
    return pa.list_(_arrow_type_from_sqlglot(inner))


def _arrow_map_type(data_type: exp.DataType) -> pa.DataType:
    if len(data_type.expressions) < _MAP_PARAM_COUNT:
        msg = "MAP types must specify key and value types"
        raise ValueError(msg)
    key_node = data_type.expressions[0]
    value_node = data_type.expressions[1]
    if not isinstance(key_node, exp.DataType) or not isinstance(value_node, exp.DataType):
        msg = "MAP types must specify key and value types"
        raise TypeError(msg)
    key_type = _arrow_type_from_sqlglot(key_node)
    value_type = _arrow_type_from_sqlglot(value_node)
    return pa.map_(key_type, value_type)


def _arrow_struct_type(data_type: exp.DataType) -> pa.DataType:
    fields: list[pa.Field] = []
    for field_expr in data_type.expressions:
        if not _is_struct_field(field_expr):
            msg = "STRUCT type must declare named fields"
            raise TypeError(msg)
        if not isinstance(field_expr.this, exp.Identifier):
            msg = "STRUCT fields must have identifier names"
            raise TypeError(msg)
        name = field_expr.this.name
        if not isinstance(field_expr.args.get("kind"), exp.DataType):
            msg = "STRUCT fields must specify a data type"
            raise TypeError(msg)
        field_type = _arrow_type_from_sqlglot(field_expr.args["kind"])
        fields.append(pa.field(name, field_type))
    return pa.struct(fields)


def _arrow_union_type(data_type: exp.DataType) -> pa.DataType:
    if not data_type.expressions:
        msg = "UNION types must specify member types"
        raise ValueError(msg)
    fields: list[pa.Field] = []
    for field_expr in data_type.expressions:
        if not _is_struct_field(field_expr):
            msg = "UNION members must be STRUCT fields"
            raise TypeError(msg)
        if not isinstance(field_expr.this, exp.Identifier):
            msg = "UNION fields must have identifier names"
            raise TypeError(msg)
        name = field_expr.this.name
        if not isinstance(field_expr.args.get("kind"), exp.DataType):
            msg = "UNION fields must specify a data type"
            raise TypeError(msg)
        field_type = _arrow_type_from_sqlglot(field_expr.args["kind"])
        fields.append(pa.field(name, field_type))
    return pa.union(fields, mode="dense")


def _arrow_bool_type(_: exp.DataType) -> pa.DataType:
    return pa.bool_()


def _arrow_int32_type(_: exp.DataType) -> pa.DataType:
    return pa.int32()


def _arrow_int64_type(_: exp.DataType) -> pa.DataType:
    return pa.int64()


def _arrow_float64_type(_: exp.DataType) -> pa.DataType:
    return pa.float64()


def _arrow_string_type(_: exp.DataType) -> pa.DataType:
    return pa.string()


def _arrow_timestamp_type(_: exp.DataType) -> pa.DataType:
    return pa.timestamp("us")


def _arrow_timestamptz_type(_: exp.DataType) -> pa.DataType:
    return pa.timestamp("us", tz="UTC")


def _arrow_big_decimal_type(_: exp.DataType) -> pa.DataType:
    return _decimal_for_precision_scale(
        _DECIMAL_DEFAULT_PRECISION,
        _DECIMAL_DEFAULT_SCALE,
    )


def _arrow_decimal_type_from_sqlglot(data_type: exp.DataType) -> pa.DataType:
    precision, scale = _decimal_precision_scale(data_type)
    if precision is None or scale is None:
        return _arrow_big_decimal_type(data_type)
    return _decimal_for_precision_scale(precision, scale)


_SQLGLOT_TYPE_HANDLERS: tuple[
    tuple[frozenset[exp.DataType.Type], Callable[[exp.DataType], pa.DataType]],
    ...,
] = (
    (_SQLGLOT_INTEGER_TYPES, _arrow_int32_type),
    (_SQLGLOT_BIGINT_TYPES, _arrow_int64_type),
    (_SQLGLOT_BIGINT_DECIMAL_TYPES, _arrow_big_decimal_type),
    (_SQLGLOT_FLOAT_TYPES, _arrow_float64_type),
    (_SQLGLOT_STRING_TYPES, _arrow_string_type),
    (_SQLGLOT_JSON_TYPES, _arrow_string_type),
    (_SQLGLOT_BOOLEAN_TYPES, _arrow_bool_type),
    (_SQLGLOT_TIMESTAMPTZ_TYPES, _arrow_timestamptz_type),
    (_SQLGLOT_TIMESTAMP_TYPES, _arrow_timestamp_type),
    (_SQLGLOT_DECIMAL_TYPES, _arrow_decimal_type_from_sqlglot),
    (_SQLGLOT_LIST_TYPES, _arrow_list_type),
    (_SQLGLOT_MAP_TYPES, _arrow_map_type),
    (_SQLGLOT_STRUCT_TYPES, _arrow_struct_type),
    (_SQLGLOT_UNION_TYPES, _arrow_union_type),
)


def _arrow_type_from_sqlglot(data_type: exp.DataType) -> pa.DataType:
    for types, handler in _SQLGLOT_TYPE_HANDLERS:
        if data_type.this in types:
            return handler(data_type)
    msg = f"Unsupported SQL type: {data_type}"
    raise ValueError(msg)


def _arrow_type_for_column_type(column_type: ColumnType) -> pa.DataType:
    if column_type in _ARROW_TYPE_MAP:
        return _ARROW_TYPE_MAP[column_type]
    normalized = normalize_column_type(column_type)
    if normalized in _ARROW_TYPE_MAP:
        return _ARROW_TYPE_MAP[normalized]
    base_type = column_type_base(normalized)
    if base_type == "DECIMAL":
        return _arrow_decimal_type(normalized)
    parsed = exp.DataType.build(normalized, dialect=_SQLGLOT_DIALECT)
    return _arrow_type_from_sqlglot(parsed)


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
    if context.iceberg_field_ids is not None:
        field_id = context.iceberg_field_ids.get(column.name)
        if field_id is not None:
            field_metadata["codeintel.iceberg_field_id"] = field_id
    return field_metadata


def _schema_metadata(context: _SchemaMetadataContext) -> dict[str, object]:
    schema_metadata: dict[str, object] = {
        "codeintel.table_key": context.table_schema.table_key,
        "codeintel.schema_hash": context.schema_hash_value,
        "codeintel.schema_digest": context.schema_digest,
        "codeintel.primary_key": list(context.table_schema.primary_key),
        "codeintel.schema_contract_version": context.contract_version,
        "codeintel.extras_policy": context.extras_policy,
        "codeintel.extras_column": context.extras_column,
    }
    if context.extras_schema:
        schema_metadata["codeintel.extras_schema"] = dict(context.extras_schema)
    if context.table_schema.description is not None:
        schema_metadata["codeintel.description"] = context.table_schema.description
    if context.provenance_payload:
        schema_metadata["codeintel.provenance"] = dict(context.provenance_payload)
    if context.iceberg_schema_id is not None:
        schema_metadata["codeintel.iceberg_schema_id"] = context.iceberg_schema_id
    if context.iceberg_name_mapping_digest is not None:
        schema_metadata["codeintel.iceberg_name_mapping_digest"] = (
            context.iceberg_name_mapping_digest
        )
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
    resolved_metadata = _normalize_contract_metadata(metadata)
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
        iceberg_field_ids=resolved_metadata.iceberg_field_ids,
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

    schema_context = _SchemaMetadataContext(
        table_schema=table_schema,
        schema_hash_value=resolved_hash,
        schema_digest=resolved_digest,
        provenance_payload=provenance_payload,
        contract_version=resolved_metadata.contract_version or ARROW_SCHEMA_CONTRACT_VERSION,
        extras_policy=resolved_metadata.extras_policy or DEFAULT_EXTRAS_POLICY,
        extras_column=resolved_metadata.extras_column or DEFAULT_EXTRAS_COLUMN,
        extras_schema=resolved_metadata.extras_schema,
        iceberg_schema_id=resolved_metadata.iceberg_schema_id,
        iceberg_name_mapping_digest=resolved_metadata.iceberg_name_mapping_digest,
    )
    schema_metadata = _schema_metadata(schema_context)

    return pa.schema(fields, metadata=_encode_metadata(schema_metadata))


def apply_contract_metadata_to_arrow_schema(
    *,
    arrow_schema: pa.Schema,
    table_schema: TableSchema,
    metadata: ArrowSchemaMetadata | None = None,
) -> pa.Schema:
    """Apply CodeIntel contract metadata to an existing Arrow schema.

    Parameters
    ----------
    arrow_schema
        Arrow schema to annotate.
    table_schema
        TableSchema defining canonical column metadata.
    metadata
        Optional contract metadata overrides.

    Returns
    -------
    pa.Schema
        Arrow schema with contract metadata applied.
    """
    resolved_metadata = _normalize_contract_metadata(metadata)
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
        iceberg_field_ids=resolved_metadata.iceberg_field_ids,
    )
    schema_context = _SchemaMetadataContext(
        table_schema=table_schema,
        schema_hash_value=resolved_hash,
        schema_digest=resolved_digest,
        provenance_payload=provenance_payload,
        contract_version=resolved_metadata.contract_version or ARROW_SCHEMA_CONTRACT_VERSION,
        extras_policy=resolved_metadata.extras_policy or DEFAULT_EXTRAS_POLICY,
        extras_column=resolved_metadata.extras_column or DEFAULT_EXTRAS_COLUMN,
        extras_schema=resolved_metadata.extras_schema,
        iceberg_schema_id=resolved_metadata.iceberg_schema_id,
        iceberg_name_mapping_digest=resolved_metadata.iceberg_name_mapping_digest,
    )
    schema_metadata = _schema_metadata(schema_context)

    fields: list[pa.Field] = []
    for column in table_schema.columns:
        try:
            field = arrow_schema.field(column.name)
        except KeyError:
            field = pa.field(
                column.name,
                _arrow_type_for_column_type(column.type),
                nullable=column.nullable,
            )
        field_metadata = _field_metadata(column, field_context)
        fields.append(field.with_metadata(_merge_metadata(field.metadata, field_metadata)))
    return pa.schema(fields, metadata=_merge_metadata(arrow_schema.metadata, schema_metadata))


def update_arrow_schema_metadata(
    *,
    schema: pa.Schema,
    updates: Mapping[str, object],
) -> pa.Schema:
    """Update schema-level Arrow metadata with CodeIntel keys.

    Parameters
    ----------
    schema
        Arrow schema to update.
    updates
        Metadata updates to merge into the schema metadata mapping.

    Returns
    -------
    pa.Schema
        Updated Arrow schema.
    """
    if not updates:
        return schema
    filtered = {key: value for key, value in updates.items() if value is not None}
    if not filtered:
        return schema
    merged = _merge_metadata(schema.metadata, filtered)
    if merged is None:
        return schema.remove_metadata()
    return schema.with_metadata(merged)


def arrow_contract_for_table_schema(
    *,
    table_schema: TableSchema,
    metadata: ArrowSchemaMetadata | None = None,
) -> pa.Schema:
    """Return a canonical Arrow schema contract for a table schema.

    Parameters
    ----------
    table_schema
        Source TableSchema.
    metadata
        Optional contract metadata overrides.

    Returns
    -------
    pa.Schema
        Arrow schema with contract metadata applied.
    """
    return arrow_schema_from_table_schema(table_schema=table_schema, metadata=metadata)


def _key_roles(schema: TableSchema) -> dict[str, str]:
    roles: dict[str, str] = {}
    for col in schema.primary_key:
        roles[col] = "primary_key"
    for index in schema.indexes:
        role = "unique_key" if index.unique else "index"
        for column in index.columns:
            roles.setdefault(column, role)
    return roles


def _encode_metadata(metadata: Mapping[str, object]) -> dict[bytes, bytes] | None:
    if not metadata:
        return None
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


def _merge_metadata(
    existing: Mapping[bytes, bytes] | None,
    updates: Mapping[str, object],
) -> dict[bytes, bytes] | None:
    decoded = _decode_metadata(existing)
    merged = dict(decoded)
    for key, value in updates.items():
        if value is None:
            continue
        merged[key] = value
    return _encode_metadata(merged)


def _decode_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, object]:
    if not metadata:
        return {}
    decoded: dict[str, object] = {}
    for key, raw in metadata.items():
        key_str = key.decode("utf-8")
        raw_str = raw.decode("utf-8")
        decoded[key_str] = _decode_metadata_value(raw_str)
    return decoded


def _decode_metadata_value(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _resolve_table_key(*, table_key: str | None, metadata: Mapping[str, object]) -> str:
    meta_value = metadata.get("codeintel.table_key")
    resolved = table_key
    if meta_value is not None:
        if not isinstance(meta_value, str):
            msg = (
                "Arrow schema metadata codeintel.table_key must be a string, "
                f"got {type(meta_value)}"
            )
            raise TypeError(msg)
        if resolved is None:
            resolved = meta_value
        elif resolved != meta_value:
            msg = f"Arrow schema table_key mismatch: {resolved!r} != {meta_value!r}"
            raise ValueError(msg)
    if resolved is None:
        msg = "table_key is required when Arrow schema metadata lacks codeintel.table_key"
        raise ValueError(msg)
    validate_table_key(resolved)
    return resolved


def table_schema_from_arrow_schema(
    *,
    arrow_schema: pa.Schema,
    table_key: str | None = None,
) -> TableSchema:
    """Convert a PyArrow schema into a TableSchema.

    Parameters
    ----------
    arrow_schema
        PyArrow schema to convert.
    table_key
        Optional table key override. When omitted, use `codeintel.table_key`
        metadata from the Arrow schema.

    Returns
    -------
    TableSchema
        TableSchema derived from the Arrow schema.
    """
    metadata = _decode_metadata(arrow_schema.metadata)
    _validate_contract_metadata(metadata)
    resolved_key = _resolve_table_key(table_key=table_key, metadata=metadata)
    schema_name, table_name = split_table_key(resolved_key)
    table_description = _metadata_str(metadata, "codeintel.description")
    primary_key = _primary_key_from_metadata(metadata)
    columns = [_column_from_field(field) for field in arrow_schema]
    if not primary_key:
        primary_key = _primary_key_from_fields(arrow_schema)
    return TableSchema(
        schema=schema_name,
        name=table_name,
        columns=columns,
        primary_key=primary_key,
        description=table_description,
    )


def table_schema_from_polars_schema(*, polars_schema: pl.Schema, table_key: str) -> TableSchema:
    """Convert a Polars schema into a TableSchema.

    Parameters
    ----------
    polars_schema
        Polars schema to convert.
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema
        TableSchema derived from the Polars schema.
    """
    return table_schema_from_arrow_schema(
        arrow_schema=polars_schema.to_arrow(),
        table_key=table_key,
    )


def table_schema_from_polars_dataframe(*, frame: pl.DataFrame, table_key: str) -> TableSchema:
    """Convert a Polars DataFrame into a TableSchema.

    Parameters
    ----------
    frame
        Polars DataFrame to derive the schema from.
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema
        TableSchema derived from the DataFrame schema.
    """
    return table_schema_from_polars_schema(polars_schema=frame.schema, table_key=table_key)


def table_schema_from_polars_lazyframe(*, frame: pl.LazyFrame, table_key: str) -> TableSchema:
    """Convert a Polars LazyFrame into a TableSchema.

    Parameters
    ----------
    frame
        Polars LazyFrame to derive the schema from.
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema
        TableSchema derived from the LazyFrame schema.
    """
    return table_schema_from_polars_schema(
        polars_schema=frame.collect_schema(),
        table_key=table_key,
    )


def _column_from_field(field: pa.Field) -> Column:
    metadata = _decode_metadata(field.metadata)
    column_type = _column_type_from_metadata(metadata)
    if column_type is None:
        column_type = _column_type_from_arrow_type(field.type)
    description = _metadata_str(metadata, "codeintel.description")
    return Column(
        name=field.name,
        type=column_type,
        nullable=field.nullable,
        description=description,
    )


def _primary_key_from_metadata(metadata: Mapping[str, object]) -> tuple[str, ...]:
    raw = metadata.get("codeintel.primary_key")
    if raw is None:
        return ()
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return tuple(raw)
    if isinstance(raw, str):
        return (raw,)
    msg = f"Arrow schema metadata codeintel.primary_key must be a list of strings, got {type(raw)}"
    raise TypeError(msg)


def _primary_key_from_fields(schema: pa.Schema) -> tuple[str, ...]:
    primary: list[str] = []
    for field in schema:
        metadata = _decode_metadata(field.metadata)
        role = metadata.get("codeintel.key_role")
        if role == "primary_key":
            primary.append(field.name)
    return tuple(primary)


def _metadata_str(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = f"Arrow metadata {key} must be a string, got {type(value)}"
    raise TypeError(msg)


def _column_type_from_metadata(metadata: Mapping[str, object]) -> ColumnType | None:
    raw = metadata.get("codeintel.column_type")
    if raw is None:
        return None
    if not isinstance(raw, str):
        msg = f"Arrow metadata codeintel.column_type must be a string, got {type(raw)}"
        raise TypeError(msg)
    try:
        return normalize_column_type(raw)
    except ValueError as exc:
        msg = f"Arrow metadata codeintel.column_type is not supported: {raw!r}"
        raise ValueError(msg) from exc


def _column_type_from_arrow_type(dtype: pa.DataType) -> ColumnType:
    """Map Arrow data types to ColumnType.

    Mapping highlights:
    - bool -> BOOLEAN
    - int8/16/32/uint8/16/32 -> INTEGER
    - int64 -> BIGINT
    - uint64 -> DECIMAL(38,0)
    - float -> DOUBLE
    - decimal -> DECIMAL (DECIMAL(38,0) when precision=38 and scale=0)
    - timestamp -> TIMESTAMP/TIMESTAMPTZ
    - date/time/duration -> TIMESTAMP
    - string/string_view -> VARCHAR
    - binary/binary_view -> VARCHAR
    - list/struct/map/union -> LIST/STRUCT/MAP/UNION types

    Returns
    -------
    ColumnType
        ColumnType corresponding to the Arrow type.

    Raises
    ------
    ValueError
        If the Arrow type cannot be mapped.
    """
    for resolver in _ARROW_TYPE_RESOLVERS:
        resolved = resolver(dtype)
        if resolved is not None:
            return resolved
    msg = f"Unsupported Arrow type for TableSchema: {dtype}"
    raise ValueError(msg)


def _boolean_column_type(dtype: pa.DataType) -> ColumnType | None:
    return "BOOLEAN" if pa.types.is_boolean(dtype) else None


def _integer_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_integer(dtype):
        return None
    if pa.types.is_int64(dtype):
        return "BIGINT"
    if pa.types.is_uint64(dtype):
        return normalize_column_type("DECIMAL(38,0)")
    return "INTEGER"


def _floating_column_type(dtype: pa.DataType) -> ColumnType | None:
    return "DOUBLE" if pa.types.is_floating(dtype) else None


def _decimal_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_decimal(dtype):
        return None
    if pa.types.is_decimal128(dtype):
        decimal = cast("pa.Decimal128Type", dtype)
        if (
            decimal.precision == _DECIMAL_DEFAULT_PRECISION
            and decimal.scale == _DECIMAL_DEFAULT_SCALE
        ):
            return normalize_column_type("DECIMAL(38,0)")
    return normalize_column_type(f"DECIMAL({dtype.precision},{dtype.scale})")


def _timestamp_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_timestamp(dtype):
        return None
    timestamp = cast("pa.TimestampType", dtype)
    if timestamp.tz:
        return "TIMESTAMPTZ"
    return "TIMESTAMP"


def _temporal_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not _is_temporal_type(dtype):
        return None
    return "TIMESTAMP"


def _string_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not _is_string_type(dtype):
        return None
    return "VARCHAR"


def _binary_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not _is_binary_type(dtype):
        return None
    return "VARCHAR"


def _dictionary_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_dictionary(dtype):
        return None
    dict_type = cast("pa.DictionaryType", dtype)
    return _column_type_from_arrow_type(dict_type.value_type)


def _struct_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_struct(dtype):
        return None
    struct_type = cast("pa.StructType", dtype)
    parts = [f"{field.name} {_column_type_from_arrow_type(field.type)}" for field in struct_type]
    return normalize_column_type(f"STRUCT({', '.join(parts)})")


def _list_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not _is_list_type(dtype):
        return None
    list_type = cast("pa.ListType", dtype)
    return normalize_column_type(f"LIST({_column_type_from_arrow_type(list_type.value_type)})")


def _map_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_map(dtype):
        return None
    map_type = cast("pa.MapType", dtype)
    key_type = _column_type_from_arrow_type(map_type.key_type)
    value_type = _column_type_from_arrow_type(map_type.item_type)
    return normalize_column_type(f"MAP({key_type}, {value_type})")


def _union_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_union(dtype):
        return None
    union_type = cast("pa.UnionType", dtype)
    parts = [f"{field.name} {_column_type_from_arrow_type(field.type)}" for field in union_type]
    return normalize_column_type(f"UNION({', '.join(parts)})")


def _null_column_type(dtype: pa.DataType) -> ColumnType | None:
    return "VARCHAR" if pa.types.is_null(dtype) else None


def _is_temporal_type(dtype: pa.DataType) -> bool:
    checks = (pa.types.is_date, pa.types.is_time, pa.types.is_duration)
    return any(check(dtype) for check in checks)


def _is_string_type(dtype: pa.DataType) -> bool:
    checks = (pa.types.is_string, pa.types.is_large_string, pa.types.is_string_view)
    return any(check(dtype) for check in checks)


def _is_binary_type(dtype: pa.DataType) -> bool:
    checks = (
        pa.types.is_binary,
        pa.types.is_large_binary,
        pa.types.is_fixed_size_binary,
        pa.types.is_binary_view,
    )
    return any(check(dtype) for check in checks)


def _is_list_type(dtype: pa.DataType) -> bool:
    checks = [
        pa.types.is_list,
        pa.types.is_large_list,
        pa.types.is_fixed_size_list,
    ]
    list_view = getattr(pa.types, "is_list_view", None)
    if callable(list_view):
        checks.append(list_view)
    large_list_view = getattr(pa.types, "is_large_list_view", None)
    if callable(large_list_view):
        checks.append(large_list_view)
    return any(check(dtype) for check in checks)


_ARROW_TYPE_RESOLVERS: tuple[
    Callable[[pa.DataType], ColumnType | None],
    ...,
] = (
    _boolean_column_type,
    _integer_column_type,
    _floating_column_type,
    _decimal_column_type,
    _timestamp_column_type,
    _temporal_column_type,
    _string_column_type,
    _binary_column_type,
    _dictionary_column_type,
    _struct_column_type,
    _list_column_type,
    _map_column_type,
    _union_column_type,
    _null_column_type,
)


def _validate_contract_metadata(metadata: Mapping[str, object]) -> None:
    version = metadata.get("codeintel.schema_contract_version")
    if version is not None:
        if not isinstance(version, str):
            msg = (
                "Arrow schema metadata codeintel.schema_contract_version must be a string, "
                f"got {type(version)}"
            )
            raise TypeError(msg)
        if version != ARROW_SCHEMA_CONTRACT_VERSION:
            msg = (
                "Arrow schema contract version mismatch: "
                f"{version!r} != {ARROW_SCHEMA_CONTRACT_VERSION!r}"
            )
            raise ValueError(msg)
    extras_policy = metadata.get("codeintel.extras_policy")
    if extras_policy is not None:
        if not isinstance(extras_policy, str):
            msg = (
                "Arrow schema metadata codeintel.extras_policy must be a string, "
                f"got {type(extras_policy)}"
            )
            raise TypeError(msg)
        if extras_policy not in EXTRAS_POLICIES:
            msg = f"Arrow schema extras_policy is not supported: {extras_policy!r}"
            raise ValueError(msg)
    extras_column = metadata.get("codeintel.extras_column")
    if extras_column is not None and not isinstance(extras_column, str):
        msg = (
            "Arrow schema metadata codeintel.extras_column must be a string, "
            f"got {type(extras_column)}"
        )
        raise TypeError(msg)
    extras_schema = metadata.get("codeintel.extras_schema")
    if extras_schema is not None and not isinstance(extras_schema, Mapping):
        msg = (
            "Arrow schema metadata codeintel.extras_schema must be a mapping, "
            f"got {type(extras_schema)}"
        )
        raise TypeError(msg)


def encode_schema_ipc(schema: pa.Schema) -> bytes:
    """Serialize an Arrow schema to IPC bytes.

    Returns
    -------
    bytes
        IPC-encoded schema bytes.

    Raises
    ------
    TypeError
        If IPC schema serialization is unavailable.
    """
    serialize_schema = getattr(pa.ipc, "serialize_schema", None)
    if callable(serialize_schema):
        buffer = cast("pa.Buffer", serialize_schema(schema))
        return buffer.to_pybytes()
    write_schema = getattr(pa.ipc, "write_schema", None)
    if callable(write_schema):
        sink = pa.BufferOutputStream()
        write_schema(schema, sink)
        return sink.getvalue().to_pybytes()
    new_stream = getattr(pa.ipc, "new_stream", None)
    if callable(new_stream):
        sink = pa.BufferOutputStream()
        writer = new_stream(sink, schema)
        close = getattr(writer, "close", None)
        if callable(close):
            close()
        return sink.getvalue().to_pybytes()
    msg = "Arrow IPC schema serialization is unavailable"
    raise TypeError(msg)


def encode_schema_ipc_b64(schema: pa.Schema) -> str:
    """Serialize an Arrow schema to base64-encoded IPC bytes.

    Returns
    -------
    str
        Base64-encoded IPC schema payload.
    """
    return base64.b64encode(encode_schema_ipc(schema)).decode("ascii")


def decode_schema_ipc(payload: bytes) -> pa.Schema:
    """Deserialize IPC bytes into an Arrow schema.

    Returns
    -------
    pyarrow.Schema
        Decoded Arrow schema.
    """
    buffer = pa.py_buffer(payload)
    return pa.ipc.read_schema(pa.BufferReader(buffer))


def decode_schema_ipc_b64(payload: str) -> pa.Schema:
    """Deserialize a base64 IPC payload into an Arrow schema.

    Returns
    -------
    pyarrow.Schema
        Decoded Arrow schema.

    Raises
    ------
    ValueError
        If the payload is not valid base64.
    """
    try:
        raw = base64.b64decode(payload)
    except (ValueError, binascii.Error) as exc:
        msg = "Invalid base64 IPC payload"
        raise ValueError(msg) from exc
    return decode_schema_ipc(raw)


def try_decode_schema_ipc(payload: bytes) -> pa.Schema | None:
    """Best-effort IPC schema decode, returning None on failure.

    Returns
    -------
    pyarrow.Schema | None
        Decoded Arrow schema, or None when decoding fails.
    """
    try:
        return decode_schema_ipc(payload)
    except (ValueError, OSError, pa.ArrowInvalid):
        return None


def try_decode_schema_ipc_b64(payload: str) -> pa.Schema | None:
    """Best-effort base64 IPC schema decode, returning None on failure.

    Returns
    -------
    pyarrow.Schema | None
        Decoded Arrow schema, or None when decoding fails.
    """
    try:
        raw = base64.b64decode(payload)
    except (ValueError, binascii.Error):
        return None
    return try_decode_schema_ipc(raw)


def arrow_schema_hash(schema: pa.Schema) -> str | None:
    """Return the CodeIntel schema hash embedded in Arrow metadata.

    Returns
    -------
    str | None
        Embedded schema hash, if present.
    """
    return _schema_metadata_value(schema, "codeintel.schema_hash")


def arrow_schema_digest(schema: pa.Schema) -> str | None:
    """Return the schema digest embedded in Arrow metadata.

    Returns
    -------
    str | None
        Embedded schema digest, if present.
    """
    return _schema_metadata_value(schema, "codeintel.schema_digest")


def _schema_metadata_value(schema: pa.Schema, key: str) -> str | None:
    metadata = schema.metadata
    if not metadata:
        return None
    raw = metadata.get(key.encode("utf-8"))
    if raw is None:
        return None
    return raw.decode("utf-8")


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
    return {
        "mode": policy.mode,
        "replace_scope": policy.replace_scope,
        "conflict_columns": list(policy.conflict_columns or ()),
        "update_columns": list(policy.update_columns or ()),
        "hash_column": policy.hash_column,
        "use_staging": policy.use_staging,
    }


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
    primary_key = tuple(
        _require_str(item, field="primary_key[]") for item in primary_key_obj if item
    )

    indexes_obj = obj.get("indexes", [])
    if not isinstance(indexes_obj, list):
        msg = "Expected list for indexes"
        raise TypeError(msg)
    indexes = [index_from_json_obj(item) for item in indexes_obj if isinstance(item, Mapping)]
    if len(indexes) != len(indexes_obj):
        msg = "Invalid index object in indexes"
        raise TypeError(msg)

    description = obj.get("description")
    description_value = description if isinstance(description, str) else None

    write_policy_obj = obj.get("write_policy")
    write_policy = _parse_write_policy(write_policy_obj) if write_policy_obj is not None else None

    return TableSchema(
        schema=schema,
        name=name,
        columns=columns,
        primary_key=primary_key,
        indexes=tuple(indexes),
        description=description_value,
        write_policy=write_policy,
    )


def to_json_schema(
    table_schema: TableSchema,
    *,
    schema_id: str | None = None,
    include_description: bool = True,
) -> dict[str, Any]:
    """Convert a TableSchema into JSON Schema 2020-12.

    Returns
    -------
    dict[str, Any]
        JSON Schema representation of the table schema.
    """
    return json_schema_from_table_schema(
        table_schema,
        schema_id=schema_id,
        include_description=include_description,
    )


def from_json_schema(
    schema_obj: Mapping[str, Any],
    *,
    table_key: str | None = None,
) -> TableSchema:
    """Convert a JSON Schema (2020-12) object into a TableSchema.

    Returns
    -------
    TableSchema
        Parsed TableSchema derived from the JSON schema object.

    Raises
    ------
    TypeError
        If the JSON schema structure is invalid.
    ValueError
        If the table key is missing or invalid.
    """
    resolved_key = table_key or _table_key_from_json_schema(schema_obj)
    if resolved_key is None:
        msg = "table_key is required when JSON schema lacks a title"
        raise ValueError(msg)
    validate_table_key(resolved_key)
    schema_name, table_name = split_table_key(resolved_key)
    properties = schema_obj.get("properties")
    if not isinstance(properties, Mapping):
        msg = "JSON schema properties must be an object"
        raise TypeError(msg)
    required_raw = schema_obj.get("required", [])
    if not isinstance(required_raw, list):
        msg = "JSON schema required must be a list"
        raise TypeError(msg)
    required = {item for item in required_raw if isinstance(item, str)}
    columns: list[Column] = []
    for name, prop in properties.items():
        if not isinstance(name, str):
            msg = "JSON schema property names must be strings"
            raise TypeError(msg)
        if not isinstance(prop, Mapping):
            msg = f"JSON schema for {name!r} must be an object"
            raise TypeError(msg)
        column_type, nullable = _column_from_json_schema(prop, required, name)
        description = prop.get("description")
        description_value = description if isinstance(description, str) else None
        columns.append(
            Column(
                name=name,
                type=column_type,
                nullable=nullable,
                description=description_value,
            )
        )
    return TableSchema(
        schema=schema_name,
        name=table_name,
        columns=columns,
    )


def _table_key_from_json_schema(schema_obj: Mapping[str, Any]) -> str | None:
    title = schema_obj.get("title")
    return title if isinstance(title, str) else None


def _column_from_json_schema(
    prop: Mapping[str, Any],
    required: set[str],
    name: str,
) -> tuple[ColumnType, bool]:
    raw_type, nullable_by_type = _json_schema_type(prop)
    nullable = name not in required or nullable_by_type
    column_type: ColumnType
    if raw_type is None:
        column_type = "JSON"
    elif raw_type == "string" and prop.get("format") == "date-time":
        column_type = "TIMESTAMP"
    elif raw_type == "array":
        column_type = _list_type_from_items(prop)
    else:
        json_type_map: dict[str, ColumnType] = {
            "boolean": "BOOLEAN",
            "integer": "INTEGER",
            "number": "DOUBLE",
            "string": "VARCHAR",
            "object": "STRUCT",
        }
        mapped_type = json_type_map.get(raw_type)
        if mapped_type is None:
            msg = f"Unsupported JSON schema type: {raw_type!r}"
            raise ValueError(msg)
        column_type = mapped_type
    return column_type, nullable


def _json_schema_type(prop: Mapping[str, Any]) -> tuple[str | None, bool]:
    raw_type = prop.get("type")
    if raw_type is None:
        return None, False
    if isinstance(raw_type, str):
        return raw_type, False
    if isinstance(raw_type, list):
        types = [item for item in raw_type if isinstance(item, str)]
        nullable = "null" in types
        candidates = [item for item in types if item != "null"]
        if not candidates:
            return None, nullable
        if len(candidates) > 1:
            msg = f"Ambiguous JSON schema types: {candidates}"
            raise ValueError(msg)
        return candidates[0], nullable
    msg = "JSON schema type must be a string or list"
    raise TypeError(msg)


def _list_type_from_items(prop: Mapping[str, Any]) -> ColumnType:
    items = prop.get("items")
    if not isinstance(items, Mapping):
        return "LIST(JSON)"
    inner_type, _ = _column_from_json_schema(items, set(), "items")
    return normalize_column_type(f"LIST({inner_type})")


def to_arrow_schema(
    table_schema: TableSchema,
    *,
    metadata: ArrowSchemaMetadata | None = None,
) -> pa.Schema:
    """Alias for arrow_schema_from_table_schema.

    Returns
    -------
    pyarrow.Schema
        Arrow schema generated from the table schema.
    """
    return arrow_schema_from_table_schema(table_schema=table_schema, metadata=metadata)


def from_arrow_schema(
    arrow_schema: pa.Schema,
    *,
    table_key: str | None = None,
) -> TableSchema:
    """Alias for table_schema_from_arrow_schema.

    Returns
    -------
    TableSchema
        Table schema derived from the Arrow schema.
    """
    return table_schema_from_arrow_schema(arrow_schema=arrow_schema, table_key=table_key)


__all__ = [
    "ARROW_FIELD_METADATA_KEYS",
    "ARROW_SCHEMA_CONTRACT_VERSION",
    "ARROW_SCHEMA_METADATA_KEYS",
    "DEFAULT_EXTRAS_COLUMN",
    "DEFAULT_EXTRAS_POLICY",
    "EXTRAS_POLICIES",
    "ArrowSchemaMetadata",
    "ArrowSchemaProvenance",
    "ExtrasPolicy",
    "apply_contract_metadata_to_arrow_schema",
    "arrow_contract_for_table_schema",
    "arrow_schema_digest",
    "arrow_schema_from_table_schema",
    "arrow_schema_hash",
    "column_from_json_obj",
    "column_to_json_obj",
    "decode_schema_ipc",
    "decode_schema_ipc_b64",
    "encode_schema_ipc",
    "encode_schema_ipc_b64",
    "from_arrow_schema",
    "from_json_schema",
    "index_from_json_obj",
    "index_to_json_obj",
    "json_schema_from_table_schema",
    "table_schema_from_arrow_schema",
    "table_schema_from_json_obj",
    "table_schema_from_polars_dataframe",
    "table_schema_from_polars_lazyframe",
    "table_schema_from_polars_schema",
    "table_schema_to_json_obj",
    "to_arrow_schema",
    "to_json_schema",
    "try_decode_schema_ipc",
    "try_decode_schema_ipc_b64",
    "update_arrow_schema_metadata",
]
