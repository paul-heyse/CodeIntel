"""PyArrow schema rendering for core TableSchema definitions."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

import pyarrow as pa
from sqlglot import exp

from codeintel.core.columnar.schema_metadata import encode_metadata
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import (
    COMPLEX_TYPE_BASES,
    Column,
    ColumnType,
    TableSchema,
    column_type_base,
    normalize_column_type,
)

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
)
_ARROW_TYPE_MAP: dict[str, pa.DataType] = {
    "BOOLEAN": pa.bool_(),
    "INTEGER": pa.int32(),
    "BIGINT": pa.int64(),
    "DOUBLE": pa.float64(),
    "VARCHAR": pa.string(),
    "BLOB": pa.binary(),
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
_SQLGLOT_BINARY_TYPES = _sqlglot_types(
    "BINARY",
    "VARBINARY",
    "BLOB",
    "BYTEA",
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


@dataclass(frozen=True, slots=True)
class _FieldMetadataContext:
    schema_hash_value: str
    schema_digest: str
    provenance_payload: Mapping[str, str]
    column_lineage: Mapping[str, Iterable[tuple[str, str]]] | None
    pii_by_column: Mapping[str, str] | None
    key_roles: Mapping[str, str]


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
        msg = f"LIST type missing value type: {data_type}"
        raise ValueError(msg)
    value_type = data_type.expressions[0]
    if not isinstance(value_type, exp.DataType):
        msg = f"LIST type value is not a DataType: {data_type}"
        raise TypeError(msg)
    return pa.list_(_arrow_type_from_sqlglot(value_type))


def _arrow_map_type(data_type: exp.DataType) -> pa.DataType:
    if len(data_type.expressions) < _MAP_PARAM_COUNT:
        msg = f"MAP type missing key/value types: {data_type}"
        raise ValueError(msg)
    key_type = data_type.expressions[0]
    value_type = data_type.expressions[1]
    if not isinstance(key_type, exp.DataType) or not isinstance(value_type, exp.DataType):
        msg = f"MAP type key/value are not DataType nodes: {data_type}"
        raise TypeError(msg)
    return pa.map_(
        _arrow_type_from_sqlglot(key_type),
        _arrow_type_from_sqlglot(value_type),
    )


def _arrow_struct_type(data_type: exp.DataType) -> pa.DataType:
    fields: list[pa.Field] = []
    for expr in data_type.expressions:
        if not isinstance(expr, exp.ColumnDef):
            msg = f"STRUCT field is not a ColumnDef: {data_type}"
            raise TypeError(msg)
        identifier = expr.this
        if not isinstance(identifier, exp.Identifier):
            msg = f"STRUCT field name is invalid: {data_type}"
            raise TypeError(msg)
        field_type = expr.kind
        if not isinstance(field_type, exp.DataType):
            msg = f"STRUCT field type is invalid: {data_type}"
            raise TypeError(msg)
        fields.append(
            pa.field(
                identifier.name,
                _arrow_type_from_sqlglot(field_type),
                nullable=True,
            )
        )
    return pa.struct(fields)


def _arrow_union_type(data_type: exp.DataType) -> pa.DataType:
    fields: list[pa.Field] = []
    for expr in data_type.expressions:
        if not isinstance(expr, exp.ColumnDef):
            msg = f"UNION field is not a ColumnDef: {data_type}"
            raise TypeError(msg)
        identifier = expr.this
        if not isinstance(identifier, exp.Identifier):
            msg = f"UNION field name is invalid: {data_type}"
            raise TypeError(msg)
        field_type = expr.kind
        if not isinstance(field_type, exp.DataType):
            msg = f"UNION field type is invalid: {data_type}"
            raise TypeError(msg)
        fields.append(
            pa.field(
                identifier.name,
                _arrow_type_from_sqlglot(field_type),
                nullable=True,
            )
        )
    return pa.union(fields, mode="sparse")


ArrowTypeHandler = Callable[[exp.DataType], pa.DataType]


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


def _arrow_binary_type(_: exp.DataType) -> pa.DataType:
    return pa.binary()


def _arrow_timestamp_type(_: exp.DataType) -> pa.DataType:
    return pa.timestamp("us")


def _arrow_timestamptz_type(_: exp.DataType) -> pa.DataType:
    return pa.timestamp("us", tz="UTC")


def _arrow_big_decimal_type(_: exp.DataType) -> pa.DataType:
    return _decimal_for_precision_scale(_DECIMAL_DEFAULT_PRECISION, _DECIMAL_DEFAULT_SCALE)


def _arrow_decimal_type_from_sqlglot(data_type: exp.DataType) -> pa.DataType:
    precision, scale = _decimal_precision_scale(data_type)
    return _decimal_for_precision_scale(
        precision or _DECIMAL_DEFAULT_PRECISION,
        scale or _DECIMAL_DEFAULT_SCALE,
    )


def _add_type_handlers(
    handlers: dict[exp.DataType.Type, ArrowTypeHandler],
    types: Iterable[exp.DataType.Type],
    handler: ArrowTypeHandler,
) -> None:
    for type_value in types:
        handlers[type_value] = handler


def _build_sqlglot_type_handlers() -> dict[exp.DataType.Type, ArrowTypeHandler]:
    handlers: dict[exp.DataType.Type, ArrowTypeHandler] = {}
    _add_type_handlers(handlers, _SQLGLOT_BOOLEAN_TYPES, _arrow_bool_type)
    _add_type_handlers(handlers, _SQLGLOT_INTEGER_TYPES, _arrow_int32_type)
    _add_type_handlers(handlers, _SQLGLOT_BIGINT_TYPES, _arrow_int64_type)
    _add_type_handlers(handlers, _SQLGLOT_BIGINT_DECIMAL_TYPES, _arrow_big_decimal_type)
    _add_type_handlers(handlers, _SQLGLOT_DECIMAL_TYPES, _arrow_decimal_type_from_sqlglot)
    _add_type_handlers(handlers, _SQLGLOT_FLOAT_TYPES, _arrow_float64_type)
    _add_type_handlers(handlers, _SQLGLOT_STRING_TYPES, _arrow_string_type)
    _add_type_handlers(handlers, _SQLGLOT_BINARY_TYPES, _arrow_binary_type)
    _add_type_handlers(handlers, _SQLGLOT_JSON_TYPES, _arrow_string_type)
    _add_type_handlers(handlers, _SQLGLOT_TIMESTAMPTZ_TYPES, _arrow_timestamptz_type)
    _add_type_handlers(handlers, _SQLGLOT_TIMESTAMP_TYPES, _arrow_timestamp_type)
    _add_type_handlers(handlers, _SQLGLOT_LIST_TYPES, _arrow_list_type)
    _add_type_handlers(handlers, _SQLGLOT_MAP_TYPES, _arrow_map_type)
    _add_type_handlers(handlers, _SQLGLOT_STRUCT_TYPES, _arrow_struct_type)
    _add_type_handlers(handlers, _SQLGLOT_UNION_TYPES, _arrow_union_type)
    return handlers


_SQLGLOT_TYPE_HANDLERS = _build_sqlglot_type_handlers()


def _arrow_type_from_sqlglot(data_type: exp.DataType) -> pa.DataType:
    handler = _SQLGLOT_TYPE_HANDLERS.get(data_type.this)
    if handler is None:
        msg = f"Unsupported SQLGlot type for Arrow conversion: {data_type}"
        raise ValueError(msg)
    return handler(data_type)


def _arrow_type_for_column_type(column_type: ColumnType) -> pa.DataType:
    normalized = normalize_column_type(str(column_type))
    base = column_type_base(normalized)
    if base == "DECIMAL":
        return _arrow_decimal_type(normalized.upper())
    if base in COMPLEX_TYPE_BASES:
        try:
            data_type = exp.DataType.build(normalized, dialect=_SQLGLOT_DIALECT)
        except (TypeError, ValueError) as exc:
            msg = f"Unsupported complex column type: {column_type}"
            raise ValueError(msg) from exc
        return _arrow_type_from_sqlglot(data_type)
    return _ARROW_TYPE_MAP.get(base, pa.string())


def arrow_type_for_column_type(column_type: ColumnType) -> pa.DataType:
    """Return the PyArrow type for a normalized ColumnType.

    Parameters
    ----------
    column_type
        Column type string to convert.

    Returns
    -------
    pyarrow.DataType
        Arrow type corresponding to the column type.
    """
    return _arrow_type_for_column_type(column_type)


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
    )
    fields = [
        pa.field(
            column.name,
            _arrow_type_for_column_type(column.type),
            nullable=column.nullable,
            metadata=encode_metadata(_field_metadata(column, field_context)),
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
    )
    schema_metadata = _schema_metadata(schema_context)

    return pa.schema(fields, metadata=encode_metadata(schema_metadata))


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
    "arrow_contract_for_table_schema",
    "arrow_schema_from_table_schema",
    "arrow_type_for_column_type",
]
