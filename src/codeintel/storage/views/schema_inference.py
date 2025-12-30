"""SQLGlot-based schema derivation for view builders."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer import annotate_types, qualify
from sqlglot.schema import MappingSchema

from codeintel.core.schemas.primitives import Column, ColumnType, TableSchema, normalize_column_type
from codeintel.core.schemas.provider import SchemaProvider
from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.views.dependencies import build_dependency_graph_from_sql, toposort
from codeintel.storage.views.discovery import DiscoveredViewBuilder, discover_view_builders
from codeintel.storage.views.inventory import view_builder_modules

if TYPE_CHECKING:
    from types import ModuleType

_logger = logging.getLogger(__name__)

_DECIMAL_INT_PRECISION = 38
_DECIMAL_INT_SCALE = 0
_DECIMAL_PARAM_COUNT = 2

_BOOLEAN_TYPES = {exp.DataType.Type.BOOLEAN}
_INTEGER_TYPES = {
    exp.DataType.Type.INT,
    exp.DataType.Type.SMALLINT,
    exp.DataType.Type.TINYINT,
    exp.DataType.Type.MEDIUMINT,
    exp.DataType.Type.UINT,
    exp.DataType.Type.USMALLINT,
    exp.DataType.Type.UTINYINT,
    exp.DataType.Type.UMEDIUMINT,
}
_BIGINT_TYPES = {
    exp.DataType.Type.BIGINT,
    exp.DataType.Type.BIGSERIAL,
    exp.DataType.Type.SERIAL,
    exp.DataType.Type.SMALLSERIAL,
}
_BIGINT_DECIMAL_TYPES = {
    exp.DataType.Type.UBIGINT,
    exp.DataType.Type.INT128,
    exp.DataType.Type.INT256,
    exp.DataType.Type.UINT128,
    exp.DataType.Type.UINT256,
}
_DECIMAL_TYPES = {
    exp.DataType.Type.DECIMAL,
    exp.DataType.Type.DECIMAL32,
    exp.DataType.Type.DECIMAL64,
    exp.DataType.Type.DECIMAL128,
    exp.DataType.Type.DECIMAL256,
    exp.DataType.Type.BIGDECIMAL,
    exp.DataType.Type.BIGNUM,
    exp.DataType.Type.UDECIMAL,
    exp.DataType.Type.DECFLOAT,
}
_FLOAT_TYPES = {exp.DataType.Type.FLOAT, exp.DataType.Type.DOUBLE, exp.DataType.Type.UDOUBLE}
_STRING_TYPES = {
    exp.DataType.Type.VARCHAR,
    exp.DataType.Type.TEXT,
    exp.DataType.Type.CHAR,
    exp.DataType.Type.BPCHAR,
    exp.DataType.Type.NCHAR,
    exp.DataType.Type.NVARCHAR,
    exp.DataType.Type.NAME,
    exp.DataType.Type.LONGTEXT,
    exp.DataType.Type.MEDIUMTEXT,
    exp.DataType.Type.TINYTEXT,
    exp.DataType.Type.FIXEDSTRING,
    exp.DataType.Type.UUID,
}
_NESTED_TYPES = {
    exp.DataType.Type.ARRAY,
    exp.DataType.Type.LIST,
    exp.DataType.Type.MAP,
    exp.DataType.Type.STRUCT,
    exp.DataType.Type.SET,
    exp.DataType.Type.UNION,
}
_JSON_TYPES = {
    exp.DataType.Type.JSON,
    exp.DataType.Type.JSONB,
    exp.DataType.Type.OBJECT,
    exp.DataType.Type.VARIANT,
    exp.DataType.Type.SUPER,
}
_TIMESTAMPTZ_TYPES = {exp.DataType.Type.TIMESTAMPTZ, exp.DataType.Type.TIMESTAMPLTZ}
_TIMESTAMP_TYPES = {
    exp.DataType.Type.TIMESTAMP,
    exp.DataType.Type.TIMESTAMPNTZ,
    exp.DataType.Type.TIMESTAMP_S,
    exp.DataType.Type.TIMESTAMP_MS,
    exp.DataType.Type.TIMESTAMP_NS,
    exp.DataType.Type.DATETIME,
    exp.DataType.Type.DATETIME2,
    exp.DataType.Type.DATETIME64,
    exp.DataType.Type.SMALLDATETIME,
    exp.DataType.Type.TIME,
    exp.DataType.Type.TIME_NS,
    exp.DataType.Type.TIMETZ,
    exp.DataType.Type.DATE,
    exp.DataType.Type.DATE32,
}


def derive_view_schemas(
    *,
    provider: SchemaProvider,
    view_keys: Iterable[str] | None = None,
    modules: tuple[ModuleType, ...] | None = None,
) -> dict[str, TableSchema]:
    """Derive TableSchema definitions for SQLGlot view builders.

    Parameters
    ----------
    provider
        Schema provider used to supply base table schemas for type inference.
    view_keys
        Optional iterable of view keys to derive; when None, derives all views
        discovered from view builder modules.
    modules
        Optional modules to scan for view builders. Defaults to canonical view modules.

    Returns
    -------
    dict[str, TableSchema]
        Mapping of view table_key to derived TableSchema definitions.
    """
    modules = modules or view_builder_modules()
    builders = discover_view_builders(modules=modules)
    if not builders:
        return {}

    builder_by_key = {builder.table_key: builder for builder in builders}
    view_sql = _view_sql_map(builders)
    if not view_sql:
        return {}

    deps = build_dependency_graph_from_sql(view_sql)
    selected = _expand_view_keys(view_keys or builder_by_key, deps)
    ordered = toposort(selected, deps)
    if not ordered:
        return {}

    schema_map = _schema_mapping(provider)
    mapping_schema = MappingSchema(schema_map)
    view_schemas: dict[str, TableSchema] = {}
    lower_to_key = {key.lower(): key for key in view_sql}

    for view_key_lower in ordered:
        original_key = lower_to_key.get(view_key_lower)
        if original_key is None:
            continue
        builder = builder_by_key.get(original_key)
        if builder is None:
            continue
        try:
            schema = _derive_view_schema(
                builder=builder,
                mapping_schema=mapping_schema,
            )
        except (TypeError, ValueError, ParseError) as exc:
            _logger.debug("Skipping view %s (schema inference failed): %s", original_key, exc)
            continue
        view_schemas[original_key] = schema
        _update_schema_mapping(schema_map, schema)
        mapping_schema = MappingSchema(schema_map)

    return view_schemas


def _view_sql_map(builders: Iterable[DiscoveredViewBuilder]) -> dict[str, str]:
    view_sql: dict[str, str] = {}
    for builder in builders:
        expression = builder.builder()
        if not isinstance(expression, exp.Expression):
            msg = f"View builder {builder.table_key} did not return a SQLGlot expression"
            raise TypeError(msg)
        view_sql[builder.table_key] = expression.sql(dialect=DUCKDB_DIALECT)
    return view_sql


def _expand_view_keys(
    view_keys: Iterable[str],
    deps: Mapping[str, frozenset[str]],
) -> set[str]:
    requested = [key.lower() for key in view_keys]
    selected: set[str] = set()
    stack = list(requested)
    while stack:
        current = stack.pop()
        if current in selected:
            continue
        selected.add(current)
        stack.extend(deps.get(current, ()))
    return selected


def _derive_view_schema(
    *,
    builder: DiscoveredViewBuilder,
    mapping_schema: MappingSchema,
) -> TableSchema:
    expression = builder.builder()
    if not isinstance(expression, exp.Expression):
        msg = f"View builder {builder.table_key} did not return a SQLGlot expression"
        raise TypeError(msg)
    qualified = qualify.qualify(
        expression,
        dialect=DUCKDB_DIALECT,
        schema=mapping_schema,
        validate_qualify_columns=False,
    )
    annotated = annotate_types.annotate_types(
        qualified,
        schema=mapping_schema,
        dialect=DUCKDB_DIALECT,
    )
    select_expr = _resolve_select(annotated)
    if select_expr is None:
        msg = f"Unable to resolve SELECT projection for {builder.table_key}"
        raise ValueError(msg)
    columns = [_column_from_select_expr(expr) for expr in select_expr.expressions]
    schema_name, table_name = split_table_key(builder.table_key)
    return TableSchema(schema=schema_name, name=table_name, columns=columns)


def _resolve_select(expression: exp.Expression) -> exp.Select | None:
    if isinstance(expression, exp.Select):
        return expression
    if isinstance(expression, exp.SetOperation):
        return _resolve_select(expression.this)
    if isinstance(expression, exp.Subquery):
        return _resolve_select(expression.this)
    return expression.find(exp.Select)


def _column_from_select_expr(expr: exp.Expression) -> Column:
    name = expr.alias_or_name
    if not name:
        msg = f"Select expression missing output name: {expr}"
        raise ValueError(msg)
    data_type = getattr(expr, "type", None)
    if not isinstance(data_type, exp.DataType):
        fallback = _infer_fallback_column_type(expr)
        if fallback is None:
            msg = f"Select expression missing type annotation: {expr}"
            raise TypeError(msg)
        column_type = fallback
    elif data_type.this == exp.DataType.Type.UNKNOWN:
        fallback = _infer_fallback_column_type(expr)
        if fallback is None:
            msg = f"Select expression has unknown type: {expr}"
            raise ValueError(msg)
        column_type = fallback
    else:
        column_type = _column_type_from_sqlglot(data_type)
    return Column(name=name, type=column_type, nullable=True)


def _infer_fallback_column_type(expr: exp.Expression) -> ColumnType | None:
    for node in expr.walk():
        if isinstance(node, exp.Anonymous):
            name = node.name or node.this
            if isinstance(name, str) and name.strip().upper() == "TO_JSON":
                return "JSON"
        if isinstance(node, (exp.Array, exp.List, exp.Struct, exp.Map)):
            return "JSON"
    return None


def _column_type_from_sqlglot(data_type: exp.DataType) -> ColumnType:
    for resolver in _SQLGLOT_TYPE_RESOLVERS:
        resolved = resolver(data_type)
        if resolved is not None:
            return resolved
    msg = f"Unsupported SQLGlot type for view schema inference: {data_type}"
    raise ValueError(msg)


def _decimal_column_type(data_type: exp.DataType) -> ColumnType:
    precision, scale = _decimal_precision_scale(data_type)
    if precision == _DECIMAL_INT_PRECISION and scale == _DECIMAL_INT_SCALE:
        return "DECIMAL(38,0)"
    return "DECIMAL"


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


def _schema_mapping(provider: SchemaProvider) -> dict[str, dict[str, dict[str, str]]]:
    mapping: dict[str, dict[str, dict[str, str]]] = {}
    for schema in provider.iter_table_schemas():
        schema_map = mapping.setdefault(schema.schema, {})
        table_map = schema_map.setdefault(schema.name, {})
        for column in schema.columns:
            table_map[column.name] = str(column.type)
    return mapping


def _update_schema_mapping(
    mapping: dict[str, dict[str, dict[str, str]]],
    schema: TableSchema,
) -> None:
    schema_map = mapping.setdefault(schema.schema, {})
    table_map = schema_map.setdefault(schema.name, {})
    for column in schema.columns:
        table_map[column.name] = str(column.type)


def _bool_type(data_type: exp.DataType) -> ColumnType | None:
    return "BOOLEAN" if data_type.this in _BOOLEAN_TYPES else None


def _integer_type(data_type: exp.DataType) -> ColumnType | None:
    return "INTEGER" if data_type.this in _INTEGER_TYPES else None


def _bigint_type(data_type: exp.DataType) -> ColumnType | None:
    return "BIGINT" if data_type.this in _BIGINT_TYPES else None


def _bigint_decimal_type(data_type: exp.DataType) -> ColumnType | None:
    return "DECIMAL(38,0)" if data_type.this in _BIGINT_DECIMAL_TYPES else None


def _float_type(data_type: exp.DataType) -> ColumnType | None:
    return "DOUBLE" if data_type.this in _FLOAT_TYPES else None


def _decimal_type(data_type: exp.DataType) -> ColumnType | None:
    if data_type.this not in _DECIMAL_TYPES:
        return None
    return _decimal_column_type(data_type)


def _string_type(data_type: exp.DataType) -> ColumnType | None:
    return "VARCHAR" if data_type.this in _STRING_TYPES else None


def _json_type(data_type: exp.DataType) -> ColumnType | None:
    return "JSON" if data_type.this in _JSON_TYPES else None


def _nested_type(data_type: exp.DataType) -> ColumnType | None:
    if data_type.this not in _NESTED_TYPES:
        return None
    sql = data_type.sql(dialect=DUCKDB_DIALECT)
    return normalize_column_type(sql)


def _timestamptz_type(data_type: exp.DataType) -> ColumnType | None:
    return "TIMESTAMPTZ" if data_type.this in _TIMESTAMPTZ_TYPES else None


def _timestamp_type(data_type: exp.DataType) -> ColumnType | None:
    return "TIMESTAMP" if data_type.this in _TIMESTAMP_TYPES else None


_SQLGLOT_TYPE_RESOLVERS: tuple[Callable[[exp.DataType], ColumnType | None], ...] = (
    _bool_type,
    _integer_type,
    _bigint_type,
    _bigint_decimal_type,
    _float_type,
    _decimal_type,
    _string_type,
    _nested_type,
    _json_type,
    _timestamptz_type,
    _timestamp_type,
)


__all__ = ["derive_view_schemas"]
