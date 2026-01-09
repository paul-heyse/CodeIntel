"""Hamilton-native view materialization using SQLGlot AST + DuckDB + Arrow."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import duckdb
import pyarrow as pa
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError, SqlglotError

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.naming import query_node, to_node_name
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import DatasetSaveSpec, SaverContext, save_dataset
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.nodes.signature_tools import set_signature
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import TagKey, TagValue
from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.observation_provider import observation_provider_for_env
from codeintel.build.tabular.conversion import (
    empty_table_from_schema,
    table_from_batches,
    tabular_to_arrow_table,
)
from codeintel.build.tabular.types import TabularInput
from codeintel.core.columnar.type_normalization import (
    normalize_binary_view_table,
    normalize_reader,
    normalize_string_view_table,
)
from codeintel.core.constants import DUCKDB_DIALECT, SCHEMAS
from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.semantic_tags import SEMANTIC_VIEW_TAG_ATTR
from codeintel.core.queries.safe import SqlIngressPolicy, UnsafeSqlError, assert_select_perimeter
from codeintel.core.schemas.resolution import resolve_table_schema
from codeintel.core.sqlglot_tools import (
    canonicalize_expression_duckdb,
    extract_column_lineage_from_ast,
    join_key_issues,
    render_sql_duckdb,
    schema_mapping_for_table_key,
)
from codeintel.core.views.discovery import discover_view_builders
from codeintel.core.views.inventory import view_builder_modules

if TYPE_CHECKING:
    from codeintel.core.views.discovery import DiscoveredViewBuilder

VIEWS_TARGET_NAME = "views"
VIEWS_DOMAIN = "views"

LOG = logging.getLogger(__name__)

_DENY_EXTERNAL_VIEW_FUNCS: frozenset[str] = frozenset(
    {
        "read_avro",
        "read_csv",
        "read_csv_auto",
        "read_delta",
        "read_excel",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_orc",
        "read_parquet",
        "read_sqlite",
        "iceberg_scan",
        "delta_scan",
        "parquet_scan",
        "sqlite_scan",
    }
)
_VIEW_SQL_POLICY = SqlIngressPolicy(
    allowed_schemas=frozenset(SCHEMAS),
    deny_functions=_DENY_EXTERNAL_VIEW_FUNCS,
)


@dataclass(frozen=True, slots=True)
class ViewPlan:
    """Execution plan for a single view."""

    table_key: str
    node_name: str
    ast: exp.Expression
    sql: str
    dependencies: tuple[str, ...]
    tags: dict[str, str]


def _table_key_from_table(table: exp.Table) -> str:
    name = table.name
    schema = table.db
    if schema:
        return f"{schema}.{name}"
    return name


def _table_keys_from_ast(ast: exp.Expression) -> tuple[str, ...]:
    cte_names = _cte_names(ast)
    keys = {
        _table_key_from_table(table)
        for table in ast.find_all(exp.Table)
        if not _is_cte_table(table, cte_names)
    }
    return tuple(sorted(key for key in keys if key))


def _cte_names(ast: exp.Expression) -> frozenset[str]:
    names = {cte.alias_or_name for cte in ast.find_all(exp.CTE) if cte.alias_or_name}
    return frozenset(names)


def _is_cte_table(table: exp.Table, cte_names: frozenset[str]) -> bool:
    if not cte_names:
        return False
    if table.db:
        return False
    return table.name in cte_names


def _rewrite_ast_tables(ast: exp.Expression) -> exp.Expression:
    def _transform(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Table):
            return node
        table_key = _table_key_from_table(node)
        alias = node.args.get("alias")
        return exp.Table(
            this=exp.Identifier(this=table_key, quoted=True),
            alias=alias,
        )

    return ast.transform(_transform)


def _render_view_sql(ast: exp.Expression) -> str:
    rewritten = _rewrite_ast_tables(ast.copy())
    return render_sql_duckdb(rewritten)


def _schema_mapping_for_dependencies(
    dependencies: Iterable[str],
) -> Mapping[str, Mapping[str, str]] | None:
    try:
        provider = get_schema_provider()
    except RuntimeError:
        return None
    merged: dict[str, Mapping[str, str]] = {}
    for table_key in dependencies:
        schema = provider.get_table_schema(table_key)
        if schema is None:
            continue
        column_types = {col.name: col.type for col in schema.columns}
        mapping = schema_mapping_for_table_key(table_key, column_types=column_types)
        if mapping is not None:
            merged.update(mapping)
    return merged or None


def _blob_column_names(
    schema_mapping: Mapping[str, Mapping[str, str]] | None,
) -> frozenset[str]:
    if not schema_mapping:
        return frozenset()
    names: set[str] = set()
    for columns in schema_mapping.values():
        for column, column_type in columns.items():
            if str(column_type).upper().startswith("BLOB"):
                names.add(column)
    return frozenset(names)


def _coerce_blob_literal_coalesce(
    ast: exp.Expression, blob_columns: frozenset[str]
) -> exp.Expression:
    if not blob_columns:
        return ast

    def _transform(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Coalesce):
            return node
        expressions = (
            [node.this, *node.expressions] if node.this is not None else list(node.expressions)
        )
        if not expressions:
            return node
        if not any(
            isinstance(expr, exp.Column) and expr.name in blob_columns for expr in expressions
        ):
            return node
        updated: list[exp.Expression] = []
        changed = False
        for expr in expressions:
            if isinstance(expr, exp.Column) and expr.name in blob_columns:
                updated.append(
                    exp.Cast(
                        this=expr,
                        to=exp.DataType.build("BLOB", dialect=DUCKDB_DIALECT),
                    )
                )
                changed = True
                continue
            if isinstance(expr, exp.Literal) and expr.is_string:
                updated.append(
                    exp.Cast(
                        this=expr,
                        to=exp.DataType.build("BLOB", dialect=DUCKDB_DIALECT),
                    )
                )
                changed = True
            else:
                updated.append(expr)
        if not changed:
            return node
        return exp.Coalesce(expressions=updated)

    return ast.transform(_transform)


def _optimize_view_ast(
    ast: exp.Expression,
    dependencies: Sequence[str],
    *,
    table_key: str,
) -> exp.Expression:
    schema_mapping = _schema_mapping_for_dependencies(dependencies)
    blob_columns = _blob_column_names(schema_mapping)
    if table_key == "docs.v_subsystem_profile":
        base = ast
    else:
        base = canonicalize_expression_duckdb(ast, schema=schema_mapping)
    optimized = _coerce_blob_literal_coalesce(base, blob_columns)
    optimized = _hoist_non_equi_join_filters(optimized)
    return _strip_nested_list_aggregates(optimized)


def _strip_nested_list_aggregates(ast: exp.Expression) -> exp.Expression:
    def _transform(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Max):
            return node
        arg = node.this
        if arg is None:
            return node
        if any(isinstance(child, exp.List) for child in arg.walk()):
            return arg
        return node

    return ast.transform(_transform)


def _hoist_non_equi_join_filters(ast: exp.Expression) -> exp.Expression:
    if not isinstance(ast, exp.Select):
        return ast
    extra_filters: list[exp.Expression] = []
    for join in ast.find_all(exp.Join):
        kind = join.args.get("kind")
        if kind is not None and str(kind).upper() != "INNER":
            continue
        on = join.args.get("on")
        if on is None:
            continue
        parts = _flatten_and_conditions(on)
        equi = [part for part in parts if _is_equi_join_condition(part)]
        non_equi = [part for part in parts if not _is_equi_join_condition(part)]
        if not non_equi or not equi:
            continue
        join.set("on", _combine_and_conditions(equi))
        extra_filters.extend(non_equi)
    if not extra_filters:
        return ast
    where = ast.args.get("where")
    combined = _combine_and_conditions(extra_filters)
    if where is None:
        ast.set("where", exp.Where(this=combined))
        return ast
    if where.this is None:
        where.set("this", combined)
        return ast
    where.set("this", _combine_and_conditions([where.this, *extra_filters]))
    return ast


def _flatten_and_conditions(expr: exp.Expression) -> list[exp.Expression]:
    if not isinstance(expr, exp.And):
        return [expr]
    left = expr.this
    right = expr.expression
    if left is None or right is None:
        return [expr]
    return [*_flatten_and_conditions(left), *_flatten_and_conditions(right)]


def _combine_and_conditions(conditions: Sequence[exp.Expression]) -> exp.Expression:
    if not conditions:
        msg = "At least one condition is required"
        raise ValueError(msg)
    iterator = iter(conditions)
    combined = next(iterator)
    for condition in iterator:
        combined = exp.And(this=combined, expression=condition)
    return combined


def _is_equi_join_condition(expr: exp.Expression) -> bool:
    if not isinstance(expr, exp.EQ):
        return False
    return isinstance(expr.this, exp.Column) and isinstance(expr.expression, exp.Column)


def _discover_registered_views() -> tuple[DiscoveredViewBuilder, ...]:
    modules = view_builder_modules()
    if not modules:
        return ()
    try:
        return discover_view_builders(modules=modules)
    except ValueError:
        return ()


def _view_ast_from_builder(*, table_key: str, builder: Callable[..., object]) -> exp.Expression:
    try:
        rendered = builder()
    except TypeError as exc:
        msg = f"View builder for {table_key} must be callable with no arguments"
        raise ValueError(msg) from exc

    ast: exp.Expression
    if isinstance(rendered, exp.Expression):
        ast = rendered
    elif isinstance(rendered, str):
        try:
            ast = parse_one(rendered, read=DUCKDB_DIALECT)
        except ParseError as exc:
            msg = f"Failed to parse SQL for view {table_key}"
            raise ValueError(msg) from exc
    else:
        msg = f"View builder for {table_key} must return SQLGlot AST or SQL string"
        raise TypeError(msg)

    if isinstance(ast, exp.Subquery):
        inner = ast.this
        if inner is None:
            msg = f"View builder for {table_key} returned an empty subquery"
            raise ValueError(msg)
        ast = inner

    if not isinstance(ast, (exp.Select, exp.SetOperation)):
        msg = f"View builder for {table_key} must return a query expression"
        raise TypeError(msg)

    return ast


def _validate_view_sql(*, table_key: str, sql: str) -> None:
    try:
        assert_select_perimeter(sql, policy=_VIEW_SQL_POLICY, enforce_safe_sql=False)
    except UnsafeSqlError as exc:
        msg = f"Unsafe SQL perimeter for view {table_key}: {exc}"
        raise ValueError(msg) from exc


def _extra_tags(tags: Mapping[str, str]) -> Mapping[TagKey, TagValue]:
    filtered = {key: value for key, value in tags.items() if key != ht.TAG_TABLE_KEY}
    return cast("Mapping[TagKey, TagValue]", filtered)


def _normalize_view_tags(tags: Mapping[str, object]) -> dict[str, str]:
    normalized = {
        str(key): _stringify_tag_value(value) for key, value in tags.items() if value is not None
    }
    if tags.get(ht.TAG_OUTPUT_KIND) == ht.OUTPUT_KIND_SEMANTIC_VIEW:
        normalized.setdefault(ht.TAG_LAYER, "semantic")
        normalized.setdefault(ht.TAG_VERSION, "1")
        normalized.setdefault(ht.TAG_KIND, "view")
    return normalized


def _stringify_tag_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, float, int)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item is not None)
    return str(value)


def _apply_semantic_attr(fn: Callable[..., object], tags: Mapping[str, str]) -> None:
    if tags.get(ht.TAG_OUTPUT_KIND) == ht.OUTPUT_KIND_SEMANTIC_VIEW:
        setattr(fn, SEMANTIC_VIEW_TAG_ATTR, dict(tags))


def _add_row_index_table(
    table: pa.Table,
    *,
    name: str,
    offset: int,
) -> pa.Table:
    if name in table.schema.names:
        return table
    index_array = pa.array(range(offset, offset + table.num_rows), type=pa.int64())
    return table.append_column(name, index_array)


def _view_node_name(table_key: str) -> str:
    return to_node_name(table_key, prefix="view")


def _view_signature(param_names: Sequence[str]) -> inspect.Signature:
    params = [
        inspect.Parameter(
            "env",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=BuildEnv,
        )
    ]
    params.extend(
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=TabularInput,
        )
        for name in param_names
    )
    return inspect.Signature(params, return_annotation=pa.Table)


def _ensure_table(value: object, *, param_name: str) -> pa.Table:
    if isinstance(value, pa.RecordBatchReader):
        reader = cast("pa.RecordBatchReader", value)
        batches = list(normalize_reader(reader))
        if not batches:
            return empty_table_from_schema(reader.schema)
        return table_from_batches(batches, schema=batches[0].schema)
    try:
        return tabular_to_arrow_table(value)
    except TypeError as exc:
        msg = f"Expected Arrow table for {param_name}, got {type(value)}"
        raise TypeError(msg) from exc


def _require_dependency(
    value: object | None,
    *,
    param_name: str,
    table_key: str,
) -> object:
    if value is None:
        msg = f"Missing dependency {param_name} for {table_key}"
        raise ValueError(msg)
    return value


def _decorate_view_node(
    fn: Callable[..., pa.Table],
    *,
    plan: ViewPlan,
) -> Callable[..., pa.Table]:
    context = SaverContext(
        domain=VIEWS_DOMAIN,
        target=VIEWS_TARGET_NAME,
        extra_tags=_extra_tags(plan.tags),
    )
    decorator = save_dataset(
        context=context,
        spec=DatasetSaveSpec(table_key=plan.table_key),
    )
    decorated = decorator(fn)
    _apply_semantic_attr(decorated, plan.tags)
    decorated.__name__ = plan.node_name
    decorated.__module__ = __name__
    decorated.__doc__ = f"Materialize view {plan.table_key}."
    return decorated


def _normalize_view_table(table: pa.Table) -> pa.Table:
    return normalize_binary_view_table(normalize_string_view_table(table))


def _execute_view_query(
    *,
    plan: ViewPlan,
    readers: Mapping[str, pa.Table],
) -> pa.Table:
    con = duckdb.connect()
    try:
        for table_key, reader in readers.items():
            table = tabular_to_arrow_table(reader)
            table = _normalize_view_table(table)
            con.register(table_key, table)
        result = con.execute(plan.sql).fetch_arrow_table()
    finally:
        con.close()
    return result


def _build_ast_view_node(
    *,
    plan: ViewPlan,
    param_by_table: Mapping[str, str],
) -> Callable[..., pa.Table]:
    def view_fn(env: BuildEnv, **kwargs: object) -> pa.Table:
        row_index_name = env.settings.dataset_row_index_name
        row_index_offset = env.settings.dataset_row_index_offset
        readers: dict[str, pa.Table] = {}
        for table_key, param_name in param_by_table.items():
            value = _require_dependency(
                kwargs.get(param_name),
                param_name=param_name,
                table_key=plan.table_key,
            )
            table = _ensure_table(value, param_name=param_name)
            if row_index_name:
                table = _add_row_index_table(
                    table,
                    name=row_index_name,
                    offset=row_index_offset,
                )
            readers[table_key] = table
        return _execute_view_query(plan=plan, readers=readers)

    view_fn = set_signature(view_fn, _view_signature(tuple(param_by_table.values())))
    view_fn.__name__ = plan.node_name
    view_fn.__module__ = __name__
    view_fn.__doc__ = f"Compute {plan.table_key} using DuckDB over Arrow."
    return _decorate_view_node(view_fn, plan=plan)


# Views are discovered from plugin modules at import time, so we keep dynamic node
# construction here (instead of TableTargetSpec) to preserve per-view tags/signatures.
_VIEW_BUILDERS = _discover_registered_views()
_VIEW_PLANS: dict[str, ViewPlan] = {}
for builder in _VIEW_BUILDERS:
    ast = _view_ast_from_builder(table_key=builder.table_key, builder=builder.builder)
    tags = _normalize_view_tags(builder.tags)
    dependencies = _table_keys_from_ast(ast)
    ast = _optimize_view_ast(ast, dependencies, table_key=builder.table_key)
    dependencies = _table_keys_from_ast(ast)
    sql = _render_view_sql(ast)
    _validate_view_sql(table_key=builder.table_key, sql=sql)
    issues = join_key_issues(ast)
    if issues:
        LOG.warning(
            "View join audit issues (table=%s): %s",
            builder.table_key,
            ", ".join(issues),
        )
    _VIEW_PLANS[builder.table_key] = ViewPlan(
        table_key=builder.table_key,
        node_name=builder.node_name,
        ast=ast,
        sql=sql,
        dependencies=dependencies,
        tags=tags,
    )

_VIEW_KEYS = frozenset(_VIEW_PLANS)


def view_plan_map() -> dict[str, ViewPlan]:
    """Return compiled view plans keyed by table key.

    Returns
    -------
    dict[str, ViewPlan]
        Mapping of view table keys to compiled view plans.
    """
    return dict(_VIEW_PLANS)


def _install_view_nodes(
    plans: Mapping[str, ViewPlan],
) -> dict[str, Callable[..., pa.Table]]:
    nodes: dict[str, Callable[..., pa.Table]] = {}
    for view_key, plan in plans.items():
        param_by_table = {
            dep: query_node(dep) if dep not in _VIEW_KEYS else _VIEW_PLANS[dep].node_name
            for dep in plan.dependencies
        }
        view_node = _build_ast_view_node(
            plan=plan,
            param_by_table=param_by_table,
        )
        globals()[plan.node_name] = view_node
        nodes[view_key] = view_node
    return nodes


_VIEW_NODES = _install_view_nodes(_VIEW_PLANS)


VIEW_TABLE_KEYS = tuple(sorted(_VIEW_PLANS))


def _view_output_nodes(catalog: DagCatalog) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for node in catalog.nodes.values():
        tags = node.tags
        if tags.get(ht.TAG_TARGET) != VIEWS_TARGET_NAME:
            continue
        if tags.get(ht.TAG_NODE_TYPE) != ht.NODE_TYPE_DATASET:
            continue
        table_key = tags.get(ht.TAG_TABLE_KEY)
        if isinstance(table_key, str) and table_key:
            outputs.setdefault(table_key, node.name)
    return outputs


def _resolve_table_columns(*, table_key: str, env: BuildEnv) -> tuple[str, ...] | None:
    try:
        provider = get_schema_provider()
    except RuntimeError:
        provider = None
    resolution = resolve_table_schema(
        table_key,
        observation_provider=observation_provider_for_env(env),
        schema_provider=provider,
    )
    table_schema = resolution.table_schema
    if table_schema is None:
        return None
    return tuple(column.name.lower() for column in table_schema.columns if column.name)


def _direct_table_dependencies(
    *,
    catalog: DagCatalog,
    node_name: str,
) -> tuple[str, ...]:
    node = catalog.nodes.get(node_name)
    if node is None:
        return ()
    deps: set[str] = set()
    for dep_name in node.deps:
        dep_node = catalog.nodes.get(dep_name)
        if dep_node is None:
            continue
        table_key = dep_node.tags.get(ht.TAG_TABLE_KEY)
        if isinstance(table_key, str) and table_key:
            deps.add(table_key)
    return tuple(sorted(deps))


def _column_lineage_from_schemas(
    *,
    env: BuildEnv,
    downstream_table: str,
    upstream_tables: Iterable[str],
) -> dict[str, frozenset[str]]:
    downstream_columns = _resolve_table_columns(table_key=downstream_table, env=env)
    if not downstream_columns:
        return {}
    upstream_columns: dict[str, tuple[str, ...]] = {}
    for table_key in upstream_tables:
        resolved = _resolve_table_columns(table_key=table_key, env=env)
        if resolved:
            upstream_columns[table_key] = resolved
    column_lineage: dict[str, frozenset[str]] = {}
    for column in downstream_columns:
        refs: set[str] = set()
        for upstream_key, columns in upstream_columns.items():
            if column in columns:
                refs.add(f"{upstream_key.lower()}.{column}")
        if refs:
            column_lineage[column] = frozenset(sorted(refs))
    return column_lineage


def _column_lineage_from_ast(plan: ViewPlan) -> dict[str, frozenset[str]]:
    try:
        raw_lineage = extract_column_lineage_from_ast(plan.ast)
    except (SqlglotError, TypeError, ValueError):
        return {}
    normalized: dict[str, frozenset[str]] = {}
    for column, refs in raw_lineage.items():
        column_key = column.lower()
        normalized[column_key] = frozenset(ref.lower() for ref in refs)
    return normalized


def _view_lineage_payload(
    env: BuildEnv,
    catalog: DagCatalog,
) -> tuple[dict[str, frozenset[str]], dict[str, dict[str, frozenset[str]]]]:
    lineage: dict[str, frozenset[str]] = {}
    column_lineage: dict[str, dict[str, frozenset[str]]] = {}
    view_nodes = _view_output_nodes(catalog)
    for view_key, node_name in view_nodes.items():
        view_key_lower = view_key.lower()
        deps = _direct_table_dependencies(catalog=catalog, node_name=node_name)
        deps_lower = tuple(dep.lower() for dep in deps if dep.lower() != view_key_lower)
        lineage[view_key_lower] = frozenset(deps_lower)
        plan = _VIEW_PLANS.get(view_key)
        column_map = _column_lineage_from_ast(plan) if plan is not None else {}
        if not column_map:
            column_map = _column_lineage_from_schemas(
                env=env,
                downstream_table=view_key,
                upstream_tables=deps,
            )
        if column_map:
            column_lineage[view_key_lower] = column_map
    return lineage, column_lineage


def view_lineage_payload(
    *,
    env: BuildEnv,
    catalog: DagCatalog,
) -> tuple[dict[str, frozenset[str]], dict[str, dict[str, frozenset[str]]]]:
    """Return view-level table and column lineage payloads.

    Returns
    -------
    tuple[dict[str, frozenset[str]], dict[str, dict[str, frozenset[str]]]]
        Table-level lineage and column-level lineage payloads.
    """
    return _view_lineage_payload(env=env, catalog=catalog)


views__table_materializations = make_table_materializations_collector(
    domain=VIEWS_DOMAIN,
    target=VIEWS_TARGET_NAME,
    table_keys=VIEW_TABLE_KEYS,
    node_name="views__table_materializations",
)


@codeintel_target(domain=VIEWS_DOMAIN, target=VIEWS_TARGET_NAME)
def t__views(
    env: BuildEnv,
    catalog: DagCatalog,
    views__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize view materialization run record.

    Parameters
    ----------
    env
        Build environment with gateway access.
    catalog
        DAG catalog for resolving view outputs.
    views__table_materializations
        Materialization results for view table outputs.

    Returns
    -------
    TargetRunRecord
        View materialization run record.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=VIEWS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=views__table_materializations,
    )


__all__ = [
    "VIEW_TABLE_KEYS",
    "t__views",
    "view_lineage_payload",
    "view_plan_map",
    "views__table_materializations",
]
