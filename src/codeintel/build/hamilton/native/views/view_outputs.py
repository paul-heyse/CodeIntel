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
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.naming import dataset_node, to_node_name
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
from codeintel.build.hamilton.tagging import TagKey, TagValue, tag_loader_query
from codeintel.build.schemas import get_schema_provider
from codeintel.build.tabular.conversion import (
    table_to_reader,
    tabular_to_arrow_reader,
    tabular_to_arrow_table,
)
from codeintel.core.columnar.dataset_scanner import scan_dataset_reader
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE, DUCKDB_DIALECT, SCHEMAS
from codeintel.core.datasets.paths import dataset_snapshot_dir
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


def _optimize_view_ast(ast: exp.Expression, dependencies: Sequence[str]) -> exp.Expression:
    schema_mapping = _schema_mapping_for_dependencies(dependencies)
    optimized = canonicalize_expression_duckdb(ast, schema=schema_mapping)
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


def _add_row_index_reader(
    reader: pa.RecordBatchReader,
    *,
    name: str,
    offset: int,
) -> pa.RecordBatchReader:
    if name in reader.schema.names:
        return reader

    def _iter_batches() -> Iterable[pa.RecordBatch]:
        current = offset
        for batch in reader:
            size = batch.num_rows
            index_array = pa.array(range(current, current + size), type=pa.int64())
            current += size
            yield batch.append_column(name, index_array)

    schema = reader.schema.append(pa.field(name, pa.int64()))
    return pa.RecordBatchReader.from_batches(schema, _iter_batches())


def _view_node_name(table_key: str) -> str:
    return to_node_name(table_key, prefix="view")


def _source_node_name(table_key: str) -> str:
    return to_node_name(f"{VIEWS_TARGET_NAME}.source.{table_key}", prefix="l")


def _loader_signature(dataset_param: str) -> inspect.Signature:
    params = [
        inspect.Parameter(
            "env",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=BuildEnv,
        ),
        inspect.Parameter(
            dataset_param,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=DatasetRef,
        ),
    ]
    return inspect.Signature(params, return_annotation=pa.RecordBatchReader)


def _build_source_loader(*, table_key: str, node_name: str) -> Callable[..., pa.RecordBatchReader]:
    dataset_param = dataset_node(table_key)

    def loader(env: BuildEnv, **kwargs: object) -> pa.RecordBatchReader:
        dataset_ref = kwargs.get(dataset_param)
        if not isinstance(dataset_ref, DatasetRef):
            msg = f"Expected DatasetRef for {dataset_param}, got {type(dataset_ref)}"
            raise TypeError(msg)
        if dataset_ref.table_key != table_key:
            msg = (
                f"DatasetRef table_key mismatch for {node_name}: "
                f"{dataset_ref.table_key} != {table_key}"
            )
            raise ValueError(msg)
        snapshot_id = dataset_ref.commit or env.commit
        if not snapshot_id:
            msg = f"Missing snapshot_id for {table_key}"
            raise ValueError(msg)
        dataset_root = env.paths.dataset_root_dir
        if dataset_root is None:
            msg = "BuildEnv.paths.dataset_root_dir is required for view loading"
            raise ValueError(msg)
        snapshot_dir = dataset_snapshot_dir(
            dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
        if not snapshot_dir.exists():
            msg = f"Missing dataset snapshot directory: {snapshot_dir}"
            raise FileNotFoundError(msg)
        reader = scan_dataset_reader(
            snapshot_dir,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        )
        if reader is None:
            msg = f"Missing dataset snapshot directory: {snapshot_dir}"
            raise FileNotFoundError(msg)
        row_index_name = env.settings.dataset_row_index_name
        if row_index_name:
            return _add_row_index_reader(
                reader,
                name=row_index_name,
                offset=env.settings.dataset_row_index_offset,
            )
        return reader

    loader = set_signature(loader, _loader_signature(dataset_param))
    loader.__name__ = node_name
    loader.__module__ = __name__
    loader.__doc__ = f"Load {table_key} as an Arrow RecordBatchReader."
    tagged = tag_loader_query(
        domain=VIEWS_DOMAIN,
        target=VIEWS_TARGET_NAME,
        table_key=table_key,
    )(loader)
    tagged.__name__ = node_name
    tagged.__module__ = __name__
    return tagged


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
            annotation=pa.RecordBatchReader,
        )
        for name in param_names
    )
    return inspect.Signature(params, return_annotation=pa.RecordBatchReader)


def _ensure_reader(value: object, *, param_name: str) -> pa.RecordBatchReader:
    try:
        return tabular_to_arrow_reader(value)
    except TypeError as exc:
        msg = f"Expected Arrow reader for {param_name}, got {type(value)}"
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
    fn: Callable[..., pa.RecordBatchReader],
    *,
    plan: ViewPlan,
) -> Callable[..., pa.RecordBatchReader]:
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


def _coerce_string_view(table: pa.Table) -> pa.Table:
    schema = table.schema
    fields: list[pa.Field] = []
    changed = False
    for field in schema:
        if pa.types.is_string_view(field.type):
            fields.append(
                pa.field(
                    field.name,
                    pa.string(),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
            changed = True
        else:
            fields.append(field)
    if not changed:
        return table
    return table.cast(pa.schema(fields, metadata=schema.metadata))


def _execute_view_query(
    *,
    plan: ViewPlan,
    readers: Mapping[str, pa.RecordBatchReader],
) -> pa.RecordBatchReader:
    con = duckdb.connect()
    try:
        for table_key, reader in readers.items():
            table = tabular_to_arrow_table(reader)
            table = _coerce_string_view(table)
            con.register(table_key, table)
        result = con.execute(plan.sql).fetch_arrow_table()
    finally:
        con.close()
    return table_to_reader(result)


def _build_ast_view_node(
    *,
    plan: ViewPlan,
    param_by_table: Mapping[str, str],
) -> Callable[..., pa.RecordBatchReader]:
    def view_fn(env: BuildEnv, **kwargs: object) -> pa.RecordBatchReader:
        _ = env
        readers: dict[str, pa.RecordBatchReader] = {}
        for table_key, param_name in param_by_table.items():
            value = _require_dependency(
                kwargs.get(param_name),
                param_name=param_name,
                table_key=plan.table_key,
            )
            readers[table_key] = _ensure_reader(value, param_name=param_name)
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
    ast = _optimize_view_ast(ast, dependencies)
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
_BASE_TABLE_KEYS = tuple(
    sorted(
        {dep for plan in _VIEW_PLANS.values() for dep in plan.dependencies if dep not in _VIEW_KEYS}
    )
)


def view_plan_map() -> dict[str, ViewPlan]:
    """Return compiled view plans keyed by table key.

    Returns
    -------
    dict[str, ViewPlan]
        Mapping of view table keys to compiled view plans.
    """
    return dict(_VIEW_PLANS)


def _install_source_loaders(
    table_keys: Iterable[str],
) -> dict[str, Callable[..., pa.RecordBatchReader]]:
    loaders: dict[str, Callable[..., pa.RecordBatchReader]] = {}
    for table_key in table_keys:
        node_name = _source_node_name(table_key)
        source_loader = _build_source_loader(table_key=table_key, node_name=node_name)
        globals()[node_name] = source_loader
        loaders[table_key] = source_loader
    return loaders


def _install_view_nodes(
    plans: Mapping[str, ViewPlan],
) -> dict[str, Callable[..., pa.RecordBatchReader]]:
    nodes: dict[str, Callable[..., pa.RecordBatchReader]] = {}
    for view_key, plan in plans.items():
        param_by_table = {
            dep: _source_node_name(dep) if dep not in _VIEW_KEYS else _VIEW_PLANS[dep].node_name
            for dep in plan.dependencies
        }
        view_node = _build_ast_view_node(
            plan=plan,
            param_by_table=param_by_table,
        )
        globals()[plan.node_name] = view_node
        nodes[view_key] = view_node
    return nodes


_SOURCE_LOADERS = _install_source_loaders(_BASE_TABLE_KEYS)
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
        observation_provider=env.gateway.schemas,
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
    "view_plan_map",
    "views__table_materializations",
]
