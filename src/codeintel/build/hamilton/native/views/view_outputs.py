"""Hamilton-native view materialization using SQLGlot AST + DuckDB relations."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import polars as pl
from sqlglot import exp

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
from codeintel.core.columnar.dataset_scanner import scan_dataset_lazyframe
from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.semantic_tags import SEMANTIC_VIEW_TAG_ATTR
from codeintel.core.schemas.resolution import resolve_table_schema
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE, SCHEMAS
from codeintel.storage.datasets.paths import dataset_snapshot_dir
from codeintel.storage.duckdb_types import DuckDBError
from codeintel.storage.metadata.sync import (
    sync_derived_lineage_columns,
    sync_derived_lineage_edges,
)
from codeintel.storage.queries.safe import SqlIngressPolicy, UnsafeSqlError, assert_select_perimeter
from codeintel.storage.sqlglot_tools import render_sql_duckdb

VIEWS_TARGET_NAME = "views"
VIEWS_DOMAIN = "views"

_VIEW_AST_PATH = Path(__file__).resolve().parents[4] / "storage" / "views" / "view_ast_map.json"

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


class ViewAstSpec(TypedDict):
    """Serialized view AST specification."""

    node_name: str
    ast: list[dict[str, object]]
    tags: dict[str, str]


@dataclass(frozen=True, slots=True)
class ViewPlan:
    """Execution plan for a single view."""

    table_key: str
    node_name: str
    ast: exp.Expression
    sql: str
    dependencies: tuple[str, ...]
    tags: dict[str, str]


def _load_view_map() -> dict[str, ViewAstSpec]:
    raw = json.loads(_VIEW_AST_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "view_ast_map.json must contain a mapping"
        raise TypeError(msg)
    view_map: dict[str, ViewAstSpec] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        node_name = value.get("node_name")
        ast = value.get("ast")
        tags = value.get("tags")
        if (
            not isinstance(node_name, str)
            or not isinstance(ast, list)
            or not isinstance(tags, dict)
        ):
            continue
        tag_map = {str(tag_key): str(tag_value) for tag_key, tag_value in tags.items()}
        view_map[key] = {
            "node_name": node_name,
            "ast": ast,
            "tags": tag_map,
        }
    return view_map


def _table_key_from_table(table: exp.Table) -> str:
    name = table.name
    schema = table.db
    if schema:
        return f"{schema}.{name}"
    return name


def _table_keys_from_ast(ast: exp.Expression) -> tuple[str, ...]:
    keys = {_table_key_from_table(table) for table in ast.find_all(exp.Table)}
    return tuple(sorted(key for key in keys if key))


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


def _validate_view_sql(*, table_key: str, sql: str) -> None:
    try:
        assert_select_perimeter(sql, policy=_VIEW_SQL_POLICY)
    except UnsafeSqlError as exc:
        msg = f"Unsafe SQL perimeter for view {table_key}: {exc}"
        raise ValueError(msg) from exc


def _extra_tags(tags: Mapping[str, str]) -> Mapping[TagKey, TagValue]:
    filtered = {key: value for key, value in tags.items() if key != ht.TAG_TABLE_KEY}
    return cast("Mapping[TagKey, TagValue]", filtered)


def _normalize_view_tags(tags: Mapping[str, str]) -> dict[str, str]:
    normalized = dict(tags)
    if tags.get(ht.TAG_OUTPUT_KIND) == ht.OUTPUT_KIND_SEMANTIC_VIEW:
        normalized.setdefault(ht.TAG_LAYER, "semantic")
        normalized.setdefault(ht.TAG_VERSION, "1")
        normalized.setdefault(ht.TAG_KIND, "view")
    return normalized


def _apply_semantic_attr(fn: Callable[..., object], tags: Mapping[str, str]) -> None:
    if tags.get(ht.TAG_OUTPUT_KIND) == ht.OUTPUT_KIND_SEMANTIC_VIEW:
        setattr(fn, SEMANTIC_VIEW_TAG_ATTR, dict(tags))


def _view_node_name(table_key: str) -> str:
    return to_node_name(table_key, prefix="view")


def _source_node_name(table_key: str) -> str:
    return to_node_name(table_key, prefix="src")


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
    return inspect.Signature(params, return_annotation=pl.LazyFrame)


def _build_source_loader(*, table_key: str, node_name: str) -> Callable[..., pl.LazyFrame]:
    dataset_param = dataset_node(table_key)

    def loader(env: BuildEnv, **kwargs: object) -> pl.LazyFrame:
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
        frame = scan_dataset_lazyframe(snapshot_dir, batch_size=DEFAULT_ARROW_BATCH_SIZE)
        if frame is None:
            msg = f"Missing dataset snapshot directory: {snapshot_dir}"
            raise FileNotFoundError(msg)
        return frame

    loader = set_signature(loader, _loader_signature(dataset_param))
    loader.__name__ = node_name
    loader.__module__ = __name__
    loader.__doc__ = f"Load {table_key} as a Polars LazyFrame."
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
            annotation=pl.LazyFrame,
        )
        for name in param_names
    )
    return inspect.Signature(params, return_annotation=pl.LazyFrame)


def _ensure_lazyframe(value: object, *, param_name: str) -> pl.LazyFrame:
    if isinstance(value, pl.LazyFrame):
        return value
    if isinstance(value, pl.DataFrame):
        return value.lazy()
    msg = f"Expected LazyFrame for {param_name}, got {type(value)}"
    raise TypeError(msg)


def _decorate_view_node(
    fn: Callable[..., pl.LazyFrame],
    *,
    plan: ViewPlan,
) -> Callable[..., pl.LazyFrame]:
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


def _build_ast_view_node(
    *,
    plan: ViewPlan,
    param_by_table: Mapping[str, str],
) -> Callable[..., pl.LazyFrame]:
    def view_fn(env: BuildEnv, **kwargs: object) -> pl.LazyFrame:
        con = env.gateway.con
        registered: list[str] = []
        try:
            for table_key, param_name in param_by_table.items():
                value = kwargs.get(param_name)
                if value is None:
                    msg = f"Missing dependency {param_name} for {plan.table_key}"
                    raise ValueError(msg)
                frame = _ensure_lazyframe(value, param_name=param_name)
                con.register(table_key, frame)
                registered.append(table_key)
            relation = con.sql(plan.sql)
            return relation.pl(lazy=True)
        finally:
            for table_key in registered:
                con.unregister(table_key)

    view_fn = set_signature(view_fn, _view_signature(tuple(param_by_table.values())))
    view_fn.__name__ = plan.node_name
    view_fn.__module__ = __name__
    view_fn.__doc__ = f"Compute {plan.table_key} using DuckDB relations."
    return _decorate_view_node(view_fn, plan=plan)


_VIEW_AST_MAP = _load_view_map()
_VIEW_KEYS = frozenset(_VIEW_AST_MAP)

_DEPENDENCIES_BY_VIEW = {
    view_key: _table_keys_from_ast(exp.Expression.load(spec["ast"]))
    for view_key, spec in _VIEW_AST_MAP.items()
}

_BASE_TABLE_KEYS = tuple(
    sorted(
        {dep for deps in _DEPENDENCIES_BY_VIEW.values() for dep in deps if dep not in _VIEW_KEYS}
    )
)

_VIEW_PLANS: dict[str, ViewPlan] = {}
for view_key, spec in _VIEW_AST_MAP.items():
    ast = exp.Expression.load(spec["ast"])
    tags = _normalize_view_tags(spec["tags"])
    sql = _render_view_sql(ast)
    _validate_view_sql(table_key=view_key, sql=sql)
    _VIEW_PLANS[view_key] = ViewPlan(
        table_key=view_key,
        node_name=spec["node_name"],
        ast=ast,
        sql=sql,
        dependencies=_DEPENDENCIES_BY_VIEW[view_key],
        tags=tags,
    )


def view_plan_map() -> dict[str, ViewPlan]:
    """Return compiled view plans keyed by table key.

    Returns
    -------
    dict[str, ViewPlan]
        Mapping of view table keys to compiled view plans.
    """
    return dict(_VIEW_PLANS)


_SOURCE_LOADERS: dict[str, Callable[..., pl.LazyFrame]] = {}
for table_key in _BASE_TABLE_KEYS:
    node_name = _source_node_name(table_key)
    loader = _build_source_loader(table_key=table_key, node_name=node_name)
    globals()[node_name] = loader
    _SOURCE_LOADERS[table_key] = loader


_VIEW_NODES: dict[str, Callable[..., pl.LazyFrame]] = {}
for view_key, plan in _VIEW_PLANS.items():
    param_by_table = {
        dep: _source_node_name(dep) if dep not in _VIEW_KEYS else _VIEW_PLANS[dep].node_name
        for dep in plan.dependencies
    }
    view_fn = _build_ast_view_node(
        plan=plan,
        param_by_table=param_by_table,
    )
    globals()[plan.node_name] = view_fn
    _VIEW_NODES[view_key] = view_fn


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
        column_map = _column_lineage_from_schemas(
            env=env,
            downstream_table=view_key,
            upstream_tables=deps,
        )
        if column_map:
            column_lineage[view_key_lower] = column_map
    return lineage, column_lineage


def _sync_view_lineage(env: BuildEnv, catalog: DagCatalog) -> None:
    repo = env.repo
    commit = env.commit
    if not repo or not commit:
        return
    lineage, column_lineage = _view_lineage_payload(env, catalog)
    try:
        sync_derived_lineage_edges(env.gateway.con, repo=repo, commit=commit, lineage=lineage)
    except DuckDBError:
        LOG.exception("Failed to sync derived lineage edges repo=%s commit=%s", repo, commit)
    try:
        sync_derived_lineage_columns(
            env.gateway.con,
            repo=repo,
            commit=commit,
            lineage=column_lineage,
        )
    except DuckDBError:
        LOG.exception("Failed to sync derived lineage columns repo=%s commit=%s", repo, commit)


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
    _sync_view_lineage(env, catalog)
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
