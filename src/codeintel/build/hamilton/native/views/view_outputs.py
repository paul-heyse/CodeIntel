"""Hamilton-native view materialization using Polars + precompiled SQL plans."""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import polars as pl

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

VIEWS_TARGET_NAME = "views"
VIEWS_DOMAIN = "views"

_VIEW_PLAN_PATH = Path(__file__).resolve().parents[4] / "storage" / "views" / "view_plan_map.json"

_MANUAL_VIEW_KEYS: frozenset[str] = frozenset(
    {
        "docs.v_data_models",
        "docs.v_data_models_normalized",
    }
)

_TABLE_ALIAS_RE = re.compile(r"[^a-zA-Z0-9_]")

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


class ViewPlanSpec(TypedDict):
    """Serialized view plan specification."""

    node_name: str
    sql: str
    dependencies: list[str]
    tags: dict[str, str]


@dataclass(frozen=True, slots=True)
class ViewPlan:
    """Execution plan for a single SQL-defined view."""

    table_key: str
    node_name: str
    sql: str
    dependencies: tuple[str, ...]
    tags: dict[str, str]


@dataclass(frozen=True, slots=True)
class _ViewInputs:
    data_models: pl.LazyFrame
    fields: pl.LazyFrame
    relationships: pl.LazyFrame


def _load_view_map() -> dict[str, ViewPlanSpec]:
    raw = json.loads(_VIEW_PLAN_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "view_plan_map.json must contain a mapping"
        raise TypeError(msg)
    view_map: dict[str, ViewPlanSpec] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        node_name = value.get("node_name")
        sql = value.get("sql")
        dependencies = value.get("dependencies")
        tags = value.get("tags")
        if (
            not isinstance(node_name, str)
            or not isinstance(sql, str)
            or not isinstance(dependencies, list)
            or not isinstance(tags, dict)
        ):
            continue
        tag_map = {str(tag_key): str(tag_value) for tag_key, tag_value in tags.items()}
        view_map[key] = {
            "node_name": node_name,
            "sql": sql,
            "dependencies": [str(dep) for dep in dependencies],
            "tags": tag_map,
        }
    return view_map


def _all_dependencies(view_map: Mapping[str, ViewPlanSpec]) -> tuple[str, ...]:
    deps: set[str] = set()
    for spec in view_map.values():
        deps.update(spec["dependencies"])
    return tuple(sorted(deps))


def _table_alias(table_key: str) -> str:
    cleaned = table_key.strip()
    cleaned = cleaned.replace("-", "_").replace(".", "__").replace("/", "__")
    cleaned = _TABLE_ALIAS_RE.sub("_", cleaned)
    cleaned = re.sub(r"_{3,}", "__", cleaned)
    cleaned = cleaned.strip("_")
    if not cleaned:
        msg = f"Invalid table key for alias: {table_key!r}"
        raise ValueError(msg)
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


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
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=pl.LazyFrame,
        )
        for name in param_names
    ]
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


def _build_sql_view_node(
    *,
    plan: ViewPlan,
    param_by_table: Mapping[str, str],
    alias_map: Mapping[str, str],
) -> Callable[..., pl.LazyFrame]:
    def view_fn(**kwargs: object) -> pl.LazyFrame:
        tables: dict[str, pl.LazyFrame] = {}
        for table_key, param_name in param_by_table.items():
            value = kwargs.get(param_name)
            if value is None:
                msg = f"Missing dependency {param_name} for {plan.table_key}"
                raise ValueError(msg)
            tables[alias_map[table_key]] = _ensure_lazyframe(value, param_name=param_name)
        ctx = pl.SQLContext(frames=tables)
        return ctx.execute(plan.sql, eager=False)

    view_fn = set_signature(view_fn, _view_signature(tuple(param_by_table.values())))
    view_fn.__name__ = plan.node_name
    view_fn.__module__ = __name__
    view_fn.__doc__ = f"Compute {plan.table_key} using Polars SQL."
    return _decorate_view_node(view_fn, plan=plan)


def _coerce_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, pl.Series):
        return value.to_list()
    if isinstance(value, list):
        return value
    return [value]


def _list_to_json(expr: pl.Expr) -> pl.Expr:
    return expr.map_elements(
        lambda value: json.dumps(_coerce_list(value)),
        return_dtype=pl.String,
    )


def _data_model_fields(
    frame: pl.LazyFrame,
    *,
    normalized: bool,
) -> pl.LazyFrame:
    if normalized:
        struct_exprs = [
            pl.col("field_name").alias("field_name"),
            pl.col("field_type").alias("field_type"),
            pl.col("required").alias("required"),
            pl.col("has_default").alias("has_default"),
            pl.col("default_expr").alias("default_expr"),
            pl.col("constraints_json").alias("constraints"),
            pl.col("source").alias("source"),
            pl.col("rel_path").alias("rel_path"),
            pl.col("lineno").alias("lineno"),
            pl.col("created_at").alias("created_at"),
        ]
    else:
        struct_exprs = [
            pl.col("field_name").alias("name"),
            pl.col("field_type").alias("type"),
            pl.col("required").alias("required"),
            pl.col("has_default").alias("has_default"),
            pl.col("default_expr").alias("default_expr"),
            pl.col("constraints_json").alias("constraints"),
            pl.col("source").alias("source"),
            pl.col("lineno").alias("lineno"),
        ]
    return (
        frame.sort("field_name")
        .with_columns(pl.struct(struct_exprs).alias("field_struct"))
        .group_by("repo", "commit", "model_id", maintain_order=True)
        .agg(pl.col("field_struct").alias("fields"))
    )


def _data_model_relationships(
    frame: pl.LazyFrame,
    *,
    normalized: bool,
) -> pl.LazyFrame:
    if normalized:
        struct_exprs = [
            pl.col("field_name").alias("field_name"),
            pl.col("target_model_id").alias("target_model_id"),
            pl.col("target_module").alias("target_module"),
            pl.col("target_model_name").alias("target_model_name"),
            pl.col("relationship_kind").alias("relationship_kind"),
            pl.col("multiplicity").alias("multiplicity"),
            pl.col("via").alias("via"),
            pl.col("evidence_json").alias("evidence"),
            pl.col("rel_path").alias("rel_path"),
            pl.col("lineno").alias("lineno"),
            pl.col("created_at").alias("created_at"),
        ]
    else:
        struct_exprs = [
            pl.col("field_name").alias("field"),
            pl.col("target_model_id").alias("target_model_id"),
            pl.col("target_model_name").alias("target_model_name"),
            pl.col("target_module").alias("target_module"),
            pl.col("multiplicity").alias("multiplicity"),
            pl.col("relationship_kind").alias("kind"),
            pl.col("via").alias("via"),
            pl.col("rel_path").alias("rel_path"),
            pl.col("lineno").alias("lineno"),
            pl.col("evidence_json").alias("evidence"),
        ]
    return (
        frame.sort("field_name")
        .with_columns(pl.struct(struct_exprs).alias("relationship_struct"))
        .group_by("repo", "commit", "source_model_id", maintain_order=True)
        .agg(pl.col("relationship_struct").alias("relationships"))
        .rename({"source_model_id": "model_id"})
    )


def _resolve_data_model_inputs(
    *,
    kwargs: Mapping[str, object],
    param_by_table: Mapping[str, str],
) -> _ViewInputs:
    return _ViewInputs(
        data_models=_ensure_lazyframe(
            kwargs.get(param_by_table["analytics.data_models"]),
            param_name=param_by_table["analytics.data_models"],
        ),
        fields=_ensure_lazyframe(
            kwargs.get(param_by_table["analytics.data_model_fields"]),
            param_name=param_by_table["analytics.data_model_fields"],
        ),
        relationships=_ensure_lazyframe(
            kwargs.get(param_by_table["analytics.data_model_relationships"]),
            param_name=param_by_table["analytics.data_model_relationships"],
        ),
    )


def _build_data_models_view(
    *,
    plan: ViewPlan,
    param_by_table: Mapping[str, str],
    normalized: bool,
) -> Callable[..., pl.LazyFrame]:
    def view_fn(**kwargs: object) -> pl.LazyFrame:
        inputs = _resolve_data_model_inputs(
            kwargs=kwargs,
            param_by_table=param_by_table,
        )
        base = inputs.data_models.with_columns(
            pl.coalesce([pl.col("base_classes_json"), pl.lit("[]")]).alias("base_classes_json")
        )
        fields = _data_model_fields(inputs.fields, normalized=normalized)
        relationships = _data_model_relationships(inputs.relationships, normalized=normalized)
        joined = (
            base.join(fields, on=["repo", "commit", "model_id"], how="left")
            .join(relationships, on=["repo", "commit", "model_id"], how="left")
            .with_columns(
                pl.col("fields").fill_null(pl.lit([])).alias("fields"),
                pl.col("relationships").fill_null(pl.lit([])).alias("relationships"),
            )
        )
        if normalized:
            return joined.select(
                [
                    "repo",
                    "commit",
                    "model_id",
                    "goid_h128",
                    "model_name",
                    "module",
                    "rel_path",
                    "model_kind",
                    "base_classes_json",
                    "fields",
                    "relationships",
                    "doc_short",
                    "doc_long",
                    "created_at",
                ]
            )
        encoded = joined.with_columns(
            _list_to_json(pl.col("fields")).alias("fields"),
            _list_to_json(pl.col("relationships")).alias("relationships"),
        )
        return encoded.select(
            [
                "repo",
                "commit",
                "model_id",
                "goid_h128",
                "model_name",
                "module",
                "rel_path",
                "model_kind",
                "base_classes_json",
                "fields",
                "relationships",
                "doc_short",
                "doc_long",
                "created_at",
            ]
        )

    view_fn = set_signature(view_fn, _view_signature(tuple(param_by_table.values())))
    view_fn.__name__ = plan.node_name
    view_fn.__module__ = __name__
    view_fn.__doc__ = f"Compute {plan.table_key} using Polars expressions."
    return _decorate_view_node(view_fn, plan=plan)


_VIEW_PLAN_MAP = _load_view_map()
_VIEW_KEYS = frozenset(_VIEW_PLAN_MAP)

_ALIAS_BY_TABLE_KEY = {key: _table_alias(key) for key in _all_dependencies(_VIEW_PLAN_MAP)}

_DEPENDENCIES_BY_VIEW = {
    view_key: tuple(spec["dependencies"]) for view_key, spec in _VIEW_PLAN_MAP.items()
}

_BASE_TABLE_KEYS = tuple(
    sorted(
        {dep for deps in _DEPENDENCIES_BY_VIEW.values() for dep in deps if dep not in _VIEW_KEYS}
    )
)


_VIEW_PLANS: dict[str, ViewPlan] = {}
for view_key, spec in _VIEW_PLAN_MAP.items():
    sql = spec["sql"]
    _validate_view_sql(table_key=view_key, sql=sql)
    tags = _normalize_view_tags(spec["tags"])
    _VIEW_PLANS[view_key] = ViewPlan(
        table_key=view_key,
        node_name=spec["node_name"],
        sql=sql,
        dependencies=_DEPENDENCIES_BY_VIEW[view_key],
        tags=tags,
    )


_SOURCE_LOADERS: dict[str, Callable[..., pl.LazyFrame]] = {}
for table_key in _BASE_TABLE_KEYS:
    node_name = _source_node_name(table_key)
    loader = _build_source_loader(table_key=table_key, node_name=node_name)
    globals()[node_name] = loader
    _SOURCE_LOADERS[table_key] = loader


_VIEW_NODES: dict[str, Callable[..., pl.LazyFrame]] = {}
for view_key, plan in _VIEW_PLANS.items():
    if view_key in _MANUAL_VIEW_KEYS:
        continue
    param_by_table = {
        dep: _source_node_name(dep) if dep not in _VIEW_KEYS else _VIEW_PLANS[dep].node_name
        for dep in plan.dependencies
    }
    view_fn = _build_sql_view_node(
        plan=plan,
        param_by_table=param_by_table,
        alias_map=_ALIAS_BY_TABLE_KEY,
    )
    globals()[plan.node_name] = view_fn
    _VIEW_NODES[view_key] = view_fn


for view_key in _MANUAL_VIEW_KEYS:
    plan = _VIEW_PLANS[view_key]
    param_by_table = {
        dep: _source_node_name(dep) if dep not in _VIEW_KEYS else _VIEW_PLANS[dep].node_name
        for dep in plan.dependencies
    }
    view_fn = _build_data_models_view(
        plan=plan,
        param_by_table=param_by_table,
        normalized=view_key.endswith("_normalized"),
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
    "views__table_materializations",
]
