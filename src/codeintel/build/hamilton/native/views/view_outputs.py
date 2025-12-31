"""Hamilton-native view materialization using Polars + SQLGlot."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import polars as pl
import pyarrow.dataset as ds
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
from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.semantic_tags import SEMANTIC_VIEW_TAG_ATTR
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.datasets.paths import dataset_snapshot_dir

VIEWS_TARGET_NAME = "views"
VIEWS_DOMAIN = "views"

_VIEW_AST_PATH = (
    Path(__file__).resolve().parents[4] / "storage" / "views" / "view_ast_map.json"
)

_MANUAL_VIEW_KEYS: frozenset[str] = frozenset(
    {
        "docs.v_data_models",
        "docs.v_data_models_normalized",
    }
)

_TABLE_ALIAS_RE = re.compile(r"[^a-zA-Z0-9_]")


class ViewAstSpec(TypedDict):
    """Serialized view definition used for SQLGlot reconstruction."""

    node_name: str
    ast: list[object]
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
        view_map[key] = {"node_name": node_name, "ast": ast, "tags": tag_map}
    return view_map


def _table_key_from_table(table: exp.Table) -> str:
    db = table.db
    if db:
        return f"{db}.{table.name}"
    return table.name


def _view_dependencies(ast: list[object], *, view_key: str) -> tuple[str, ...]:
    expression = exp.Expression.load(ast)
    deps = {_table_key_from_table(table) for table in expression.find_all(exp.Table)}
    deps.discard(view_key)
    return tuple(sorted(deps))


def _all_dependencies(view_map: Mapping[str, ViewAstSpec]) -> tuple[str, ...]:
    deps: set[str] = set()
    for view_key, spec in view_map.items():
        deps.update(_view_dependencies(spec["ast"], view_key=view_key))
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


def _rewrite_sql(ast: list[object], alias_map: Mapping[str, str]) -> str:
    expression = exp.Expression.load(ast)
    for table in expression.find_all(exp.Table):
        table_key = _table_key_from_table(table)
        alias = alias_map.get(table_key)
        if alias is None:
            continue
        table.set("this", exp.to_identifier(alias))
        table.set("db", None)
        table.set("catalog", None)
    return expression.sql(dialect="duckdb")


def _extra_tags(tags: Mapping[str, str]) -> Mapping[TagKey, TagValue]:
    filtered = {key: value for key, value in tags.items() if key != ht.TAG_TABLE_KEY}
    return cast("Mapping[TagKey, TagValue]", filtered)


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
        dataset = ds.dataset(str(snapshot_dir), format="parquet", partitioning="hive")
        return pl.scan_pyarrow_dataset(dataset, batch_size=DEFAULT_ARROW_BATCH_SIZE)

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
        ctx = pl.SQLContext(**tables)
        return ctx.execute(plan.sql)

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
            pl.coalesce([pl.col("base_classes_json"), pl.lit("[]")]).alias(
                "base_classes_json"
            )
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


_VIEW_AST_MAP = _load_view_map()
_VIEW_KEYS = frozenset(_VIEW_AST_MAP)

_ALIAS_BY_TABLE_KEY = {key: _table_alias(key) for key in _all_dependencies(_VIEW_AST_MAP)}

_DEPENDENCIES_BY_VIEW = {
    view_key: _view_dependencies(spec["ast"], view_key=view_key)
    for view_key, spec in _VIEW_AST_MAP.items()
}

_BASE_TABLE_KEYS = tuple(
    sorted(
        {
            dep
            for deps in _DEPENDENCIES_BY_VIEW.values()
            for dep in deps
            if dep not in _VIEW_KEYS
        }
    )
)


_VIEW_PLANS: dict[str, ViewPlan] = {}
for view_key, spec in _VIEW_AST_MAP.items():
    sql = _rewrite_sql(spec["ast"], _ALIAS_BY_TABLE_KEY)
    _VIEW_PLANS[view_key] = ViewPlan(
        table_key=view_key,
        node_name=_view_node_name(view_key),
        sql=sql,
        dependencies=_DEPENDENCIES_BY_VIEW[view_key],
        tags=spec["tags"],
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
        dep: _source_node_name(dep) if dep not in _VIEW_KEYS else _view_node_name(dep)
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
        dep: _source_node_name(dep) if dep not in _VIEW_KEYS else _view_node_name(dep)
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
    """Finalize view materialization run record."""
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
