"""Shared helpers to materialize Parquet-backed graphs as rustworkx stores.

This module provides functions to load various graph types from
Parquet datasets into rustworkx graph stores. View-registry
fallthrough is intentionally disallowed in this layer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.build.graphs.assembly import table_to_reader
from codeintel.build.graphs.assembly.finalize import (
    GraphFinalizeArtifacts,
    finalize_graph_plan,
)
from codeintel.build.graphs.engine.datasets import (
    GraphRunMetadata,
    GraphViewFactory,
    GraphViewScanOptions,
    graph_execution_context,
    graph_run_metadata,
)
from codeintel.build.graphs.engine.protocol import GraphKind
from codeintel.build.graphs.rx.build_from_edges import (
    BuildStoreOptions,
    EdgeBuildSpec,
    build_store_from_edge_tuples,
)
from codeintel.build.graphs.rx.metadata import GraphMetadata, apply_graph_metadata
from codeintel.build.graphs.rx.policies import (
    DEFAULT_NUMERIC_POLICY,
    GraphWeightPolicy,
    weight_policy_for_kind,
)
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.finalize_ops import finalize_reader, finalize_spec_for_table
from codeintel.core.columnar.iter import iter_array_values, iter_tuples
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.columnar.plan_builder import (
    TablePlanOptions,
    build_grouped_rollup_plan,
    build_table_plan,
)
from codeintel.core.columnar.plan_kernels import (
    GroupByMaxJoinBackSpec,
    StableDedupeSpec,
    group_by_max_join_back,
    stable_dedupe_with_ties,
)
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan
from codeintel.core.data_models.ids import as_int
from codeintel.core.data_models.ids import normalize_decimal_id as normalize_decimal
from codeintel.core.schemas.primitives import resolve_canonical_sort_keys
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from codeintel.core.columnar.dedupe_ops import DedupeTier
    from codeintel.core.columnar.execution_context import ExecutionContext
    from codeintel.core.columnar.streaming import ScanTelemetry

log = logging.getLogger(__name__)

_INTERNAL_PLAN_TABLE_KEY = "internal.plan_materialize"


def _ensure_dataset_root(dataset_root: Path | None, table_key: str) -> Path | None:
    if dataset_root is None:
        log.warning("Dataset root not configured; cannot load %s", table_key)
        return None
    return dataset_root


def _view_factory(
    dataset_root: Path,
    *,
    repo: str | None,
    commit: str,
) -> GraphViewFactory:
    return GraphViewFactory.for_snapshot(dataset_root, repo=repo, commit=commit)


def _column_index(names: list[str], column: str) -> int | None:
    try:
        return names.index(column)
    except ValueError:
        return None


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _determinism_for_table(table_key: str) -> DedupeTier:
    schema = get_schema_service().get_table_schema(table_key)
    if schema is not None:
        policy = schema.finalize_policy
        if policy is not None and policy.dedupe is not None and policy.dedupe.tier is not None:
            return policy.dedupe.tier
    canonical_keys = resolve_canonical_sort_keys(schema)
    if canonical_keys == ():
        return "throughput"
    if canonical_keys:
        return "canonical"
    return "stable_set"


def _ordering_keys_for_table(table_key: str) -> tuple[str, ...] | None:
    schema = get_schema_service().get_table_schema(table_key)
    if schema is None:
        return None
    keys = resolve_canonical_sort_keys(schema)
    if not keys:
        return None
    return tuple(keys)


def _scan_options_for_table(
    table_key: str,
    *,
    base: GraphViewScanOptions | None = None,
) -> GraphViewScanOptions:
    determinism = _determinism_for_table(table_key)
    provenance = determinism == "canonical"
    execution_ctx = graph_execution_context(determinism=determinism, provenance=provenance)
    if base is None:
        return GraphViewScanOptions(provenance=provenance, execution_ctx=execution_ctx)
    return replace(
        base,
        provenance=base.provenance or provenance,
        execution_ctx=execution_ctx,
    )


def _finalize_graph_table(
    plan: Plan,
    *,
    table_key: str,
    determinism: DedupeTier,
    ctx: ExecutionContext | None,
    artifacts: GraphFinalizeArtifacts | None = None,
) -> pa.Table:
    result = finalize_graph_plan(
        plan,
        table_key=table_key,
        determinism=determinism,
        ctx=ctx,
        artifacts=artifacts,
    )
    return result.good


def _order_by_if_canonical(
    determinism: DedupeTier,
    *,
    keys: Sequence[str],
) -> tuple[SortKey, ...]:
    if determinism != "canonical":
        return ()
    return tuple(cast("SortKey", (key, "ascending")) for key in keys)


def _graph_kind_name(kind: GraphKind) -> str:
    raw = getattr(kind, "name", None)
    if isinstance(raw, str):
        return raw
    return str(kind)


def _apply_graph_run_metadata(
    store: RxGraphStore,
    *,
    kind: GraphKind,
    run_metadata: GraphRunMetadata | None,
    ordering_keys: Sequence[str] | None = None,
) -> None:
    if run_metadata is None:
        return
    apply_graph_metadata(
        store.graph,
        GraphMetadata(
            weight_policy=store.weight_policy.name,
            graph_kind=_graph_kind_name(kind),
            determinism_tier=run_metadata.determinism_tier,
            scan_profile=run_metadata.scan_profile,
            ordering_keys=tuple(ordering_keys) if ordering_keys else None,
        ),
    )


def _aggregate_edge_counts(
    table: pa.Table,
    *,
    src: str,
    dst: str,
    order_by: Sequence[SortKey] = (),
) -> pa.Table:
    if src not in table.column_names or dst not in table.column_names:
        return table
    plan = build_table_plan(
        table=table,
        options=TablePlanOptions(
            filter_expr=E.and_(E.is_valid(src), E.is_valid(dst)),
            projection={src: E.field(src), dst: E.field(dst)},
        ),
    )
    plan = build_grouped_rollup_plan(
        plan,
        keys=(src, dst),
        aggregates=[(src, "count", None, "weight")],
        order_by=order_by,
    )
    return _plan_to_table(plan)


def _filter_edge_table(
    table: pa.Table,
    *,
    src: str,
    dst: str,
    order_by: Sequence[SortKey] = (),
) -> pa.Table:
    if src not in table.column_names or dst not in table.column_names:
        return pa.Table.from_arrays(
            [pa.array([], type=pa.string()), pa.array([], type=pa.string())],
            names=[src, dst],
        )
    plan = build_table_plan(
        table=table,
        options=TablePlanOptions(
            filter_expr=E.and_(E.is_valid(src), E.is_valid(dst)),
            projection={src: E.field(src), dst: E.field(dst)},
            order_by=order_by,
        ),
    )
    return _plan_to_table(plan)


def _rename_weight_column(table: pa.Table, *, count_col: str) -> pa.Table:
    if count_col == "weight" or count_col not in table.column_names:
        return table
    if "weight" in table.column_names:
        table = table.drop_columns(["weight"])
    names = ["weight" if name == count_col else name for name in table.column_names]
    return table.rename_columns(names)


def _iter_table_tuples(
    table: pa.Table,
    *,
    columns: Sequence[str],
) -> Iterable[tuple[object, ...]]:
    yield from iter_tuples(table_to_reader(table), columns=columns)


def _node_ids_from_table[NodeId: Hashable](
    table: pa.Table,
    *,
    columns: Sequence[str],
    normalize: Callable[[object], NodeId | None],
) -> set[NodeId]:
    node_ids: set[NodeId] = set()
    for column in columns:
        if column not in table.column_names:
            continue
        for value in iter_array_values(table[column]):
            node_id = normalize(value)
            if node_id is not None:
                node_ids.add(node_id)
    return node_ids


@dataclass(frozen=True)
class _EdgeTableSpec:
    src: str
    dst: str
    directed: bool
    weight_policy: GraphWeightPolicy
    normalize: Callable[[object], Hashable | None]
    aggregate_edges: bool = False


@dataclass(frozen=True)
class _ImportGraphInputs:
    edge_counts: pa.Table
    node_ids: set[str]
    fallback_layers: dict[str, int]
    run_metadata: GraphRunMetadata | None


@dataclass(frozen=True)
class _ConfigBipartiteInputs:
    allowed_modules: set[str]
    config_table: pa.Table
    config_determinism: DedupeTier
    run_metadata: GraphRunMetadata | None


@dataclass(frozen=True)
class _SymbolModuleInputs:
    edge_table: pa.Table
    module_lookup: pa.Table
    edge_determinism: DedupeTier
    run_metadata: GraphRunMetadata | None


@dataclass(frozen=True)
class _CallGraphInputs:
    edge_table: pa.Table
    node_ids: set[int]
    node_attrs: dict[int, dict[str, object]]
    run_metadata: GraphRunMetadata | None


def _artifacts_for_scan(
    factory: GraphViewFactory,
    *,
    determinism: DedupeTier,
    scan_options: GraphViewScanOptions,
    scan_telemetry: ScanTelemetry | None = None,
) -> GraphFinalizeArtifacts:
    return GraphFinalizeArtifacts(
        dataset_root=factory.dataset_root,
        snapshot_id=factory.snapshot_id,
        run_metadata=graph_run_metadata(
            determinism=determinism,
            execution_ctx=scan_options.execution_ctx,
        ),
        scan_telemetry=scan_telemetry,
        manifest_dir=scan_options.manifest_dir,
        manifest_options=scan_options.manifest_options,
    )


def _edge_table_to_store[NodeId: Hashable](
    table: pa.Table,
    *,
    spec: _EdgeTableSpec,
    node_ids: Iterable[NodeId] | None = None,
    node_attrs: Mapping[NodeId, Mapping[str, object]] | None = None,
) -> RxGraphStore:
    node_list = list(node_ids) if node_ids is not None else None
    normalized_attrs = _normalize_node_attrs(node_attrs)
    build_spec = EdgeBuildSpec(
        directed=spec.directed,
        weight_policy=spec.weight_policy,
        numeric_policy=DEFAULT_NUMERIC_POLICY,
        src_fn=spec.normalize,
        dst_fn=spec.normalize,
    )
    return build_store_from_edge_tuples(
        _iter_table_tuples(table, columns=(spec.src, spec.dst, "weight")),
        spec=build_spec,
        options=BuildStoreOptions(
            stable_nodes=True,
            aggregate_edges=spec.aggregate_edges,
            node_ids=node_list,
            node_attrs=normalized_attrs,
            node_hint=len(node_list) if node_list is not None else None,
            edge_hint=table.num_rows,
        ),
    )


def _normalize_node_attrs[NodeId: Hashable](
    node_attrs: Mapping[NodeId, Mapping[str, object]] | None,
) -> Mapping[Hashable, Mapping[str, object]] | None:
    if not node_attrs:
        return None
    normalized: dict[Hashable, Mapping[str, object]] = {}
    for node_id, attrs in node_attrs.items():
        normalized[node_id] = dict(attrs)
    return normalized


def _load_call_graph_inputs(factory: GraphViewFactory) -> _CallGraphInputs | None:
    edge_table_key = "graph.call_graph_edges"
    edge_determinism = _determinism_for_table(edge_table_key)
    edge_scan_options = _scan_options_for_table(edge_table_key)
    edge_result = factory.load_plan_with_telemetry(
        table_key=edge_table_key,
        scan_options=edge_scan_options,
    )
    if edge_result is None:
        return None
    edge_artifacts = _artifacts_for_scan(
        factory,
        determinism=edge_determinism,
        scan_options=edge_scan_options,
        scan_telemetry=edge_result.scan_telemetry,
    )
    edge_table = _finalize_graph_table(
        edge_result.plan,
        table_key=edge_table_key,
        determinism=edge_determinism,
        ctx=edge_scan_options.execution_ctx,
        artifacts=edge_artifacts,
    )
    edge_table = _aggregate_edge_counts(
        edge_table,
        src="caller_goid_h128",
        dst="callee_goid_h128",
        order_by=_order_by_if_canonical(
            edge_determinism,
            keys=("caller_goid_h128", "callee_goid_h128"),
        ),
    )
    node_ids = _node_ids_from_table(
        edge_table,
        columns=("caller_goid_h128", "callee_goid_h128"),
        normalize=normalize_decimal,
    )
    node_attrs: dict[int, dict[str, object]] = {}
    node_table_key = "graph.call_graph_nodes"
    node_determinism = _determinism_for_table(node_table_key)
    node_scan_options = _scan_options_for_table(node_table_key)
    node_result = factory.load_plan_with_telemetry(
        table_key=node_table_key,
        scan_options=node_scan_options,
    )
    if node_result is not None:
        node_artifacts = _artifacts_for_scan(
            factory,
            determinism=node_determinism,
            scan_options=node_scan_options,
            scan_telemetry=node_result.scan_telemetry,
        )
        node_attrs = _call_node_attrs(
            factory,
            table_to_reader(
                _finalize_graph_table(
                    node_result.plan,
                    table_key=node_table_key,
                    determinism=node_determinism,
                    ctx=node_scan_options.execution_ctx,
                    artifacts=node_artifacts,
                )
            ),
        )
    return _CallGraphInputs(
        edge_table=edge_table,
        node_ids=node_ids,
        node_attrs=node_attrs,
        run_metadata=edge_artifacts.run_metadata,
    )


def _load_import_edge_inputs(factory: GraphViewFactory) -> _ImportGraphInputs | None:
    edge_table_key = "graph.import_graph_edges"
    determinism = _determinism_for_table(edge_table_key)
    edge_scan_options = _scan_options_for_table(edge_table_key)
    edge_result = factory.load_plan_with_telemetry(
        table_key=edge_table_key,
        scan_options=edge_scan_options,
    )
    if edge_result is None:
        return None
    edge_artifacts = _artifacts_for_scan(
        factory,
        determinism=determinism,
        scan_options=edge_scan_options,
        scan_telemetry=edge_result.scan_telemetry,
    )
    edge_table = _finalize_graph_table(
        edge_result.plan,
        table_key=edge_table_key,
        determinism=determinism,
        ctx=edge_scan_options.execution_ctx,
        artifacts=edge_artifacts,
    )
    edge_counts = _aggregate_edge_counts(
        edge_table,
        src="src_module",
        dst="dst_module",
        order_by=_order_by_if_canonical(
            determinism,
            keys=("src_module", "dst_module"),
        ),
    )
    node_ids = _node_ids_from_table(
        edge_counts,
        columns=("src_module", "dst_module"),
        normalize=_coerce_str,
    )
    fallback_layers = _fallback_layer_by_module(edge_table)
    return _ImportGraphInputs(
        edge_counts=edge_counts,
        node_ids=node_ids,
        fallback_layers=fallback_layers,
        run_metadata=edge_artifacts.run_metadata,
    )


def _load_import_module_attrs(factory: GraphViewFactory) -> dict[str, dict[str, int]]:
    module_table_key = "graph.import_modules"
    determinism = _determinism_for_table(module_table_key)
    module_scan_options = _scan_options_for_table(module_table_key)
    module_result = factory.load_plan_with_telemetry(
        table_key=module_table_key,
        scan_options=module_scan_options,
    )
    if module_result is None:
        return {}
    module_artifacts = _artifacts_for_scan(
        factory,
        determinism=determinism,
        scan_options=module_scan_options,
        scan_telemetry=module_result.scan_telemetry,
    )
    module_table = _finalize_graph_table(
        module_result.plan,
        table_key=module_table_key,
        determinism=determinism,
        ctx=module_scan_options.execution_ctx,
        artifacts=module_artifacts,
    )
    return _module_attrs_from_reader(factory, table_to_reader(module_table))


def _load_config_bipartite_inputs(
    factory: GraphViewFactory,
) -> _ConfigBipartiteInputs | None:
    modules_table_key = "core.modules"
    modules_determinism = _determinism_for_table(modules_table_key)
    modules_scan_options = _scan_options_for_table(
        modules_table_key,
        base=GraphViewScanOptions(apply_filter=False),
    )
    modules_result = factory.load_plan_with_telemetry(
        table_key=modules_table_key,
        scan_options=modules_scan_options,
    )
    if modules_result is None:
        return None
    modules_artifacts = _artifacts_for_scan(
        factory,
        determinism=modules_determinism,
        scan_options=modules_scan_options,
        scan_telemetry=modules_result.scan_telemetry,
    )
    modules_table = _finalize_graph_table(
        modules_result.plan,
        table_key=modules_table_key,
        determinism=modules_determinism,
        ctx=modules_scan_options.execution_ctx,
        artifacts=modules_artifacts,
    )
    allowed_modules = _allowed_modules_from_table(
        modules_table,
        repo=factory.scan_context.repo,
        commit=factory.scan_context.commit,
        order_by=_order_by_if_canonical(
            modules_determinism,
            keys=("module",),
        ),
    )
    config_table_key = "analytics.config_values"
    config_determinism = _determinism_for_table(config_table_key)
    config_scan_options = _scan_options_for_table(
        config_table_key,
        base=GraphViewScanOptions(apply_filter=False),
    )
    config_result = factory.load_plan_with_telemetry(
        table_key=config_table_key,
        scan_options=config_scan_options,
    )
    if config_result is None:
        return None
    config_artifacts = _artifacts_for_scan(
        factory,
        determinism=config_determinism,
        scan_options=config_scan_options,
        scan_telemetry=config_result.scan_telemetry,
    )
    config_table = _finalize_graph_table(
        config_result.plan,
        table_key=config_table_key,
        determinism=config_determinism,
        ctx=config_scan_options.execution_ctx,
        artifacts=config_artifacts,
    )
    return _ConfigBipartiteInputs(
        allowed_modules=allowed_modules,
        config_table=config_table,
        config_determinism=config_determinism,
        run_metadata=config_artifacts.run_metadata,
    )


def _load_symbol_module_inputs(
    factory: GraphViewFactory,
    *,
    repo: str,
    commit: str,
) -> _SymbolModuleInputs | None:
    edge_table_key = "graph.symbol_use_edges"
    edge_determinism = _determinism_for_table(edge_table_key)
    edge_scan_options = _scan_options_for_table(edge_table_key)
    edge_result = factory.load_plan_with_telemetry(
        table_key=edge_table_key,
        scan_options=edge_scan_options,
    )
    if edge_result is None:
        return None
    edge_artifacts = _artifacts_for_scan(
        factory,
        determinism=edge_determinism,
        scan_options=edge_scan_options,
        scan_telemetry=edge_result.scan_telemetry,
    )
    edge_table = _finalize_graph_table(
        edge_result.plan,
        table_key=edge_table_key,
        determinism=edge_determinism,
        ctx=edge_scan_options.execution_ctx,
        artifacts=edge_artifacts,
    )
    modules_table_key = "core.modules"
    modules_determinism = _determinism_for_table(modules_table_key)
    module_scan_options = _scan_options_for_table(
        modules_table_key,
        base=GraphViewScanOptions(apply_filter=False),
    )
    module_result = factory.load_plan_with_telemetry(
        table_key=modules_table_key,
        scan_options=module_scan_options,
    )
    if module_result is None:
        return None
    module_artifacts = _artifacts_for_scan(
        factory,
        determinism=modules_determinism,
        scan_options=module_scan_options,
        scan_telemetry=module_result.scan_telemetry,
    )
    module_table = _finalize_graph_table(
        module_result.plan,
        table_key=modules_table_key,
        determinism=modules_determinism,
        ctx=module_scan_options.execution_ctx,
        artifacts=module_artifacts,
    )
    module_lookup = _module_lookup_table(module_table, repo=repo, commit=commit)
    if module_lookup.num_rows == 0:
        return None
    return _SymbolModuleInputs(
        edge_table=edge_table,
        module_lookup=module_lookup,
        edge_determinism=edge_determinism,
        run_metadata=edge_artifacts.run_metadata,
    )


def _iter_scoped_rows(
    factory: GraphViewFactory,
    reader: pa.RecordBatchReader,
) -> Iterable[tuple[object, ...]]:
    names = list(reader.schema.names)
    repo_idx = _column_index(names, "repo")
    commit_idx = _column_index(names, "commit")
    repo = factory.scan_context.repo
    commit = factory.scan_context.commit
    for row in factory.iter_tuples(reader):
        if repo_idx is not None and repo is not None:
            row_repo = row[repo_idx]
            if row_repo is not None and str(row_repo) != repo:
                continue
        if commit_idx is not None and commit is not None:
            row_commit = row[commit_idx]
            if row_commit is not None and str(row_commit) != commit:
                continue
        yield row


def _empty_graph(*, directed: bool, kind: GraphKind) -> RxGraphStore:
    policy = weight_policy_for_kind(kind)
    return (
        RxGraphStore.directed(weight_policy=policy)
        if directed
        else RxGraphStore.undirected(weight_policy=policy)
    )


def _call_node_attrs(
    factory: GraphViewFactory,
    reader: pa.RecordBatchReader,
) -> dict[int, dict[str, object]]:
    node_attrs: dict[int, dict[str, object]] = {}
    for node_raw, kind in factory.iter_tuples(reader):
        node_id = normalize_decimal(node_raw)
        if node_id is None:
            continue
        attrs = node_attrs.setdefault(node_id, {})
        if kind is not None:
            attrs["kind"] = str(kind)
    return node_attrs


def _module_lookup_table(
    table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> pa.Table:
    if (
        table.num_rows == 0
        or "path" not in table.column_names
        or "module" not in table.column_names
    ):
        return pa.Table.from_arrays(
            [pa.array([], type=pa.string()), pa.array([], type=pa.string())],
            names=["path", "module"],
        )
    filters: list[object] = [E.is_valid("path"), E.is_valid("module")]
    if repo is not None and "repo" in table.column_names:
        filters.append(E.or_(E.is_null("repo"), E.field("repo") == E.scalar(repo)))
    if commit is not None and "commit" in table.column_names:
        filters.append(E.or_(E.is_null("commit"), E.field("commit") == E.scalar(commit)))
    projection: dict[str, object] = {
        "path": E.cast(E.field("path"), "string"),
        "module": E.cast(E.field("module"), "string"),
    }
    specificity_expr = E.scalar(0)
    if "repo" in table.column_names:
        specificity_expr += E.cast(E.is_valid("repo"), "int8")
    if "commit" in table.column_names:
        specificity_expr += E.cast(E.is_valid("commit"), "int8")
    projection["specificity"] = specificity_expr
    plan = build_table_plan(
        table=table,
        options=TablePlanOptions(
            filter_expr=E.and_(*filters),
            projection=projection,
        ),
    )
    filtered = _plan_to_table(plan)
    if filtered.num_rows == 0:
        return filtered.select(["path", "module"])
    winners = group_by_max_join_back(
        filtered,
        spec=GroupByMaxJoinBackSpec(
            key_columns=("path",),
            score_column="specificity",
            allowed_columns=("module",),
        ),
    )
    deduped = stable_dedupe_with_ties(
        winners,
        spec=StableDedupeSpec(
            key_columns=("path",),
            order_by=(("specificity", "descending"),),
            tie_breakers=(("module", "ascending"),),
            require_tie_breakers=True,
            hash_tiebreaker=True,
        ),
    )
    return deduped.select(["path", "module"])


def _symbol_module_edge_counts(
    edge_table: pa.Table,
    module_lookup: pa.Table,
    *,
    order_by: Sequence[SortKey] = (),
) -> pa.Table:
    if edge_table.num_rows == 0 or module_lookup.num_rows == 0:
        return pa.Table.from_arrays(
            [
                pa.array([], type=pa.string()),
                pa.array([], type=pa.string()),
                pa.array([], type=pa.float64()),
            ],
            names=["use_module", "def_module", "weight"],
        )
    if "use_path" not in edge_table.column_names or "def_path" not in edge_table.column_names:
        return pa.Table.from_arrays(
            [
                pa.array([], type=pa.string()),
                pa.array([], type=pa.string()),
                pa.array([], type=pa.float64()),
            ],
            names=["use_module", "def_module", "weight"],
        )
    edge_plan = build_table_plan(
        table=edge_table,
        options=TablePlanOptions(
            projection={
                "use_path": E.cast(E.field("use_path"), "string"),
                "def_path": E.cast(E.field("def_path"), "string"),
            },
            filter_expr=E.and_(E.is_valid("use_path"), E.is_valid("def_path")),
        ),
    )
    module_plan = build_table_plan(
        table=module_lookup,
        options=TablePlanOptions(
            projection={
                "path": E.cast(E.field("path"), "string"),
                "module": E.cast(E.field("module"), "string"),
            },
            filter_expr=E.and_(E.is_valid("path"), E.is_valid("module")),
        ),
    )
    def_join = edge_plan.hash_join(
        right=module_plan,
        spec=HashJoinSpec(
            left_keys=["def_path"],
            right_keys=["path"],
            how="inner",
            left_output=["use_path", "def_path"],
            right_output=["module"],
        ),
    )
    def_join = def_join.project({"use_path": E.field("use_path"), "def_module": E.field("module")})
    use_join = def_join.hash_join(
        right=module_plan,
        spec=HashJoinSpec(
            left_keys=["use_path"],
            right_keys=["path"],
            how="inner",
            left_output=["use_path", "def_module"],
            right_output=["module"],
        ),
    )
    use_join = use_join.project(
        {
            "use_module": E.field("module"),
            "def_module": E.field("def_module"),
        }
    )
    use_join = use_join.filter(E.and_(E.is_valid("use_module"), E.is_valid("def_module")))
    use_join = use_join.filter(E.field("use_module") != E.field("def_module"))
    use_join = build_grouped_rollup_plan(
        use_join,
        keys=("use_module", "def_module"),
        aggregates=[("use_module", "count", None, "weight")],
        order_by=order_by,
    )
    return _plan_to_table(use_join)


def _maybe_to_gpu_graph(store: RxGraphStore, *, use_gpu: bool) -> RxGraphStore:
    """
    No-op for rustworkx-backed execution (CPU-only).

    Parameters
    ----------
    store : RxGraphStore
        Graph store to optionally prepare for GPU execution.
    use_gpu : bool
        Whether GPU execution was requested.

    Returns
    -------
    RxGraphStore
        The original graph store.
    """
    if use_gpu:
        log.debug("GPU backend requested; rustworkx execution is CPU-only.")
    return store


def module_attrs_from_row(
    module: object,
    scc_id: object | None,
    component_size: object | None,
    layer: object | None,
) -> tuple[str, dict[str, int]]:
    """
    Build a normalized node attribute mapping for an import module row.

    Parameters
    ----------
    module :
        Module identifier from the import_modules table.
    scc_id :
        Strongly connected component identifier.
    component_size :
        Size of the SCC.
    layer :
        Condensation DAG layer.

    Returns
    -------
    tuple[str, dict[str, int]]
        Normalized module name and attribute dictionary.
    """
    module_name = str(module)
    attrs: dict[str, int] = {}
    scc_value = as_int(scc_id)
    if scc_value is not None:
        attrs["scc_id"] = scc_value
    comp_size_value = as_int(component_size)
    if comp_size_value is not None:
        attrs["component_size"] = comp_size_value
    layer_value = as_int(layer)
    if layer_value is not None:
        attrs["layer"] = layer_value
    return module_name, attrs


def _module_attrs_from_reader(
    factory: GraphViewFactory,
    reader: pa.RecordBatchReader,
) -> dict[str, dict[str, int]]:
    attrs_by_module: dict[str, dict[str, int]] = {}
    for module_row in factory.iter_tuples(reader):
        module_name, attrs = module_attrs_from_row(*module_row)
        attrs_by_module[module_name] = attrs
    return attrs_by_module


def _apply_fallback_layers(
    attrs_by_module: dict[str, dict[str, int]],
    fallback_layers: Mapping[str, int],
) -> None:
    for module, layer in fallback_layers.items():
        attrs = attrs_by_module.setdefault(module, {})
        if "layer" not in attrs:
            attrs["layer"] = layer


def _fallback_layer_by_module(table: pa.Table) -> dict[str, int]:
    if (
        table.num_rows == 0
        or "src_module" not in table.column_names
        or "module_layer" not in table.column_names
    ):
        return {}
    plan = build_table_plan(
        table=table,
        options=TablePlanOptions(
            filter_expr=E.and_(E.is_valid("src_module"), E.is_valid("module_layer")),
            projection={
                "src_module": E.field("src_module"),
                "module_layer": E.field("module_layer"),
            },
        ),
    )
    plan = build_grouped_rollup_plan(
        plan,
        keys=("src_module",),
        aggregates=[("module_layer", "max", None, "module_layer_max")],
        order_by=(("src_module", "ascending"),),
    )
    grouped = _plan_to_table(plan)
    result: dict[str, int] = {}
    for src_module, layer in _iter_table_tuples(
        grouped,
        columns=("src_module", "module_layer_max"),
    ):
        if src_module is None:
            continue
        layer_value = as_int(layer)
        if layer_value is None:
            continue
        result[str(src_module)] = layer_value
    return result


def load_call_graph(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build a call graph store of caller -> callee edges.

    Nodes are GOID integers; parallel edges are aggregated via `weight`.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    RxGraphStore
        Directed call graph store with weighted edges and isolated nodes preserved.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.call_graph_edges")
    if dataset_root is None:
        return _empty_graph(directed=True, kind=GraphKind.CALL_GRAPH)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    inputs = _load_call_graph_inputs(factory)
    if inputs is None:
        return _empty_graph(directed=True, kind=GraphKind.CALL_GRAPH)
    store = _edge_table_to_store(
        inputs.edge_table,
        spec=_EdgeTableSpec(
            src="caller_goid_h128",
            dst="callee_goid_h128",
            directed=True,
            weight_policy=weight_policy_for_kind(GraphKind.CALL_GRAPH),
            normalize=normalize_decimal,
        ),
        node_ids=inputs.node_ids or None,
        node_attrs=inputs.node_attrs or None,
    )
    _apply_graph_run_metadata(
        store,
        kind=GraphKind.CALL_GRAPH,
        run_metadata=inputs.run_metadata,
        ordering_keys=_ordering_keys_for_table("graph.call_graph_edges"),
    )
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def load_import_graph(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build a directed import graph store of module -> module edges.

    Edge weights represent aggregated edge counts when multiple edges exist.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    RxGraphStore
        Directed import graph store with weights capturing edge multiplicity.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.import_graph_edges")
    if dataset_root is None:
        return _empty_graph(directed=True, kind=GraphKind.IMPORT_GRAPH)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    edge_inputs = _load_import_edge_inputs(factory)
    if edge_inputs is None:
        return _empty_graph(directed=True, kind=GraphKind.IMPORT_GRAPH)
    module_attrs = _load_import_module_attrs(factory)
    _apply_fallback_layers(module_attrs, edge_inputs.fallback_layers)
    store = _edge_table_to_store(
        edge_inputs.edge_counts,
        spec=_EdgeTableSpec(
            src="src_module",
            dst="dst_module",
            directed=True,
            weight_policy=weight_policy_for_kind(GraphKind.IMPORT_GRAPH),
            normalize=_coerce_str,
        ),
        node_ids=edge_inputs.node_ids or None,
        node_attrs=module_attrs or None,
    )
    _apply_graph_run_metadata(
        store,
        kind=GraphKind.IMPORT_GRAPH,
        run_metadata=edge_inputs.run_metadata,
        ordering_keys=_ordering_keys_for_table("graph.import_graph_edges"),
    )
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def parse_reference_modules(ref_modules: object, allowed_modules: set[str]) -> list[str]:
    """Normalize reference modules input into a filtered list.

    Returns
    -------
    list[str]
        Allowed module names parsed from input.
    """
    modules: list[str] = []
    if isinstance(ref_modules, Mapping):
        ref_modules = ref_modules.get("reference_modules")
    if isinstance(ref_modules, list):
        modules = [str(mod) for mod in ref_modules]
    elif isinstance(ref_modules, str):
        try:
            parsed = json.loads(ref_modules)
            if isinstance(parsed, list):
                modules = [str(mod) for mod in parsed]
        except (json.JSONDecodeError, TypeError, ValueError):
            modules = []
    if allowed_modules:
        return [module for module in modules if module in allowed_modules]
    return modules


@dataclass
class ConfigGraphStats:
    total_rows: int = 0
    empty_refs: int = 0
    parsed_modules: int = 0
    kept_modules: int = 0
    dropped_modules: int = 0


def _allowed_modules_from_table(
    table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
    order_by: Sequence[SortKey] = (),
) -> set[str]:
    if table.num_rows == 0 or "module" not in table.column_names:
        return set()
    filters: list[object] = [E.is_valid("module")]
    if repo is not None and "repo" in table.column_names:
        filters.append(E.field("repo") == E.scalar(repo))
    if commit is not None and "commit" in table.column_names:
        filters.append(E.field("commit") == E.scalar(commit))
    plan = build_table_plan(
        table=table,
        options=TablePlanOptions(
            filter_expr=E.and_(*filters),
            projection={"module": E.field("module")},
        ),
    )
    plan = build_grouped_rollup_plan(
        plan,
        keys=("module",),
        aggregates=[("module", "count", None, "module_count")],
        order_by=order_by,
    )
    filtered = _plan_to_table(plan)
    module_reader = table_to_reader(filtered)
    return {str(module) for (module,) in iter_tuples(module_reader, columns=["module"]) if module}


def _allowed_modules_from_reader(
    factory: GraphViewFactory,
    modules_reader: pa.RecordBatchReader,
) -> set[str]:
    schema_names = list(modules_reader.schema.names)
    if "module" not in schema_names:
        return set()
    repo = factory.scan_context.repo
    commit = factory.scan_context.commit
    include_repo = repo is not None and "repo" in schema_names
    include_commit = commit is not None and "commit" in schema_names
    columns = ["module"]
    if include_repo:
        columns.append("repo")
    if include_commit:
        columns.append("commit")
    allowed: set[str] = set()
    repo_index = columns.index("repo") if include_repo else -1
    commit_index = columns.index("commit") if include_commit else -1
    for values in iter_tuples(modules_reader, columns=columns):
        module = values[0]
        if module is None:
            continue
        if include_repo and values[repo_index] != repo:
            continue
        if include_commit and values[commit_index] != commit:
            continue
        allowed.add(str(module))
    return allowed


def _config_bipartite_edges(
    table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
    allowed_modules: set[str],
    order_by: Sequence[SortKey] = (),
) -> tuple[
    list[tuple[Hashable, Hashable, float]],
    dict[Hashable, dict[str, object]],
    ConfigGraphStats,
]:
    stats = ConfigGraphStats()
    if table.num_rows == 0 or "key" not in table.column_names or "extras" not in table.column_names:
        return [], {}, stats
    filters: list[object] = []
    if repo is not None and "repo" in table.column_names:
        filters.append(E.field("repo") == E.scalar(repo))
    if commit is not None and "commit" in table.column_names:
        filters.append(E.field("commit") == E.scalar(commit))
    filter_expr = E.and_(*filters) if filters else None
    plan = build_table_plan(
        table=table,
        options=TablePlanOptions(
            filter_expr=filter_expr,
            projection={"key": E.field("key"), "extras": E.field("extras")},
            order_by=order_by,
        ),
    )
    filtered = _plan_to_table(plan)
    edges: list[tuple[Hashable, Hashable, float]] = []
    node_attrs: dict[Hashable, dict[str, object]] = {}
    for key, ref_modules in iter_tuples(
        table_to_reader(filtered),
        columns=["key", "extras"],
    ):
        stats.total_rows += 1
        if key is None or ref_modules is None:
            stats.empty_refs += 1
            continue
        key_node = ("c", str(key))
        node_attrs.setdefault(key_node, {"bipartite": 0})
        raw_modules = parse_reference_modules(ref_modules, set())
        stats.parsed_modules += len(raw_modules)
        filtered_modules = (
            [module for module in raw_modules if module in allowed_modules]
            if allowed_modules
            else raw_modules
        )
        if allowed_modules and raw_modules and not filtered_modules:
            filtered_modules = raw_modules
        stats.kept_modules += len(filtered_modules)
        stats.dropped_modules += len(raw_modules) - len(filtered_modules)
        for module_name in filtered_modules:
            module_node = ("m", module_name)
            node_attrs.setdefault(module_node, {"bipartite": 1})
            edges.append((key_node, module_node, 1.0))
    return edges, node_attrs, stats


def load_config_module_bipartite(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build a bipartite graph store of config keys <-> modules.

    Keys are ("c", key); modules are ("m", module). Edge weight equals one per
    reference occurrence.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    RxGraphStore
        Undirected bipartite graph store for configuration references.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "analytics.config_values")
    if dataset_root is None:
        return _empty_graph(directed=False, kind=GraphKind.CONFIG_MODULE_BIPARTITE)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    inputs = _load_config_bipartite_inputs(factory)
    if inputs is None:
        return _empty_graph(directed=False, kind=GraphKind.CONFIG_MODULE_BIPARTITE)
    edges, node_attrs, stats = _config_bipartite_edges(
        inputs.config_table,
        repo=repo,
        commit=commit,
        allowed_modules=inputs.allowed_modules,
        order_by=_order_by_if_canonical(
            inputs.config_determinism,
            keys=("key",),
        ),
    )
    if not edges:
        return _empty_graph(directed=False, kind=GraphKind.CONFIG_MODULE_BIPARTITE)
    spec = EdgeBuildSpec(
        directed=False,
        weight_policy=weight_policy_for_kind(GraphKind.CONFIG_MODULE_BIPARTITE),
        numeric_policy=DEFAULT_NUMERIC_POLICY,
    )
    options = BuildStoreOptions(
        stable_nodes=True,
        aggregate_edges=True,
        node_attrs=node_attrs,
        node_hint=len(node_attrs),
        edge_hint=len(edges),
    )
    store = build_store_from_edge_tuples(edges, spec=spec, options=options)
    _apply_graph_run_metadata(
        store,
        kind=GraphKind.CONFIG_MODULE_BIPARTITE,
        run_metadata=inputs.run_metadata,
        ordering_keys=_ordering_keys_for_table("analytics.config_values"),
    )
    graph = store.graph
    log.info(
        "Config bipartite built: rows=%d empty_refs=%d allowed_modules=%d "
        "parsed_modules=%d kept_modules=%d dropped_modules=%d graph_nodes=%d edges=%d",
        stats.total_rows,
        stats.empty_refs,
        len(inputs.allowed_modules),
        stats.parsed_modules,
        stats.kept_modules,
        stats.dropped_modules,
        graph.num_nodes(),
        graph.num_edges(),
    )
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def load_symbol_module_graph(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build an undirected weighted graph store of module-level symbol coupling.

    Edge weights count shared symbol def/use pairs between modules.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    RxGraphStore
        Undirected graph store where weights reflect shared symbol relations.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.symbol_use_edges")
    if dataset_root is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_MODULE_GRAPH)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    inputs = _load_symbol_module_inputs(factory, repo=repo, commit=commit)
    if inputs is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_MODULE_GRAPH)
    edge_counts = _symbol_module_edge_counts(
        inputs.edge_table,
        inputs.module_lookup,
        order_by=_order_by_if_canonical(
            inputs.edge_determinism,
            keys=("use_module", "def_module"),
        ),
    )
    node_ids = _node_ids_from_table(
        edge_counts,
        columns=("use_module", "def_module"),
        normalize=_coerce_str,
    )
    store = _edge_table_to_store(
        edge_counts,
        spec=_EdgeTableSpec(
            src="use_module",
            dst="def_module",
            directed=False,
            weight_policy=weight_policy_for_kind(GraphKind.SYMBOL_MODULE_GRAPH),
            normalize=_coerce_str,
            aggregate_edges=True,
        ),
        node_ids=node_ids or None,
    )
    _apply_graph_run_metadata(
        store,
        kind=GraphKind.SYMBOL_MODULE_GRAPH,
        run_metadata=inputs.run_metadata,
        ordering_keys=_ordering_keys_for_table("graph.symbol_use_edges"),
    )
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def load_symbol_function_graph(
    dataset_root: Path | None,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build an undirected weighted graph store of function-level symbol coupling (GOIDs).

    Edge weights count shared symbol def/use pairs between functions when available.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    commit : str
        Commit hash anchoring the snapshot.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    RxGraphStore
        Undirected graph store linking functions by shared symbol usage.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.symbol_use_edges")
    if dataset_root is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_FUNCTION_GRAPH)
    factory = _view_factory(dataset_root, repo=None, commit=commit)
    edge_table_key = "graph.symbol_use_edges"
    edge_determinism = _determinism_for_table(edge_table_key)
    edge_scan_options = _scan_options_for_table(edge_table_key)
    edge_result = factory.load_plan_with_telemetry(
        table_key=edge_table_key,
        scan_options=edge_scan_options,
    )
    if edge_result is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_FUNCTION_GRAPH)
    edge_artifacts = _artifacts_for_scan(
        factory,
        determinism=edge_determinism,
        scan_options=edge_scan_options,
        scan_telemetry=edge_result.scan_telemetry,
    )

    edge_table = _finalize_graph_table(
        edge_result.plan,
        table_key=edge_table_key,
        determinism=edge_determinism,
        ctx=edge_scan_options.execution_ctx,
        artifacts=edge_artifacts,
    )
    edge_table = _aggregate_edge_counts(
        edge_table,
        src="use_goid_h128",
        dst="def_goid_h128",
        order_by=_order_by_if_canonical(
            edge_determinism,
            keys=("use_goid_h128", "def_goid_h128"),
        ),
    )
    if edge_table.num_rows:
        plan = build_table_plan(
            table=edge_table,
            options=TablePlanOptions(
                filter_expr=E.field("use_goid_h128") != E.field("def_goid_h128"),
            ),
        )
        edge_table = _plan_to_table(plan)
    node_ids = _node_ids_from_table(
        edge_table,
        columns=("use_goid_h128", "def_goid_h128"),
        normalize=normalize_decimal,
    )
    store = _edge_table_to_store(
        edge_table,
        spec=_EdgeTableSpec(
            src="use_goid_h128",
            dst="def_goid_h128",
            directed=False,
            weight_policy=weight_policy_for_kind(GraphKind.SYMBOL_FUNCTION_GRAPH),
            normalize=normalize_decimal,
            aggregate_edges=True,
        ),
        node_ids=node_ids or None,
    )
    _apply_graph_run_metadata(
        store,
        kind=GraphKind.SYMBOL_FUNCTION_GRAPH,
        run_metadata=edge_artifacts.run_metadata,
        ordering_keys=_ordering_keys_for_table(edge_table_key),
    )
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def _plan_to_table(plan: Plan) -> pa.Table:
    execution_ctx = resolve_execution_context(None)
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    result = finalize_reader(
        reader,
        spec=finalize_spec_for_table(_INTERNAL_PLAN_TABLE_KEY, mode="tolerant"),
    )
    return result.good


__all__ = [
    "as_int",
    "load_call_graph",
    "load_config_module_bipartite",
    "load_import_graph",
    "load_symbol_function_graph",
    "load_symbol_module_graph",
    "module_attrs_from_row",
    "normalize_decimal",
    "parse_reference_modules",
]
