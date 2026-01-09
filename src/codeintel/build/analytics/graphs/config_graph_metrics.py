"""Config bipartite/projection graph metrics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.build.analytics.compute.graphs import (
    build_projection_graph,
    log_empty_graph,
    log_projection_skipped,
    projection_metrics,
)
from codeintel.build.analytics.graphs.constants import MAX_BETWEENNESS_NODES
from codeintel.build.analytics.utilities.datasets import validate_contract_rows
from codeintel.build.analytics.utilities.snapshot import snapshot_plan
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store, graph_node_count
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.build.schemas import get_contract_for_table_key
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import materialize_plan
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.row_models import columns_for_table_key
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

    from codeintel.build.analytics.compute.graphs import ProjectionMetrics
    from codeintel.build.graphs.runtime.context import GraphContext


CONFIG_GRAPH_METRICS_KEYS_TABLE_KEY = "analytics.config_graph_metrics_keys"
CONFIG_GRAPH_METRICS_MODULES_TABLE_KEY = "analytics.config_graph_metrics_modules"
CONFIG_PROJECTION_KEY_EDGES_TABLE_KEY = "analytics.config_projection_key_edges"
CONFIG_PROJECTION_MODULE_EDGES_TABLE_KEY = "analytics.config_projection_module_edges"


def _columns_for_table(table_key: str) -> tuple[str, ...]:
    columns = columns_for_table_key(table_key)
    if not columns:
        msg = f"No schema columns registered for {table_key}"
        raise ValueError(msg)
    return tuple(columns)


CONFIG_GRAPH_METRICS_KEYS_COLS = _columns_for_table(CONFIG_GRAPH_METRICS_KEYS_TABLE_KEY)
CONFIG_GRAPH_METRICS_MODULES_COLS = _columns_for_table(CONFIG_GRAPH_METRICS_MODULES_TABLE_KEY)
CONFIG_PROJECTION_KEY_EDGES_COLS = _columns_for_table(CONFIG_PROJECTION_KEY_EDGES_TABLE_KEY)
CONFIG_PROJECTION_MODULE_EDGES_COLS = _columns_for_table(CONFIG_PROJECTION_MODULE_EDGES_TABLE_KEY)

NODE_ID_INDEX = 2


@dataclass(frozen=True)
class ProjectionContext:
    """Projection execution context."""

    repo: str
    commit: str
    created_at: datetime
    graph_ctx: GraphContext


@dataclass(frozen=True)
class ProjectionTargets:
    """Dataset targets for projection metrics."""

    node_table_key: str
    edge_table_key: str


@dataclass(frozen=True)
class _ProjectionPlan:
    graph: GraphInput
    keys: set[Hashable]
    modules: set[Hashable]
    context: ProjectionContext
    key_targets: ProjectionTargets
    module_targets: ProjectionTargets


def _projection_rows(
    *,
    proj: GraphInput,
    metrics: ProjectionMetrics,
    context: ProjectionContext,
    targets: ProjectionTargets,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    node_contract, edge_contract, node_id_col, src_col, dst_col = _projection_contracts(targets)

    proj_store = ensure_store(proj)
    node_dicts = [
        {
            "repo": context.repo,
            "commit": context.commit,
            node_id_col: _projection_node_id(node),
            "degree": metrics.degree.get(node, 0),
            "weighted_degree": metrics.weighted_degree.get(node, 0.0),
            "betweenness": metrics.betweenness.get(node, 0.0),
            "closeness": metrics.closeness.get(node, 0.0),
            "community_id": metrics.community_id.get(node),
            "created_at": context.created_at,
        }
        for node in proj_store.node_ids()
    ]
    edge_dicts = []
    for src_idx, dst_idx in proj_store.graph.edge_list():
        src_id = proj_store.index_to_id[src_idx]
        dst_id = proj_store.index_to_id[dst_idx]
        payload = proj_store.graph.get_edge_data(src_idx, dst_idx)
        edge_dicts.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                src_col: _projection_node_id(src_id),
                dst_col: _projection_node_id(dst_id),
                "weight": _coerce_edge_weight(payload),
                "created_at": context.created_at,
            }
        )
    node_rows = validate_contract_rows(
        node_contract.table_key,
        node_dicts,
        validation_profile="schema-only",
    )
    edge_rows = validate_contract_rows(
        edge_contract.table_key,
        edge_dicts,
        validation_profile="schema-only",
    )
    node_serializer = row_serializer_for_table_key(node_contract.table_key)
    edge_serializer = row_serializer_for_table_key(edge_contract.table_key)
    return (
        [node_serializer(row) for row in node_rows],
        [edge_serializer(row) for row in edge_rows],
    )


def _projection_contracts(
    targets: ProjectionTargets,
) -> tuple[DatasetContract, DatasetContract, str, str, str]:
    node_contract = get_contract_for_table_key(targets.node_table_key)
    edge_contract = get_contract_for_table_key(targets.edge_table_key)
    node_columns = node_contract.schema.column_names() if node_contract.schema else []
    edge_columns = edge_contract.schema.column_names() if edge_contract.schema else []
    node_id_col = node_columns[NODE_ID_INDEX] if len(node_columns) > NODE_ID_INDEX else "node"
    src_col = next((col for col in edge_columns if col.startswith("src_")), "src")
    dst_col = next((col for col in edge_columns if col.startswith("dst_")), "dst")
    return node_contract, edge_contract, node_id_col, src_col, dst_col


def _projection_node_id(node: object) -> str:
    if isinstance(node, tuple) and len(node) > 1:
        return str(node[1])
    return str(node)


def _coerce_edge_weight(value: object) -> float:
    if value is None:
        return 1.0
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 1.0
    return 1.0


def _matches_optional_scope(value: object, expected: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return str(value) == expected


def _parse_reference_modules(ref_modules: object) -> list[str]:
    if isinstance(ref_modules, list):
        return [str(mod) for mod in ref_modules]
    if isinstance(ref_modules, str):
        try:
            parsed = json.loads(ref_modules)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(mod) for mod in parsed]
    return []


def _flatten_reference_modules(raw: object) -> list[str]:
    if isinstance(raw, list):
        modules: list[str] = []
        for item in raw:
            modules.extend(_parse_reference_modules(item))
        return modules
    return _parse_reference_modules(raw)


def _normalize_reference_modules(
    raw: object,
    *,
    allowed_modules: set[str] | None,
) -> list[str]:
    modules = _flatten_reference_modules(raw)
    if not modules:
        return []
    if allowed_modules is None:
        return modules
    filtered = [module for module in modules if module in allowed_modules]
    return filtered or modules


def _row_matches_scope(
    row: Mapping[str, object],
    *,
    repo: str | None,
    commit: str | None,
) -> bool:
    return (repo is None or _matches_optional_scope(row.get("repo"), repo)) and (
        commit is None or _matches_optional_scope(row.get("commit"), commit)
    )


def _config_reference_rowset(
    table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> pa.Table:
    required = {"key", "extras"}
    missing = [name for name in required if name not in table.column_names]
    if missing:
        msg = f"Missing config reference columns: {missing}"
        raise ValueError(msg)
    plan = snapshot_plan(table, repo=repo, commit=commit, columns=("key", "extras"))
    plan = plan.filter(E.is_valid("key"))
    plan = plan.project(
        {
            "key": E.field("key"),
            "reference_modules": E.field(("extras", "reference_modules")),
        }
    )
    plan = plan.aggregate(
        keys=[E.field("key")],
        aggregates=[("reference_modules", "list", None, "reference_modules")],
    )
    plan = plan.order_by(sort_keys=[("key", "ascending")])
    return materialize_plan(plan, use_threads=True)


def _add_bipartite_edge(graph: RxGraphStore, *, key: str, module: str) -> None:
    key_node = ("c", key)
    module_node = ("m", module)
    graph.set_node_attrs(key_node, {"bipartite": 0})
    graph.set_node_attrs(module_node, {"bipartite": 1})
    graph.add_weighted_edge(key_node, module_node, weight=1.0)


def _config_bipartite_from_rows(
    config_value_rows: Iterable[Mapping[str, object]],
    *,
    allowed_modules: set[str] | None,
    repo: str | None,
    commit: str | None,
) -> RxGraphStore:
    graph = RxGraphStore.undirected()
    for row in config_value_rows:
        if not _row_matches_scope(row, repo=repo, commit=commit):
            continue
        key = row.get("key")
        if key is None:
            continue
        reference_modules = _reference_modules_from_row(row)
        modules = _normalize_reference_modules(
            reference_modules,
            allowed_modules=allowed_modules,
        )
        if not modules:
            continue
        key_value = str(key)
        for module_name in modules:
            _add_bipartite_edge(graph, key=key_value, module=str(module_name))
    return graph


def _rows_from_tabular(
    rows: Iterable[Mapping[str, object]] | pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> list[dict[str, object]]:
    if isinstance(rows, pa.Table):
        table = _config_reference_rowset(cast("pa.Table", rows), repo=repo, commit=commit)
        return [dict(row) for row in iter_rows(table)]
    return [dict(row) for row in rows]


def _reference_modules_from_row(row: Mapping[str, object]) -> object:
    if "reference_modules" in row:
        return row.get("reference_modules")
    extras = row.get("extras")
    if isinstance(extras, Mapping):
        return extras.get("reference_modules")
    return None


def _partition_bipartite_nodes(store: RxGraphStore) -> tuple[set[Hashable], set[Hashable]]:
    keys = {node for node in store.node_ids() if store.get_node_attrs(node).get("bipartite") == 0}
    modules = set(store.node_ids()) - keys
    return keys, modules


def build_config_module_bipartite(
    config_value_rows: Iterable[Mapping[str, object]] | pa.Table,
    *,
    allowed_modules: set[str] | None = None,
    repo: str | None = None,
    commit: str | None = None,
) -> GraphInput:
    """Build a bipartite graph of config keys to modules from config values rows.

    Parameters
    ----------
    config_value_rows
        Rows from analytics.config_values or analytics.config_references.
    allowed_modules
        Optional module allowlist to filter reference modules.
    repo
        Optional repo identifier for filtering.
    commit
        Optional commit identifier for filtering.

    Returns
    -------
    GraphInput
        Undirected bipartite graph with config keys and modules.
    """
    rows = _rows_from_tabular(config_value_rows, repo=repo, commit=commit)
    return _config_bipartite_from_rows(
        rows,
        allowed_modules=allowed_modules,
        repo=repo,
        commit=commit,
    )


def _projection_payload(
    *,
    graph: GraphInput,
    nodes: set[Hashable],
    context: ProjectionContext,
    label: str,
    targets: ProjectionTargets,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    proj = build_projection_graph(
        graph,
        nodes,
        label=label,
    )
    metrics = projection_metrics(
        graph,
        nodes,
        context.graph_ctx,
        projection=proj,
        label=label,
    )
    return _projection_rows(
        proj=proj,
        metrics=metrics,
        context=context,
        targets=targets,
    )


@dataclass(frozen=True)
class ConfigGraphMetricsResult:
    """Result from config graph metrics computation.

    Attributes
    ----------
    key_rows
        Tuple rows for analytics.config_graph_metrics_keys table, or None.
    module_rows
        Tuple rows for analytics.config_graph_metrics_modules table, or None.
    key_edge_rows
        Tuple rows for analytics.config_projection_key_edges table, or None.
    module_edge_rows
        Tuple rows for analytics.config_projection_module_edges table, or None.
    """

    key_rows: tuple[tuple[object, ...], ...] | None
    module_rows: tuple[tuple[object, ...], ...] | None
    key_edge_rows: tuple[tuple[object, ...], ...] | None
    module_edge_rows: tuple[tuple[object, ...], ...] | None


def _empty_config_graph_metrics_result() -> ConfigGraphMetricsResult:
    return ConfigGraphMetricsResult(
        key_rows=None,
        module_rows=None,
        key_edge_rows=None,
        module_edge_rows=None,
    )


def _finalize_rows(
    rows: list[tuple[object, ...]],
) -> tuple[tuple[object, ...], ...] | None:
    return tuple(rows) if rows else None


def _build_projection_plan(
    *,
    repo: str,
    commit: str,
    graph: GraphInput,
    runtime_opts: GraphRuntimeOptions,
) -> _ProjectionPlan | None:
    if graph_node_count(graph) == 0:
        log_empty_graph("config_module_bipartite", graph)
        return None
    ctx = resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=runtime_opts.use_gpu,
            now=datetime.now(UTC),
            betweenness_cap=MAX_BETWEENNESS_NODES,
            pagerank_weight="weight",
            betweenness_weight="weight",
        )
    )
    store = ensure_store(graph)
    keys, modules = _partition_bipartite_nodes(store)
    if len(keys) == 0 or len(modules) == 0:
        log_projection_skipped(
            "config_projection",
            "missing partition",
            nodes=0,
            graph_nodes=graph_node_count(graph),
        )
        return None

    projection_ctx = ProjectionContext(
        repo=repo,
        commit=commit,
        created_at=ctx.resolved_now(),
        graph_ctx=ctx,
    )
    key_targets = ProjectionTargets(
        node_table_key=CONFIG_GRAPH_METRICS_KEYS_TABLE_KEY,
        edge_table_key=CONFIG_PROJECTION_KEY_EDGES_TABLE_KEY,
    )
    module_targets = ProjectionTargets(
        node_table_key=CONFIG_GRAPH_METRICS_MODULES_TABLE_KEY,
        edge_table_key=CONFIG_PROJECTION_MODULE_EDGES_TABLE_KEY,
    )
    return _ProjectionPlan(
        graph=graph,
        keys=keys,
        modules=modules,
        context=projection_ctx,
        key_targets=key_targets,
        module_targets=module_targets,
    )


def compute_config_graph_metrics_result(
    *,
    repo: str,
    commit: str,
    config_value_rows: Iterable[Mapping[str, object]] | pa.Table,
    allowed_modules: set[str] | None = None,
    runtime: GraphRuntimeOptions | None = None,
) -> ConfigGraphMetricsResult:
    """Compute config graph metrics rows without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/ArrowDatasetSaver.

    Parameters
    ----------
    repo
        Repository identifier anchoring the metrics.
    commit
        Commit hash anchoring the metrics snapshot.
    config_value_rows
        Config value rows containing reference modules.
    allowed_modules
        Optional module allowlist for reference modules.
    runtime
        Optional runtime options used to set graph execution preferences.

    Returns
    -------
    ConfigGraphMetricsResult
        Container with rows for all four config graph metrics tables.
    """
    runtime_opts = runtime or GraphRuntimeOptions()
    graph = build_config_module_bipartite(
        config_value_rows,
        allowed_modules=allowed_modules,
        repo=repo,
        commit=commit,
    )
    plan = _build_projection_plan(
        repo=repo,
        commit=commit,
        graph=graph,
        runtime_opts=runtime_opts,
    )
    if plan is None:
        return _empty_config_graph_metrics_result()
    key_rows, key_edges = _projection_payload(
        graph=plan.graph,
        nodes=plan.keys,
        context=plan.context,
        label="config_keys",
        targets=plan.key_targets,
    )
    module_rows, module_edges = _projection_payload(
        graph=plan.graph,
        nodes=plan.modules,
        context=plan.context,
        label="config_modules",
        targets=plan.module_targets,
    )

    return ConfigGraphMetricsResult(
        key_rows=_finalize_rows(key_rows),
        module_rows=_finalize_rows(module_rows),
        key_edge_rows=_finalize_rows(key_edges),
        module_edge_rows=_finalize_rows(module_edges),
    )
