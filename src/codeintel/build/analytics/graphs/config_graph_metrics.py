"""Config bipartite/projection graph metrics."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
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
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store, graph_node_count
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.build.schemas import get_contract_for_table_key
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import and_kleene, equal_mask
from codeintel.core.schemas.row_models import columns_for_table_key
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key

if TYPE_CHECKING:
    from collections.abc import Hashable

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


def _projection_rows(
    *,
    proj: GraphInput,
    metrics: ProjectionMetrics,
    context: ProjectionContext,
    targets: ProjectionTargets,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    node_contract = get_contract_for_table_key(targets.node_table_key)
    edge_contract = get_contract_for_table_key(targets.edge_table_key)
    node_columns = node_contract.schema.column_names() if node_contract.schema else []
    edge_columns = edge_contract.schema.column_names() if edge_contract.schema else []
    node_id_col = node_columns[NODE_ID_INDEX] if len(node_columns) > NODE_ID_INDEX else "node"
    src_col = next((col for col in edge_columns if col.startswith("src_")), "src")
    dst_col = next((col for col in edge_columns if col.startswith("dst_")), "dst")

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


def _normalize_reference_modules(
    raw: object,
    *,
    allowed_modules: set[str] | None,
) -> list[str]:
    modules = _parse_reference_modules(raw)
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


def _filter_table_by_scope(
    table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> pa.Table:
    if repo is None and commit is None:
        return table
    mask: pa.Array | pa.ChunkedArray | None = None
    if repo is not None and "repo" in table.column_names:
        mask = equal_mask(table["repo"], pa.scalar(repo))
    if commit is not None and "commit" in table.column_names:
        commit_mask = equal_mask(table["commit"], pa.scalar(commit))
        mask = commit_mask if mask is None else and_kleene(mask, commit_mask)
    if mask is None:
        return table
    return safe_filter(table, mask)


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
        modules = _normalize_reference_modules(
            row.get("reference_modules"),
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
        table = _filter_table_by_scope(cast("pa.Table", rows), repo=repo, commit=commit)
        return list(iter_rows(table))
    return [dict(row) for row in rows]


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
        Rows from analytics.config_values.
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
    if graph_node_count(graph) == 0:
        log_empty_graph("config_module_bipartite", graph)
        return ConfigGraphMetricsResult(
            key_rows=None, module_rows=None, key_edge_rows=None, module_edge_rows=None
        )
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
    keys = {node for node in store.node_ids() if store.get_node_attrs(node).get("bipartite") == 0}
    modules = set(store.node_ids()) - keys
    if len(keys) == 0 or len(modules) == 0:
        log_projection_skipped(
            "config_projection",
            "missing partition",
            nodes=0,
            graph_nodes=graph_node_count(graph),
        )
        return ConfigGraphMetricsResult(
            key_rows=None, module_rows=None, key_edge_rows=None, module_edge_rows=None
        )

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

    key_rows, key_edges = _projection_payload(
        graph=graph,
        nodes=keys,
        context=projection_ctx,
        label="config_keys",
        targets=key_targets,
    )
    module_rows, module_edges = _projection_payload(
        graph=graph,
        nodes=modules,
        context=projection_ctx,
        label="config_modules",
        targets=module_targets,
    )

    return ConfigGraphMetricsResult(
        key_rows=tuple(key_rows) if key_rows else None,
        module_rows=tuple(module_rows) if module_rows else None,
        key_edge_rows=tuple(key_edges) if key_edges else None,
        module_edge_rows=tuple(module_edges) if module_edges else None,
    )
