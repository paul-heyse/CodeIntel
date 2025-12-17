"""Native Hamilton implementation for config_data_flow target.

This module provides the Hamilton native nodes for config data flow tracking:
- `t__config_data_flow__compute`: Pure compute node for config tracking
- Config data flow materialization via SaveToDecorator
- Config graph metrics materialization via SaveToDecorator

Phase 4: Analytics domain migration with Hamilton-native DAG-visible I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.graphs.config_data_flow import (
    CONFIG_DATA_FLOW_COLS,
    ConfigDataFlowResult,
    compute_config_data_flow_result,
)
from codeintel.analytics.graphs.config_graph_metrics import (
    CONFIG_GRAPH_METRICS_KEYS_COLS,
    CONFIG_GRAPH_METRICS_MODULES_COLS,
    CONFIG_PROJECTION_KEY_EDGES_COLS,
    CONFIG_PROJECTION_MODULE_EDGES_COLS,
    ConfigGraphMetricsResult,
    compute_config_graph_metrics_result,
)
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService
from codeintel.graphs.runtime import GraphRuntimeOptions, resolve_graph_runtime
from codeintel.hamilton.records import TargetRunRecord

if TYPE_CHECKING:
    from codeintel.analytics.parsing.ast_cache import FunctionAst

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class ConfigDataFlowComputeResult:
    """Combined result from config data flow and graph metrics computation.

    Attributes
    ----------
    data_flow
        Result from config data flow computation.
    graph_metrics
        Result from config graph metrics computation.
    error
        Error message if computation failed.
    """

    data_flow: ConfigDataFlowResult | None
    graph_metrics: ConfigGraphMetricsResult | None
    error: str | None = None


@tag(domain="analytics", target="config_data_flow", node_type="compute")
def t__config_data_flow__compute(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
    t__goids: TargetRunRecord,
) -> ConfigDataFlowComputeResult:
    """Track configuration key usage and data flow at the function level.

    Compute config data flow and graph metrics, returning rows for
    DAG-visible materialization.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__call_graph
        Upstream call_graph target result (for dependency).
    t__goids
        Upstream goids target result (for dependency).

    Returns
    -------
    ConfigDataFlowComputeResult
        Combined result with data flow and graph metrics rows.

    Notes
    -----
    The tracking includes:
    - Config key reads at function level
    - Config key propagation through calls
    - Function-level config dependencies
    - Config graph metrics for keys and modules
    """
    if t__call_graph.status != "succeeded":
        return ConfigDataFlowComputeResult(
            data_flow=None,
            graph_metrics=None,
            error=f"Upstream call_graph target failed: {t__call_graph.error}",
        )

    if t__goids.status != "succeeded":
        return ConfigDataFlowComputeResult(
            data_flow=None,
            graph_metrics=None,
            error=f"Upstream goids target failed: {t__goids.error}",
        )

    try:
        # Get graph runtime for call graph
        call_graph: nx.DiGraph = nx.DiGraph()
        try:
            graph_runtime = resolve_graph_runtime(
                env.gateway,
                env.snapshot,
                GraphRuntimeOptions(),
            )
            if graph_runtime is not None:
                call_graph = graph_runtime.ensure_call_graph()
        except (RuntimeError, ValueError, OSError) as exc:
            log.warning("Failed to load call graph: %s", exc)

        # Load function ASTs
        ast_by_goid: dict[int, FunctionAst] = {}
        missing_goids: set[int] = set()
        try:
            catalog = CatalogService.from_db(
                env.gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
            )
            request = FunctionAstLoadRequest(
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
                repo_root=env.snapshot.repo_root,
                catalog_provider=catalog,
            )
            ast_by_goid, missing_goids = load_function_asts(env.gateway, request)
        except (RuntimeError, ValueError, OSError) as exc:
            log.warning("Failed to load function ASTs: %s", exc)

        # Compute config data flow (pure compute, no persistence)
        data_flow_result = compute_config_data_flow_result(
            env.gateway,
            env.snapshot,
            call_graph=call_graph,
            ast_by_goid=ast_by_goid,
            missing_goids=missing_goids,
        )

        # Compute config graph metrics (pure compute, no persistence)
        graph_metrics_result = compute_config_graph_metrics_result(
            env.gateway,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            runtime=None,  # Use default runtime
        )

        return ConfigDataFlowComputeResult(
            data_flow=data_flow_result,
            graph_metrics=graph_metrics_result,
        )

    except Exception as exc:
        log.exception("Config data flow computation failed")
        return ConfigDataFlowComputeResult(
            data_flow=None,
            graph_metrics=None,
            error=str(exc),
        )


# --- SaveToDecorator nodes for each table ---


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.config_data_flow"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("config_data_flow"),
    table_key=value("analytics.config_data_flow"),
    columns=value(tuple(CONFIG_DATA_FLOW_COLS)),
)
@tag(domain="analytics", target="config_data_flow", node_type="compute")
def config_data_flow__rows(
    t__config_data_flow__compute: ConfigDataFlowComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.config_data_flow table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the config_data_flow table, or None if skipped/failed.
    """
    if t__config_data_flow__compute.data_flow is None:
        return None
    return t__config_data_flow__compute.data_flow.rows


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.config_graph_metrics_keys"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("config_data_flow"),
    table_key=value("analytics.config_graph_metrics_keys"),
    columns=value(CONFIG_GRAPH_METRICS_KEYS_COLS),
)
@tag(domain="analytics", target="config_data_flow", node_type="compute")
def config_graph_metrics_keys__rows(
    t__config_data_flow__compute: ConfigDataFlowComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.config_graph_metrics_keys table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the config_graph_metrics_keys table, or None if skipped/failed.
    """
    if t__config_data_flow__compute.graph_metrics is None:
        return None
    return t__config_data_flow__compute.graph_metrics.key_rows


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.config_graph_metrics_modules"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("config_data_flow"),
    table_key=value("analytics.config_graph_metrics_modules"),
    columns=value(CONFIG_GRAPH_METRICS_MODULES_COLS),
)
@tag(domain="analytics", target="config_data_flow", node_type="compute")
def config_graph_metrics_modules__rows(
    t__config_data_flow__compute: ConfigDataFlowComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.config_graph_metrics_modules table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the config_graph_metrics_modules table, or None if skipped/failed.
    """
    if t__config_data_flow__compute.graph_metrics is None:
        return None
    return t__config_data_flow__compute.graph_metrics.module_rows


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.config_projection_key_edges"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("config_data_flow"),
    table_key=value("analytics.config_projection_key_edges"),
    columns=value(CONFIG_PROJECTION_KEY_EDGES_COLS),
)
@tag(domain="analytics", target="config_data_flow", node_type="compute")
def config_projection_key_edges__rows(
    t__config_data_flow__compute: ConfigDataFlowComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.config_projection_key_edges table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the config_projection_key_edges table, or None if skipped/failed.
    """
    if t__config_data_flow__compute.graph_metrics is None:
        return None
    return t__config_data_flow__compute.graph_metrics.key_edge_rows


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.config_projection_module_edges"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("config_data_flow"),
    table_key=value("analytics.config_projection_module_edges"),
    columns=value(CONFIG_PROJECTION_MODULE_EDGES_COLS),
)
@tag(domain="analytics", target="config_data_flow", node_type="compute")
def config_projection_module_edges__rows(
    t__config_data_flow__compute: ConfigDataFlowComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.config_projection_module_edges table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the config_projection_module_edges table, or None if skipped/failed.
    """
    if t__config_data_flow__compute.graph_metrics is None:
        return None
    return t__config_data_flow__compute.graph_metrics.module_edge_rows


@dataclass(frozen=True)
class _ConfigMaterializations:
    """Bundle of materialization results for config_data_flow target."""

    data_flow: dict[str, Any]
    keys: dict[str, Any]
    modules: dict[str, Any]
    key_edges: dict[str, Any]
    module_edges: dict[str, Any]


@tag(domain="analytics", target="config_data_flow", node_type="compute")
def config_data_flow__materializations(
    m__analytics__config_data_flow: dict[str, Any],
    m__analytics__config_graph_metrics_keys: dict[str, Any],
    m__analytics__config_graph_metrics_modules: dict[str, Any],
    m__analytics__config_projection_key_edges: dict[str, Any],
    m__analytics__config_projection_module_edges: dict[str, Any],
) -> _ConfigMaterializations:
    """Bundle materialization results for the materialize node.

    Parameters
    ----------
    m__analytics__config_data_flow
        Materialization metadata for config_data_flow table.
    m__analytics__config_graph_metrics_keys
        Materialization metadata for config_graph_metrics_keys table.
    m__analytics__config_graph_metrics_modules
        Materialization metadata for config_graph_metrics_modules table.
    m__analytics__config_projection_key_edges
        Materialization metadata for config_projection_key_edges table.
    m__analytics__config_projection_module_edges
        Materialization metadata for config_projection_module_edges table.

    Returns
    -------
    _ConfigMaterializations
        Bundled materialization metadata.
    """
    return _ConfigMaterializations(
        data_flow=m__analytics__config_data_flow,
        keys=m__analytics__config_graph_metrics_keys,
        modules=m__analytics__config_graph_metrics_modules,
        key_edges=m__analytics__config_projection_key_edges,
        module_edges=m__analytics__config_projection_module_edges,
    )


@tag(domain="analytics", target="config_data_flow", node_type="materialize")
def t__config_data_flow(
    env: BuildEnv,
    graph: TargetGraph,
    t__config_data_flow__compute: ConfigDataFlowComputeResult,
    config_data_flow__materializations: _ConfigMaterializations,
) -> TargetRunRecord:
    """Materialize config data flow target.

    Combines all materializations into a single TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__config_data_flow__compute
        Computed config data flow result from the compute node.
    config_data_flow__materializations
        Bundled materialization metadata for all tables.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    if t__config_data_flow__compute.error:
        return TargetRunRecord(
            target="config_data_flow",
            plugin_name="native:config_data_flow",
            status="failed",
            input_hash="",
            options_hash=None,
            duration_ms=0.0,
            row_counts={},
            error=t__config_data_flow__compute.error,
            datasets=(),
            artifacts=(),
        )

    mat = config_data_flow__materializations
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name="config_data_flow",
        materializations={
            "analytics.config_data_flow": mat.data_flow,
            "analytics.config_graph_metrics_keys": mat.keys,
            "analytics.config_graph_metrics_modules": mat.modules,
            "analytics.config_projection_key_edges": mat.key_edges,
            "analytics.config_projection_module_edges": mat.module_edges,
        },
    )


__all__ = [
    "ConfigDataFlowComputeResult",
    "config_data_flow__materializations",
    "config_data_flow__rows",
    "config_graph_metrics_keys__rows",
    "config_graph_metrics_modules__rows",
    "config_projection_key_edges__rows",
    "config_projection_module_edges__rows",
    "t__config_data_flow",
    "t__config_data_flow__compute",
]
