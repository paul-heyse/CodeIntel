"""Native Hamilton implementations for config graph analytics targets.

This module consolidates config-related graph analytics:

- ``config_data_flow``: Config key extraction and propagation.
- ``cfg_dfg_metrics``: CFG/DFG metrics derived from graph tables.

Phase 4: Analytics domain migration with Hamilton-native DAG-visible I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from codeintel.analytics.cfg_dfg.compute import (
    CfgMetricsResult,
    DfgMetricsResult,
    compute_cfg_metrics_pure,
    compute_dfg_metrics_pure,
)
from codeintel.analytics.graphs.config_data_flow import compute_config_data_flow_result
from codeintel.analytics.graphs.config_graph_metrics import compute_config_graph_metrics_result
from codeintel.analytics.resources import ProviderRegistryOptions, build_registry
from codeintel.analytics.resources.asts import AstProvider
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.graph_runtime_options import load_graph_runtime_options
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.patterns.savers import (
    SaverContext,
    TableSaveSpec,
    save_rows,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import options_hash_for_target
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.build.hashing import InputHashOptions
from codeintel.build.targets import TargetGraph
from codeintel.core.hamilton.records import TargetRunRecord
from codeintel.core.resources import ResourceNotFoundError
from codeintel.graphs.runtime import resolve_graph_runtime
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.analytics.graphs.config_data_flow import ConfigDataFlowResult
    from codeintel.analytics.graphs.config_graph_metrics import ConfigGraphMetricsResult
    from codeintel.analytics.parsing.ast_cache import FunctionAst

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

CONFIG_DATA_FLOW_TARGET_NAME = "config_data_flow"
CFG_DFG_METRICS_TARGET_NAME = "cfg_dfg_metrics"

CONFIG_DATA_FLOW_TABLE_KEY = "analytics.config_data_flow"
CONFIG_GRAPH_METRICS_KEYS_TABLE_KEY = "analytics.config_graph_metrics_keys"
CONFIG_GRAPH_METRICS_MODULES_TABLE_KEY = "analytics.config_graph_metrics_modules"
CONFIG_PROJECTION_KEY_EDGES_TABLE_KEY = "analytics.config_projection_key_edges"
CONFIG_PROJECTION_MODULE_EDGES_TABLE_KEY = "analytics.config_projection_module_edges"
CONFIG_DATA_FLOW_TABLE_KEYS = (
    CONFIG_DATA_FLOW_TABLE_KEY,
    CONFIG_GRAPH_METRICS_KEYS_TABLE_KEY,
    CONFIG_GRAPH_METRICS_MODULES_TABLE_KEY,
    CONFIG_PROJECTION_KEY_EDGES_TABLE_KEY,
    CONFIG_PROJECTION_MODULE_EDGES_TABLE_KEY,
)

CFG_FUNCTION_METRICS_TABLE_KEY = "analytics.cfg_function_metrics"
CFG_BLOCK_METRICS_TABLE_KEY = "analytics.cfg_block_metrics"
CFG_FUNCTION_METRICS_EXT_TABLE_KEY = "analytics.cfg_function_metrics_ext"
DFG_FUNCTION_METRICS_TABLE_KEY = "analytics.dfg_function_metrics"
DFG_BLOCK_METRICS_TABLE_KEY = "analytics.dfg_block_metrics"
DFG_FUNCTION_METRICS_EXT_TABLE_KEY = "analytics.dfg_function_metrics_ext"
CFG_DFG_METRICS_TABLE_KEYS = (
    CFG_FUNCTION_METRICS_TABLE_KEY,
    CFG_BLOCK_METRICS_TABLE_KEY,
    CFG_FUNCTION_METRICS_EXT_TABLE_KEY,
    DFG_FUNCTION_METRICS_TABLE_KEY,
    DFG_BLOCK_METRICS_TABLE_KEY,
    DFG_FUNCTION_METRICS_EXT_TABLE_KEY,
)
CONFIG_DATA_FLOW_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=CONFIG_DATA_FLOW_TARGET_NAME,
    hash_options_node="config_data_flow__hash_options",
)
CFG_DFG_METRICS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=CFG_DFG_METRICS_TARGET_NAME,
    hash_options_node="cfg_dfg_metrics__hash_options",
)


@tag_helper(domain="analytics")
def gateway(env: BuildEnv) -> StorageGateway:
    """Expose the storage gateway for config graph nodes.

    Returns
    -------
    StorageGateway
        Storage gateway for the current build environment.
    """
    return env.gateway


@tag_helper(domain="analytics", target=CONFIG_DATA_FLOW_TARGET_NAME)
def config_data_flow__hash_options(env: BuildEnv) -> InputHashOptions:
    """Build hash inputs for config_data_flow execution.

    Returns
    -------
    InputHashOptions
        Hash inputs for manifest-based skip evaluation.
    """
    return InputHashOptions(
        options_hash=options_hash_for_target(env, CONFIG_DATA_FLOW_TARGET_NAME),
        manifests=env.manifest_index,
    )


@tag_helper(domain="analytics", target=CONFIG_DATA_FLOW_TARGET_NAME)
def config_data_flow__skip(
    env: BuildEnv,
    graph: TargetGraph,
    config_data_flow__hash_options: InputHashOptions,
) -> bool:
    """Return True when config_data_flow should be skipped.

    Returns
    -------
    bool
        True when the target should be skipped.
    """
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        CONFIG_DATA_FLOW_TARGET_NAME,
        hash_options=config_data_flow__hash_options,
    )
    return executor.should_skip()


@tag_helper(domain="analytics", target=CFG_DFG_METRICS_TARGET_NAME)
def cfg_dfg_metrics__hash_options(env: BuildEnv) -> InputHashOptions:
    """Build hash inputs for cfg_dfg_metrics execution.

    Returns
    -------
    InputHashOptions
        Hash inputs for manifest-based skip evaluation.
    """
    return InputHashOptions(
        options_hash=options_hash_for_target(env, CFG_DFG_METRICS_TARGET_NAME),
        manifests=env.manifest_index,
    )


@tag_helper(domain="analytics", target=CFG_DFG_METRICS_TARGET_NAME)
def cfg_dfg_metrics__skip(
    env: BuildEnv,
    graph: TargetGraph,
    cfg_dfg_metrics__hash_options: InputHashOptions,
) -> bool:
    """Return True when cfg_dfg_metrics should be skipped.

    Returns
    -------
    bool
        True when the target should be skipped.
    """
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        CFG_DFG_METRICS_TARGET_NAME,
        hash_options=cfg_dfg_metrics__hash_options,
    )
    return executor.should_skip()


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


@tag_compute(domain="analytics", target=CONFIG_DATA_FLOW_TARGET_NAME)
def t__config_data_flow__compute(
    env: BuildEnv,
    gateway: StorageGateway,
    t__call_graph: TargetRunRecord,
    t__goids: TargetRunRecord,
    *,
    config_data_flow__skip: bool,
) -> ConfigDataFlowComputeResult:
    """Track configuration key usage and data flow at the function level.

    Compute config data flow and graph metrics, returning rows for
    DAG-visible materialization.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    gateway
        Storage gateway for analytics queries.
    t__call_graph
        Upstream call_graph target result (for dependency).
    t__goids
        Upstream goids target result (for dependency).
    config_data_flow__skip
        Skip flag derived from manifest-based input hash evaluation.
    config_data_flow__skip
        Skip flag derived from manifest-based input hash evaluation.

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

    if config_data_flow__skip:
        return ConfigDataFlowComputeResult(data_flow=None, graph_metrics=None)

    try:
        # Get graph runtime for call graph
        call_graph: nx.DiGraph = nx.DiGraph()
        try:
            graph_runtime = resolve_graph_runtime(
                gateway,
                env.snapshot,
                load_graph_runtime_options(env, target_name=CONFIG_DATA_FLOW_TARGET_NAME),
            )
            if graph_runtime is not None:
                call_graph = graph_runtime.ensure_call_graph()
        except (RuntimeError, ValueError, OSError) as exc:
            log.warning("Failed to load call graph: %s", exc)

        # Load function ASTs
        ast_by_goid: dict[int, FunctionAst] = {}
        missing_goids: set[int] = set()
        registry = build_registry(
            gateway=gateway,
            snapshot=env.snapshot,
            registry_options=ProviderRegistryOptions(
                include_graphs=False,
                include_asts=True,
            ),
        )
        try:
            ast_data = registry.require(AstProvider).get()
            ast_by_goid = ast_data.function_ast_map
            missing_goids = ast_data.missing_function_goids
        except (ResourceNotFoundError, RuntimeError, ValueError, OSError) as exc:
            log.warning("Failed to load function ASTs: %s", exc)

        # Compute config data flow (pure compute, no persistence)
        data_flow_result = compute_config_data_flow_result(
            gateway,
            env.snapshot,
            call_graph=call_graph,
            ast_by_goid=ast_by_goid,
            missing_goids=missing_goids,
        )

        # Compute config graph metrics (pure compute, no persistence)
        graph_metrics_result = compute_config_graph_metrics_result(
            gateway,
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


@save_rows(
    context=CONFIG_DATA_FLOW_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CONFIG_DATA_FLOW_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=CONFIG_DATA_FLOW_TARGET_NAME,
    target_="config_data_flow__rows",
)
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


@save_rows(
    context=CONFIG_DATA_FLOW_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CONFIG_GRAPH_METRICS_KEYS_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=CONFIG_DATA_FLOW_TARGET_NAME,
    target_="config_graph_metrics_keys__rows",
)
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


@save_rows(
    context=CONFIG_DATA_FLOW_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CONFIG_GRAPH_METRICS_MODULES_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=CONFIG_DATA_FLOW_TARGET_NAME,
    target_="config_graph_metrics_modules__rows",
)
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


@save_rows(
    context=CONFIG_DATA_FLOW_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CONFIG_PROJECTION_KEY_EDGES_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=CONFIG_DATA_FLOW_TARGET_NAME,
    target_="config_projection_key_edges__rows",
)
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


@save_rows(
    context=CONFIG_DATA_FLOW_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CONFIG_PROJECTION_MODULE_EDGES_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=CONFIG_DATA_FLOW_TARGET_NAME,
    target_="config_projection_module_edges__rows",
)
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


config_data_flow__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=CONFIG_DATA_FLOW_TARGET_NAME,
    table_keys=CONFIG_DATA_FLOW_TABLE_KEYS,
)


@codeintel_target(domain="analytics", target=CONFIG_DATA_FLOW_TARGET_NAME)
def t__config_data_flow(
    env: BuildEnv,
    graph: TargetGraph,
    t__config_data_flow__compute: ConfigDataFlowComputeResult,
    config_data_flow__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Config key usage flow through functions.

    Combines all materializations into a single TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__config_data_flow__compute
        Computed config data flow result from the compute node.
    config_data_flow__table_materializations
        Materialization metadata for config data flow tables.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    if t__config_data_flow__compute.error:
        options_hash = options_hash_for_target(env, CONFIG_DATA_FLOW_TARGET_NAME)
        return TargetRunRecord(
            target=CONFIG_DATA_FLOW_TARGET_NAME,
            plugin_name=f"native:{CONFIG_DATA_FLOW_TARGET_NAME}",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__config_data_flow__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            graph=graph,
            target_name=CONFIG_DATA_FLOW_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=config_data_flow__table_materializations,
    )


# ---------------------------------------------------------------------------
# cfg_dfg_metrics target
# ---------------------------------------------------------------------------


_CFG_DFG_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, CfgMetricsResult, DfgMetricsResult)


@tag_compute(domain="analytics", target=CFG_DFG_METRICS_TARGET_NAME)
def t__cfg_dfg_metrics__compute_cfg(
    env: BuildEnv,
    gateway: StorageGateway,
    *,
    cfg_dfg_metrics__skip: bool,
) -> CfgMetricsResult | None:
    """Compute CFG metrics for all functions in the snapshot.

    Returns
    -------
    CfgMetricsResult | None
        Computed metrics, or None when the target is skipped.
    """
    if cfg_dfg_metrics__skip:
        return None
    return compute_cfg_metrics_pure(
        gateway,
        env.snapshot.repo,
        env.snapshot.commit,
    )


@tag_compute(domain="analytics", target=CFG_DFG_METRICS_TARGET_NAME)
def t__cfg_dfg_metrics__compute_dfg(
    env: BuildEnv,
    gateway: StorageGateway,
    *,
    cfg_dfg_metrics__skip: bool,
) -> DfgMetricsResult | None:
    """Compute DFG metrics for all functions in the snapshot.

    Returns
    -------
    DfgMetricsResult | None
        Computed metrics, or None when the target is skipped.
    """
    if cfg_dfg_metrics__skip:
        return None
    return compute_dfg_metrics_pure(
        gateway,
        env.snapshot.repo,
        env.snapshot.commit,
    )


@save_rows(
    context=CFG_DFG_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CFG_FUNCTION_METRICS_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=CFG_DFG_METRICS_TARGET_NAME,
    target_="cfg_function_metrics__rows",
)
def cfg_function_metrics__rows(
    t__cfg_dfg_metrics__compute_cfg: CfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.cfg_function_metrics.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when the target is skipped.
    """
    if t__cfg_dfg_metrics__compute_cfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_cfg.fn_rows)


@save_rows(
    context=CFG_DFG_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CFG_BLOCK_METRICS_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=CFG_DFG_METRICS_TARGET_NAME,
    target_="cfg_block_metrics__rows",
)
def cfg_block_metrics__rows(
    t__cfg_dfg_metrics__compute_cfg: CfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.cfg_block_metrics.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when the target is skipped.
    """
    if t__cfg_dfg_metrics__compute_cfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_cfg.block_rows)


@save_rows(
    context=CFG_DFG_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CFG_FUNCTION_METRICS_EXT_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=CFG_DFG_METRICS_TARGET_NAME,
    target_="cfg_function_metrics_ext__rows",
)
def cfg_function_metrics_ext__rows(
    t__cfg_dfg_metrics__compute_cfg: CfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.cfg_function_metrics_ext.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when the target is skipped.
    """
    if t__cfg_dfg_metrics__compute_cfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_cfg.ext_rows)


@save_rows(
    context=CFG_DFG_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=DFG_FUNCTION_METRICS_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=CFG_DFG_METRICS_TARGET_NAME,
    target_="dfg_function_metrics__rows",
)
def dfg_function_metrics__rows(
    t__cfg_dfg_metrics__compute_dfg: DfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.dfg_function_metrics.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when the target is skipped.
    """
    if t__cfg_dfg_metrics__compute_dfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_dfg.fn_rows)


@save_rows(
    context=CFG_DFG_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=DFG_BLOCK_METRICS_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=CFG_DFG_METRICS_TARGET_NAME,
    target_="dfg_block_metrics__rows",
)
def dfg_block_metrics__rows(
    t__cfg_dfg_metrics__compute_dfg: DfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.dfg_block_metrics.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when the target is skipped.
    """
    if t__cfg_dfg_metrics__compute_dfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_dfg.block_rows)


@save_rows(
    context=CFG_DFG_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=DFG_FUNCTION_METRICS_EXT_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=CFG_DFG_METRICS_TARGET_NAME,
    target_="dfg_function_metrics_ext__rows",
)
def dfg_function_metrics_ext__rows(
    t__cfg_dfg_metrics__compute_dfg: DfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.dfg_function_metrics_ext.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when the target is skipped.
    """
    if t__cfg_dfg_metrics__compute_dfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_dfg.ext_rows)


cfg_dfg_metrics__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=CFG_DFG_METRICS_TARGET_NAME,
    table_keys=CFG_DFG_METRICS_TABLE_KEYS,
)


@codeintel_target(domain="analytics", target=CFG_DFG_METRICS_TARGET_NAME)
def t__cfg_dfg_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    cfg_dfg_metrics__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Control-flow and data-flow graph metrics per function.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            graph=graph,
            target_name=CFG_DFG_METRICS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=cfg_dfg_metrics__table_materializations,
    )


__all__ = [
    "ConfigDataFlowComputeResult",
    "cfg_block_metrics__rows",
    "cfg_dfg_metrics__hash_options",
    "cfg_dfg_metrics__skip",
    "cfg_dfg_metrics__table_materializations",
    "cfg_function_metrics__rows",
    "cfg_function_metrics_ext__rows",
    "config_data_flow__hash_options",
    "config_data_flow__rows",
    "config_data_flow__skip",
    "config_data_flow__table_materializations",
    "config_graph_metrics_keys__rows",
    "config_graph_metrics_modules__rows",
    "config_projection_key_edges__rows",
    "config_projection_module_edges__rows",
    "dfg_block_metrics__rows",
    "dfg_function_metrics__rows",
    "dfg_function_metrics_ext__rows",
    "t__cfg_dfg_metrics",
    "t__cfg_dfg_metrics__compute_cfg",
    "t__cfg_dfg_metrics__compute_dfg",
    "t__config_data_flow",
    "t__config_data_flow__compute",
]
