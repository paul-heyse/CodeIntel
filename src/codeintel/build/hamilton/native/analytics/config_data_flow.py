"""Native Hamilton implementation for config_data_flow target.

This module provides the Hamilton native nodes for config data flow tracking:
- `t__config_data_flow__compute`: Pure compute node for config tracking
- `t__config_data_flow`: Materialize node that writes all tables

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
from hamilton.function_modifiers import tag

from codeintel.analytics.graphs import compute_config_data_flow, compute_config_graph_metrics
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService
from codeintel.graphs.runtime import GraphRuntimeOptions, resolve_graph_runtime

if TYPE_CHECKING:
    from codeintel.analytics.parsing.ast_cache import FunctionAst

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class ConfigDataFlowResult:
    """Result from config data flow computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    error
        Error message if computation failed.
    """

    success: bool
    error: str | None = None


@tag(domain="analytics", target="config_data_flow", node_type="compute")
def t__config_data_flow__compute(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
    t__goids: TargetRunRecord,
) -> ConfigDataFlowResult:
    """Track configuration key usage and data flow at the function level.

    This is a compute node that calls the config data flow computation
    which handles both computation and persistence internally.

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
    ConfigDataFlowResult
        Result indicating success or failure.

    Notes
    -----
    The tracking includes:
    - Config key reads at function level
    - Config key propagation through calls
    - Function-level config dependencies
    """
    if t__call_graph.status != "succeeded":
        return ConfigDataFlowResult(
            success=False,
            error=f"Upstream call_graph target failed: {t__call_graph.error}",
        )

    if t__goids.status != "succeeded":
        return ConfigDataFlowResult(
            success=False,
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

        # Compute config data flow (handles persistence internally)
        compute_config_data_flow(
            env.gateway,
            env.snapshot,
            call_graph=call_graph,
            ast_by_goid=ast_by_goid,
            missing_goids=missing_goids,
        )

        # Compute config graph metrics (handles persistence internally)
        compute_config_graph_metrics(
            env.gateway,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            runtime=None,  # Use default runtime
        )

        return ConfigDataFlowResult(success=True)

    except Exception as exc:
        log.exception("Config data flow computation failed")
        return ConfigDataFlowResult(
            success=False,
            error=str(exc),
        )


@tag(domain="analytics", target="config_data_flow", node_type="materialize")
def t__config_data_flow(
    env: BuildEnv,
    graph: TargetGraph,
    t__config_data_flow__compute: ConfigDataFlowResult,
) -> TargetRunRecord:
    """Materialize config data flow target.

    This is the entry point for the config_data_flow target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__config_data_flow__compute
        Computed config data flow result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following tables:
    - analytics.config_data_flow
    - analytics.config_graph_metrics_keys
    - analytics.config_graph_metrics_modules
    - analytics.config_projection_key_edges
    - analytics.config_projection_module_edges
    """
    executor = NativeTargetExecutor.for_target(env, graph, "config_data_flow")

    if executor.should_skip():
        return executor.skip()

    if not t__config_data_flow__compute.success:
        return executor.fail(
            RuntimeError(t__config_data_flow__compute.error or "Config data flow failed")
        )

    def compute() -> dict[str, int]:
        # Data flow is persisted during compute - return empty counts
        return {
            "analytics.config_data_flow": 0,
            "analytics.config_graph_metrics_keys": 0,
            "analytics.config_graph_metrics_modules": 0,
            "analytics.config_projection_key_edges": 0,
            "analytics.config_projection_module_edges": 0,
        }

    return executor.execute(compute)


__all__ = [
    "ConfigDataFlowResult",
    "t__config_data_flow",
    "t__config_data_flow__compute",
]
