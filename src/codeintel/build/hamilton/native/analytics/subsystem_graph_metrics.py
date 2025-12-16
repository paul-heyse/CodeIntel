"""Native Hamilton implementation for subsystem_graph_metrics target.

This module provides the Hamilton native nodes for subsystem graph metrics:
- `t__subsystem_graph_metrics__compute`: Pure compute node for graph metrics
- `t__subsystem_graph_metrics`: Materialize node that writes the table

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hamilton.function_modifiers import tag

from codeintel.analytics.graphs.subsystem_graph_metrics import (
    compute_subsystem_graph_metrics,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.graphs.runtime import GraphRuntimeOptions, resolve_graph_runtime

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class SubsystemGraphMetricsResult:
    """Result from subsystem graph metrics computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    row_count
        Number of rows written.
    error
        Error message if computation failed.
    """

    success: bool
    row_count: int = 0
    error: str | None = None


@tag(domain="analytics", target="subsystem_graph_metrics", node_type="compute")
def t__subsystem_graph_metrics__compute(
    env: BuildEnv,
    t__subsystems: TargetRunRecord,
) -> SubsystemGraphMetricsResult:
    """Compute graph metrics for subsystems.

    This is a compute node that calls the subsystem graph metrics computation
    which handles both computation and persistence internally.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__subsystems
        Upstream subsystems target result (for dependency).

    Returns
    -------
    SubsystemGraphMetricsResult
        Result indicating success or failure with row count.

    Notes
    -----
    The metrics computed include:
    - Subsystem coupling metrics
    - Inter-subsystem dependencies
    - Subsystem centrality measures
    """
    if t__subsystems.status != "succeeded":
        return SubsystemGraphMetricsResult(
            success=False,
            error=f"Upstream subsystems target failed: {t__subsystems.error}",
        )

    try:
        # Get graph runtime
        try:
            graph_runtime = resolve_graph_runtime(
                env.gateway,
                env.snapshot,
                GraphRuntimeOptions(),
            )
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to resolve graph runtime: %s", exc)
            graph_runtime = None

        # Compute subsystem graph metrics (handles persistence internally)
        log.info(
            "Computing subsystem graph metrics for %s@%s",
            env.snapshot.repo,
            env.snapshot.commit,
        )
        compute_subsystem_graph_metrics(
            env.gateway,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            runtime=graph_runtime,
        )

        # Get row count
        row = env.gateway.execute(
            """
            SELECT COUNT(*) FROM analytics.subsystem_graph_metrics
            WHERE repo = ? AND commit = ?
            """,
            [env.snapshot.repo, env.snapshot.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return SubsystemGraphMetricsResult(
            success=True,
            row_count=row_count,
        )

    except Exception as exc:
        log.exception("Subsystem graph metrics computation failed")
        return SubsystemGraphMetricsResult(
            success=False,
            error=str(exc),
        )


@tag(domain="analytics", target="subsystem_graph_metrics", node_type="materialize")
def t__subsystem_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystem_graph_metrics__compute: SubsystemGraphMetricsResult,
) -> TargetRunRecord:
    """Materialize subsystem graph metrics target.

    This is the entry point for the subsystem_graph_metrics target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__subsystem_graph_metrics__compute
        Computed subsystem graph metrics result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following table:
    - analytics.subsystem_graph_metrics
    """
    executor = NativeTargetExecutor.for_target(env, graph, "subsystem_graph_metrics")

    if executor.should_skip():
        return executor.skip()

    if not t__subsystem_graph_metrics__compute.success:
        return executor.fail(
            RuntimeError(
                t__subsystem_graph_metrics__compute.error or "Subsystem graph metrics failed"
            )
        )

    def compute() -> dict[str, int]:
        return {
            "analytics.subsystem_graph_metrics": t__subsystem_graph_metrics__compute.row_count
        }

    return executor.execute(compute)


__all__ = [
    "SubsystemGraphMetricsResult",
    "t__subsystem_graph_metrics",
    "t__subsystem_graph_metrics__compute",
]
