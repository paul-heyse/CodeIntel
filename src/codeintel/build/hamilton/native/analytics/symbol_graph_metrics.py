"""Native Hamilton implementation for symbol_graph_metrics target.

This module provides the Hamilton native nodes for symbol graph metrics:
- `t__symbol_graph_metrics__compute`: Pure compute node for graph metrics
- `t__symbol_graph_metrics`: Materialize node that writes both tables

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hamilton.function_modifiers import tag

from codeintel.analytics.graphs.symbol_graph_metrics import (
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.graphs.runtime import GraphRuntimeOptions, resolve_graph_runtime

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class SymbolGraphMetricsResult:
    """Result from symbol graph metrics computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    row_counts
        Row counts per table produced.
    error
        Error message if computation failed.
    """

    success: bool
    row_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@tag(domain="analytics", target="symbol_graph_metrics", node_type="compute")
def t__symbol_graph_metrics__compute(
    env: BuildEnv,
    t__symbol_uses: TargetRunRecord,
) -> SymbolGraphMetricsResult:
    """Compute graph metrics from symbol usage patterns.

    This is a compute node that calls the symbol graph metrics computation
    which handles both computation and persistence internally.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__symbol_uses
        Upstream symbol_uses target result (for dependency).

    Returns
    -------
    SymbolGraphMetricsResult
        Result indicating success or failure with row counts.

    Notes
    -----
    The metrics computed include:
    - Symbol coupling metrics
    - Cross-module symbol flow
    - Symbol centrality measures
    """
    if t__symbol_uses.status != "succeeded":
        return SymbolGraphMetricsResult(
            success=False,
            error=f"Upstream symbol_uses target failed: {t__symbol_uses.error}",
        )

    row_counts: dict[str, int] = {}

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

        repo = env.snapshot.repo
        commit = env.snapshot.commit

        # Compute module metrics (handles persistence internally)
        try:
            log.info("Computing symbol graph metrics (modules) for %s@%s", repo, commit)
            compute_symbol_graph_metrics_modules(
                env.gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
            row = env.gateway.execute(
                """
                SELECT COUNT(*) FROM analytics.symbol_graph_metrics_modules
                WHERE repo = ? AND commit = ?
                """,
                [repo, commit],
            ).fetchone()
            row_counts["analytics.symbol_graph_metrics_modules"] = int(row[0]) if row else 0
        except (RuntimeError, ValueError, OSError) as exc:
            log.warning("Symbol graph metrics (modules) failed: %s", exc)

        # Compute function metrics (handles persistence internally)
        try:
            log.info("Computing symbol graph metrics (functions) for %s@%s", repo, commit)
            compute_symbol_graph_metrics_functions(
                env.gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
            row = env.gateway.execute(
                """
                SELECT COUNT(*) FROM analytics.symbol_graph_metrics_functions
                WHERE repo = ? AND commit = ?
                """,
                [repo, commit],
            ).fetchone()
            row_counts["analytics.symbol_graph_metrics_functions"] = int(row[0]) if row else 0
        except (RuntimeError, ValueError, OSError) as exc:
            log.warning("Symbol graph metrics (functions) failed: %s", exc)

        log.info("Symbol graph metrics completed: %s", row_counts)
        return SymbolGraphMetricsResult(
            success=True,
            row_counts=row_counts,
        )

    except Exception as exc:
        log.exception("Symbol graph metrics computation failed")
        return SymbolGraphMetricsResult(
            success=False,
            error=str(exc),
        )


@tag(domain="analytics", target="symbol_graph_metrics", node_type="materialize")
def t__symbol_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__symbol_graph_metrics__compute: SymbolGraphMetricsResult,
) -> TargetRunRecord:
    """Materialize symbol graph metrics target.

    This is the entry point for the symbol_graph_metrics target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__symbol_graph_metrics__compute
        Computed symbol graph metrics result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following tables:
    - analytics.symbol_graph_metrics_functions
    - analytics.symbol_graph_metrics_modules
    """
    executor = NativeTargetExecutor.for_target(env, graph, "symbol_graph_metrics")

    if executor.should_skip():
        return executor.skip()

    if not t__symbol_graph_metrics__compute.success:
        return executor.fail(
            RuntimeError(t__symbol_graph_metrics__compute.error or "Symbol graph metrics failed")
        )

    def compute() -> dict[str, int]:
        return dict(t__symbol_graph_metrics__compute.row_counts)

    return executor.execute(compute)


__all__ = [
    "SymbolGraphMetricsResult",
    "t__symbol_graph_metrics",
    "t__symbol_graph_metrics__compute",
]
