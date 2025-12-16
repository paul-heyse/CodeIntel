"""Native Hamilton implementation for function_metrics target.

This module provides the Hamilton native nodes for function metrics computation:
- `t__function_metrics__compute`: Pure compute node for function metrics
- `t__function_metrics`: Materialize node that writes both tables

The compute node calls `compute_function_metrics_and_types` from
`codeintel.analytics.functions` which handles both computation and persistence.

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hamilton.function_modifiers import tag

from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_metrics_and_types,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class FunctionMetricsResult:
    """Result from function metrics computation.

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
    row_counts: dict[str, int]
    error: str | None = None


@tag(domain="analytics", target="function_metrics", node_type="compute")
def t__function_metrics__compute(env: BuildEnv) -> FunctionMetricsResult:
    """Compute function metrics and type coverage for all functions.

    This is a compute node that calls the function metrics computation
    which handles both computation and persistence internally.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    FunctionMetricsResult
        Result containing row counts for both tables.

    Notes
    -----
    The metrics computed include:
    - Lines of code (LOC, SLOC)
    - Cyclomatic complexity
    - Nesting depth
    - Type annotation coverage
    """
    try:
        options = FunctionAnalyticsOptions()
        result = compute_function_metrics_and_types(
            env.gateway,
            env.snapshot,
            options=options,
        )
        return FunctionMetricsResult(
            success=True,
            row_counts={
                "analytics.function_metrics": result.get("metrics_rows", 0),
                "analytics.function_types": result.get("types_rows", 0),
            },
        )
    except Exception as exc:
        log.exception("Function metrics computation failed")
        return FunctionMetricsResult(
            success=False,
            row_counts={},
            error=str(exc),
        )


@tag(domain="analytics", target="function_metrics", node_type="materialize")
def t__function_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__function_metrics__compute: FunctionMetricsResult,
) -> TargetRunRecord:
    """Materialize function metrics target.

    This is the entry point for the function_metrics target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__function_metrics__compute
        Computed function metrics result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following tables:
    - analytics.function_metrics
    - analytics.function_types
    """
    executor = NativeTargetExecutor.for_target(env, graph, "function_metrics")

    if executor.should_skip():
        return executor.skip()

    if not t__function_metrics__compute.success:
        return executor.fail(
            RuntimeError(t__function_metrics__compute.error or "Function metrics failed")
        )

    def compute() -> dict[str, int]:
        return dict(t__function_metrics__compute.row_counts)

    return executor.execute(compute)


__all__ = [
    "FunctionMetricsResult",
    "t__function_metrics",
    "t__function_metrics__compute",
]
