"""Native Hamilton implementation for function_history target.

This module provides the Hamilton native nodes for function history metrics:
- `t__function_history__compute`: Pure compute node for function history
- `t__function_history`: Materialize node that writes the table

The compute node calls `build_function_history_rows` from
`codeintel.analytics.functions.function_history` which returns row tuples.
The materialize node uses `materialize_rows` to persist the data to DuckDB
with proper asset tracking.
"""

from __future__ import annotations

import logging

from hamilton.function_modifiers import tag

from codeintel.analytics.functions.function_history import (
    FUNCTION_HISTORY_COLS,
    build_function_history_rows,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.build.targets import TargetGraph

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@tag(domain="analytics", target="function_history", node_type="compute")
def t__function_history__compute(env: BuildEnv) -> tuple[tuple[object, ...], ...]:
    """Compute function history metrics for all functions.

    This is a pure compute node with no side effects. It computes git history
    and churn metrics for each function and returns row data.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Row tuples matching FUNCTION_HISTORY_COLS schema.

    Notes
    -----
    The metrics computed include:
    - Function creation and last modification dates
    - Commit count and author count
    - Lines added and deleted (churn)
    - Stability bucket classification
    """
    return build_function_history_rows(env.gateway, env.snapshot)


@tag(domain="analytics", target="function_history", node_type="materialize")
def t__function_history(
    env: BuildEnv,
    graph: TargetGraph,
    t__function_history__compute: tuple[tuple[object, ...], ...],
) -> TargetRunRecord:
    """Materialize function history table to DuckDB.

    This is the only side-effect boundary for this target. It writes
    the computed history metrics to DuckDB and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__function_history__compute
        Computed function history rows from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following table:
    - analytics.function_history
    """
    executor = NativeTargetExecutor.for_target(env, graph, "function_history")

    if executor.should_skip():
        return executor.skip()

    def compute() -> dict[str, int]:
        # Ensure table exists
        backend = env.gateway.policy
        backend.ensure_table("analytics.function_history")

        ctx = MaterializationContext(
            gateway=env.gateway,
            snapshot=env.snapshot,
            validate=env.validate_outputs,
            owner_target="function_history",
            input_hash=executor.input_hash,
        )

        # Materialize function history table
        ref = materialize_rows(
            ctx,
            "analytics.function_history",
            t__function_history__compute,
            FUNCTION_HISTORY_COLS,
        )

        return {"analytics.function_history": ref.row_count or 0}

    return executor.execute(compute)


# Export node names for Hamilton discovery
__all__ = [
    "t__function_history",
    "t__function_history__compute",
]
