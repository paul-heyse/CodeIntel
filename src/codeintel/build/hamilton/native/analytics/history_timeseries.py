"""Native Hamilton implementation for history_timeseries target.

This module provides the Hamilton native nodes for history timeseries metrics:
- `t__history_timeseries__compute`: Pure compute node for history timeseries
- `t__history_timeseries`: Materialize node that writes the table

The history_timeseries target is a special multi-commit analysis feature that
requires explicit configuration (snapshot resolver and commit list). When these
are not available, the target returns empty results gracefully.

For full functionality, this target typically runs via the HistoryTimeseriesPlugin
which receives configuration through the plugin parameter mechanism.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

from codeintel.analytics.history.history_timeseries import (
    HISTORY_TIMESERIES_COLS,
    build_history_timeseries_rows,
)
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.manifest_hook import TargetRunRecord
    from codeintel.build.targets import TargetGraph


log = logging.getLogger(__name__)


@tag(domain="analytics", target="history_timeseries", node_type="compute")
def t__history_timeseries__compute(env: BuildEnv) -> tuple[tuple[object, ...], ...]:
    """Compute history timeseries metrics across commits.

    This is a pure compute node with no side effects. It aggregates analytics
    across multiple commits for function and module entities.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Row tuples matching HISTORY_TIMESERIES_COLS schema.

    Notes
    -----
    This target requires special multi-commit configuration:
    - A snapshot resolver to access different commit databases
    - A list of commits to include in the analysis

    When these are not configured, this node returns an empty tuple and
    the target completes successfully with zero rows.

    For full history timeseries functionality, use the HistoryTimeseriesPlugin
    which receives configuration through the plugin parameter mechanism.
    """
    # History timeseries requires multi-commit configuration that is typically
    # provided via plugin parameters. When running through Hamilton native,
    # we would need commits and a db_resolver, which aren't available in BuildEnv.
    #
    # In the standard Hamilton flow, we return empty since the full configuration
    # isn't available. The HistoryTimeseriesPlugin handles the configured case.
    _ = env  # env would be used if multi-commit configuration were available
    log.info(
        "history_timeseries: Multi-commit configuration not available via BuildEnv. "
        "For full functionality, use HistoryTimeseriesPlugin with explicit configuration."
    )
    return ()


@tag(domain="analytics", target="history_timeseries", node_type="materialize")
def t__history_timeseries(
    env: BuildEnv,
    graph: TargetGraph,
    t__history_timeseries__compute: tuple[tuple[object, ...], ...],
) -> TargetRunRecord:
    """Materialize history timeseries table to DuckDB.

    This is the only side-effect boundary for this target. It writes
    the computed history metrics to DuckDB and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__history_timeseries__compute
        Computed history timeseries rows from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following table:
    - analytics.history_timeseries
    """
    executor = NativeTargetExecutor.for_target(env, graph, "history_timeseries")

    if executor.should_skip():
        return executor.skip()

    def compute() -> dict[str, int]:
        # Ensure table exists
        backend = DuckDBPolicyBackend(env.gateway)
        backend.ensure_table("analytics.history_timeseries")

        ctx = MaterializationContext(
            gateway=env.gateway,
            snapshot=env.snapshot,
            validate=env.validate_outputs,
            owner_target="history_timeseries",
            input_hash=executor.input_hash,
        )

        # Materialize history timeseries table
        ref = materialize_rows(
            ctx,
            "analytics.history_timeseries",
            t__history_timeseries__compute,
            HISTORY_TIMESERIES_COLS,
        )

        return {"analytics.history_timeseries": ref.row_count or 0}

    return executor.execute(compute)


# Export node names for Hamilton discovery
__all__ = [
    "HISTORY_TIMESERIES_COLS",
    "build_history_timeseries_rows",
    "t__history_timeseries",
    "t__history_timeseries__compute",
]
