"""Native Hamilton implementation for history_timeseries target.

This module provides the Hamilton native nodes for history timeseries metrics:
- `t__history_timeseries__compute`: Pure compute node for history timeseries
- `t__history_timeseries`: Materialize node that writes the table

The history_timeseries target is a special multi-commit analysis feature that
requires explicit configuration (snapshot resolver and commit list). When these
are not available, the target returns empty results gracefully.

This target requires multi-commit configuration that is not yet wired into
``BuildEnv``. Until that is implemented, this target returns an empty result
set (zero rows) and succeeds.
"""

from __future__ import annotations

import logging
from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.history.history_timeseries import (
    HISTORY_TIMESERIES_COLS,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.targets import TargetGraph

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.history_timeseries"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("history_timeseries"),
    table_key=value("analytics.history_timeseries"),
    columns=value(tuple(HISTORY_TIMESERIES_COLS)),
)
@tag(
    domain="analytics",
    target="history_timeseries",
    node_type="compute",
    target_="t__history_timeseries__compute",
)
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

    Full multi-commit functionality will be enabled when BuildEnv is extended
    with a commit list and a snapshot resolver.
    """
    # History timeseries requires multi-commit configuration that is typically
    # provided via plugin parameters. When running through Hamilton native,
    # we would need commits and a db_resolver, which aren't available in BuildEnv.
    #
    # We return empty since the full configuration isn't available yet.
    _ = env  # env would be used if multi-commit configuration were available
    log.info(
        "history_timeseries: Multi-commit configuration not available via BuildEnv; "
        "returning empty result set."
    )
    return ()


@tag(domain="analytics", target="history_timeseries", node_type="materialize")
def t__history_timeseries(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__history_timeseries: dict[str, Any],
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
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="history_timeseries",
        expected_table_key="analytics.history_timeseries",
        materialization=m__analytics__history_timeseries,
    )


# Export node names for Hamilton discovery
__all__ = [
    "HISTORY_TIMESERIES_COLS",
    "t__history_timeseries",
    "t__history_timeseries__compute",
]
