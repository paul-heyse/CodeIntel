"""Native Hamilton implementations for history-derived analytics targets.

This module consolidates targets that derive analytics from git history:

- ``function_history``: Per-function creation/modification/churn metrics.
- ``history_timeseries``: Multi-commit timeseries analytics (currently stubbed).

The compute node calls `build_function_history_rows` from
`codeintel.analytics.functions.function_history` which returns row tuples.
The materialize node uses `materialize_rows` to persist the data to DuckDB
with proper asset tracking.
"""

from __future__ import annotations

import logging
from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.functions.function_history import (
    FUNCTION_HISTORY_COLS,
    build_function_history_rows,
)
from codeintel.analytics.history.history_timeseries import HISTORY_TIMESERIES_COLS
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.function_history"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("function_history"),
    table_key=value("analytics.function_history"),
    columns=value(tuple(FUNCTION_HISTORY_COLS)),
)
@tag(domain="analytics", target="function_history", node_type="compute", target_="t__function_history__compute")
def t__function_history__compute(env: BuildEnv, graph: TargetGraph) -> tuple[tuple[object, ...], ...] | None:
    """Compute function history metrics for all functions.

    This is a pure compute node with no side effects. It computes git history
    and churn metrics for each function and returns row data.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for manifest-driven skip checks.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples matching FUNCTION_HISTORY_COLS schema, or None when skipped.

    Notes
    -----
    The metrics computed include:
    - Function creation and last modification dates
    - Commit count and author count
    - Lines added and deleted (churn)
    - Stability bucket classification
    """
    target = graph.get("function_history")
    if target is not None:
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=None,
            manifests=env.manifest_index,
        )
        if should_skip_native_target(env, target, input_hash):
            return None
    return build_function_history_rows(env.gateway, env.snapshot)


@tag(domain="analytics", target="function_history", node_type="materialize")
def t__function_history(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__function_history: dict[str, Any],
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
    m__analytics__function_history
        Materialization metadata for analytics.function_history.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following table:
    - analytics.function_history
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="function_history",
        expected_table_key="analytics.function_history",
        materialization=m__analytics__function_history,
    )


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

    Full multi-commit functionality is not yet wired into ``BuildEnv``, so this
    node currently returns an empty result set and succeeds.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Empty row set until multi-commit configuration is available.
    """
    _ = env
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

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
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
    "t__function_history",
    "t__function_history__compute",
    "t__history_timeseries",
    "t__history_timeseries__compute",
]
