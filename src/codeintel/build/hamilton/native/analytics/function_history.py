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
from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.functions.function_history import (
    FUNCTION_HISTORY_COLS,
    build_function_history_rows,
)
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
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="function_history",
        expected_table_key="analytics.function_history",
        materialization=m__analytics__function_history,
    )


# Export node names for Hamilton discovery
__all__ = [
    "t__function_history",
    "t__function_history__compute",
]
