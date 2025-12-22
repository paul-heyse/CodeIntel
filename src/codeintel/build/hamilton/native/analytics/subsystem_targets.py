"""Native Hamilton implementation for the `subsystems` analytics target.

This target is implemented as a native execution boundary that reuses the
canonical subsystem inference pipeline in `codeintel.analytics.subsystems`.

The subsystem pipeline materializes both:
- ``analytics.subsystems``
- ``analytics.subsystem_modules``
"""

from __future__ import annotations

import logging

from codeintel.analytics.subsystems.materialize import build_subsystems
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.target_override_tables import SUBSYSTEMS_OVERRIDE_TABLES
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
    register_output_targets,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize
from codeintel.build.hamilton.templates import executor_materialize
from codeintel.build.targets import TargetGraph
from codeintel.storage.queries.safe import count_rows_for_snapshot

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ExecutionResult)

SUBSYSTEMS_TARGET_NAME = "subsystems"

SUBSYSTEMS_TABLE_KEY = "analytics.subsystems"
SUBSYSTEM_MODULES_TABLE_KEY = "analytics.subsystem_modules"
SUBSYSTEMS_TABLE_KEYS = (SUBSYSTEMS_TABLE_KEY, SUBSYSTEM_MODULES_TABLE_KEY)

register_output_targets(
    make_output_target(
        name=SUBSYSTEMS_TARGET_NAME,
        module="analytics",
        description="Architectural subsystem inference.",
        options=TargetSpecOptions(
            table_keys=SUBSYSTEMS_TABLE_KEYS,
            override_tables=SUBSYSTEMS_OVERRIDE_TABLES,
        ),
    ),
)


@tag_compute(domain="analytics", target=SUBSYSTEMS_TARGET_NAME)
def t__subsystems__compute(
    env: BuildEnv,
    graph: TargetGraph,
    t__import_graph: TargetRunRecord,
    t__semantic_roles: TargetRunRecord,
) -> ExecutionResult:
    """Compute subsystems by executing the subsystem inference pipeline.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup and skip detection.
    t__import_graph
        Upstream import graph record.
    t__semantic_roles
        Upstream semantic roles record.

    Returns
    -------
    ExecutionResult
        Success/failure with row counts for produced tables.
    """
    executor = NativeTargetExecutor.for_target(env, graph, SUBSYSTEMS_TARGET_NAME)
    if executor.should_skip():
        return ExecutionResult.ok()

    if t__import_graph.status != "succeeded":
        return ExecutionResult.failed(
            f"Upstream import_graph target failed: {t__import_graph.error or 'unknown error'}"
        )

    if t__semantic_roles.status != "succeeded":
        return ExecutionResult.failed(
            f"Upstream semantic_roles target failed: {t__semantic_roles.error or 'unknown error'}"
        )

    try:
        build_subsystems(env.gateway, env.snapshot)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        log.exception("subsystems: build_subsystems failed")
        return ExecutionResult.failed(str(exc))

    row_counts = {
        SUBSYSTEMS_TABLE_KEY: count_rows_for_snapshot(
            env.gateway.con,
            SUBSYSTEMS_TABLE_KEY,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        ),
        SUBSYSTEM_MODULES_TABLE_KEY: count_rows_for_snapshot(
            env.gateway.con,
            SUBSYSTEM_MODULES_TABLE_KEY,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        ),
    }
    log.info("subsystems: materialized row_counts=%s", row_counts)
    return ExecutionResult.ok(table_counts=row_counts)


@tag_materialize(domain="analytics", target=SUBSYSTEMS_TARGET_NAME)
def t__subsystems(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystems__compute: ExecutionResult,
) -> TargetRunRecord:
    """Materialize a TargetRunRecord for subsystems from a compute result.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup and skip detection.
    t__subsystems__compute
        Compute result containing success/failure status and row counts.

    Returns
    -------
    TargetRunRecord
        Final execution record for the target.
    """
    return executor_materialize(
        env=env,
        graph=graph,
        target_name=SUBSYSTEMS_TARGET_NAME,
        compute_result=t__subsystems__compute,
    )


__all__ = [
    "t__subsystems",
    "t__subsystems__compute",
]
