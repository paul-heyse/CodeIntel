"""Native Hamilton implementation for the `subsystems` analytics target.

This target is implemented as a native execution boundary that reuses the
canonical subsystem inference pipeline in `codeintel.analytics.subsystems`.

The subsystem pipeline materializes both:
- ``analytics.subsystems``
- ``analytics.subsystem_modules``
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

from codeintel.analytics.subsystems.materialize import build_subsystems
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.targets import TargetGraph

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

SUBSYSTEMS_TARGET_NAME = "subsystems"

SUBSYSTEMS_TABLE_KEY = "analytics.subsystems"
SUBSYSTEM_MODULES_TABLE_KEY = "analytics.subsystem_modules"
SUBSYSTEMS_TABLE_KEYS = (SUBSYSTEMS_TABLE_KEY, SUBSYSTEM_MODULES_TABLE_KEY)

TARGET_SPECS = (
    make_output_target(
        name=SUBSYSTEMS_TARGET_NAME,
        module="analytics",
        description="Architectural subsystem inference.",
        options=TargetSpecOptions(table_keys=SUBSYSTEMS_TABLE_KEYS),
    ),
)

_COUNT_ROWS_QUERY_BY_TABLE_KEY: dict[str, str] = {
    SUBSYSTEMS_TABLE_KEY: (
        "SELECT COUNT(*) FROM analytics.subsystems WHERE repo = ? AND commit = ?"
    ),
    SUBSYSTEM_MODULES_TABLE_KEY: (
        "SELECT COUNT(*) FROM analytics.subsystem_modules WHERE repo = ? AND commit = ?"
    ),
}


def _count_rows(gateway: StorageGateway, table_key: str, *, repo: str, commit: str) -> int:
    query = _COUNT_ROWS_QUERY_BY_TABLE_KEY.get(table_key)
    if query is None:
        msg = f"Unsupported table_key for row count query: {table_key}"
        raise ValueError(msg)
    row = gateway.execute(query, [repo, commit]).fetchone()
    return int(row[0]) if row else 0


@tag(domain="analytics", target=SUBSYSTEMS_TARGET_NAME, node_type="materialize")
def t__subsystems(
    env: BuildEnv,
    graph: TargetGraph,
    t__import_graph: TargetRunRecord,
    t__semantic_roles: TargetRunRecord,
) -> TargetRunRecord:
    """Materialize subsystems and subsystem membership tables.

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
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, SUBSYSTEMS_TARGET_NAME)

    if executor.should_skip():
        return executor.skip()

    if t__import_graph.status != "succeeded":
        return executor.fail(
            RuntimeError(f"Upstream import_graph target failed: {t__import_graph.error}")
        )

    if t__semantic_roles.status != "succeeded":
        return executor.fail(
            RuntimeError(f"Upstream semantic_roles target failed: {t__semantic_roles.error}")
        )

    def compute() -> dict[str, int]:
        build_subsystems(env.gateway, env.snapshot)
        repo = env.snapshot.repo
        commit = env.snapshot.commit
        row_counts = {
            SUBSYSTEMS_TABLE_KEY: _count_rows(
                env.gateway,
                SUBSYSTEMS_TABLE_KEY,
                repo=repo,
                commit=commit,
            ),
            SUBSYSTEM_MODULES_TABLE_KEY: _count_rows(
                env.gateway,
                SUBSYSTEM_MODULES_TABLE_KEY,
                repo=repo,
                commit=commit,
            ),
        }
        log.info("subsystems: materialized row_counts=%s", row_counts)
        return row_counts

    return executor.execute(compute)


__all__ = [
    "t__subsystems",
]
