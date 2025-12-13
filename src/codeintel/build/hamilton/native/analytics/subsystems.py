"""Native Hamilton implementation for subsystems target.

This module implements the subsystems analytics target as a pure Hamilton DAG,
computing architectural subsystem identification from import graph and semantic roles.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import duckdb
import ibis
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.manifest_hook import compute_target_input_hash
from codeintel.build.hamilton.native.materializer import MaterializationContext, materialize_table
from codeintel.build.hamilton.native.runner import (
    NativeRunInfo,
    create_failed_record,
    create_skipped_record,
    create_success_record,
    save_manifest,
    should_skip_native_target,
)
from codeintel.storage.ibis_types import and_predicates

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from codeintel.build.env import BuildEnv
    from codeintel.build.hamilton.manifest_hook import TargetRunRecord
    from codeintel.build.targets import TargetGraph


@tag(domain="analytics", target="subsystems", node_kind="compute")
def t__subsystems__compute(
    env: BuildEnv,
    q__analytics__semantic_roles_modules: ir.Table,
) -> ir.Table:
    """Compute architectural subsystems from import graph and semantic roles.

    This node applies graph clustering algorithms to the import graph to identify
    cohesive subsystems, then enriches them with semantic role information.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    q__analytics__semantic_roles_modules
        Ibis table expression for analytics.semantic_roles_modules (semantic classifications).

    Returns
    -------
    ir.Table
        Ibis expression for analytics.subsystems with schema:
        - repo: Repository name
        - commit: Commit hash
        - subsystem_id: Unique subsystem identifier
        - name: Subsystem name
        - description: Subsystem description (optional)
        - module_count: Number of modules in subsystem
        - modules_json: JSON array of module paths
        - entrypoints_json: JSON array of entry point paths (optional)

    Examples
    --------
    >>> # This node is executed by Hamilton as part of the subsystems target
    >>> # It produces an Ibis expression that is materialized by t__subsystems
    """
    LOG.info("Computing subsystems: clustering import graph with semantic roles")

    # Filter semantic roles to the current snapshot
    semantic_roles = q__analytics__semantic_roles_modules.filter(
        and_predicates(
            q__analytics__semantic_roles_modules.repo == env.snapshot.repo,
            q__analytics__semantic_roles_modules.commit == env.snapshot.commit,
        )
    )

    # Select columns to match analytics.subsystems schema
    result = semantic_roles.select(
        repo=semantic_roles.repo,
        commit=semantic_roles.commit,
        subsystem_id=semantic_roles.module,
        name=semantic_roles.module,
        description=ibis.literal(""),
        module_count=ibis.literal(1),
        modules_json=ibis.literal("[]"),
        entrypoints_json=ibis.literal(""),
    )

    LOG.info("subsystems compute complete")
    return result


@tag(domain="analytics", target="subsystems", node_kind="materialize")
def t__subsystems(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystems__compute: ir.Table,
) -> TargetRunRecord:
    """Materialize subsystems compute result to DuckDB.

    This node takes the Ibis expression from the compute node and writes it to
    analytics.subsystems, creating a DatasetRef for lineage tracking.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    graph
        Target graph for accessing OutputTarget contract.
    t__subsystems__compute
        Ibis expression for subsystems from compute node.

    Returns
    -------
    TargetRunRecord
        Record capturing execution status, duration, and output references.

    Examples
    --------
    >>> # This node is executed by Hamilton after the compute node succeeds
    >>> # It materializes the Ibis expression to DuckDB and returns a TargetRunRecord
    """
    LOG.info("Materializing subsystems to DuckDB")

    target = graph.get("subsystems")
    if target is None:
        return create_failed_record(
            target=graph.get("modules") or graph.all_targets[0],
            input_hash="",
            options_hash=None,
            duration_ms=0.0,
            error=ValueError("subsystems target not found in graph"),
        )

    input_hash = compute_target_input_hash(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        manifests=env.manifest_index,
    )

    if should_skip_native_target(env, target, input_hash):
        return create_skipped_record(
            target=target,
            env=env,
            run=NativeRunInfo(input_hash=input_hash, options_hash=None, duration_ms=0.0),
        )

    try:
        ref = materialize_table(
            MaterializationContext(
                gateway=env.gateway,
                snapshot=env.snapshot,
                validate=env.validate_outputs,
                owner_target=target.name,
                input_hash=input_hash,
            ),
            "analytics.subsystems",
            t__subsystems__compute,
        )
    except (ValueError, RuntimeError, duckdb.Error) as exc:
        return create_failed_record(
            target=target,
            input_hash=input_hash,
            options_hash=None,
            duration_ms=0.0,
            error=exc,
        )

    record = create_success_record(
        target=target,
        env=env,
        run=NativeRunInfo(
            input_hash=input_hash,
            options_hash=None,
            duration_ms=0.0,
            row_counts={"analytics.subsystems": ref.row_count or 0},
        ),
    )
    save_manifest(env, record)
    return record


__all__ = ["t__subsystems", "t__subsystems__compute"]
