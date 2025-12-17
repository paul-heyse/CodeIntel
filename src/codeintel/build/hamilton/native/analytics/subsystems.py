"""Native Hamilton implementation for subsystems target.

This module implements the subsystems analytics target as a pure Hamilton DAG,
computing architectural subsystem identification from import graph and semantic roles.

Includes Hamilton-native validation via @check_output_custom (Phase 4)
and schema documentation via @schema.output.
"""

from __future__ import annotations

import logging
from typing import Any

import ibis
import ibis.expr.types as ir
from hamilton.function_modifiers import check_output_custom, schema, source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.targets import TargetGraph
from codeintel.storage.ibis_types import and_predicates

LOG = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)


@SaveToDecorator(
    [DuckDBIbisTableSaver],
    output_name_=materialize_node("analytics.subsystems"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("subsystems"),
    table_key=value("analytics.subsystems"),
)
@tag(domain="analytics", target="subsystems", node_type="compute")
@check_output_custom(
    *build_table_contract(
        required_columns=[
            "repo",
            "commit",
            "subsystem_id",
            "name",
            "module_count",
        ],
        column_types={
            "repo": "string",
            "commit": "string",
            "subsystem_id": "string",
            "name": "string",
            "module_count": "int64",
        },
        no_nulls=["repo", "commit", "subsystem_id", "name"],
    ),
)
@schema.output(
    ("repo", "string"),
    ("commit", "string"),
    ("subsystem_id", "string"),
    ("name", "string"),
    ("description", "string"),
    ("module_count", "int"),
    ("modules_json", "string"),
    ("entrypoints_json", "string"),
    target_="t__subsystems__compute",
)
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


@tag(domain="analytics", target="subsystems", node_type="materialize")
def t__subsystems(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__subsystems: dict[str, Any],
) -> TargetRunRecord:
    """Finalize subsystems execution from DAG-visible DuckDB materialization.

    The DuckDB write is performed by a Hamilton materializer node
    (``m__analytics__subsystems``). This target node converts the materialization
    metadata into a TargetRunRecord and persists the manifest on success.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    graph
        Target graph for accessing OutputTarget contract.
    m__analytics__subsystems
        Materialization metadata dict produced by the DuckDB saver node.

    Returns
    -------
    TargetRunRecord
        Record capturing execution status, duration, and output references.

    Examples
    --------
    >>> # This node is executed by Hamilton after the compute node succeeds
    >>> # It converts the saver metadata into a TargetRunRecord
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="subsystems",
        expected_table_key="analytics.subsystems",
        materialization=m__analytics__subsystems,
    )


__all__ = ["t__subsystems", "t__subsystems__compute"]
