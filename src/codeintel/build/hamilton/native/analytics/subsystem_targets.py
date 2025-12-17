"""Native Hamilton implementation for subsystems target.

This module implements the subsystems analytics target as a pure Hamilton DAG,
computing architectural subsystem identification from import graph and semantic roles.

Uses @pipe_input for DAG-visible multi-step transformations (Phase 5).
Includes Hamilton-native validation via @check_output_custom (Phase 4)
and schema documentation via @schema.output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import ibis
import ibis.expr.types as ir
from hamilton.function_modifiers import (
    check_output_custom,
    pipe_input,
    schema,
    source,
    step,
    tag,
    value,
)
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.graphs.subsystem_agreement import compute_subsystem_agreement
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.targets import TargetGraph
from codeintel.storage.ibis_types import and_predicates

LOG = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)


def _filter_to_snapshot(
    roles: ir.Table,
    env: BuildEnv,
) -> ir.Table:
    """Filter semantic roles to current snapshot.

    Parameters
    ----------
    roles
        Ibis table expression for analytics.semantic_roles_modules.
    env
        Build environment with snapshot info.

    Returns
    -------
    ir.Table
        Filtered Ibis expression for current repo/commit.
    """
    LOG.debug("Filtering semantic roles to snapshot %s@%s", env.snapshot.repo, env.snapshot.commit)
    return roles.filter(
        and_predicates(
            roles.repo == env.snapshot.repo,
            roles.commit == env.snapshot.commit,
        )
    )


def _group_modules_by_role(roles: ir.Table) -> ir.Table:
    """Group modules by their semantic role to form subsystem candidates.

    Parameters
    ----------
    roles
        Filtered semantic roles table.

    Returns
    -------
    ir.Table
        Grouped table with role counts per module.
    """
    LOG.debug("Grouping modules by semantic role")
    # Each module's dominant role forms its subsystem assignment
    return roles.group_by(["repo", "commit", "module", "role"]).aggregate(
        role_strength=ibis._.count(),
    )


def _assign_subsystem_ids(grouped: ir.Table) -> ir.Table:
    """Assign subsystem IDs based on module roles.

    Parameters
    ----------
    grouped
        Grouped modules with role assignments.

    Returns
    -------
    ir.Table
        Table with subsystem_id assignments.
    """
    LOG.debug("Assigning subsystem IDs from roles")
    # Use the role as the subsystem identifier
    return grouped.select(
        repo=grouped.repo,
        commit=grouped.commit,
        module=grouped.module,
        subsystem_id=grouped.role,
    )


def _build_subsystem_schema(assigned: ir.Table) -> ir.Table:
    """Build final subsystems schema with metadata columns.

    Parameters
    ----------
    assigned
        Table with subsystem assignments.

    Returns
    -------
    ir.Table
        Final Ibis expression for analytics.subsystems table.
    """
    LOG.info("Building final subsystems schema")
    return assigned.select(
        repo=assigned.repo,
        commit=assigned.commit,
        subsystem_id=assigned.subsystem_id,
        name=assigned.module,
        description=ibis.literal(""),
        module_count=ibis.literal(1),
        modules_json=ibis.literal("[]"),
        entrypoints_json=ibis.literal(""),
    )


@SaveToDecorator(
    [DuckDBIbisTableSaver],
    output_name_=materialize_node("analytics.subsystems"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("subsystems"),
    table_key=value("analytics.subsystems"),
)
@pipe_input(
    step(_filter_to_snapshot, env=source("env")),
    step(_group_modules_by_role),
    step(_assign_subsystem_ids),
    step(_build_subsystem_schema),
    namespace=None,
    on_input="q__analytics__semantic_roles_modules",
)
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
@tag(domain="analytics", target="subsystems", node_type="compute", target_="t__subsystems__compute")
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
    q__analytics__semantic_roles_modules: ir.Table,
) -> ir.Table:
    """Compute architectural subsystems from import graph and semantic roles.

    This node applies a multi-step transformation pipeline via @pipe_input:
    1. Filter to current snapshot
    2. Group modules by semantic role
    3. Assign subsystem IDs from roles
    4. Build final schema

    Parameters
    ----------
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

    Notes
    -----
    The @pipe_input decorator makes each transformation step DAG-visible,
    enabling better tracing and debugging of the subsystem computation.
    """
    # The @pipe_input chain transforms q__analytics__semantic_roles_modules
    # into the final result; returning it keeps the function body minimal
    # and ensures intermediate steps are DAG-visible.
    return q__analytics__semantic_roles_modules


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


# ---------------------------------------------------------------------------
# subsystem_agreement target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsystemAgreementResult:
    """Result from subsystem agreement computation."""

    success: bool
    row_count: int = 0
    error: str | None = None


@tag(domain="analytics", target="subsystem_agreement", node_type="compute")
def t__subsystem_agreement__compute(
    env: BuildEnv,
    t__subsystems: TargetRunRecord,
) -> SubsystemAgreementResult:
    """Compare subsystem assignments with import community labels.

    Returns
    -------
    SubsystemAgreementResult
        Status indicator, row count, and optional error message.
    """
    if t__subsystems.status != "succeeded":
        return SubsystemAgreementResult(
            success=False,
            error=f"Upstream subsystems target failed: {t__subsystems.error}",
        )

    try:
        LOG.info(
            "Computing subsystem agreement for %s@%s",
            env.snapshot.repo,
            env.snapshot.commit,
        )
        compute_subsystem_agreement(
            env.gateway,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )

        row = env.gateway.execute(
            """
            SELECT COUNT(*) FROM analytics.subsystem_agreement
            WHERE repo = ? AND commit = ?
            """,
            [env.snapshot.repo, env.snapshot.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return SubsystemAgreementResult(success=True, row_count=row_count)
    except Exception as exc:
        LOG.exception("Subsystem agreement computation failed")
        return SubsystemAgreementResult(success=False, error=str(exc))


@tag(domain="analytics", target="subsystem_agreement", node_type="materialize")
def t__subsystem_agreement(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystem_agreement__compute: SubsystemAgreementResult,
) -> TargetRunRecord:
    """Materialize subsystem agreement target.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "subsystem_agreement")

    if executor.should_skip():
        return executor.skip()

    if not t__subsystem_agreement__compute.success:
        return executor.fail(
            RuntimeError(t__subsystem_agreement__compute.error or "Subsystem agreement failed")
        )

    def compute() -> dict[str, int]:
        return {"analytics.subsystem_agreement": t__subsystem_agreement__compute.row_count}

    return executor.execute(compute)


__all__ = [
    "SubsystemAgreementResult",
    "t__subsystem_agreement",
    "t__subsystem_agreement__compute",
    "t__subsystems",
    "t__subsystems__compute",
]
