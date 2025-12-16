"""Native Hamilton implementation for graph_validation target.

This module implements graph validation as a native Hamilton pipeline with:
- t__graph_validation__check: Run validation checks on all graphs
- t__graph_validation: Materialize and return TargetRunRecord

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, SupportsInt, cast

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.ibis_types import filter_by, ibis_bool

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class GraphValidationResult:
    """Result from graph validation.

    Attributes
    ----------
    success
        Whether validation passed (no errors).
    error_count
        Number of validation errors found.
    errors
        List of validation error messages.
    table_counts
        Row counts per output (validation errors).
    error
        Fatal error message if validation failed.
    """

    success: bool
    error_count: int = 0
    errors: list[str] = field(default_factory=list)
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@tag(node_type="helper")
def _validate_call_graph_integrity(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[str]:
    """Validate call graph edge integrity.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    list[str]
        List of validation errors.
    """
    errors: list[str] = []

    try:
        edges = gateway.ibis.table("graph.call_graph_edges")
        nodes = gateway.ibis.table("graph.call_graph_nodes")

        scoped_edges = filter_by(edges, edges.repo == repo, edges.commit == commit)

        caller_join = scoped_edges.left_join(
            nodes, predicates=[(scoped_edges.caller_goid_h128, nodes.goid_h128)]
        )
        orphan_callers_expr = caller_join.filter(ibis_bool(nodes.goid_h128.isnull())).count()
        orphan_callers = int(cast("SupportsInt", orphan_callers_expr.execute()))
        if orphan_callers > 0:
            errors.append(f"Found {orphan_callers} call graph edges with orphan caller GOIDs")

        callee_join = scoped_edges.left_join(
            nodes, predicates=[(scoped_edges.callee_goid_h128, nodes.goid_h128)]
        )
        orphan_callees_expr = callee_join.filter(
            ibis_bool(scoped_edges.callee_goid_h128.notnull())
            & ibis_bool(nodes.goid_h128.isnull())
        ).count()
        orphan_callees = int(cast("SupportsInt", orphan_callees_expr.execute()))
        if orphan_callees > 0:
            log.debug(
                "validation: %d call graph edges have unresolved callee GOIDs",
                orphan_callees,
            )
    except DuckDBError as exc:
        log.debug("validation: Could not validate call graph: %s", exc)

    return errors


@tag(node_type="helper")
def _validate_import_graph_integrity(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[str]:
    """Validate import graph integrity.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    list[str]
        List of validation errors.
    """
    errors: list[str] = []

    try:
        edges = gateway.ibis.table("graph.import_graph_edges")
        modules = gateway.ibis.table("graph.import_modules")
        scoped_edges = filter_by(edges, edges.repo == repo, edges.commit == commit)

        joined = scoped_edges.left_join(
            modules,
            predicates=[
                (scoped_edges.src_module, modules.module),
                (scoped_edges.repo, modules.repo),
                (scoped_edges.commit, modules.commit),
            ],
        )
        orphan_src_expr = joined.filter(ibis_bool(modules.module.isnull())).count()
        orphan_src = int(cast("SupportsInt", orphan_src_expr.execute()))
        if orphan_src > 0:
            errors.append(f"Found {orphan_src} import edges with missing source modules")

    except DuckDBError as exc:
        log.debug("validation: Could not validate import graph: %s", exc)

    return errors


@tag(node_type="helper")
def _validate_cfg_integrity(
    gateway: StorageGateway,
    _repo: str,
    _commit: str,
) -> list[str]:
    """Validate CFG integrity.

    Parameters
    ----------
    gateway
        Storage gateway.
    _repo
        Repository identifier (unused, CFG tables don't scope by repo).
    _commit
        Commit SHA (unused, CFG tables don't scope by commit).

    Returns
    -------
    list[str]
        List of validation errors.
    """
    errors: list[str] = []

    try:
        edges = gateway.ibis.table("graph.cfg_edges")
        blocks = gateway.ibis.table("graph.cfg_blocks")

        joined = edges.left_join(
            blocks,
            predicates=[
                (edges.src_block_id, blocks.block_id),
                (edges.function_goid_h128, blocks.function_goid_h128),
            ],
        )
        orphan_edges_expr = joined.filter(ibis_bool(blocks.block_id.isnull())).count()
        orphan_edges = int(cast("SupportsInt", orphan_edges_expr.execute()))
        if orphan_edges > 0:
            errors.append(f"Found {orphan_edges} CFG edges with missing source blocks")

    except DuckDBError as exc:
        log.debug("validation: Could not validate CFG: %s", exc)

    return errors


@tag(domain="graphs", target="graph_validation", node_type="compute")
def t__graph_validation__check(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
    t__import_graph: TargetRunRecord,
    t__cfg: TargetRunRecord,
) -> GraphValidationResult:
    """Run validation checks on all graph data.

    This is the compute node for the graph_validation target. It validates
    the integrity of call graph, import graph, and CFG data.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__call_graph
        Upstream call_graph target result (for dependency).
    t__import_graph
        Upstream import_graph target result (for dependency).
    t__cfg
        Upstream cfg target result (for dependency).

    Returns
    -------
    GraphValidationResult
        Result containing validation errors.

    Notes
    -----
    This target produces no tables - it only logs validation errors.
    """
    deps = [("call_graph", t__call_graph), ("import_graph", t__import_graph), ("cfg", t__cfg)]
    for name, record in deps:
        if record.status != "succeeded":
            return GraphValidationResult(
                success=False,
                error=f"Upstream {name} target failed: {record.error}",
            )

    try:
        gateway = env.gateway
        repo = env.snapshot.repo
        commit = env.snapshot.commit

        all_errors: list[str] = []

        call_graph_errors = _validate_call_graph_integrity(gateway, repo, commit)
        all_errors.extend(call_graph_errors)

        import_graph_errors = _validate_import_graph_integrity(gateway, repo, commit)
        all_errors.extend(import_graph_errors)

        cfg_errors = _validate_cfg_integrity(gateway, repo, commit)
        all_errors.extend(cfg_errors)

        for error in all_errors:
            log.warning("graph_validation: %s", error)

        log.info(
            "graph_validation: Completed with %d issues found for repo=%s commit=%s",
            len(all_errors),
            repo,
            commit,
        )

        return GraphValidationResult(
            success=len(all_errors) == 0,
            error_count=len(all_errors),
            errors=all_errors,
            table_counts={"analytics.graph_validation": len(all_errors)},
        )

    except Exception as exc:
        log.exception("Graph validation failed: %s", exc)
        return GraphValidationResult(
            success=False,
            error=str(exc),
        )


@tag(domain="graphs", target="graph_validation", node_type="materialize")
def t__graph_validation(
    env: BuildEnv,
    graph: TargetGraph,
    t__graph_validation__check: GraphValidationResult,
) -> TargetRunRecord:
    """Materialize graph validation target.

    This is the entry point for the graph_validation target. It orchestrates
    validation checks and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__graph_validation__check
        Validation result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "graph_validation")

    if executor.should_skip():
        return executor.skip()

    if t__graph_validation__check.error:
        return executor.fail(RuntimeError(t__graph_validation__check.error))

    if not t__graph_validation__check.success:
        errors_msg = "\n".join(t__graph_validation__check.errors)
        return executor.fail(RuntimeError(f"Graph validation failed:\n{errors_msg}"))

    def compute() -> dict[str, int]:
        return dict(t__graph_validation__check.table_counts)

    return executor.execute(compute)


__all__ = [
    "GraphValidationResult",
    "t__graph_validation",
    "t__graph_validation__check",
]
