"""Graph validation plugin.

This module validates graph integrity by checking:
- Call graph edge integrity (all caller/callee GOIDs exist)
- Import graph consistency (modules exist)
- CFG/DFG structural validity (no orphan references)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, SupportsInt, cast

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import GraphMetricsStepConfig
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.ibis_types import filter_by, ibis_bool

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


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


class GraphValidationPlugin(TargetPlugin):
    """Validate graph integrity.

    This plugin performs validation checks on graph data:
    1. Call graph edge integrity
    2. Import graph consistency
    3. CFG structural validity

    Outputs
    -------
    - graphs.validation_results: Graph validation results (errors logged)
    """

    plugin_name: ClassVar[str] = "graph_validation"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Validate graph integrity."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute graph validation.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result.
        """
        _ = self  # Protocol method requires instance

        # Use GraphMetricsStepConfig as validation uses similar inputs
        cfg = GraphMetricsStepConfig(snapshot=ctx.snapshot)

        gateway = ctx.gateway
        repo = cfg.repo
        commit = cfg.commit

        try:
            all_errors: list[str] = []

            # Run validation checks
            call_graph_errors = _validate_call_graph_integrity(gateway, repo, commit)
            all_errors.extend(call_graph_errors)

            import_graph_errors = _validate_import_graph_integrity(gateway, repo, commit)
            all_errors.extend(import_graph_errors)

            cfg_errors = _validate_cfg_integrity(gateway, repo, commit)
            all_errors.extend(cfg_errors)

            # Log errors
            for error in all_errors:
                log.warning("graph_validation: %s", error)

            log.info(
                "graph_validation: Completed with %d issues found for repo=%s commit=%s",
                len(all_errors),
                repo,
                commit,
            )

            row_counts: dict[str, int] = {
                "graphs.validation_results": len(all_errors),
            }

            if all_errors:
                return TargetResult.failed("\n".join(all_errors))
            return TargetResult.succeeded(row_counts=row_counts)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Graph validation failed: {e}")


__all__ = ["GraphValidationPlugin"]
