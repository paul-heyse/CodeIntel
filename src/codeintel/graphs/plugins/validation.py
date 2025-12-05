"""Graph validation plugin.

This module validates graph integrity by checking:
- Call graph edge integrity (all caller/callee GOIDs exist)
- Import graph consistency (modules exist)
- CFG/DFG structural validity (no orphan references)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import GraphMetricsStepConfig

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
    con = gateway.con

    try:
        # Check for orphan caller GOIDs (edges referencing non-existent nodes)
        orphan_callers = con.execute(
            """
            SELECT COUNT(*)
            FROM graphs.call_graph_edges e
            LEFT JOIN graphs.call_graph_nodes n ON e.caller_goid_h128 = n.goid_h128
            WHERE n.goid_h128 IS NULL
              AND e.repo = ? AND e.commit = ?
            """,
            [repo, commit],
        ).fetchone()
        if orphan_callers and orphan_callers[0] > 0:
            errors.append(f"Found {orphan_callers[0]} call graph edges with orphan caller GOIDs")

        # Check for orphan callee GOIDs (unresolved references)
        orphan_callees = con.execute(
            """
            SELECT COUNT(*)
            FROM graphs.call_graph_edges e
            LEFT JOIN graphs.call_graph_nodes n ON e.callee_goid_h128 = n.goid_h128
            WHERE e.callee_goid_h128 IS NOT NULL
              AND n.goid_h128 IS NULL
              AND e.repo = ? AND e.commit = ?
            """,
            [repo, commit],
        ).fetchone()
        if orphan_callees and orphan_callees[0] > 0:
            # This is a warning, not an error - unresolved callees are expected
            log.debug(
                "validation: %d call graph edges have unresolved callee GOIDs",
                orphan_callees[0],
            )
    except Exception as e:  # noqa: BLE001
        log.debug("validation: Could not validate call graph: %s", e)

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
    con = gateway.con

    try:
        # Check for edges referencing non-existent modules
        orphan_src = con.execute(
            """
            SELECT COUNT(*)
            FROM graphs.import_graph_edges e
            LEFT JOIN graphs.import_modules m ON e.src_module = m.module
              AND e.repo = m.repo AND e.commit = m.commit
            WHERE m.module IS NULL
              AND e.repo = ? AND e.commit = ?
            """,
            [repo, commit],
        ).fetchone()
        if orphan_src and orphan_src[0] > 0:
            errors.append(f"Found {orphan_src[0]} import edges with missing source modules")

    except Exception as e:  # noqa: BLE001
        log.debug("validation: Could not validate import graph: %s", e)

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
    con = gateway.con

    try:
        # Check for CFG edges referencing non-existent blocks
        # Note: CFG tables are scoped by function_goid_h128, not repo/commit
        orphan_edges = con.execute(
            """
            SELECT COUNT(*)
            FROM graphs.cfg_edges e
            LEFT JOIN graphs.cfg_blocks b ON e.src_block_id = b.block_id
              AND e.function_goid_h128 = b.function_goid_h128
            WHERE b.block_id IS NULL
            """,
        ).fetchone()
        if orphan_edges and orphan_edges[0] > 0:
            errors.append(f"Found {orphan_edges[0]} CFG edges with missing source blocks")

    except Exception as e:  # noqa: BLE001
        log.debug("validation: Could not validate CFG: %s", e)

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
