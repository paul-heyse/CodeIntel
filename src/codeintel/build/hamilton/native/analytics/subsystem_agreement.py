"""Native Hamilton implementation for subsystem_agreement target.

This module provides the Hamilton native nodes for subsystem agreement:
- `t__subsystem_agreement__compute`: Pure compute node for agreement
- `t__subsystem_agreement`: Materialize node that writes the table

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hamilton.function_modifiers import tag

from codeintel.analytics.graphs.subsystem_agreement import compute_subsystem_agreement
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class SubsystemAgreementResult:
    """Result from subsystem agreement computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    row_count
        Number of rows written.
    error
        Error message if computation failed.
    """

    success: bool
    row_count: int = 0
    error: str | None = None


@tag(domain="analytics", target="subsystem_agreement", node_type="compute")
def t__subsystem_agreement__compute(
    env: BuildEnv,
    t__subsystems: TargetRunRecord,
) -> SubsystemAgreementResult:
    """Compare subsystem assignments with import community labels.

    This is a compute node that calls the subsystem agreement computation
    which handles both computation and persistence internally.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__subsystems
        Upstream subsystems target result (for dependency).

    Returns
    -------
    SubsystemAgreementResult
        Result indicating success or failure with row count.

    Notes
    -----
    The comparison checks consistency between:
    - Inferred subsystem assignments
    - Import graph community detection
    - Identifies disagreement areas
    """
    if t__subsystems.status != "succeeded":
        return SubsystemAgreementResult(
            success=False,
            error=f"Upstream subsystems target failed: {t__subsystems.error}",
        )

    try:
        # Compute subsystem agreement (handles persistence internally)
        log.info(
            "Computing subsystem agreement for %s@%s",
            env.snapshot.repo,
            env.snapshot.commit,
        )
        compute_subsystem_agreement(
            env.gateway,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )

        # Get row count
        row = env.gateway.execute(
            """
            SELECT COUNT(*) FROM analytics.subsystem_agreement
            WHERE repo = ? AND commit = ?
            """,
            [env.snapshot.repo, env.snapshot.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return SubsystemAgreementResult(
            success=True,
            row_count=row_count,
        )

    except Exception as exc:
        log.exception("Subsystem agreement computation failed")
        return SubsystemAgreementResult(
            success=False,
            error=str(exc),
        )


@tag(domain="analytics", target="subsystem_agreement", node_type="materialize")
def t__subsystem_agreement(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystem_agreement__compute: SubsystemAgreementResult,
) -> TargetRunRecord:
    """Materialize subsystem agreement target.

    This is the entry point for the subsystem_agreement target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__subsystem_agreement__compute
        Computed subsystem agreement result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following table:
    - analytics.subsystem_agreement
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
]
