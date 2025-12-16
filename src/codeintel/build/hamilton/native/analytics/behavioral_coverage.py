"""Native Hamilton implementation for behavioral_coverage target.

This module provides the Hamilton native nodes for behavioral coverage:
- `t__behavioral_coverage__compute`: Pure compute node for behavioral coverage
- `t__behavioral_coverage`: Materialize node that writes the table

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hamilton.function_modifiers import tag

from codeintel.analytics.testing.profiles.builder import build_behavioral_coverage
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class BehavioralCoverageResult:
    """Result from behavioral coverage computation.

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


@tag(domain="analytics", target="behavioral_coverage", node_type="compute")
def t__behavioral_coverage__compute(
    env: BuildEnv,
    t__test_profile: TargetRunRecord,
) -> BehavioralCoverageResult:
    """Assign heuristic behavior tags to tests.

    This is a compute node that calls the behavioral coverage builder
    which handles both computation and persistence internally.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__test_profile
        Upstream test_profile target result (for dependency).

    Returns
    -------
    BehavioralCoverageResult
        Result indicating success or failure with row count.

    Notes
    -----
    The classifications include:
    - Unit tests vs integration tests
    - Behavioral patterns and coverage types
    """
    if t__test_profile.status != "succeeded":
        return BehavioralCoverageResult(
            success=False,
            error=f"Upstream test_profile target failed: {t__test_profile.error}",
        )

    try:
        # Build behavioral coverage (handles persistence internally)
        # Note: llm_runner is not available in native mode
        build_behavioral_coverage(
            env.gateway,
            env.snapshot,
            llm_runner=None,
        )

        # Get row count
        row = env.gateway.execute(
            """
            SELECT COUNT(*) FROM analytics.behavioral_coverage
            WHERE repo = ? AND commit = ?
            """,
            [env.snapshot.repo, env.snapshot.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return BehavioralCoverageResult(
            success=True,
            row_count=row_count,
        )

    except Exception as exc:
        log.exception("Behavioral coverage computation failed")
        return BehavioralCoverageResult(
            success=False,
            error=str(exc),
        )


@tag(domain="analytics", target="behavioral_coverage", node_type="materialize")
def t__behavioral_coverage(
    env: BuildEnv,
    graph: TargetGraph,
    t__behavioral_coverage__compute: BehavioralCoverageResult,
) -> TargetRunRecord:
    """Materialize behavioral coverage target.

    This is the entry point for the behavioral_coverage target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__behavioral_coverage__compute
        Computed behavioral coverage result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following table:
    - analytics.behavioral_coverage
    """
    executor = NativeTargetExecutor.for_target(env, graph, "behavioral_coverage")

    if executor.should_skip():
        return executor.skip()

    if not t__behavioral_coverage__compute.success:
        return executor.fail(
            RuntimeError(t__behavioral_coverage__compute.error or "Behavioral coverage failed")
        )

    def compute() -> dict[str, int]:
        return {"analytics.behavioral_coverage": t__behavioral_coverage__compute.row_count}

    return executor.execute(compute)


__all__ = [
    "BehavioralCoverageResult",
    "t__behavioral_coverage",
    "t__behavioral_coverage__compute",
]
