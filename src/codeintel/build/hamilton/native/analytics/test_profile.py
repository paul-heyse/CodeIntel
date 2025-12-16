"""Native Hamilton implementation for test_profile target.

This module provides the Hamilton native nodes for test profiles:
- `t__test_profile__compute`: Pure compute node for test profiles
- `t__test_profile`: Materialize node that writes the table

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hamilton.function_modifiers import tag

from codeintel.analytics.testing.profiles.builder import build_test_profile
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class TestProfileResult:
    """Result from test profile computation.

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


@tag(domain="analytics", target="test_profile", node_type="compute")
def t__test_profile__compute(
    env: BuildEnv,
    t__coverage_test_edges: TargetRunRecord,
) -> TestProfileResult:
    """Build per-test profiles with coverage and subsystem context.

    This is a compute node that calls the test profile builder
    which handles both computation and persistence internally.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__coverage_test_edges
        Upstream coverage_test_edges target result (for dependency).

    Returns
    -------
    TestProfileResult
        Result indicating success or failure with row count.

    Notes
    -----
    The profiles include:
    - Coverage context for each test
    - Subsystem associations
    - Test metadata aggregation
    """
    if t__coverage_test_edges.status != "succeeded":
        return TestProfileResult(
            success=False,
            error=f"Upstream coverage_test_edges target failed: {t__coverage_test_edges.error}",
        )

    try:
        # Build test profiles (handles persistence internally)
        build_test_profile(env.gateway, env.snapshot)

        # Get row count
        row = env.gateway.execute(
            """
            SELECT COUNT(*) FROM analytics.test_profile
            WHERE repo = ? AND commit = ?
            """,
            [env.snapshot.repo, env.snapshot.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return TestProfileResult(
            success=True,
            row_count=row_count,
        )

    except Exception as exc:
        log.exception("Test profile computation failed")
        return TestProfileResult(
            success=False,
            error=str(exc),
        )


@tag(domain="analytics", target="test_profile", node_type="materialize")
def t__test_profile(
    env: BuildEnv,
    graph: TargetGraph,
    t__test_profile__compute: TestProfileResult,
) -> TargetRunRecord:
    """Materialize test profile target.

    This is the entry point for the test_profile target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__test_profile__compute
        Computed test profile result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following table:
    - analytics.test_profile
    """
    executor = NativeTargetExecutor.for_target(env, graph, "test_profile")

    if executor.should_skip():
        return executor.skip()

    if not t__test_profile__compute.success:
        return executor.fail(
            RuntimeError(t__test_profile__compute.error or "Test profile failed")
        )

    def compute() -> dict[str, int]:
        return {"analytics.test_profile": t__test_profile__compute.row_count}

    return executor.execute(compute)


__all__ = [
    "TestProfileResult",
    "t__test_profile",
    "t__test_profile__compute",
]
