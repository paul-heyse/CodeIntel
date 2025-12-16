"""Native Hamilton implementation for coverage_test_edges target.

This module provides the Hamilton native nodes for test coverage edges:
- `t__coverage_test_edges__compute`: Pure compute node for coverage edges
- `t__coverage_test_edges`: Materialize node that writes the table

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hamilton.function_modifiers import tag

from codeintel.analytics.testing import compute_test_coverage_edges
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class CoverageTestEdgesResult:
    """Result from coverage test edges computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    error
        Error message if computation failed.
    """

    success: bool
    error: str | None = None


@tag(domain="analytics", target="coverage_test_edges", node_type="compute")
def t__coverage_test_edges__compute(
    env: BuildEnv,
    t__goids: TargetRunRecord,
) -> CoverageTestEdgesResult:
    """Compute test-to-function coverage edges.

    This is a compute node that calls the test coverage edges computation
    which handles both computation and persistence internally.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__goids
        Upstream goids target result (for dependency).

    Returns
    -------
    CoverageTestEdgesResult
        Result indicating success or failure.

    Notes
    -----
    The edges computed include:
    - Test to function mapping
    - Coverage relationship edges
    - Test impact analysis foundation
    """
    if t__goids.status != "succeeded":
        return CoverageTestEdgesResult(
            success=False,
            error=f"Upstream goids target failed: {t__goids.error}",
        )

    try:
        # Load catalog
        try:
            catalog = CatalogService.from_db(
                env.gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
            )
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            catalog = None

        # Compute test coverage edges (handles persistence internally)
        compute_test_coverage_edges(
            env.gateway,
            env.snapshot,
            catalog_provider=catalog,
        )

        return CoverageTestEdgesResult(success=True)

    except Exception as exc:
        log.exception("Coverage test edges computation failed")
        return CoverageTestEdgesResult(
            success=False,
            error=str(exc),
        )


@tag(domain="analytics", target="coverage_test_edges", node_type="materialize")
def t__coverage_test_edges(
    env: BuildEnv,
    graph: TargetGraph,
    t__coverage_test_edges__compute: CoverageTestEdgesResult,
) -> TargetRunRecord:
    """Materialize coverage test edges target.

    This is the entry point for the coverage_test_edges target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__coverage_test_edges__compute
        Computed coverage edges result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following table:
    - analytics.test_coverage_edges
    """
    executor = NativeTargetExecutor.for_target(env, graph, "coverage_test_edges")

    if executor.should_skip():
        return executor.skip()

    if not t__coverage_test_edges__compute.success:
        return executor.fail(
            RuntimeError(t__coverage_test_edges__compute.error or "Coverage test edges failed")
        )

    def compute() -> dict[str, int]:
        # Edges are persisted during compute - return empty count
        return {"analytics.test_coverage_edges": 0}

    return executor.execute(compute)


__all__ = [
    "CoverageTestEdgesResult",
    "t__coverage_test_edges",
    "t__coverage_test_edges__compute",
]
