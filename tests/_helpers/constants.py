"""Central constants for test helpers.

This module provides the single source of truth for commonly used test constants.
All test helper modules should import from here rather than defining their own.
"""

from __future__ import annotations

# =============================================================================
# Repository and Commit Defaults
# =============================================================================

DEFAULT_REPO: str = "demo/repo"
"""Default repository identifier for tests."""

DEFAULT_COMMIT: str = "deadbeef"
"""Default commit hash for tests."""

DEFAULT_RUN_ID: str = "test-run-001"
"""Default run identifier for plugin execution tests."""

LAYERED_DAG_SHAPES: tuple[tuple[int, ...], ...] = ((2, 3, 2), (3, 3, 3), (2, 2, 2, 2))
"""Common layered DAG shapes for parameterized graph tests."""

BRIDGE_COUNTS: tuple[int, ...] = (1, 2, 3)
"""Bridge edge counts used for articulation/bridging sweeps."""

WEIGHTED_CYCLE_SIZES: tuple[int, ...] = (3, 4, 5)
"""Cycle sizes for weighted cycle/centrality sweeps."""


__all__ = [
    "BRIDGE_COUNTS",
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "DEFAULT_RUN_ID",
    "LAYERED_DAG_SHAPES",
    "WEIGHTED_CYCLE_SIZES",
]
