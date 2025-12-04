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


__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "DEFAULT_RUN_ID",
]
