"""Compatibility aliases for canonical test identifiers.

The canonical defaults live in `tests._helpers.constants`. This module remains
as a backward-compatible import path for older helpers.
"""

from __future__ import annotations

from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO, DEFAULT_RUN_ID

__all__ = ["DEFAULT_COMMIT", "DEFAULT_REPO", "DEFAULT_RUN_ID"]
