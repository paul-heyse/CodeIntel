"""Compatibility layer for resolver imports used in legacy tests.

This module re-exports the public resolver API from the current implementation
under ``codeintel.build.readiness`` (ResolutionResult/Reason) and
``codeintel.build.hamilton.executor`` (BuildResolver) to preserve compatibility
with older import paths.
"""

from __future__ import annotations

from codeintel.build.hamilton.executor import BuildResolver  # noqa: F401
from codeintel.build.readiness import ResolutionReason, ResolutionResult  # noqa: F401

__all__ = [
    "BuildResolver",
    "ResolutionReason",
    "ResolutionResult",
]
