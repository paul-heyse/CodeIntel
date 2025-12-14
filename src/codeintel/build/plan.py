"""Compatibility layer for plan generation imports used in legacy tests.

This module re-exports the public plan API from the current implementation
under ``codeintel.build.hamilton.planner`` to preserve compatibility with
older import paths.
"""

from __future__ import annotations

from codeintel.build.hamilton.planner import (  # noqa: F401
    BuildPlan,
    PlanGenerator,
    PlanStage,
    PlanStep,
    format_duration,
)

__all__ = [
    "BuildPlan",
    "PlanGenerator",
    "PlanStage",
    "PlanStep",
    "format_duration",
]
