"""Build system for computing minimal execution plans.

This package provides the target graph, state validation, resolution,
plan generation, execution, and readiness infrastructure for the
CodeIntel build system.

Key concepts:

- **OutputTarget**: A discrete output that can be requested and validated
- **TargetGraph**: Complete dependency graph of all output targets
- **OutputManifest**: Record of a target's computation with input/output hashes
- **BuildRunRecord**: Record of a build system run for observability

Import patterns::

    # Basic imports (no heavy dependencies)
    from codeintel.build import get_target_graph, OutputTarget, TargetGraph

    # For execution (import from submodules to avoid circular imports)
    from codeintel.build.executor import BuildExecutor, BuildResult
    from codeintel.build.plan import BuildPlan, PlanGenerator
    from codeintel.build.resolver import BuildResolver
    from codeintel.build.state import StateValidator
    from codeintel.build.readiness import DatabaseReadinessView

CLI usage::

    codeintel build run --all          # Build all targets
    codeintel build status             # Show target status
    codeintel build history            # Show run history

Use ``get_target_graph()`` to access the singleton target graph instance.
"""

from __future__ import annotations

# Core types that have minimal dependencies (see docstring for full import patterns)
from codeintel.build.hashing import compute_input_hash, compute_options_hash
from codeintel.build.manifest import BuildRunRecord, OutputManifest
from codeintel.build.operations import OperationTargets, get_targets_for_operation
from codeintel.build.registry import build_target_graph, get_target_graph
from codeintel.build.targets import OutputTarget, TargetGraph, TargetModule

__all__ = [
    "BuildRunRecord",
    "OperationTargets",
    "OutputManifest",
    "OutputTarget",
    "TargetGraph",
    "TargetModule",
    "build_target_graph",
    "compute_input_hash",
    "compute_options_hash",
    "get_target_graph",
    "get_targets_for_operation",
]
