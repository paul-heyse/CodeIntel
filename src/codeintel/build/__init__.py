"""Build system for computing minimal execution plans.

This package provides the target graph, state validation, Hamilton-based
execution, and readiness infrastructure for the CodeIntel build system.

Key concepts:

- **OutputTarget**: A discrete output that can be requested and validated
- **TargetGraph**: Complete dependency graph of all output targets
- **OutputContract**: Tables and artifacts a target produces (single source of truth)
- **OutputManifest**: Record of a target's computation with input/output hashes
- **BuildRunRecord**: Record of a build system run for observability
- **TargetExecutionContext**: Everything a plugin needs for execution
- **BuildError**: Rich error hierarchy with actionable hints

Import patterns::


    from codeintel.build import get_target_graph, OutputTarget, TargetGraph


    from codeintel.build.contracts import OutputContract, ArtifactSpec, TableSchema
    from codeintel.build.resources import TargetResources, TargetExecution
    from codeintel.build.parameters import TargetParameters


    from codeintel.build.context import TargetExecutionContext, TargetResult


    from codeintel.build.protocols import ToolRunner, ScipIndexer, TypeChecker
    from codeintel.build.providers import create_default_providers


    from codeintel.build.errors import BuildError, BuildErrorCollection


    from codeintel.build.hamilton import HamiltonBuildExecutor, HamiltonBuildResult
    from codeintel.build.state import StateValidator
    from codeintel.build.readiness import DatabaseReadinessView


    from codeintel.build.config import load_build_config, BuildConfig

CLI usage::

    codeintel build run --all
    codeintel build status
    codeintel build history

Use ``get_target_graph()`` to access the singleton target graph instance.
"""

from __future__ import annotations

from codeintel.build.context import TargetExecutionContext, TargetResult
from codeintel.build.contracts import EMPTY_CONTRACT, ArtifactSpec, OutputContract
from codeintel.build.errors import BuildError, BuildErrorCollection
from codeintel.build.hashing import compute_input_hash, compute_options_hash
from codeintel.build.manifest import BuildRunRecord, OutputManifest
from codeintel.build.operations import OperationTargets, get_targets_for_operation
from codeintel.build.parameters import EMPTY_PARAMETERS, TargetParameters
from codeintel.build.plugin import TargetPlugin, TargetPluginProtocol
from codeintel.build.registry import build_target_graph, get_target_graph
from codeintel.build.resources import (
    DEFAULT_EXECUTION,
    DEFAULT_RESOURCES,
    TargetExecution,
    TargetResources,
)
from codeintel.build.run_config import BuildRunConfig
from codeintel.build.targets import OutputTarget, TargetGraph, TargetModule
from codeintel.build.unified_registry import UnifiedRegistry, get_unified_registry

__all__ = [
    "DEFAULT_EXECUTION",
    "DEFAULT_RESOURCES",
    "EMPTY_CONTRACT",
    "EMPTY_PARAMETERS",
    "ArtifactSpec",
    "BuildError",
    "BuildErrorCollection",
    "BuildRunConfig",
    "BuildRunRecord",
    "OperationTargets",
    "OutputContract",
    "OutputManifest",
    "OutputTarget",
    "TargetExecution",
    "TargetExecutionContext",
    "TargetGraph",
    "TargetModule",
    "TargetParameters",
    "TargetPlugin",
    "TargetPluginProtocol",
    "TargetResources",
    "TargetResult",
    "UnifiedRegistry",
    "build_target_graph",
    "compute_input_hash",
    "compute_options_hash",
    "get_target_graph",
    "get_targets_for_operation",
    "get_unified_registry",
]
