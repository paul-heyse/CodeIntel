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

from importlib import import_module
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
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

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_EXECUTION": ("codeintel.build.resources", "DEFAULT_EXECUTION"),
    "DEFAULT_RESOURCES": ("codeintel.build.resources", "DEFAULT_RESOURCES"),
    "EMPTY_CONTRACT": ("codeintel.build.contracts", "EMPTY_CONTRACT"),
    "EMPTY_PARAMETERS": ("codeintel.build.parameters", "EMPTY_PARAMETERS"),
    "ArtifactSpec": ("codeintel.build.contracts", "ArtifactSpec"),
    "BuildError": ("codeintel.build.errors", "BuildError"),
    "BuildErrorCollection": ("codeintel.build.errors", "BuildErrorCollection"),
    "BuildRunConfig": ("codeintel.build.run_config", "BuildRunConfig"),
    "BuildRunRecord": ("codeintel.build.manifest", "BuildRunRecord"),
    "OperationTargets": ("codeintel.build.operations", "OperationTargets"),
    "OutputContract": ("codeintel.build.contracts", "OutputContract"),
    "OutputManifest": ("codeintel.build.manifest", "OutputManifest"),
    "OutputTarget": ("codeintel.build.targets", "OutputTarget"),
    "TargetExecution": ("codeintel.build.resources", "TargetExecution"),
    "TargetExecutionContext": ("codeintel.build.context", "TargetExecutionContext"),
    "TargetGraph": ("codeintel.build.targets", "TargetGraph"),
    "TargetModule": ("codeintel.build.targets", "TargetModule"),
    "TargetParameters": ("codeintel.build.parameters", "TargetParameters"),
    "TargetPlugin": ("codeintel.build.plugin", "TargetPlugin"),
    "TargetPluginProtocol": ("codeintel.build.plugin", "TargetPluginProtocol"),
    "TargetResources": ("codeintel.build.resources", "TargetResources"),
    "TargetResult": ("codeintel.build.context", "TargetResult"),
    "UnifiedRegistry": ("codeintel.build.unified_registry", "UnifiedRegistry"),
    "build_target_graph": ("codeintel.build.registry", "build_target_graph"),
    "compute_input_hash": ("codeintel.build.hashing", "compute_input_hash"),
    "compute_options_hash": ("codeintel.build.hashing", "compute_options_hash"),
    "get_target_graph": ("codeintel.build.registry", "get_target_graph"),
    "get_targets_for_operation": ("codeintel.build.operations", "get_targets_for_operation"),
    "get_unified_registry": ("codeintel.build.unified_registry", "get_unified_registry"),
}


def __getattr__(name: str) -> object:
    """Lazily import build symbols to avoid import-time cycles.

    Returns
    -------
    object
        Requested attribute loaded from its defining module.

    Raises
    ------
    AttributeError
        If the requested attribute is not registered for lazy loading.
    """
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
