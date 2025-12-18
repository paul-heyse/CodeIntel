"""Build system for computing minimal execution plans.

This package provides the target graph, state validation, Hamilton-based
execution, and readiness infrastructure for the CodeIntel build system.

Key concepts:

- **OutputTarget**: A discrete output that can be requested and validated
- **TargetGraph**: Complete dependency graph of all output targets
- **OutputContract**: Tables and artifacts a target produces (single source of truth)
- **OutputManifest**: Record of a target's computation with input/output hashes
- **BuildRunRecord**: Record of a build system run for observability
- **BuildError**: Rich error hierarchy with actionable hints

Import patterns::


    from codeintel.build import load_target_system, OutputTarget, TargetGraph


    from codeintel.build.contracts import OutputContract, ArtifactSpec, TableSchema
    from codeintel.build.resources import TargetResources, TargetExecution
    from codeintel.build.parameters import TargetParameters


    from codeintel.build.hamilton.env import BuildEnv


    from codeintel.build.errors import BuildError, BuildErrorCollection


    from codeintel.build.hamilton import HamiltonBuildExecutor, HamiltonBuildResult
    from codeintel.build.state import StateValidator


    from codeintel.build.config import load_build_config, BuildConfig

CLI usage::

    codeintel build run --all
    codeintel build status
    codeintel build history

Use ``load_target_system().graph`` to access the singleton target graph instance.
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
    "OutputContract",
    "OutputManifest",
    "OutputTarget",
    "TargetExecution",
    "TargetGraph",
    "TargetModule",
    "TargetParameters",
    "TargetResources",
    "TargetSystem",
    "compute_input_hash",
    "compute_options_hash",
    "load_target_system",
]

if TYPE_CHECKING:
    from codeintel.build.contracts import EMPTY_CONTRACT, ArtifactSpec, OutputContract
    from codeintel.build.errors import BuildError, BuildErrorCollection
    from codeintel.build.hashing import compute_input_hash, compute_options_hash
    from codeintel.build.parameters import EMPTY_PARAMETERS, TargetParameters
    from codeintel.build.resources import (
        DEFAULT_EXECUTION,
        DEFAULT_RESOURCES,
        TargetExecution,
        TargetResources,
    )
    from codeintel.build.run_config import BuildRunConfig
    from codeintel.build.target_system import TargetSystem, load_target_system
    from codeintel.build.targets import OutputTarget, TargetGraph, TargetModule
    from codeintel.core.build_manifest import BuildRunRecord, OutputManifest

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_EXECUTION": ("codeintel.build.resources", "DEFAULT_EXECUTION"),
    "DEFAULT_RESOURCES": ("codeintel.build.resources", "DEFAULT_RESOURCES"),
    "EMPTY_CONTRACT": ("codeintel.build.contracts", "EMPTY_CONTRACT"),
    "EMPTY_PARAMETERS": ("codeintel.build.parameters", "EMPTY_PARAMETERS"),
    "ArtifactSpec": ("codeintel.build.contracts", "ArtifactSpec"),
    "BuildError": ("codeintel.build.errors", "BuildError"),
    "BuildErrorCollection": ("codeintel.build.errors", "BuildErrorCollection"),
    "BuildRunConfig": ("codeintel.build.run_config", "BuildRunConfig"),
    "BuildRunRecord": ("codeintel.core.build_manifest", "BuildRunRecord"),
    "OutputContract": ("codeintel.build.contracts", "OutputContract"),
    "OutputManifest": ("codeintel.core.build_manifest", "OutputManifest"),
    "OutputTarget": ("codeintel.build.targets", "OutputTarget"),
    "TargetExecution": ("codeintel.build.resources", "TargetExecution"),
    "TargetGraph": ("codeintel.build.targets", "TargetGraph"),
    "TargetModule": ("codeintel.build.targets", "TargetModule"),
    "TargetParameters": ("codeintel.build.parameters", "TargetParameters"),
    "TargetResources": ("codeintel.build.resources", "TargetResources"),
    "compute_input_hash": ("codeintel.build.hashing", "compute_input_hash"),
    "compute_options_hash": ("codeintel.build.hashing", "compute_options_hash"),
    "TargetSystem": ("codeintel.build.target_system", "TargetSystem"),
    "load_target_system": ("codeintel.build.target_system", "load_target_system"),
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
