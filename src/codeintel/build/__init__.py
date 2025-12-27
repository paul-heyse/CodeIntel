"""Build system for computing minimal execution plans.

This package provides the Hamilton-derived catalog, state validation,
execution, and readiness infrastructure for the CodeIntel build system.

Key concepts:

- **DagCatalog**: Immutable catalog derived from the Hamilton DAG + tags
- **TargetDescriptor**: Target metadata derived from tags + contracts
- **OutputContract**: Output identities a target produces (tables + artifacts)
- **OutputManifest**: Record of a target's computation with input/output hashes
- **BuildRunRecord**: Record of a build system run for observability
- **BuildError**: Rich error hierarchy with actionable hints

Import patterns::


    from codeintel.build import DagCatalog, TargetDescriptor
    from codeintel.build.target_metadata import get_target_metadata_service


    from codeintel.build.contracts import OutputContract, ArtifactSpec, TableOutputDescriptor
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

Use ``get_target_metadata_service().system.catalog`` for the canonical DAG catalog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.core.imports.lazy import lazy_import

__all__ = [
    "DEFAULT_EXECUTION",
    "DEFAULT_RESOURCES",
    "EMPTY_CONTRACT",
    "EMPTY_PARAMETERS",
    "ArtifactSpec",
    "BuildError",
    "BuildErrorCollection",
    "BuildRunRecord",
    "DagCatalog",
    "ExecutionPolicy",
    "OutputContract",
    "OutputManifest",
    "TargetDescriptor",
    "TargetExecution",
    "TargetMetadataService",
    "TargetModule",
    "TargetParameters",
    "TargetResources",
    "compute_input_hash",
    "compute_options_hash",
    "get_target_metadata_service",
]

if TYPE_CHECKING:
    from codeintel.build.contracts import EMPTY_CONTRACT, ArtifactSpec, OutputContract
    from codeintel.build.errors import BuildError, BuildErrorCollection
    from codeintel.build.execution_policy import ExecutionPolicy
    from codeintel.build.hamilton.dag_catalog import DagCatalog, TargetDescriptor
    from codeintel.build.hashing import compute_input_hash, compute_options_hash
    from codeintel.build.parameters import EMPTY_PARAMETERS, TargetParameters
    from codeintel.build.resources import (
        DEFAULT_EXECUTION,
        DEFAULT_RESOURCES,
        TargetExecution,
        TargetResources,
    )
    from codeintel.build.target_metadata import TargetMetadataService, get_target_metadata_service
    from codeintel.build.targets import TargetModule
    from codeintel.core.build_manifest import BuildRunRecord, OutputManifest

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_EXECUTION": ("codeintel.build.resources", "DEFAULT_EXECUTION"),
    "DEFAULT_RESOURCES": ("codeintel.build.resources", "DEFAULT_RESOURCES"),
    "EMPTY_CONTRACT": ("codeintel.build.contracts", "EMPTY_CONTRACT"),
    "EMPTY_PARAMETERS": ("codeintel.build.parameters", "EMPTY_PARAMETERS"),
    "ArtifactSpec": ("codeintel.build.contracts", "ArtifactSpec"),
    "BuildError": ("codeintel.build.errors", "BuildError"),
    "BuildErrorCollection": ("codeintel.build.errors", "BuildErrorCollection"),
    "ExecutionPolicy": ("codeintel.build.execution_policy", "ExecutionPolicy"),
    "BuildRunRecord": ("codeintel.core.build_manifest", "BuildRunRecord"),
    "DagCatalog": ("codeintel.build.hamilton.dag_catalog", "DagCatalog"),
    "OutputContract": ("codeintel.build.contracts", "OutputContract"),
    "OutputManifest": ("codeintel.core.build_manifest", "OutputManifest"),
    "TargetExecution": ("codeintel.build.resources", "TargetExecution"),
    "TargetDescriptor": ("codeintel.build.hamilton.dag_catalog", "TargetDescriptor"),
    "TargetModule": ("codeintel.build.targets", "TargetModule"),
    "TargetParameters": ("codeintel.build.parameters", "TargetParameters"),
    "TargetResources": ("codeintel.build.resources", "TargetResources"),
    "compute_input_hash": ("codeintel.build.hashing", "compute_input_hash"),
    "compute_options_hash": ("codeintel.build.hashing", "compute_options_hash"),
    "TargetMetadataService": ("codeintel.build.target_metadata", "TargetMetadataService"),
    "get_target_metadata_service": (
        "codeintel.build.target_metadata",
        "get_target_metadata_service",
    ),
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
        module = lazy_import(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
