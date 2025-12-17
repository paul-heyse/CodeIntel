"""Unified target registrations.

This module provides a central location for registering targets with their
implementations. This replaces the scattered registrations across the legacy
registry and plugin registry layers.

The goal is atomic registration: each target is registered with its
implementation in a single call, preventing mismatches.

Migration Status
----------------
This module now registers targets with their plugin classes and/or native
Hamilton modules, enabling the UnifiedRegistry to serve as the single source
of truth for all target implementations.

Example
-------
>>> from codeintel.build.registrations import register_all_targets
>>> from codeintel.build.unified_registry import UnifiedRegistry
>>> registry = UnifiedRegistry()
>>> register_all_targets(registry)
>>> len(registry)
45
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from codeintel.build.unified_registry import UnifiedRegistry


# -----------------------------------------------------------------------------
# Lazy Module Access
#
# These functions provide cached module access to break circular dependencies.
# Using importlib + lru_cache avoids global statements and provides lazy loading.
# -----------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _get_module(name: str) -> ModuleType:
    """Load a module lazily with caching.

    Parameters
    ----------
    name
        Fully qualified module name.

    Returns
    -------
    ModuleType
        The loaded module.
    """
    return importlib.import_module(name)


def _build_registry() -> ModuleType:
    """Get the build registry module.

    Returns
    -------
    ModuleType
        The codeintel.build.registry module.
    """
    return _get_module("codeintel.build.registry")


__all__ = [
    "register_all_targets",
    "register_analytics_targets",
    "register_export_targets",
    "register_graph_targets",
    "register_ingestion_targets",
]


def register_all_targets(registry: UnifiedRegistry) -> None:
    """Register all targets with their implementations.

    This function registers all targets and then validates that each target
    has at least one implementation (plugin or native module).

    Parameters
    ----------
    registry
        The unified registry to populate.

    Raises
    ------
    ValueError
        If any target is registered without an implementation.
    """
    register_ingestion_targets(registry)
    register_graph_targets(registry)
    register_analytics_targets(registry)
    register_export_targets(registry)

    # Validate all targets have implementations
    orphans = [name for name in registry if not registry.has_implementation(name)]
    if orphans:
        msg = f"Targets registered without implementations: {', '.join(sorted(orphans))}"
        raise ValueError(msg)


def register_ingestion_targets(registry: UnifiedRegistry) -> None:
    """Register ingestion module targets.

    Ingestion targets parse and extract data from source repositories.

    Phase 2: All ingestion targets migrated to native Hamilton modules.

    Parameters
    ----------
    registry
        The unified registry to populate.
    """
    # Get modules lazily
    targets = _build_registry()

    native_module = "codeintel.build.hamilton.native.ingestion"
    native_targets = (
        targets.MODULES_TARGET,
        targets.AST_TARGET,
        targets.CST_TARGET,
        targets.COVERAGE_INGEST_TARGET,
        targets.TESTS_INGEST_TARGET,
        targets.DOCSTRINGS_TARGET,
        targets.CONFIG_INGEST_TARGET,
        targets.SCIP_TARGET,
        targets.TYPING_TARGET,
    )
    for target in native_targets:
        registry.register(target, native_module=native_module)


def register_graph_targets(registry: UnifiedRegistry) -> None:
    """Register graph module targets.

    Graph targets build and analyze code graphs (call, import, CFG/DFG).

    Phase 3: All graph targets migrated to native Hamilton modules.

    Parameters
    ----------
    registry
        The unified registry to populate.
    """
    # Get modules lazily
    targets = _build_registry()

    graphs_module = "codeintel.build.hamilton.native.graphs"
    analytics_module = "codeintel.build.hamilton.native.analytics"

    graphs_targets = (
        targets.GOIDS_TARGET,
        targets.CALL_GRAPH_TARGET,
        targets.IMPORT_GRAPH_TARGET,
        targets.CFG_TARGET,
        targets.DFG_TARGET,
        targets.SYMBOL_USES_TARGET,
        targets.GRAPH_VALIDATION_TARGET,
        targets.GRAPH_METRICS_TARGET,
        targets.CALL_GRAPH_VIEWS_TARGET,
    )
    for target in graphs_targets:
        registry.register(target, native_module=graphs_module)

    analytics_graph_targets = (
        targets.CFG_DFG_METRICS_TARGET,
        targets.SYMBOL_GRAPH_METRICS_TARGET,
        targets.TEST_GRAPH_METRICS_TARGET,
    )
    for target in analytics_graph_targets:
        registry.register(target, native_module=analytics_module)


def register_analytics_targets(registry: UnifiedRegistry) -> None:
    """Register analytics module targets.

    Analytics targets compute metrics, detect patterns, and analyze code.

    Phase 4: All analytics targets migrated to native Hamilton modules.

    Parameters
    ----------
    registry
        The unified registry to populate.
    """
    # Get modules lazily
    targets = _build_registry()

    native_module = "codeintel.build.hamilton.native.analytics"
    native_targets = (
        targets.FUNCTION_METRICS_TARGET,
        targets.FUNCTION_EFFECTS_TARGET,
        targets.FUNCTION_CONTRACTS_TARGET,
        targets.FUNCTION_HISTORY_TARGET,
        targets.HISTORY_TIMESERIES_TARGET,
        targets.COVERAGE_TEST_EDGES_TARGET,
        targets.DATA_MODELS_TARGET,
        targets.DATA_MODEL_USAGE_TARGET,
        targets.CONFIG_DATA_FLOW_TARGET,
        targets.SEMANTIC_ROLES_TARGET,
        targets.SUBSYSTEM_GRAPH_METRICS_TARGET,
        targets.SUBSYSTEM_AGREEMENT_TARGET,
        targets.TEST_PROFILE_TARGET,
        targets.BEHAVIORAL_COVERAGE_TARGET,
        targets.ENTRYPOINTS_TARGET,
        targets.EXTERNAL_DEPS_TARGET,
        targets.PROFILES_TARGET,
        targets.FUNCTION_AST_FEATURES_TARGET,
        targets.RISK_FACTORS_TARGET,
        targets.COVERAGE_FUNCTIONS_TARGET,
        targets.HOTSPOTS_TARGET,
        targets.SUBSYSTEMS_TARGET,
    )
    for target in native_targets:
        registry.register(target, native_module=native_module)


def register_export_targets(registry: UnifiedRegistry) -> None:
    """Register export module targets.

    Export targets produce output files (JSONL, Parquet, etc.).

    Parameters
    ----------
    registry
        The unified registry to populate.
    """
    # Get modules lazily
    targets = _build_registry()

    native_module = "codeintel.build.hamilton.native.export"
    native_targets = (
        targets.EXPORT_JSONL_TARGET,
        targets.EXPORT_PARQUET_TARGET,
    )
    for target in native_targets:
        registry.register(target, native_module=native_module)
