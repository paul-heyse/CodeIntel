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


def _analytics_plugins() -> ModuleType:
    """Get the analytics plugins module.

    Returns
    -------
    ModuleType
        The codeintel.build.plugins.analytics module.
    """
    return _get_module("codeintel.build.plugins.analytics")


def _graphs_plugins() -> ModuleType:
    """Get the graphs plugins module.

    Returns
    -------
    ModuleType
        The codeintel.build.plugins.graphs module.
    """
    return _get_module("codeintel.build.plugins.graphs")


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

    # Native targets (migrated to Hamilton pipelines in Phase 2)
    registry.register(
        targets.MODULES_TARGET,
        native_module="codeintel.build.hamilton.native.ingestion.modules",
    )
    registry.register(
        targets.AST_TARGET,
        native_module="codeintel.build.hamilton.native.ingestion.ast",
    )
    registry.register(
        targets.CST_TARGET,
        native_module="codeintel.build.hamilton.native.ingestion.cst",
    )
    registry.register(
        targets.COVERAGE_INGEST_TARGET,
        native_module="codeintel.build.hamilton.native.ingestion.coverage",
    )
    registry.register(
        targets.TESTS_INGEST_TARGET,
        native_module="codeintel.build.hamilton.native.ingestion.tests",
    )
    registry.register(
        targets.DOCSTRINGS_TARGET,
        native_module="codeintel.build.hamilton.native.ingestion.docstrings",
    )
    registry.register(
        targets.CONFIG_INGEST_TARGET,
        native_module="codeintel.build.hamilton.native.ingestion.config",
    )
    registry.register(
        targets.SCIP_TARGET,
        native_module="codeintel.build.hamilton.native.ingestion.scip",
    )
    registry.register(
        targets.TYPING_TARGET,
        native_module="codeintel.build.hamilton.native.ingestion.typing",
    )


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

    # Native targets (migrated from plugin to Hamilton pipeline in Phase 3)
    registry.register(
        targets.GOIDS_TARGET,
        native_module="codeintel.build.hamilton.native.graphs.goids",
    )
    registry.register(
        targets.CALL_GRAPH_TARGET,
        native_module="codeintel.build.hamilton.native.graphs.call_graph",
    )
    registry.register(
        targets.IMPORT_GRAPH_TARGET,
        native_module="codeintel.build.hamilton.native.graphs.import_graph",
    )
    registry.register(
        targets.CFG_TARGET,
        native_module="codeintel.build.hamilton.native.graphs.cfg_dfg",
    )
    registry.register(
        targets.DFG_TARGET,
        native_module="codeintel.build.hamilton.native.graphs.cfg_dfg",
    )
    # Native target (migrated from plugin to Hamilton pipeline)
    registry.register(
        targets.CFG_DFG_METRICS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.cfg_dfg",
    )
    registry.register(
        targets.SYMBOL_USES_TARGET,
        native_module="codeintel.build.hamilton.native.graphs.symbol_uses",
    )
    registry.register(
        targets.GRAPH_VALIDATION_TARGET,
        native_module="codeintel.build.hamilton.native.graphs.graph_validation",
    )
    registry.register(
        targets.GRAPH_METRICS_TARGET,
        native_module="codeintel.build.hamilton.native.graphs.graph_metrics",
    )
    # Native target (migrated from plugin to Hamilton pipeline in Phase 4)
    registry.register(
        targets.SYMBOL_GRAPH_METRICS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.symbol_graph_metrics",
    )
    # Native target (migrated from plugin to Hamilton pipeline)
    registry.register(
        targets.TEST_GRAPH_METRICS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.test_graph_metrics",
    )

    # Native targets (migrated to Hamilton pipelines)
    registry.register(
        targets.CALL_GRAPH_VIEWS_TARGET,
        native_module="codeintel.build.hamilton.native.graphs.call_graph_views",
    )


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

    # Native targets (migrated from plugin to Hamilton pipeline in Phase 4)
    registry.register(
        targets.FUNCTION_METRICS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.function_metrics",
    )
    registry.register(
        targets.FUNCTION_EFFECTS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.function_effects",
    )
    registry.register(
        targets.FUNCTION_CONTRACTS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.function_contracts",
    )
    # Native target (migrated from plugin to Hamilton pipeline)
    registry.register(
        targets.FUNCTION_HISTORY_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.function_history",
    )
    # Native target (migrated from plugin to Hamilton pipeline)
    registry.register(
        targets.HISTORY_TIMESERIES_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.history_timeseries",
    )
    # Native target (migrated from plugin to Hamilton pipeline in Phase 4)
    registry.register(
        targets.COVERAGE_TEST_EDGES_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.coverage_test_edges",
    )
    # Native target (migrated from plugin to Hamilton pipeline)
    registry.register(
        targets.DATA_MODELS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.data_models",
    )
    # Native target (migrated from plugin to Hamilton pipeline)
    registry.register(
        targets.DATA_MODEL_USAGE_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.data_models",
    )
    # Native targets (migrated from plugin to Hamilton pipeline in Phase 4)
    registry.register(
        targets.CONFIG_DATA_FLOW_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.config_data_flow",
    )
    registry.register(
        targets.SEMANTIC_ROLES_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.semantic_roles",
    )
    registry.register(
        targets.SUBSYSTEM_GRAPH_METRICS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.subsystem_graph_metrics",
    )
    registry.register(
        targets.SUBSYSTEM_AGREEMENT_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.subsystem_agreement",
    )
    registry.register(
        targets.TEST_PROFILE_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.test_profile",
    )
    registry.register(
        targets.BEHAVIORAL_COVERAGE_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.behavioral_coverage",
    )
    # Native targets (migrated from plugin to Hamilton pipeline)
    registry.register(
        targets.ENTRYPOINTS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.entrypoints",
    )
    registry.register(
        targets.EXTERNAL_DEPS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.dependencies",
    )
    # Native targets (migrated from plugin to Hamilton pipeline in Phase 4)
    registry.register(
        targets.PROFILES_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.profiles",
    )
    registry.register(
        targets.FUNCTION_AST_FEATURES_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.ast_features",
    )

    # Native targets (migrated to Hamilton pipelines - Phase 1.5 + Phase 4)
    registry.register(
        targets.RISK_FACTORS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.risk_factors",
    )
    registry.register(
        targets.COVERAGE_FUNCTIONS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.coverage_functions",
    )
    registry.register(
        targets.HOTSPOTS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.hotspots",
    )
    registry.register(
        targets.SUBSYSTEMS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.subsystems",
    )


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

    # Native targets (migrated to Hamilton pipelines)
    registry.register(
        targets.EXPORT_JSONL_TARGET,
        native_module="codeintel.build.hamilton.native.export.export_jsonl",
    )
    registry.register(
        targets.EXPORT_PARQUET_TARGET,
        native_module="codeintel.build.hamilton.native.export.export_parquet",
    )
