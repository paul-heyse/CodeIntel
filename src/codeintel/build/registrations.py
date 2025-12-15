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


def _ingestion_plugins() -> ModuleType:
    """Get the ingestion plugins module.

    Returns
    -------
    ModuleType
        The codeintel.build.plugins.ingestion module.
    """
    return _get_module("codeintel.build.plugins.ingestion")


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

    Parameters
    ----------
    registry
        The unified registry to populate.
    """
    # Get modules lazily
    plugins = _ingestion_plugins()
    targets = _build_registry()

    # Plugin-based targets
    registry.register(targets.MODULES_TARGET, plugin=plugins.RepoScanPlugin)
    registry.register(targets.AST_TARGET, plugin=plugins.AstExtractPlugin)
    registry.register(targets.CST_TARGET, plugin=plugins.CstExtractPlugin)
    registry.register(targets.COVERAGE_INGEST_TARGET, plugin=plugins.CoverageIngestPlugin)
    registry.register(targets.TESTS_INGEST_TARGET, plugin=plugins.TestsIngestPlugin)
    registry.register(targets.DOCSTRINGS_TARGET, plugin=plugins.DocstringsIngestPlugin)
    registry.register(targets.CONFIG_INGEST_TARGET, plugin=plugins.ConfigIngestPlugin)

    # Native targets (migrated to Hamilton pipelines)
    # These have both plugin fallback and native implementation
    registry.register(
        targets.SCIP_TARGET,
        plugin=plugins.ScipIngestPlugin,
        native_module="codeintel.build.hamilton.native.ingestion.scip",
    )
    registry.register(
        targets.TYPING_TARGET,
        plugin=plugins.TypingIngestPlugin,
        native_module="codeintel.build.hamilton.native.ingestion.typing",
    )


def register_graph_targets(registry: UnifiedRegistry) -> None:
    """Register graph module targets.

    Graph targets build and analyze code graphs (call, import, CFG/DFG).

    Parameters
    ----------
    registry
        The unified registry to populate.
    """
    # Get modules lazily
    analytics = _analytics_plugins()
    graphs = _graphs_plugins()
    targets = _build_registry()

    # Plugin-based targets
    registry.register(targets.GOIDS_TARGET, plugin=graphs.GoidBuilderPlugin)
    registry.register(targets.CALL_GRAPH_TARGET, plugin=graphs.CallGraphPlugin)
    registry.register(targets.IMPORT_GRAPH_TARGET, plugin=graphs.ImportGraphPlugin)
    registry.register(targets.CFG_TARGET, plugin=graphs.CfgDfgPlugin)
    registry.register(targets.DFG_TARGET, plugin=graphs.CfgDfgPlugin)
    # Native target (migrated from plugin to Hamilton pipeline)
    registry.register(
        targets.CFG_DFG_METRICS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.cfg_dfg",
    )
    registry.register(targets.SYMBOL_USES_TARGET, plugin=graphs.SymbolUsesPlugin)
    registry.register(targets.GRAPH_VALIDATION_TARGET, plugin=graphs.GraphValidationPlugin)
    registry.register(targets.GRAPH_METRICS_TARGET, plugin=graphs.CoreMetricsPlugin)
    registry.register(
        targets.SYMBOL_GRAPH_METRICS_TARGET, plugin=analytics.SymbolGraphMetricsPlugin
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

    Parameters
    ----------
    registry
        The unified registry to populate.
    """
    # Get modules lazily
    plugins = _analytics_plugins()
    targets = _build_registry()

    # Plugin-based targets
    registry.register(targets.FUNCTION_METRICS_TARGET, plugin=plugins.FunctionMetricsPlugin)
    registry.register(targets.FUNCTION_EFFECTS_TARGET, plugin=plugins.FunctionEffectsPlugin)
    registry.register(targets.FUNCTION_CONTRACTS_TARGET, plugin=plugins.FunctionContractsPlugin)
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
    registry.register(targets.COVERAGE_TEST_EDGES_TARGET, plugin=plugins.CoverageTestEdgesPlugin)
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
    registry.register(targets.CONFIG_DATA_FLOW_TARGET, plugin=plugins.ConfigDataFlowPlugin)
    registry.register(targets.SEMANTIC_ROLES_TARGET, plugin=plugins.SemanticRolesPlugin)
    registry.register(
        targets.SUBSYSTEM_GRAPH_METRICS_TARGET, plugin=plugins.SubsystemGraphMetricsPlugin
    )
    registry.register(targets.SUBSYSTEM_AGREEMENT_TARGET, plugin=plugins.SubsystemAgreementPlugin)
    registry.register(targets.TEST_PROFILE_TARGET, plugin=plugins.TestProfilePlugin)
    registry.register(targets.BEHAVIORAL_COVERAGE_TARGET, plugin=plugins.BehavioralCoveragePlugin)
    # Native targets (migrated from plugin to Hamilton pipeline)
    registry.register(
        targets.ENTRYPOINTS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.entrypoints",
    )
    registry.register(
        targets.EXTERNAL_DEPS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.dependencies",
    )
    registry.register(targets.PROFILES_TARGET, plugin=plugins.ProfilesPlugin)
    registry.register(
        targets.FUNCTION_AST_FEATURES_TARGET, plugin=plugins.FunctionAstFeaturesPlugin
    )

    # Native targets (migrated to Hamilton pipelines)
    # These have both plugin fallback and native implementation
    registry.register(
        targets.RISK_FACTORS_TARGET,
        plugin=plugins.RiskFactorsPlugin,
        native_module="codeintel.build.hamilton.native.analytics.risk_factors",
    )
    # Native target (plugin removed in Phase 3-4)
    registry.register(
        targets.COVERAGE_FUNCTIONS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.coverage_functions",
    )
    registry.register(
        targets.HOTSPOTS_TARGET,
        plugin=plugins.HotspotsPlugin,
        native_module="codeintel.build.hamilton.native.analytics.hotspots",
    )
    registry.register(
        targets.SUBSYSTEMS_TARGET,
        plugin=plugins.SubsystemsPlugin,
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
