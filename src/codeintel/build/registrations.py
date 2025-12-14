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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.build.unified_registry import UnifiedRegistry

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
    # Import targets (deferred to avoid circular imports)
    # Import plugin classes (deferred to avoid circular imports)
    from codeintel.build.plugins.ingestion import (  # noqa: PLC0415
        AstExtractPlugin,
        ConfigIngestPlugin,
        CoverageIngestPlugin,
        CstExtractPlugin,
        DocstringsIngestPlugin,
        RepoScanPlugin,
        ScipIngestPlugin,
        TestsIngestPlugin,
        TypingIngestPlugin,
    )
    from codeintel.build.registry import (  # noqa: PLC0415
        AST_TARGET,
        CONFIG_INGEST_TARGET,
        COVERAGE_INGEST_TARGET,
        CST_TARGET,
        DOCSTRINGS_TARGET,
        MODULES_TARGET,
        SCIP_TARGET,
        TESTS_INGEST_TARGET,
        TYPING_TARGET,
    )

    # Plugin-based targets
    registry.register(MODULES_TARGET, plugin=RepoScanPlugin)
    registry.register(AST_TARGET, plugin=AstExtractPlugin)
    registry.register(CST_TARGET, plugin=CstExtractPlugin)
    registry.register(COVERAGE_INGEST_TARGET, plugin=CoverageIngestPlugin)
    registry.register(TESTS_INGEST_TARGET, plugin=TestsIngestPlugin)
    registry.register(DOCSTRINGS_TARGET, plugin=DocstringsIngestPlugin)
    registry.register(CONFIG_INGEST_TARGET, plugin=ConfigIngestPlugin)

    # Native targets (migrated to Hamilton pipelines)
    # These have both plugin fallback and native implementation
    registry.register(
        SCIP_TARGET,
        plugin=ScipIngestPlugin,
        native_module="codeintel.build.hamilton.native.ingestion.scip",
    )
    registry.register(
        TYPING_TARGET,
        plugin=TypingIngestPlugin,
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
    # Import targets (deferred to avoid circular imports)
    # Import plugin classes (deferred to avoid circular imports)
    from codeintel.build.plugins.analytics import (  # noqa: PLC0415
        SymbolGraphMetricsPlugin,
    )
    from codeintel.build.plugins.graphs import (  # noqa: PLC0415
        CallGraphPlugin,
        CfgDfgPlugin,
        CoreMetricsPlugin,
        GoidBuilderPlugin,
        GraphValidationPlugin,
        ImportGraphPlugin,
        SymbolUsesPlugin,
    )
    from codeintel.build.registry import (  # noqa: PLC0415
        CALL_GRAPH_TARGET,
        CALL_GRAPH_VIEWS_TARGET,
        CFG_DFG_METRICS_TARGET,
        CFG_TARGET,
        DFG_TARGET,
        GOIDS_TARGET,
        GRAPH_METRICS_TARGET,
        GRAPH_VALIDATION_TARGET,
        IMPORT_GRAPH_TARGET,
        SYMBOL_GRAPH_METRICS_TARGET,
        SYMBOL_USES_TARGET,
        TEST_GRAPH_METRICS_TARGET,
    )

    # Plugin-based targets
    registry.register(GOIDS_TARGET, plugin=GoidBuilderPlugin)
    registry.register(CALL_GRAPH_TARGET, plugin=CallGraphPlugin)
    registry.register(IMPORT_GRAPH_TARGET, plugin=ImportGraphPlugin)
    registry.register(CFG_TARGET, plugin=CfgDfgPlugin)
    registry.register(DFG_TARGET, plugin=CfgDfgPlugin)
    # Native target (migrated from plugin to Hamilton pipeline)
    registry.register(
        CFG_DFG_METRICS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.cfg_dfg",
    )
    registry.register(SYMBOL_USES_TARGET, plugin=SymbolUsesPlugin)
    registry.register(GRAPH_VALIDATION_TARGET, plugin=GraphValidationPlugin)
    registry.register(GRAPH_METRICS_TARGET, plugin=CoreMetricsPlugin)
    registry.register(SYMBOL_GRAPH_METRICS_TARGET, plugin=SymbolGraphMetricsPlugin)
    # Native target (migrated from plugin to Hamilton pipeline)
    registry.register(
        TEST_GRAPH_METRICS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.test_graph_metrics",
    )

    # Native targets (migrated to Hamilton pipelines)
    registry.register(
        CALL_GRAPH_VIEWS_TARGET,
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
    # Import targets (deferred to avoid circular imports)
    # Import plugin classes (deferred to avoid circular imports)
    from codeintel.build.plugins.analytics import (  # noqa: PLC0415
        BehavioralCoveragePlugin,
        ConfigDataFlowPlugin,
        CoverageFunctionsPlugin,
        CoverageTestEdgesPlugin,
        DataModelUsagePlugin,
        FunctionAstFeaturesPlugin,
        FunctionContractsPlugin,
        FunctionEffectsPlugin,
        FunctionHistoryPlugin,
        FunctionMetricsPlugin,
        HistoryTimeseriesPlugin,
        HotspotsPlugin,
        ProfilesPlugin,
        RiskFactorsPlugin,
        SemanticRolesPlugin,
        SubsystemAgreementPlugin,
        SubsystemGraphMetricsPlugin,
        SubsystemsPlugin,
        TestProfilePlugin,
    )
    from codeintel.build.registry import (  # noqa: PLC0415
        BEHAVIORAL_COVERAGE_TARGET,
        CONFIG_DATA_FLOW_TARGET,
        COVERAGE_FUNCTIONS_TARGET,
        COVERAGE_TEST_EDGES_TARGET,
        DATA_MODEL_USAGE_TARGET,
        DATA_MODELS_TARGET,
        ENTRYPOINTS_TARGET,
        EXTERNAL_DEPS_TARGET,
        FUNCTION_AST_FEATURES_TARGET,
        FUNCTION_CONTRACTS_TARGET,
        FUNCTION_EFFECTS_TARGET,
        FUNCTION_HISTORY_TARGET,
        FUNCTION_METRICS_TARGET,
        HISTORY_TIMESERIES_TARGET,
        HOTSPOTS_TARGET,
        PROFILES_TARGET,
        RISK_FACTORS_TARGET,
        SEMANTIC_ROLES_TARGET,
        SUBSYSTEM_AGREEMENT_TARGET,
        SUBSYSTEM_GRAPH_METRICS_TARGET,
        SUBSYSTEMS_TARGET,
        TEST_PROFILE_TARGET,
    )

    # Plugin-based targets
    registry.register(FUNCTION_METRICS_TARGET, plugin=FunctionMetricsPlugin)
    registry.register(FUNCTION_EFFECTS_TARGET, plugin=FunctionEffectsPlugin)
    registry.register(FUNCTION_CONTRACTS_TARGET, plugin=FunctionContractsPlugin)
    registry.register(FUNCTION_HISTORY_TARGET, plugin=FunctionHistoryPlugin)
    registry.register(HISTORY_TIMESERIES_TARGET, plugin=HistoryTimeseriesPlugin)
    registry.register(COVERAGE_TEST_EDGES_TARGET, plugin=CoverageTestEdgesPlugin)
    # Native target (migrated from plugin to Hamilton pipeline)
    registry.register(
        DATA_MODELS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.data_models",
    )
    registry.register(DATA_MODEL_USAGE_TARGET, plugin=DataModelUsagePlugin)
    registry.register(CONFIG_DATA_FLOW_TARGET, plugin=ConfigDataFlowPlugin)
    registry.register(SEMANTIC_ROLES_TARGET, plugin=SemanticRolesPlugin)
    registry.register(SUBSYSTEM_GRAPH_METRICS_TARGET, plugin=SubsystemGraphMetricsPlugin)
    registry.register(SUBSYSTEM_AGREEMENT_TARGET, plugin=SubsystemAgreementPlugin)
    registry.register(TEST_PROFILE_TARGET, plugin=TestProfilePlugin)
    registry.register(BEHAVIORAL_COVERAGE_TARGET, plugin=BehavioralCoveragePlugin)
    # Native targets (migrated from plugin to Hamilton pipeline)
    registry.register(
        ENTRYPOINTS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.entrypoints",
    )
    registry.register(
        EXTERNAL_DEPS_TARGET,
        native_module="codeintel.build.hamilton.native.analytics.dependencies",
    )
    registry.register(PROFILES_TARGET, plugin=ProfilesPlugin)
    registry.register(FUNCTION_AST_FEATURES_TARGET, plugin=FunctionAstFeaturesPlugin)

    # Native targets (migrated to Hamilton pipelines)
    # These have both plugin fallback and native implementation
    registry.register(
        RISK_FACTORS_TARGET,
        plugin=RiskFactorsPlugin,
        native_module="codeintel.build.hamilton.native.analytics.risk_factors",
    )
    registry.register(
        COVERAGE_FUNCTIONS_TARGET,
        plugin=CoverageFunctionsPlugin,
        native_module="codeintel.build.hamilton.native.analytics.coverage_functions",
    )
    registry.register(
        HOTSPOTS_TARGET,
        plugin=HotspotsPlugin,
        native_module="codeintel.build.hamilton.native.analytics.hotspots",
    )
    registry.register(
        SUBSYSTEMS_TARGET,
        plugin=SubsystemsPlugin,
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
    # Import targets (deferred to avoid circular imports)
    from codeintel.build.registry import (  # noqa: PLC0415
        EXPORT_JSONL_TARGET,
        EXPORT_PARQUET_TARGET,
    )

    # Native targets (migrated to Hamilton pipelines)
    registry.register(
        EXPORT_JSONL_TARGET,
        native_module="codeintel.build.hamilton.native.export.export_jsonl",
    )
    registry.register(
        EXPORT_PARQUET_TARGET,
        native_module="codeintel.build.hamilton.native.export.export_parquet",
    )
