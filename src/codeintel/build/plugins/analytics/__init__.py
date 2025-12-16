"""Analytics plugins using the TargetPlugin protocol.

This package contains all analytics plugins that implement the
TargetPlugin protocol from codeintel.build.plugin. Plugins are
registered in the build registry (codeintel.build.unified_registry).

**Phase 4 Migration Complete**

All analytics plugins have been migrated to native Hamilton modules.
The original plugin implementations are no longer used. This module
now exports stub classes that emit deprecation warnings when used.

For the actual implementations, see:
``codeintel.build.hamilton.native.analytics``

Migrated plugins:
- BehavioralCoveragePlugin -> behavioral_coverage.py
- ConfigDataFlowPlugin -> config_data_flow.py
- CoverageTestEdgesPlugin -> coverage_test_edges.py
- FunctionAstFeaturesPlugin -> ast_features.py
- FunctionContractsPlugin -> function_contracts.py
- FunctionEffectsPlugin -> function_effects.py
- FunctionMetricsPlugin -> function_metrics.py
- HotspotsPlugin -> hotspots.py
- ProfilesPlugin -> profiles.py
- RiskFactorsPlugin -> risk_factors.py
- SemanticRolesPlugin -> semantic_roles.py
- SubsystemAgreementPlugin -> subsystem_agreement.py
- SubsystemGraphMetricsPlugin -> subsystem_graph_metrics.py
- SubsystemsPlugin -> subsystems.py
- SymbolGraphMetricsPlugin -> symbol_graph_metrics.py
- TestProfilePlugin -> test_profile.py

Previously migrated (Phase 3):
- CfgDfgMetricsPlugin -> cfg_dfg.py
- CoverageFunctionsPlugin -> coverage_functions.py
- DataModelsPlugin -> data_models.py
- DataModelUsagePlugin -> data_models.py
- EntrypointsPlugin -> entrypoints.py
- ExternalDepsPlugin -> dependencies.py
- FunctionHistoryPlugin -> function_history.py
- HistoryTimeseriesPlugin -> history_timeseries.py
- TestGraphMetricsPlugin -> test_graph_metrics.py
"""

from __future__ import annotations

# Import stub classes for backward compatibility
from codeintel.build.plugins.analytics.stubs import (
    BehavioralCoveragePlugin,
    ConfigDataFlowPlugin,
    CoverageTestEdgesPlugin,
    FunctionAstFeaturesPlugin,
    FunctionContractsPlugin,
    FunctionEffectsPlugin,
    FunctionMetricsPlugin,
    HotspotsPlugin,
    ProfilesPlugin,
    RiskFactorsPlugin,
    SemanticRolesPlugin,
    SubsystemAgreementPlugin,
    SubsystemGraphMetricsPlugin,
    SubsystemsPlugin,
    SymbolGraphMetricsPlugin,
    TestProfilePlugin,
)

__all__ = [
    "BehavioralCoveragePlugin",
    "ConfigDataFlowPlugin",
    "CoverageTestEdgesPlugin",
    "FunctionAstFeaturesPlugin",
    "FunctionContractsPlugin",
    "FunctionEffectsPlugin",
    "FunctionMetricsPlugin",
    "HotspotsPlugin",
    "ProfilesPlugin",
    "RiskFactorsPlugin",
    "SemanticRolesPlugin",
    "SubsystemAgreementPlugin",
    "SubsystemGraphMetricsPlugin",
    "SubsystemsPlugin",
    "SymbolGraphMetricsPlugin",
    "TestProfilePlugin",
]
