"""Plugin registration for the unified analytics system.

This module instantiates analytics plugins as TargetPlugin instances.
These plugins are discovered and executed by the build executor via ALL_PLUGINS.

Migration Note
--------------
Analytics plugins implement TargetPlugin. The build executor executes them
directly via `plugin.execute(ctx)` rather than through a separate registry.
"""

from __future__ import annotations

from codeintel.analytics.plugins.config_data_flow import ConfigDataFlowPlugin
from codeintel.analytics.plugins.coverage import (
    CoverageFunctionsPlugin,
    CoverageTestEdgesPlugin,
)
from codeintel.analytics.plugins.data_models import (
    DataModelsPlugin,
    DataModelUsagePlugin,
)
from codeintel.analytics.plugins.dependencies import ExternalDepsPlugin
from codeintel.analytics.plugins.entrypoints import EntrypointsPlugin
from codeintel.analytics.plugins.functions import (
    FunctionAstFeaturesPlugin,
    FunctionContractsPlugin,
    FunctionEffectsPlugin,
    FunctionHistoryPlugin,
    FunctionMetricsPlugin,
)
from codeintel.analytics.plugins.history import HistoryTimeseriesPlugin
from codeintel.analytics.plugins.hotspots import HotspotsPlugin
from codeintel.analytics.plugins.profiles import ProfilesPlugin
from codeintel.analytics.plugins.risk import RiskFactorsPlugin
from codeintel.analytics.plugins.semantic_roles import SemanticRolesPlugin
from codeintel.analytics.plugins.subsystems import SubsystemsPlugin
from codeintel.analytics.plugins.tests import (
    BehavioralCoveragePlugin,
    TestProfilePlugin,
)

# Singleton plugin instances - instantiate once, register once
FUNCTION_METRICS_PLUGIN = FunctionMetricsPlugin()
FUNCTION_AST_FEATURES_PLUGIN = FunctionAstFeaturesPlugin()
FUNCTION_EFFECTS_PLUGIN = FunctionEffectsPlugin()
FUNCTION_CONTRACTS_PLUGIN = FunctionContractsPlugin()
FUNCTION_HISTORY_PLUGIN = FunctionHistoryPlugin()
COVERAGE_FUNCTIONS_PLUGIN = CoverageFunctionsPlugin()
COVERAGE_TEST_EDGES_PLUGIN = CoverageTestEdgesPlugin()
TEST_PROFILE_PLUGIN = TestProfilePlugin()
BEHAVIORAL_COVERAGE_PLUGIN = BehavioralCoveragePlugin()
HOTSPOTS_PLUGIN = HotspotsPlugin()
SUBSYSTEMS_PLUGIN = SubsystemsPlugin()
SEMANTIC_ROLES_PLUGIN = SemanticRolesPlugin()
DATA_MODELS_PLUGIN = DataModelsPlugin()
DATA_MODEL_USAGE_PLUGIN = DataModelUsagePlugin()
ENTRYPOINTS_PLUGIN = EntrypointsPlugin()
EXTERNAL_DEPS_PLUGIN = ExternalDepsPlugin()
PROFILES_PLUGIN = ProfilesPlugin()
HISTORY_TIMESERIES_PLUGIN = HistoryTimeseriesPlugin()
RISK_FACTORS_PLUGIN = RiskFactorsPlugin()
CONFIG_DATA_FLOW_PLUGIN = ConfigDataFlowPlugin()

# All plugins - used by build executor for discovery
# Note: Graph plugins are in graphs.plugins and are executed via
# the graph plugin system, not this registration module.
ALL_PLUGINS = (
    FUNCTION_METRICS_PLUGIN,
    FUNCTION_AST_FEATURES_PLUGIN,
    FUNCTION_EFFECTS_PLUGIN,
    FUNCTION_CONTRACTS_PLUGIN,
    FUNCTION_HISTORY_PLUGIN,
    COVERAGE_FUNCTIONS_PLUGIN,
    COVERAGE_TEST_EDGES_PLUGIN,
    TEST_PROFILE_PLUGIN,
    BEHAVIORAL_COVERAGE_PLUGIN,
    HOTSPOTS_PLUGIN,
    SUBSYSTEMS_PLUGIN,
    SEMANTIC_ROLES_PLUGIN,
    DATA_MODELS_PLUGIN,
    DATA_MODEL_USAGE_PLUGIN,
    ENTRYPOINTS_PLUGIN,
    EXTERNAL_DEPS_PLUGIN,
    PROFILES_PLUGIN,
    HISTORY_TIMESERIES_PLUGIN,
    RISK_FACTORS_PLUGIN,
    CONFIG_DATA_FLOW_PLUGIN,
)

__all__ = [
    "ALL_PLUGINS",
    "BEHAVIORAL_COVERAGE_PLUGIN",
    "CONFIG_DATA_FLOW_PLUGIN",
    "COVERAGE_FUNCTIONS_PLUGIN",
    "COVERAGE_TEST_EDGES_PLUGIN",
    "DATA_MODELS_PLUGIN",
    "DATA_MODEL_USAGE_PLUGIN",
    "ENTRYPOINTS_PLUGIN",
    "EXTERNAL_DEPS_PLUGIN",
    "FUNCTION_AST_FEATURES_PLUGIN",
    "FUNCTION_CONTRACTS_PLUGIN",
    "FUNCTION_EFFECTS_PLUGIN",
    "FUNCTION_HISTORY_PLUGIN",
    "FUNCTION_METRICS_PLUGIN",
    "HISTORY_TIMESERIES_PLUGIN",
    "HOTSPOTS_PLUGIN",
    "PROFILES_PLUGIN",
    "RISK_FACTORS_PLUGIN",
    "SEMANTIC_ROLES_PLUGIN",
    "SUBSYSTEMS_PLUGIN",
    "TEST_PROFILE_PLUGIN",
]
