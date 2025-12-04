"""Migrated analytics plugins using the new protocol.

This package contains all analytics plugins migrated to the new
unified plugin protocol.
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
from codeintel.analytics.plugins.registration import (
    ALL_PLUGINS,
    BEHAVIORAL_COVERAGE_PLUGIN,
    CONFIG_DATA_FLOW_PLUGIN,
    COVERAGE_FUNCTIONS_PLUGIN,
    COVERAGE_TEST_EDGES_PLUGIN,
    DATA_MODEL_USAGE_PLUGIN,
    DATA_MODELS_PLUGIN,
    ENTRYPOINTS_PLUGIN,
    EXTERNAL_DEPS_PLUGIN,
    FUNCTION_AST_FEATURES_PLUGIN,
    FUNCTION_CONTRACTS_PLUGIN,
    FUNCTION_EFFECTS_PLUGIN,
    FUNCTION_HISTORY_PLUGIN,
    FUNCTION_METRICS_PLUGIN,
    HISTORY_TIMESERIES_PLUGIN,
    HOTSPOTS_PLUGIN,
    PROFILES_PLUGIN,
    RISK_FACTORS_PLUGIN,
    SEMANTIC_ROLES_PLUGIN,
    SUBSYSTEMS_PLUGIN,
    TEST_PROFILE_PLUGIN,
    ensure_plugins_registered,
    register_all_plugins,
)
from codeintel.analytics.plugins.risk import RiskFactorsPlugin
from codeintel.analytics.plugins.semantic_roles import SemanticRolesPlugin
from codeintel.analytics.plugins.subsystems import SubsystemsPlugin
from codeintel.analytics.plugins.tests import (
    BehavioralCoveragePlugin,
    TestProfilePlugin,
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
    "BehavioralCoveragePlugin",
    "ConfigDataFlowPlugin",
    "CoverageFunctionsPlugin",
    "CoverageTestEdgesPlugin",
    "DataModelUsagePlugin",
    "DataModelsPlugin",
    "EntrypointsPlugin",
    "ExternalDepsPlugin",
    "FunctionAstFeaturesPlugin",
    "FunctionContractsPlugin",
    "FunctionEffectsPlugin",
    "FunctionHistoryPlugin",
    "FunctionMetricsPlugin",
    "HistoryTimeseriesPlugin",
    "HotspotsPlugin",
    "ProfilesPlugin",
    "RiskFactorsPlugin",
    "SemanticRolesPlugin",
    "SubsystemsPlugin",
    "TestProfilePlugin",
    "ensure_plugins_registered",
    "register_all_plugins",
]
