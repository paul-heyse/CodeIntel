"""Plugin registration for the unified analytics system.

This module instantiates analytics plugins that are now used by the
build system. The legacy registry registration is deprecated.

NOTE: The analytics plugins now implement TargetPlugin instead of
AnalyticsPluginProtocol. They are registered via codeintel.build.plugin_registry
rather than the legacy analytics registry.
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

# All plugins in registration order
# Note: Graph metrics plugins are now in graphs.plugins.metrics and
# are registered separately via the graph plugin system.
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


# Track registration state
class _RegistrationState:
    """Thread-safe state holder for plugin registration."""

    registered: bool = False


def register_all_plugins() -> None:
    """Register all analytics plugins with the global registry.

    DEPRECATED: The analytics plugins now use the build system's
    plugin_registry. This function is a no-op for backward compatibility.
    """
    # No-op: plugins are now registered via codeintel.build.plugin_registry
    _RegistrationState.registered = True


def ensure_plugins_registered() -> None:
    """Ensure all plugins are registered.

    This is an alias for register_all_plugins() for clarity in calling code.
    """
    register_all_plugins()


def reset_registration_state() -> None:
    """Reset the registration state.

    Primarily useful for testing to ensure clean state between tests.
    """
    _RegistrationState.registered = False


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
    "ensure_plugins_registered",
    "register_all_plugins",
]
