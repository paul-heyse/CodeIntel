"""Plugin registration for the unified analytics system.

This module instantiates and registers all analytics plugins with the
global registry. It also provides backward-compatible constants for
pipeline steps that reference plugins by name.
"""

from __future__ import annotations

from codeintel.analytics.core.plugins.config_data_flow import ConfigDataFlowPlugin
from codeintel.analytics.core.plugins.coverage import (
    CoverageFunctionsPlugin,
    CoverageTestEdgesPlugin,
)
from codeintel.analytics.core.plugins.data_models import (
    DataModelsPlugin,
    DataModelUsagePlugin,
)
from codeintel.analytics.core.plugins.dependencies import ExternalDepsPlugin
from codeintel.analytics.core.plugins.entrypoints import EntrypointsPlugin
from codeintel.analytics.core.plugins.functions import (
    FunctionAstFeaturesPlugin,
    FunctionContractsPlugin,
    FunctionEffectsPlugin,
    FunctionHistoryPlugin,
    FunctionMetricsPlugin,
)
from codeintel.analytics.core.plugins.graphs import CoreGraphMetricsPlugin
from codeintel.analytics.core.plugins.history import HistoryTimeseriesPlugin
from codeintel.analytics.core.plugins.hotspots import HotspotsPlugin
from codeintel.analytics.core.plugins.profiles import ProfilesPlugin
from codeintel.analytics.core.plugins.risk import RiskFactorsPlugin
from codeintel.analytics.core.plugins.semantic_roles import SemanticRolesPlugin
from codeintel.analytics.core.plugins.subsystems import SubsystemsPlugin
from codeintel.analytics.core.plugins.tests import (
    BehavioralCoveragePlugin,
    TestProfilePlugin,
)
from codeintel.analytics.core.registry import get_registry

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
CORE_GRAPH_METRICS_PLUGIN = CoreGraphMetricsPlugin()

# All plugins in registration order
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
    CORE_GRAPH_METRICS_PLUGIN,
)

# Track registration state
_REGISTERED = False


def register_all_plugins() -> None:
    """Register all plugins with the global registry.

    This function is idempotent - calling it multiple times has no effect
    after the first registration.
    """
    global _REGISTERED  # noqa: PLW0603
    if _REGISTERED:
        return

    registry = get_registry()
    for plugin in ALL_PLUGINS:
        registry.register(plugin)

    _REGISTERED = True


def ensure_plugins_registered() -> None:
    """Ensure all plugins are registered.

    This is an alias for register_all_plugins() for clarity in calling code.
    """
    register_all_plugins()


# Default plugin names for backward compatibility
DEFAULT_ANALYTICS_PLUGINS: tuple[str, ...] = tuple(
    plugin.metadata.name for plugin in ALL_PLUGINS if plugin.metadata.enabled_by_default
)


__all__ = [
    "ALL_PLUGINS",
    "BEHAVIORAL_COVERAGE_PLUGIN",
    "CONFIG_DATA_FLOW_PLUGIN",
    "CORE_GRAPH_METRICS_PLUGIN",
    "COVERAGE_FUNCTIONS_PLUGIN",
    "COVERAGE_TEST_EDGES_PLUGIN",
    "DATA_MODELS_PLUGIN",
    "DATA_MODEL_USAGE_PLUGIN",
    "DEFAULT_ANALYTICS_PLUGINS",
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
