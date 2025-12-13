"""Analytics plugins using the TargetPlugin protocol.

This package contains all analytics plugins that implement the
TargetPlugin protocol from codeintel.build.plugin. Plugins are
registered in the build registry (codeintel.build.plugin_registry).

Example
-------
>>> from codeintel.build.plugin_registry import get_plugin_for_target
>>> plugin = get_plugin_for_target("function_metrics")
>>> result = await plugin.execute(ctx)
"""

from __future__ import annotations

from codeintel.analytics.plugins.cfg_dfg import CfgDfgMetricsPlugin
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
from codeintel.analytics.plugins.subsystem_metrics import (
    SubsystemAgreementPlugin,
    SubsystemGraphMetricsPlugin,
)
from codeintel.analytics.plugins.subsystems import SubsystemsPlugin
from codeintel.analytics.plugins.symbol_graph_metrics import SymbolGraphMetricsPlugin
from codeintel.analytics.plugins.tests import (
    BehavioralCoveragePlugin,
    TestGraphMetricsPlugin,
    TestProfilePlugin,
)

__all__ = [
    "BehavioralCoveragePlugin",
    "CfgDfgMetricsPlugin",
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
    "SubsystemAgreementPlugin",
    "SubsystemGraphMetricsPlugin",
    "SubsystemsPlugin",
    "SymbolGraphMetricsPlugin",
    "TestGraphMetricsPlugin",
    "TestProfilePlugin",
]
