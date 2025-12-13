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

from codeintel.build.plugins.analytics.cfg_dfg import CfgDfgMetricsPlugin
from codeintel.build.plugins.analytics.config_data_flow import ConfigDataFlowPlugin
from codeintel.build.plugins.analytics.coverage import (
    CoverageFunctionsPlugin,
    CoverageTestEdgesPlugin,
)
from codeintel.build.plugins.analytics.data_models import (
    DataModelsPlugin,
    DataModelUsagePlugin,
)
from codeintel.build.plugins.analytics.dependencies import ExternalDepsPlugin
from codeintel.build.plugins.analytics.entrypoints import EntrypointsPlugin
from codeintel.build.plugins.analytics.functions import (
    FunctionAstFeaturesPlugin,
    FunctionContractsPlugin,
    FunctionEffectsPlugin,
    FunctionHistoryPlugin,
    FunctionMetricsPlugin,
)
from codeintel.build.plugins.analytics.history import HistoryTimeseriesPlugin
from codeintel.build.plugins.analytics.hotspots import HotspotsPlugin
from codeintel.build.plugins.analytics.profiles import ProfilesPlugin
from codeintel.build.plugins.analytics.risk import RiskFactorsPlugin
from codeintel.build.plugins.analytics.semantic_roles import SemanticRolesPlugin
from codeintel.build.plugins.analytics.subsystem_metrics import (
    SubsystemAgreementPlugin,
    SubsystemGraphMetricsPlugin,
)
from codeintel.build.plugins.analytics.subsystems import SubsystemsPlugin
from codeintel.build.plugins.analytics.symbol_graph_metrics import SymbolGraphMetricsPlugin
from codeintel.build.plugins.analytics.tests import (
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
