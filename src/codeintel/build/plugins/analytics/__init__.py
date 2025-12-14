"""Analytics plugins using the TargetPlugin protocol.

This package contains all analytics plugins that implement the
TargetPlugin protocol from codeintel.build.plugin. Plugins are
registered in the build registry (codeintel.build.unified_registry).

Note: Several plugins have been removed in favor of Hamilton native modules
(see ``codeintel.build.hamilton.native.analytics``). The removed plugins are:
- CfgDfgMetricsPlugin
- CoverageFunctionsPlugin
- DataModelsPlugin
- DataModelUsagePlugin
- EntrypointsPlugin
- ExternalDepsPlugin
- FunctionHistoryPlugin
- HistoryTimeseriesPlugin
- TestGraphMetricsPlugin
"""

from __future__ import annotations

from codeintel.build.plugins.analytics.config_data_flow import ConfigDataFlowPlugin
from codeintel.build.plugins.analytics.coverage import CoverageTestEdgesPlugin
from codeintel.build.plugins.analytics.functions import (
    FunctionAstFeaturesPlugin,
    FunctionContractsPlugin,
    FunctionEffectsPlugin,
    FunctionMetricsPlugin,
)
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
