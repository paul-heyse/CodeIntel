"""Native analytics modules.

This package contains pure Hamilton implementations of analytics targets,
replacing plugin wrappers with explicit compute + materialize node pipelines.

Phase 4 Migration
-----------------
The following modules have been migrated from plugins to native Hamilton:

Already Native (Phase 1.5):
- risk_factors
- hotspots

Already Native (Phase 3):
- function_history
- history_timeseries
- subsystems
- entrypoints
- dependencies (external_deps)
- data_models
- coverage_functions
- cfg_dfg (cfg_dfg_metrics)
- test_graph_metrics

Newly Migrated (Phase 4):
- function_metrics
- ast_features (function_ast_features)
- function_effects
- function_contracts
- coverage_test_edges
- test_profile
- behavioral_coverage
- semantic_roles
- subsystem_graph_metrics
- subsystem_agreement
- config_data_flow
- profiles
- symbol_graph_metrics
"""

from __future__ import annotations

from codeintel.build.hamilton.native.analytics.ast_features import (
    AstFeaturesResult,
    t__function_ast_features,
    t__function_ast_features__compute,
)
from codeintel.build.hamilton.native.analytics.behavioral_coverage import (
    BehavioralCoverageResult,
    t__behavioral_coverage,
    t__behavioral_coverage__compute,
)
from codeintel.build.hamilton.native.analytics.cfg_dfg import (
    t__cfg_dfg_metrics,
    t__cfg_dfg_metrics__compute_cfg,
    t__cfg_dfg_metrics__compute_dfg,
)
from codeintel.build.hamilton.native.analytics.config_data_flow import (
    ConfigDataFlowResult,
    t__config_data_flow,
    t__config_data_flow__compute,
)
from codeintel.build.hamilton.native.analytics.coverage_functions import (
    t__coverage_functions,
    t__coverage_functions__compute,
)
from codeintel.build.hamilton.native.analytics.coverage_test_edges import (
    CoverageTestEdgesResult,
    t__coverage_test_edges,
    t__coverage_test_edges__compute,
)
from codeintel.build.hamilton.native.analytics.data_models import (
    t__data_model_usage,
    t__data_model_usage__compute,
    t__data_models,
    t__data_models__compute,
)
from codeintel.build.hamilton.native.analytics.dependencies import (
    t__external_deps,
    t__external_deps__compute_calls,
)
from codeintel.build.hamilton.native.analytics.entrypoints import (
    t__entrypoints,
    t__entrypoints__compute,
)
from codeintel.build.hamilton.native.analytics.function_contracts import (
    FunctionContractsResult,
    t__function_contracts,
    t__function_contracts__compute,
)
from codeintel.build.hamilton.native.analytics.function_effects import (
    FunctionEffectsResult,
    t__function_effects,
    t__function_effects__compute,
)
from codeintel.build.hamilton.native.analytics.function_history import (
    t__function_history,
    t__function_history__compute,
)
from codeintel.build.hamilton.native.analytics.function_metrics import (
    FunctionMetricsResult,
    t__function_metrics,
    t__function_metrics__compute,
)
from codeintel.build.hamilton.native.analytics.history_timeseries import (
    t__history_timeseries,
    t__history_timeseries__compute,
)
from codeintel.build.hamilton.native.analytics.hotspots import (
    hotspots__modules_complexity,
    t__hotspots,
    t__hotspots__compute,
)
from codeintel.build.hamilton.native.analytics.profiles import (
    ProfilesResult,
    t__profiles,
    t__profiles__compute,
)
from codeintel.build.hamilton.native.analytics.risk_factors import (
    risk_factors__fan_in,
    risk_factors__fan_out,
    t__risk_factors,
    t__risk_factors__compute,
)
from codeintel.build.hamilton.native.analytics.semantic_roles import (
    SemanticRolesResult,
    t__semantic_roles,
    t__semantic_roles__compute,
)
from codeintel.build.hamilton.native.analytics.subsystem_agreement import (
    SubsystemAgreementResult,
    t__subsystem_agreement,
    t__subsystem_agreement__compute,
)
from codeintel.build.hamilton.native.analytics.subsystem_graph_metrics import (
    SubsystemGraphMetricsResult,
    t__subsystem_graph_metrics,
    t__subsystem_graph_metrics__compute,
)
from codeintel.build.hamilton.native.analytics.subsystems import (
    t__subsystems,
    t__subsystems__compute,
)
from codeintel.build.hamilton.native.analytics.symbol_graph_metrics import (
    SymbolGraphMetricsResult,
    t__symbol_graph_metrics,
    t__symbol_graph_metrics__compute,
)
from codeintel.build.hamilton.native.analytics.test_graph_metrics import (
    t__test_graph_metrics,
    t__test_graph_metrics__compute,
)
from codeintel.build.hamilton.native.analytics.test_profile import (
    TestProfileResult,
    t__test_profile,
    t__test_profile__compute,
)

__all__ = [
    "AstFeaturesResult",
    "BehavioralCoverageResult",
    "ConfigDataFlowResult",
    "CoverageTestEdgesResult",
    "FunctionContractsResult",
    "FunctionEffectsResult",
    "FunctionMetricsResult",
    "ProfilesResult",
    "SemanticRolesResult",
    "SubsystemAgreementResult",
    "SubsystemGraphMetricsResult",
    "SymbolGraphMetricsResult",
    "TestProfileResult",
    "hotspots__modules_complexity",
    "risk_factors__fan_in",
    "risk_factors__fan_out",
    "t__behavioral_coverage",
    "t__behavioral_coverage__compute",
    "t__cfg_dfg_metrics",
    "t__cfg_dfg_metrics__compute_cfg",
    "t__cfg_dfg_metrics__compute_dfg",
    "t__config_data_flow",
    "t__config_data_flow__compute",
    "t__coverage_functions",
    "t__coverage_functions__compute",
    "t__coverage_test_edges",
    "t__coverage_test_edges__compute",
    "t__data_model_usage",
    "t__data_model_usage__compute",
    "t__data_models",
    "t__data_models__compute",
    "t__entrypoints",
    "t__entrypoints__compute",
    "t__external_deps",
    "t__external_deps__compute_calls",
    "t__function_ast_features",
    "t__function_ast_features__compute",
    "t__function_contracts",
    "t__function_contracts__compute",
    "t__function_effects",
    "t__function_effects__compute",
    "t__function_history",
    "t__function_history__compute",
    "t__function_metrics",
    "t__function_metrics__compute",
    "t__history_timeseries",
    "t__history_timeseries__compute",
    "t__hotspots",
    "t__hotspots__compute",
    "t__profiles",
    "t__profiles__compute",
    "t__risk_factors",
    "t__risk_factors__compute",
    "t__semantic_roles",
    "t__semantic_roles__compute",
    "t__subsystem_agreement",
    "t__subsystem_agreement__compute",
    "t__subsystem_graph_metrics",
    "t__subsystem_graph_metrics__compute",
    "t__subsystems",
    "t__subsystems__compute",
    "t__symbol_graph_metrics",
    "t__symbol_graph_metrics__compute",
    "t__test_graph_metrics",
    "t__test_graph_metrics__compute",
    "t__test_profile",
    "t__test_profile__compute",
]
