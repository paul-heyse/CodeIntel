"""Native analytics modules.

This package contains pure Hamilton implementations of analytics targets,
replacing legacy wrappers with explicit compute + materialize node pipelines.

Phase 4 Migration
-----------------
The following modules have been migrated from legacy wrappers to native Hamilton:

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

from codeintel.build.hamilton.native.analytics.classification_targets import (
    SemanticRolesResult,
    TestProfileComputeResult,
    semantic_roles__functions_rows,
    semantic_roles__modules_rows,
    t__semantic_roles,
    t__semantic_roles__compute,
    t__test_profile,
    t__test_profile__compute,
    test_profile__rows,
)
from codeintel.build.hamilton.native.analytics.config_graph_targets import (
    ConfigDataFlowComputeResult,
    t__cfg_dfg_metrics,
    t__cfg_dfg_metrics__compute_cfg,
    t__cfg_dfg_metrics__compute_dfg,
    t__config_data_flow,
    t__config_data_flow__compute,
)
from codeintel.build.hamilton.native.analytics.coverage_targets import (
    t__behavioral_coverage,
    t__behavioral_coverage__compute,
    t__coverage_functions,
    t__coverage_functions__compute,
    t__coverage_test_edges,
    t__coverage_test_edges__compute,
)
from codeintel.build.hamilton.native.analytics.dependency_targets import (
    t__entrypoints,
    t__entrypoints__compute,
    t__external_deps,
    t__external_deps__compute_calls,
)
from codeintel.build.hamilton.native.analytics.function_detail_targets import (
    FunctionContractsResult,
    FunctionEffectsResult,
    function_contracts__rows,
    function_effects__rows,
    t__function_contracts,
    t__function_contracts__compute,
    t__function_effects,
    t__function_effects__compute,
)
from codeintel.build.hamilton.native.analytics.function_metrics import (
    FunctionAnalyticsResult,
    function_metrics__metrics_rows,
    function_metrics__types_rows,
    function_metrics__validation_rows,
    t__function_metrics,
    t__function_metrics__compute,
)
from codeintel.build.hamilton.native.analytics.hotspots import (
    hotspots__rows,
    t__hotspots,
    t__hotspots__compute,
)
from codeintel.build.hamilton.native.analytics.metadata_targets import (
    AstFeaturesResult,
    ProfilesComputeResult,
    t__data_model_usage,
    t__data_model_usage__compute,
    t__data_models,
    t__data_models__compute,
    t__function_ast_features,
    t__function_ast_features__compute,
    t__profiles,
    t__profiles__compute,
)
from codeintel.build.hamilton.native.analytics.metrics_targets import (
    t__function_history,
    t__function_history__compute,
    t__history_timeseries,
    t__history_timeseries__compute,
    t__subsystem_agreement,
    t__subsystem_agreement__compute,
    t__subsystem_graph_metrics,
    t__subsystem_graph_metrics__compute,
    t__symbol_graph_metrics,
    t__symbol_graph_metrics__compute,
    t__test_graph_metrics,
    t__test_graph_metrics__compute,
    test_graph_metrics__functions_rows,
    test_graph_metrics__tests_rows,
)
from codeintel.build.hamilton.native.analytics.risk_factors import (
    risk_factors__fan_in,
    risk_factors__fan_out,
    t__risk_factors,
    t__risk_factors__compute,
)
from codeintel.build.hamilton.native.analytics.subsystem_cache_targets import (
    SubsystemCachesComputeResult,
    subsystem_coverage_cache__rows,
    subsystem_profile_cache__rows,
    t__subsystem_caches,
    t__subsystem_caches__compute,
)
from codeintel.build.hamilton.native.analytics.subsystem_targets import (
    t__subsystems,
)

__all__ = [
    "AstFeaturesResult",
    "ConfigDataFlowComputeResult",
    "FunctionAnalyticsResult",
    "FunctionContractsResult",
    "FunctionEffectsResult",
    "ProfilesComputeResult",
    "SemanticRolesResult",
    "SubsystemCachesComputeResult",
    "TestProfileComputeResult",
    "function_contracts__rows",
    "function_effects__rows",
    "function_metrics__metrics_rows",
    "function_metrics__types_rows",
    "function_metrics__validation_rows",
    "hotspots__rows",
    "risk_factors__fan_in",
    "risk_factors__fan_out",
    "semantic_roles__functions_rows",
    "semantic_roles__modules_rows",
    "subsystem_coverage_cache__rows",
    "subsystem_profile_cache__rows",
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
    "t__subsystem_caches",
    "t__subsystem_caches__compute",
    "t__subsystem_graph_metrics",
    "t__subsystem_graph_metrics__compute",
    "t__subsystems",
    "t__symbol_graph_metrics",
    "t__symbol_graph_metrics__compute",
    "t__test_graph_metrics",
    "t__test_graph_metrics__compute",
    "t__test_profile",
    "t__test_profile__compute",
    "test_graph_metrics__functions_rows",
    "test_graph_metrics__tests_rows",
    "test_profile__rows",
]
