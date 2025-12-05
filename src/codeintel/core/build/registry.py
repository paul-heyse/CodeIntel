"""Target registration and graph construction for the build system.

This module defines all known output targets and their dependencies,
enabling the build system to compute minimal execution plans.

Targets are organized by module (ingestion, graphs, analytics) and
registered in the singleton target graph.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from codeintel.core.build.targets import OutputTarget, TargetGraph

log = logging.getLogger(__name__)

# =============================================================================
# Ingestion Targets
# =============================================================================

MODULES_TARGET = OutputTarget(
    name="modules",
    module="ingestion",
    plugin="repo_scan",
    tables=("core.modules", "core.file_state", "core.repo_map"),
    dependencies=(),
    description="Repository module and file index from scanning.",
)

AST_TARGET = OutputTarget(
    name="ast",
    module="ingestion",
    plugin="ast_extract",
    tables=("core.ast_nodes", "core.ast_metrics"),
    dependencies=("modules",),
    description="Python AST extraction and metrics.",
)

CST_TARGET = OutputTarget(
    name="cst",
    module="ingestion",
    plugin="cst_extract",
    tables=("core.cst_nodes",),
    dependencies=("modules",),
    description="Concrete syntax tree extraction.",
)

SCIP_TARGET = OutputTarget(
    name="scip",
    module="ingestion",
    plugin="scip_ingest",
    tables=("core.goids", "core.goid_crosswalk"),
    dependencies=("modules",),
    description="SCIP index ingestion and GOID generation.",
)

TYPING_TARGET = OutputTarget(
    name="typing",
    module="ingestion",
    plugin="typing_ingest",
    tables=("analytics.typedness", "analytics.static_diagnostics"),
    dependencies=("modules",),
    description="Type annotation analysis and static diagnostics.",
)

COVERAGE_INGEST_TARGET = OutputTarget(
    name="coverage_ingest",
    module="ingestion",
    plugin="coverage_ingest",
    tables=("analytics.coverage_lines",),
    dependencies=("modules",),
    description="Line-level test coverage ingestion.",
)

TESTS_INGEST_TARGET = OutputTarget(
    name="tests_ingest",
    module="ingestion",
    plugin="tests_ingest",
    tables=("analytics.test_catalog",),
    dependencies=("modules",),
    description="Test catalog ingestion from pytest.",
)

DOCSTRINGS_TARGET = OutputTarget(
    name="docstrings",
    module="ingestion",
    plugin="docstrings_ingest",
    tables=("core.docstrings",),
    dependencies=("ast",),
    description="Docstring extraction and parsing.",
)

CONFIG_INGEST_TARGET = OutputTarget(
    name="config_ingest",
    module="ingestion",
    plugin="config_ingest",
    tables=("analytics.config_values",),
    dependencies=("modules",),
    description="Configuration file parsing and reference tracking.",
)

# =============================================================================
# Graph Targets
# =============================================================================

GOIDS_TARGET = OutputTarget(
    name="goids",
    module="graphs",
    plugin="goid_builder",
    tables=("core.goids", "core.goid_crosswalk"),
    dependencies=("scip", "ast"),
    description="GOID resolution and crosswalk construction.",
)

CALL_GRAPH_TARGET = OutputTarget(
    name="call_graph",
    module="graphs",
    plugin="call_graph",
    tables=("graph.call_graph_nodes", "graph.call_graph_edges"),
    dependencies=("goids", "scip"),
    description="Function call graph construction.",
)

IMPORT_GRAPH_TARGET = OutputTarget(
    name="import_graph",
    module="graphs",
    plugin="import_graph",
    tables=("graph.import_graph_edges", "graph.import_modules"),
    dependencies=("modules",),
    description="Module import graph construction.",
)

CFG_TARGET = OutputTarget(
    name="cfg",
    module="graphs",
    plugin="cfg_builder",
    tables=("graph.cfg_blocks", "graph.cfg_edges"),
    dependencies=("goids", "ast"),
    description="Control flow graph construction per function.",
)

DFG_TARGET = OutputTarget(
    name="dfg",
    module="graphs",
    plugin="dfg_builder",
    tables=("graph.dfg_edges",),
    dependencies=("cfg",),
    description="Data flow graph construction per function.",
)

SYMBOL_USES_TARGET = OutputTarget(
    name="symbol_uses",
    module="graphs",
    plugin="symbol_uses",
    tables=("graph.symbol_use_edges",),
    dependencies=("scip",),
    description="Symbol definition-to-use edge extraction.",
)

GRAPH_VALIDATION_TARGET = OutputTarget(
    name="graph_validation",
    module="graphs",
    plugin="graph_validation",
    tables=("analytics.graph_validation",),
    dependencies=("call_graph", "import_graph", "cfg"),
    description="Graph integrity validation checks.",
)

# =============================================================================
# Analytics Targets
# =============================================================================

HOTSPOTS_TARGET = OutputTarget(
    name="hotspots",
    module="analytics",
    plugin="hotspots",
    tables=("analytics.hotspots",),
    dependencies=("modules",),
    description="File hotspot analysis based on churn.",
)

FUNCTION_METRICS_TARGET = OutputTarget(
    name="function_metrics",
    module="analytics",
    plugin="function_metrics",
    tables=("analytics.function_metrics", "analytics.function_types"),
    dependencies=("goids", "ast"),
    description="Function structural metrics and type annotations.",
)

FUNCTION_EFFECTS_TARGET = OutputTarget(
    name="function_effects",
    module="analytics",
    plugin="function_effects",
    tables=("analytics.function_effects",),
    dependencies=("function_metrics",),
    description="Function purity and side-effect analysis.",
)

FUNCTION_CONTRACTS_TARGET = OutputTarget(
    name="function_contracts",
    module="analytics",
    plugin="function_contracts",
    tables=("analytics.function_contracts",),
    dependencies=("function_metrics", "docstrings"),
    description="Inferred function pre/postconditions.",
)

FUNCTION_HISTORY_TARGET = OutputTarget(
    name="function_history",
    module="analytics",
    plugin="function_history",
    tables=("analytics.function_history",),
    dependencies=("goids",),
    description="Function git history and churn metrics.",
)

HISTORY_TIMESERIES_TARGET = OutputTarget(
    name="history_timeseries",
    module="analytics",
    plugin="history_timeseries",
    tables=("analytics.history_timeseries",),
    dependencies=("function_history",),
    description="Historical metrics timeseries for trending.",
)

COVERAGE_FUNCTIONS_TARGET = OutputTarget(
    name="coverage_functions",
    module="analytics",
    plugin="coverage_functions",
    tables=("analytics.coverage_functions",),
    dependencies=("goids", "coverage_ingest"),
    description="Per-function coverage aggregation.",
)

COVERAGE_TEST_EDGES_TARGET = OutputTarget(
    name="coverage_test_edges",
    module="analytics",
    plugin="coverage_test_edges",
    tables=("analytics.test_coverage_edges",),
    dependencies=("coverage_functions", "tests_ingest"),
    description="Test-to-function coverage edges.",
)

DATA_MODELS_TARGET = OutputTarget(
    name="data_models",
    module="analytics",
    plugin="data_models",
    tables=(
        "analytics.data_models",
        "analytics.data_model_fields",
        "analytics.data_model_relationships",
    ),
    dependencies=("goids", "ast"),
    description="Data model extraction (dataclasses, Pydantic, etc.).",
)

DATA_MODEL_USAGE_TARGET = OutputTarget(
    name="data_model_usage",
    module="analytics",
    plugin="data_model_usage",
    tables=("analytics.data_model_usage",),
    dependencies=("data_models", "call_graph"),
    description="Function-level data model usage tracking.",
)

CONFIG_DATA_FLOW_TARGET = OutputTarget(
    name="config_data_flow",
    module="analytics",
    plugin="config_data_flow",
    tables=("analytics.config_data_flow",),
    dependencies=("config_ingest", "call_graph"),
    description="Config key usage flow through functions.",
)

RISK_FACTORS_TARGET = OutputTarget(
    name="risk_factors",
    module="analytics",
    plugin="risk_factors",
    tables=("analytics.goid_risk_factors",),
    dependencies=(
        "function_metrics",
        "coverage_functions",
        "hotspots",
        "typing",
    ),
    description="Composite risk factors per function.",
)

GRAPH_METRICS_TARGET = OutputTarget(
    name="graph_metrics",
    module="analytics",
    plugin="graph_metrics",
    tables=(
        "analytics.graph_metrics_functions",
        "analytics.graph_metrics_functions_ext",
        "analytics.graph_metrics_modules",
        "analytics.graph_metrics_modules_ext",
    ),
    dependencies=("call_graph", "import_graph"),
    description="Graph topology metrics for functions and modules.",
)

SEMANTIC_ROLES_TARGET = OutputTarget(
    name="semantic_roles",
    module="analytics",
    plugin="semantic_roles",
    tables=("analytics.semantic_roles_functions", "analytics.semantic_roles_modules"),
    dependencies=("function_metrics", "call_graph"),
    description="Semantic role classification (handler, utility, etc.).",
)

SUBSYSTEMS_TARGET = OutputTarget(
    name="subsystems",
    module="analytics",
    plugin="subsystems",
    tables=(
        "analytics.subsystems",
        "analytics.subsystem_modules",
        "analytics.subsystem_graph_metrics",
    ),
    dependencies=("import_graph", "semantic_roles"),
    description="Architectural subsystem inference.",
)

TEST_PROFILE_TARGET = OutputTarget(
    name="test_profile",
    module="analytics",
    plugin="test_profile",
    tables=("analytics.test_profile",),
    dependencies=("coverage_test_edges", "tests_ingest"),
    description="Per-test profile with coverage and characteristics.",
)

BEHAVIORAL_COVERAGE_TARGET = OutputTarget(
    name="behavioral_coverage",
    module="analytics",
    plugin="behavioral_coverage",
    tables=("analytics.behavioral_coverage",),
    dependencies=("test_profile",),
    description="Behavioral coverage tagging from test patterns.",
)

ENTRYPOINTS_TARGET = OutputTarget(
    name="entrypoints",
    module="analytics",
    plugin="entrypoints",
    tables=("analytics.entrypoints", "analytics.entrypoint_tests"),
    dependencies=("goids", "semantic_roles", "test_profile"),
    description="External entrypoint detection (HTTP, CLI, etc.).",
)

EXTERNAL_DEPS_TARGET = OutputTarget(
    name="external_deps",
    module="analytics",
    plugin="external_deps",
    tables=("analytics.external_dependencies", "analytics.external_dependency_calls"),
    dependencies=("call_graph",),
    description="External library dependency analysis.",
)

PROFILES_TARGET = OutputTarget(
    name="profiles",
    module="analytics",
    plugin="profiles",
    tables=(
        "analytics.function_profile",
        "analytics.file_profile",
        "analytics.module_profile",
    ),
    dependencies=(
        "risk_factors",
        "graph_metrics",
        "function_history",
        "semantic_roles",
        "docstrings",
    ),
    description="Denormalized profile tables for querying.",
)

FUNCTION_AST_FEATURES_TARGET = OutputTarget(
    name="function_ast_features",
    module="analytics",
    plugin="function_ast_features",
    tables=("analytics.function_ast_features",),
    dependencies=("goids", "ast"),
    description="AST-derived semantic features for functions.",
)

# =============================================================================
# Target Registration
# =============================================================================

ALL_TARGETS: tuple[OutputTarget, ...] = (
    # Ingestion
    MODULES_TARGET,
    AST_TARGET,
    CST_TARGET,
    SCIP_TARGET,
    TYPING_TARGET,
    COVERAGE_INGEST_TARGET,
    TESTS_INGEST_TARGET,
    DOCSTRINGS_TARGET,
    CONFIG_INGEST_TARGET,
    # Graphs
    GOIDS_TARGET,
    CALL_GRAPH_TARGET,
    IMPORT_GRAPH_TARGET,
    CFG_TARGET,
    DFG_TARGET,
    SYMBOL_USES_TARGET,
    GRAPH_VALIDATION_TARGET,
    # Analytics
    HOTSPOTS_TARGET,
    FUNCTION_METRICS_TARGET,
    FUNCTION_EFFECTS_TARGET,
    FUNCTION_CONTRACTS_TARGET,
    FUNCTION_HISTORY_TARGET,
    HISTORY_TIMESERIES_TARGET,
    COVERAGE_FUNCTIONS_TARGET,
    COVERAGE_TEST_EDGES_TARGET,
    DATA_MODELS_TARGET,
    DATA_MODEL_USAGE_TARGET,
    CONFIG_DATA_FLOW_TARGET,
    RISK_FACTORS_TARGET,
    GRAPH_METRICS_TARGET,
    SEMANTIC_ROLES_TARGET,
    SUBSYSTEMS_TARGET,
    TEST_PROFILE_TARGET,
    BEHAVIORAL_COVERAGE_TARGET,
    ENTRYPOINTS_TARGET,
    EXTERNAL_DEPS_TARGET,
    PROFILES_TARGET,
    FUNCTION_AST_FEATURES_TARGET,
)


def build_target_graph() -> TargetGraph:
    """Construct the complete target graph from all registered targets.

    Returns
    -------
    TargetGraph
        Graph with all targets registered and validated.

    Raises
    ------
    ValueError
        If the graph has validation errors (cycles, missing deps).
    """
    graph = TargetGraph()
    for target in ALL_TARGETS:
        graph.register(target)
    errors = graph.validate()
    if errors:
        error_msg = "\n".join(errors)
        msg = f"Target graph validation failed:\n{error_msg}"
        raise ValueError(msg)
    return graph


@lru_cache(maxsize=1)
def get_target_graph() -> TargetGraph:
    """Get the singleton target graph instance.

    Returns
    -------
    TargetGraph
        The singleton target graph with all registered targets.
    """
    return build_target_graph()


__all__ = [
    "ALL_TARGETS",
    "AST_TARGET",
    "BEHAVIORAL_COVERAGE_TARGET",
    "CALL_GRAPH_TARGET",
    "CFG_TARGET",
    "CONFIG_DATA_FLOW_TARGET",
    "CONFIG_INGEST_TARGET",
    "COVERAGE_FUNCTIONS_TARGET",
    "COVERAGE_INGEST_TARGET",
    "COVERAGE_TEST_EDGES_TARGET",
    "CST_TARGET",
    "DATA_MODELS_TARGET",
    "DATA_MODEL_USAGE_TARGET",
    "DFG_TARGET",
    "DOCSTRINGS_TARGET",
    "ENTRYPOINTS_TARGET",
    "EXTERNAL_DEPS_TARGET",
    "FUNCTION_AST_FEATURES_TARGET",
    "FUNCTION_CONTRACTS_TARGET",
    "FUNCTION_EFFECTS_TARGET",
    "FUNCTION_HISTORY_TARGET",
    "FUNCTION_METRICS_TARGET",
    "GOIDS_TARGET",
    "GRAPH_METRICS_TARGET",
    "GRAPH_VALIDATION_TARGET",
    "HISTORY_TIMESERIES_TARGET",
    "HOTSPOTS_TARGET",
    "IMPORT_GRAPH_TARGET",
    "MODULES_TARGET",
    "PROFILES_TARGET",
    "RISK_FACTORS_TARGET",
    "SCIP_TARGET",
    "SEMANTIC_ROLES_TARGET",
    "SUBSYSTEMS_TARGET",
    "SYMBOL_USES_TARGET",
    "TESTS_INGEST_TARGET",
    "TEST_PROFILE_TARGET",
    "TYPING_TARGET",
    "build_target_graph",
    "get_target_graph",
]
