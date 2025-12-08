"""Target registration and graph construction for the build system.

This module defines all known output targets and their dependencies,
enabling the build system to compute minimal execution plans.

Targets are organized by module (ingestion, graphs, analytics) and
registered in the singleton target graph.

The OutputTarget is now the single source of truth for what each target
produces. Targets can be defined with:
1. Full contracts (new pattern) - OutputContract with TableSchema
2. Legacy tables (deprecated) - Just table key strings

During migration, both patterns coexist. TABLE_SCHEMAS can be derived
from target contracts using derive_schemas_from_targets().
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.contracts import ArtifactSpec, OutputContract
from codeintel.build.resources import (
    CPU_INTENSIVE_EXECUTION,
    TOOL_EXECUTION,
    TargetResources,
)
from codeintel.build.targets import OutputTarget, TargetGraph

if TYPE_CHECKING:
    from codeintel.config.datasets.primitives import TableSchema

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
    resources=TargetResources(tracker=True, modules=True),
    execution=CPU_INTENSIVE_EXECUTION,
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
    contract=OutputContract(
        artifacts=(
            ArtifactSpec("scip_index", "{scip_dir}/index.scip", "SCIP index file"),
            ArtifactSpec("scip_json", "{scip_dir}/index.json", "SCIP JSON export"),
        ),
    ),
    tables=("core.goids", "core.goid_crosswalk"),  # Legacy: will migrate to contract
    dependencies=("modules",),
    resources=TargetResources(
        tracker=True,
        modules=True,
        tools=("scip-python", "scip"),
    ),
    execution=TOOL_EXECUTION,
    description="SCIP index ingestion and GOID generation.",
)

TYPING_TARGET = OutputTarget(
    name="typing",
    module="ingestion",
    plugin="typing_ingest",
    tables=("analytics.typedness", "analytics.static_diagnostics"),
    dependencies=("modules",),
    resources=TargetResources(
        tracker=True,
        modules=True,
        tools=("pyright", "pyrefly", "ruff"),
    ),
    execution=TOOL_EXECUTION,
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
    plugin="callgraph",
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
    plugin="cfg_dfg",
    tables=("graph.cfg_blocks", "graph.cfg_edges"),
    dependencies=("goids", "ast"),
    description="Control flow graph construction per function.",
)

DFG_TARGET = OutputTarget(
    name="dfg",
    module="graphs",
    plugin="",  # No separate plugin - CFG target runs cfg_dfg which produces both
    tables=("graph.dfg_edges",),
    dependencies=("cfg",),
    description="Data flow graph construction per function.",
)

CFG_DFG_METRICS_TARGET = OutputTarget(
    name="cfg_dfg_metrics",
    module="analytics",
    plugin="cfg_dfg_metrics",
    tables=(
        "analytics.cfg_function_metrics",
        "analytics.cfg_block_metrics",
        "analytics.cfg_function_metrics_ext",
        "analytics.dfg_function_metrics",
        "analytics.dfg_block_metrics",
        "analytics.dfg_function_metrics_ext",
    ),
    dependencies=("cfg", "dfg"),
    description="Control-flow and data-flow graph metrics per function.",
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
    module="graphs",
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
    ),
    dependencies=("import_graph", "semantic_roles"),
    description="Architectural subsystem inference.",
)

SUBSYSTEM_GRAPH_METRICS_TARGET = OutputTarget(
    name="subsystem_graph_metrics",
    module="analytics",
    plugin="subsystem_graph_metrics",
    tables=("analytics.subsystem_graph_metrics",),
    dependencies=("subsystems", "graph_metrics"),
    description="Graph metrics for subsystems.",
)

SUBSYSTEM_AGREEMENT_TARGET = OutputTarget(
    name="subsystem_agreement",
    module="analytics",
    plugin="subsystem_agreement",
    tables=("analytics.subsystem_agreement",),
    dependencies=("subsystems", "graph_metrics"),
    description="Subsystem vs import community agreement.",
)

TEST_PROFILE_TARGET = OutputTarget(
    name="test_profile",
    module="analytics",
    plugin="test_profile",
    tables=("analytics.test_profile",),
    dependencies=("coverage_test_edges", "tests_ingest"),
    description="Per-test profile with coverage and characteristics.",
)

TEST_GRAPH_METRICS_TARGET = OutputTarget(
    name="test_graph_metrics",
    module="analytics",
    plugin="test_graph_metrics",
    tables=(
        "analytics.test_graph_metrics_tests",
        "analytics.test_graph_metrics_functions",
    ),
    dependencies=("coverage_test_edges",),
    description="Graph metrics from test-function bipartite graph.",
)

SYMBOL_GRAPH_METRICS_TARGET = OutputTarget(
    name="symbol_graph_metrics",
    module="analytics",
    plugin="symbol_graph_metrics",
    tables=(
        "analytics.symbol_graph_metrics_functions",
        "analytics.symbol_graph_metrics_modules",
    ),
    dependencies=("symbol_uses",),
    description="Graph metrics from symbol usage patterns.",
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
# Export Targets
# =============================================================================

EXPORT_JSONL_TARGET = OutputTarget(
    name="export_jsonl",
    module="export",
    plugin="export_jsonl",
    tables=(),  # Produces files, not tables
    dependencies=("profiles",),  # Depends on analytics being complete
    description="Export datasets to JSONL format for Document Output.",
)

EXPORT_PARQUET_TARGET = OutputTarget(
    name="export_parquet",
    module="export",
    plugin="export_parquet",
    tables=(),  # Produces files, not tables
    dependencies=("profiles",),  # Depends on analytics being complete
    description="Export datasets to Parquet format for Document Output.",
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
    CFG_DFG_METRICS_TARGET,
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
    SUBSYSTEM_GRAPH_METRICS_TARGET,
    SUBSYSTEM_AGREEMENT_TARGET,
    TEST_PROFILE_TARGET,
    TEST_GRAPH_METRICS_TARGET,
    SYMBOL_GRAPH_METRICS_TARGET,
    BEHAVIORAL_COVERAGE_TARGET,
    ENTRYPOINTS_TARGET,
    EXTERNAL_DEPS_TARGET,
    PROFILES_TARGET,
    FUNCTION_AST_FEATURES_TARGET,
    # Export
    EXPORT_JSONL_TARGET,
    EXPORT_PARQUET_TARGET,
)


def build_target_graph(targets: tuple[OutputTarget, ...] | None = None) -> TargetGraph:
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
    active_targets = targets or ALL_TARGETS
    graph = TargetGraph()
    for target in active_targets:
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


def derive_schemas_from_targets(
    targets: tuple[OutputTarget, ...],
) -> dict[str, TableSchema]:
    """Derive TABLE_SCHEMAS from target contracts.

    This function extracts TableSchema definitions from OutputTarget
    contracts, enabling schema derivation from the build system.

    During migration, this is used alongside static TABLE_SCHEMAS.
    Eventually, TABLE_SCHEMAS will be fully derived from targets.

    Parameters
    ----------
    targets
        Tuple of OutputTargets to extract schemas from.

    Returns
    -------
    dict[str, TableSchema]
        Mapping of table key to TableSchema.
    """
    schemas: dict[str, TableSchema] = {}

    for target in targets:
        for table in target.contract.tables:
            key = table.fq_name
            if key in schemas:
                log.warning(
                    "Duplicate schema for %s from targets %s",
                    key,
                    target.name,
                )
            schemas[key] = table

    return schemas


def get_all_target_table_keys(targets: tuple[OutputTarget, ...] | None = None) -> frozenset[str]:
    """Return all table keys declared by any target.

    This includes both contract tables and legacy tables.

    Returns
    -------
    frozenset[str]
        Set of all table keys.
    """
    keys: set[str] = set()
    for target in targets or ALL_TARGETS:
        # Add contract tables
        keys.update(target.contract.table_keys)
        # Add legacy tables
        keys.update(target.tables)
    return frozenset(keys)


def get_target_by_table(
    table_key: str, *, targets: tuple[OutputTarget, ...] | None = None
) -> OutputTarget | None:
    """Find the target that produces a given table.

    Parameters
    ----------
    table_key
        Fully-qualified table name.
    targets
        Optional set of targets to search (defaults to ALL_TARGETS).

    Returns
    -------
    OutputTarget | None
        Target that produces this table, or None.
    """
    for target in targets or ALL_TARGETS:
        if table_key in target.table_keys:
            return target
    return None


__all__ = [
    "ALL_TARGETS",
    "AST_TARGET",
    "BEHAVIORAL_COVERAGE_TARGET",
    "CALL_GRAPH_TARGET",
    "CFG_DFG_METRICS_TARGET",
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
    "EXPORT_JSONL_TARGET",
    "EXPORT_PARQUET_TARGET",
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
    "SUBSYSTEM_AGREEMENT_TARGET",
    "SUBSYSTEM_GRAPH_METRICS_TARGET",
    "SYMBOL_GRAPH_METRICS_TARGET",
    "SYMBOL_USES_TARGET",
    "TESTS_INGEST_TARGET",
    "TEST_GRAPH_METRICS_TARGET",
    "TEST_PROFILE_TARGET",
    "TYPING_TARGET",
    "build_target_graph",
    "derive_schemas_from_targets",
    "get_all_target_table_keys",
    "get_target_by_table",
    "get_target_graph",
]
