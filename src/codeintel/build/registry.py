"""Target registration and graph construction for the build system.

Hamilton-First Architecture
---------------------------
This module now uses Hamilton as the source of truth for target dependencies.
Use `get_target_graph()` to get a TargetGraph with Hamilton-derived dependencies.

The static `*_TARGET` constants remain for `registrations.py` compatibility,
but dependencies are derived from the actual Hamilton DAG, not static
declarations.

Each OutputTarget defines its output tables via an OutputContract with
TableSchema definitions. TABLE_SCHEMAS can be derived from target
contracts using derive_schemas_from_targets().
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.contracts import ArtifactSpec, OutputContract
from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.introspect import target_graph_from_hamilton
from codeintel.build.resources import (
    CPU_INTENSIVE_EXECUTION,
    TOOL_EXECUTION,
    TargetResources,
)
from codeintel.build.schemas.declared_schemas import TABLE_SCHEMAS as _DATASET_TABLE_SCHEMAS
from codeintel.build.targets import OutputTarget

if TYPE_CHECKING:
    from codeintel.build.targets import TargetGraph
    from codeintel.core.schemas.primitives import TableSchema

log = logging.getLogger(__name__)


MODULES_TARGET = OutputTarget(
    name="modules",
    module="ingestion",
    plugin="repo_scan",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["core.modules"],
            _DATASET_TABLE_SCHEMAS["core.file_state"],
            _DATASET_TABLE_SCHEMAS["core.repo_map"],
        )
    ),
    dependencies=(),
    description="Repository module and file index from scanning.",
)

AST_TARGET = OutputTarget(
    name="ast",
    module="ingestion",
    plugin="ast_extract",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["core.ast_nodes"],
            _DATASET_TABLE_SCHEMAS["core.ast_metrics"],
        )
    ),
    dependencies=("modules",),
    resources=TargetResources(tracker=True, modules=True),
    execution=CPU_INTENSIVE_EXECUTION,
    description="Python AST extraction and metrics.",
)

CST_TARGET = OutputTarget(
    name="cst",
    module="ingestion",
    plugin="cst_extract",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["core.cst_nodes"],)),
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
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["analytics.typedness"],
            _DATASET_TABLE_SCHEMAS["analytics.static_diagnostics"],
        )
    ),
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
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.coverage_lines"],)),
    dependencies=("modules",),
    description="Line-level test coverage ingestion.",
)

TESTS_INGEST_TARGET = OutputTarget(
    name="tests_ingest",
    module="ingestion",
    plugin="tests_ingest",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.test_catalog"],)),
    dependencies=("modules",),
    description="Test catalog ingestion from pytest.",
)

DOCSTRINGS_TARGET = OutputTarget(
    name="docstrings",
    module="ingestion",
    plugin="docstrings_ingest",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["core.docstrings"],)),
    dependencies=("ast",),
    description="Docstring extraction and parsing.",
)

CONFIG_INGEST_TARGET = OutputTarget(
    name="config_ingest",
    module="ingestion",
    plugin="config_ingest",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.config_values"],)),
    dependencies=("modules",),
    description="Configuration file parsing and reference tracking.",
)


GOIDS_TARGET = OutputTarget(
    name="goids",
    module="graphs",
    plugin="goid_builder",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["core.goids"],
            _DATASET_TABLE_SCHEMAS["core.goid_crosswalk"],
        )
    ),
    dependencies=("scip", "ast"),
    description="GOID resolution and crosswalk construction.",
)

CALL_GRAPH_TARGET = OutputTarget(
    name="call_graph",
    module="graphs",
    plugin="callgraph",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["graph.call_graph_nodes"],
            _DATASET_TABLE_SCHEMAS["graph.call_graph_edges"],
        )
    ),
    dependencies=("goids", "scip"),
    description="Function call graph construction.",
)

CALL_GRAPH_VIEWS_TARGET = OutputTarget(
    name="call_graph_views",
    module="graphs",
    plugin="",  # Native target, no plugin
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["graph.v_function_call_counts"],
            _DATASET_TABLE_SCHEMAS["graph.v_call_depth_stats"],
        )
    ),
    dependencies=("call_graph",),
    description="Derived views over call graph for analytics.",
)

IMPORT_GRAPH_TARGET = OutputTarget(
    name="import_graph",
    module="graphs",
    plugin="import_graph",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["graph.import_modules"],
            _DATASET_TABLE_SCHEMAS["graph.import_graph_edges"],
        )
    ),
    dependencies=("modules",),
    description="Module import graph construction.",
)

CFG_TARGET = OutputTarget(
    name="cfg",
    module="graphs",
    plugin="cfg_dfg",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["graph.cfg_blocks"],
            _DATASET_TABLE_SCHEMAS["graph.cfg_edges"],
            _DATASET_TABLE_SCHEMAS["graph.dfg_edges"],
        )
    ),
    dependencies=("goids", "ast"),
    description="Control flow graph construction per function.",
)

DFG_TARGET = OutputTarget(
    name="dfg",
    module="graphs",
    plugin="",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["graph.dfg_edges"],)),
    dependencies=("cfg",),
    description="Data flow graph construction per function.",
)

CFG_DFG_METRICS_TARGET = OutputTarget(
    name="cfg_dfg_metrics",
    module="analytics",
    plugin="cfg_dfg_metrics",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["analytics.cfg_function_metrics"],
            _DATASET_TABLE_SCHEMAS["analytics.cfg_block_metrics"],
            _DATASET_TABLE_SCHEMAS["analytics.cfg_function_metrics_ext"],
            _DATASET_TABLE_SCHEMAS["analytics.dfg_function_metrics"],
            _DATASET_TABLE_SCHEMAS["analytics.dfg_block_metrics"],
            _DATASET_TABLE_SCHEMAS["analytics.dfg_function_metrics_ext"],
        )
    ),
    dependencies=("cfg", "dfg"),
    description="Control-flow and data-flow graph metrics per function.",
)

SYMBOL_USES_TARGET = OutputTarget(
    name="symbol_uses",
    module="graphs",
    plugin="symbol_uses",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["graph.symbol_use_edges"],)),
    dependencies=("scip",),
    description="Symbol definition-to-use edge extraction.",
)

GRAPH_VALIDATION_TARGET = OutputTarget(
    name="graph_validation",
    module="graphs",
    plugin="graph_validation",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.graph_validation"],)),
    dependencies=("call_graph", "import_graph", "cfg"),
    description="Graph integrity validation checks.",
)


HOTSPOTS_TARGET = OutputTarget(
    name="hotspots",
    module="analytics",
    plugin="hotspots",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.hotspots"],)),
    dependencies=("modules",),
    description="File hotspot analysis based on churn.",
)

FUNCTION_METRICS_TARGET = OutputTarget(
    name="function_metrics",
    module="analytics",
    plugin="function_metrics",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["analytics.function_metrics"],
            _DATASET_TABLE_SCHEMAS["analytics.function_types"],
        )
    ),
    dependencies=("goids", "ast"),
    description="Function structural metrics and type annotations.",
)

FUNCTION_EFFECTS_TARGET = OutputTarget(
    name="function_effects",
    module="analytics",
    plugin="function_effects",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.function_effects"],)),
    dependencies=("function_metrics",),
    description="Function purity and side-effect analysis.",
)

FUNCTION_CONTRACTS_TARGET = OutputTarget(
    name="function_contracts",
    module="analytics",
    plugin="function_contracts",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.function_contracts"],)),
    dependencies=("function_metrics", "docstrings"),
    description="Inferred function pre/postconditions.",
)

FUNCTION_HISTORY_TARGET = OutputTarget(
    name="function_history",
    module="analytics",
    plugin="function_history",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.function_history"],)),
    dependencies=("goids",),
    description="Function git history and churn metrics.",
)

HISTORY_TIMESERIES_TARGET = OutputTarget(
    name="history_timeseries",
    module="analytics",
    plugin="history_timeseries",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.history_timeseries"],)),
    dependencies=("function_history",),
    description="Historical metrics timeseries for trending.",
)

COVERAGE_FUNCTIONS_TARGET = OutputTarget(
    name="coverage_functions",
    module="analytics",
    plugin="coverage_functions",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.coverage_functions"],)),
    dependencies=("goids", "coverage_ingest"),
    description="Per-function coverage aggregation.",
)

COVERAGE_TEST_EDGES_TARGET = OutputTarget(
    name="coverage_test_edges",
    module="analytics",
    plugin="coverage_test_edges",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.test_coverage_edges"],)),
    dependencies=("coverage_functions", "tests_ingest"),
    description="Test-to-function coverage edges.",
)

DATA_MODELS_TARGET = OutputTarget(
    name="data_models",
    module="analytics",
    plugin="data_models",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.data_models"],)),
    dependencies=("goids", "ast"),
    description="Data model extraction (dataclasses, Pydantic, etc.).",
)

DATA_MODEL_USAGE_TARGET = OutputTarget(
    name="data_model_usage",
    module="analytics",
    plugin="data_model_usage",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.data_model_usage"],)),
    dependencies=("data_models", "call_graph"),
    description="Function-level data model usage tracking.",
)

CONFIG_DATA_FLOW_TARGET = OutputTarget(
    name="config_data_flow",
    module="analytics",
    plugin="config_data_flow",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["analytics.config_data_flow"],
            _DATASET_TABLE_SCHEMAS["analytics.config_graph_metrics_keys"],
            _DATASET_TABLE_SCHEMAS["analytics.config_graph_metrics_modules"],
            _DATASET_TABLE_SCHEMAS["analytics.config_projection_key_edges"],
            _DATASET_TABLE_SCHEMAS["analytics.config_projection_module_edges"],
        )
    ),
    dependencies=("config_ingest", "call_graph"),
    description="Config key usage flow through functions.",
)

RISK_FACTORS_TARGET = OutputTarget(
    name="risk_factors",
    module="analytics",
    plugin="risk_factors",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.goid_risk_factors"],)),
    dependencies=(
        "call_graph",
        "function_metrics",
    ),
    description="Composite risk factors per function.",
)

GRAPH_METRICS_TARGET = OutputTarget(
    name="graph_metrics",
    module="graphs",
    plugin="graph_metrics",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["analytics.graph_metrics_functions"],
            _DATASET_TABLE_SCHEMAS["analytics.graph_metrics_functions_ext"],
            _DATASET_TABLE_SCHEMAS["analytics.graph_metrics_modules"],
            _DATASET_TABLE_SCHEMAS["analytics.graph_metrics_modules_ext"],
        )
    ),
    dependencies=("call_graph", "import_graph"),
    description="Graph topology metrics for functions and modules.",
)

SEMANTIC_ROLES_TARGET = OutputTarget(
    name="semantic_roles",
    module="analytics",
    plugin="semantic_roles",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["analytics.semantic_roles_functions"],
            _DATASET_TABLE_SCHEMAS["analytics.semantic_roles_modules"],
        )
    ),
    dependencies=("function_metrics", "call_graph"),
    description="Semantic role classification (handler, utility, etc.).",
)

SUBSYSTEMS_TARGET = OutputTarget(
    name="subsystems",
    module="analytics",
    plugin="subsystems",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["analytics.subsystems"],
            _DATASET_TABLE_SCHEMAS["analytics.subsystem_modules"],
        )
    ),
    dependencies=("import_graph", "semantic_roles"),
    description="Architectural subsystem inference.",
)

SUBSYSTEM_GRAPH_METRICS_TARGET = OutputTarget(
    name="subsystem_graph_metrics",
    module="analytics",
    plugin="subsystem_graph_metrics",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.subsystem_graph_metrics"],)),
    dependencies=("subsystems", "graph_metrics"),
    description="Graph metrics for subsystems.",
)

SUBSYSTEM_AGREEMENT_TARGET = OutputTarget(
    name="subsystem_agreement",
    module="analytics",
    plugin="subsystem_agreement",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.subsystem_agreement"],)),
    dependencies=("subsystems", "graph_metrics"),
    description="Subsystem vs import community agreement.",
)

TEST_PROFILE_TARGET = OutputTarget(
    name="test_profile",
    module="analytics",
    plugin="test_profile",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.test_profile"],)),
    dependencies=("coverage_test_edges", "tests_ingest"),
    description="Per-test profile with coverage and characteristics.",
)

TEST_GRAPH_METRICS_TARGET = OutputTarget(
    name="test_graph_metrics",
    module="analytics",
    plugin="test_graph_metrics",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["analytics.test_graph_metrics_tests"],
            _DATASET_TABLE_SCHEMAS["analytics.test_graph_metrics_functions"],
        )
    ),
    dependencies=("coverage_test_edges",),
    description="Graph metrics from test-function bipartite graph.",
)

SYMBOL_GRAPH_METRICS_TARGET = OutputTarget(
    name="symbol_graph_metrics",
    module="analytics",
    plugin="symbol_graph_metrics",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["analytics.symbol_graph_metrics_modules"],
            _DATASET_TABLE_SCHEMAS["analytics.symbol_graph_metrics_functions"],
        )
    ),
    dependencies=("symbol_uses",),
    description="Graph metrics from symbol usage patterns.",
)

BEHAVIORAL_COVERAGE_TARGET = OutputTarget(
    name="behavioral_coverage",
    module="analytics",
    plugin="behavioral_coverage",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.behavioral_coverage"],)),
    dependencies=("test_profile",),
    description="Behavioral coverage tagging from test patterns.",
)

ENTRYPOINTS_TARGET = OutputTarget(
    name="entrypoints",
    module="analytics",
    plugin="entrypoints",
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.entrypoints"],)),
    dependencies=("goids", "semantic_roles", "test_profile"),
    description="External entrypoint detection (HTTP, CLI, etc.).",
)

EXTERNAL_DEPS_TARGET = OutputTarget(
    name="external_deps",
    module="analytics",
    plugin="external_deps",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["analytics.external_dependencies"],
            _DATASET_TABLE_SCHEMAS["analytics.external_dependency_calls"],
        )
    ),
    dependencies=("call_graph",),
    description="External library dependency analysis.",
)

PROFILES_TARGET = OutputTarget(
    name="profiles",
    module="analytics",
    plugin="profiles",
    contract=OutputContract(
        tables=(
            _DATASET_TABLE_SCHEMAS["analytics.function_profile"],
            _DATASET_TABLE_SCHEMAS["analytics.file_profile"],
            _DATASET_TABLE_SCHEMAS["analytics.module_profile"],
        )
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
    contract=OutputContract(tables=(_DATASET_TABLE_SCHEMAS["analytics.function_ast_features"],)),
    dependencies=("goids", "ast"),
    description="AST-derived semantic features for functions.",
)


EXPORT_JSONL_TARGET = OutputTarget(
    name="export_jsonl",
    module="export",
    plugin="export_jsonl",
    contract=OutputContract(
        artifacts=(
            ArtifactSpec(
                "jsonl_export",
                "{export_dir}/codeintel.jsonl",
                "JSONL export of analytics datasets",
            ),
        )
    ),
    dependencies=("profiles",),
    description="Export datasets to JSONL format for Document Output.",
)

EXPORT_PARQUET_TARGET = OutputTarget(
    name="export_parquet",
    module="export",
    plugin="export_parquet",
    contract=OutputContract(
        artifacts=(
            ArtifactSpec(
                "parquet_export",
                "{export_dir}/codeintel.parquet",
                "Parquet export of analytics datasets",
            ),
        )
    ),
    dependencies=("profiles",),
    description="Export datasets to Parquet format for Document Output.",
)


ALL_TARGETS: tuple[OutputTarget, ...] = (
    MODULES_TARGET,
    AST_TARGET,
    CST_TARGET,
    SCIP_TARGET,
    TYPING_TARGET,
    COVERAGE_INGEST_TARGET,
    TESTS_INGEST_TARGET,
    DOCSTRINGS_TARGET,
    CONFIG_INGEST_TARGET,
    GOIDS_TARGET,
    CALL_GRAPH_TARGET,
    CALL_GRAPH_VIEWS_TARGET,
    IMPORT_GRAPH_TARGET,
    CFG_TARGET,
    DFG_TARGET,
    CFG_DFG_METRICS_TARGET,
    SYMBOL_USES_TARGET,
    GRAPH_VALIDATION_TARGET,
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
    EXPORT_JSONL_TARGET,
    EXPORT_PARQUET_TARGET,
)


@lru_cache(maxsize=1)
def get_target_graph() -> TargetGraph:
    """Get the singleton target graph with Hamilton-derived dependencies.

    This is the canonical way to get a TargetGraph. Dependencies are derived
    from the actual Hamilton DAG, ensuring they match execution reality.

    Returns
    -------
    TargetGraph
        The singleton target graph with Hamilton-derived dependencies.

    Examples
    --------
    >>> graph = get_target_graph()
    >>> "modules" in graph
    True
    >>> deps = graph.dependencies_of("goids")
    >>> "scip" in deps
    True
    """
    runtime = build_driver()
    return target_graph_from_hamilton(runtime)


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

    Returns
    -------
    frozenset[str]
        Set of all table keys from target contracts.
    """
    keys: set[str] = set()
    for target in targets or ALL_TARGETS:
        keys.update(target.table_keys)
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
    "derive_schemas_from_targets",
    "get_all_target_table_keys",
    "get_target_by_table",
    "get_target_graph",
]
