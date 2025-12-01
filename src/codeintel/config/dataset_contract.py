"""Single source of truth for dataset contracts (tables + docs views)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast

from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
from codeintel.storage import rows as row_models
from codeintel.storage.views import DERIVED_DOCS_VIEWS

RowToTuple = Callable[[Mapping[str, object]], tuple[object, ...]]
RowDictType = type[object]


@dataclass(frozen=True)
class RowBinding:
    """Connect a DuckDB table key to a TypedDict row model and serializer."""

    row_type: RowDictType
    to_tuple: RowToTuple


@dataclass(frozen=True)
class DatasetContract:
    """Metadata describing a logical dataset backed by a DuckDB table or view.

    Attributes
    ----------
    table_key
        Fully qualified DuckDB identifier, e.g. "analytics.function_profile".
    name
        Logical dataset name, e.g. "function_profile".
    schema
        Statically defined TableSchema when the dataset is backed by a table;
        None when the dataset is a view.
    row_binding
        Optional binding to a TypedDict row model and serializer.
    json_schema_id
        Optional JSON Schema identifier (without .json) used for export validation.
    jsonl_filename
        Default filename for JSONL exports (may be None when not exported).
    parquet_filename
        Default filename for Parquet exports (may be None when not exported).
    is_view
        True when this dataset is a docs.* view instead of a base table.
    owner_package
        Optional package ownership derived from schema prefix (core, analytics, graphs, qa, docs).
    tags
        Classification tags applied to the dataset (e.g., base_table, docs_view, read_only).
    description
        Optional human-readable description of the dataset's purpose.
    family
        Optional dataset family inferred from the schema prefix (e.g., "core",
        "analytics", "docs").
    owner
        Optional team or individual owner for stewardship and escalation.
    freshness_sla
        Optional freshness expectation (e.g., "daily", "hourly").
    retention_policy
        Optional retention policy descriptor (e.g., "90d").
    stable_id
        Optional stable identifier for comparing contracts across versions.
    schema_version
        Optional schema version string for change tracking.
    upstream_dependencies
        Optional tuple of other dataset names this dataset depends on.
    """

    table_key: str
    name: str
    schema: TableSchema | None
    row_binding: RowBinding | None = None
    json_schema_id: str | None = None
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    is_view: bool = False
    owner_package: Literal["core", "analytics", "graphs", "qa", "docs"] | None = None
    tags: frozenset[str] = frozenset()
    description: str | None = None
    family: str | None = None
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    stable_id: str | None = None
    schema_version: str | None = None
    upstream_dependencies: tuple[str, ...] = ()
    validation_profile: Literal["strict", "lenient"] = "strict"

    def has_row_binding(self) -> bool:
        """
        Return True when this dataset has a TypedDict row binding.

        Returns
        -------
        bool
            True when a row binding is configured.
        """
        return self.row_binding is not None

    def require_row_binding(self) -> RowBinding:
        """
        Return the row binding or raise a clear error if missing.

        Returns
        -------
        RowBinding
            Configured row binding for this dataset.

        Raises
        ------
        KeyError
            If no row binding is configured for this dataset.
        """
        if self.row_binding is None:
            message = f"Dataset {self.name} ({self.table_key}) has no row binding"
            raise KeyError(message)
        return self.row_binding

    def capabilities(self) -> dict[str, bool]:
        """
        Return capability flags derived from attached metadata.

        Returns
        -------
        dict[str, bool]
            Flags for validation and export support.
        """
        docs_view = self.table_key.startswith("docs.")
        read_only = self.is_view or docs_view or "read_only" in self.tags
        return {
            "can_validate": self.json_schema_id is not None,
            "can_export_jsonl": self.jsonl_filename is not None,
            "can_export_parquet": self.parquet_filename is not None,
            "has_row_binding": self.row_binding is not None,
            "is_view": self.is_view,
            "docs_view": docs_view,
            "read_only": read_only,
        }


def _row_binding(
    row_type: RowDictType,
    to_tuple: Callable[..., tuple[object, ...]],
) -> RowBinding:
    return RowBinding(row_type=row_type, to_tuple=cast("RowToTuple", to_tuple))


def _metadata_for_name(name: str) -> dict[str, object]:
    return {
        "description": _DESCRIPTION_BY_DATASET_NAME.get(name),
        "owner": _OWNER_BY_DATASET_NAME.get(name),
        "freshness_sla": _FRESHNESS_BY_DATASET_NAME.get(name),
        "retention_policy": _RETENTION_BY_DATASET_NAME.get(name),
        "upstream_dependencies": _DEPENDENCIES_BY_DATASET_NAME.get(name, ()),
        "stable_id": _STABLE_ID_BY_DATASET_NAME.get(name, name),
        "schema_version": _SCHEMA_VERSION_BY_DATASET_NAME.get(name, "1"),
        "validation_profile": _VALIDATION_PROFILE_BY_DATASET_NAME.get(name, "strict"),
    }


ROW_BINDINGS_BY_TABLE_KEY: Final[dict[str, RowBinding]] = {
    "analytics.coverage_lines": _row_binding(
        row_type=row_models.CoverageLineRow,
        to_tuple=row_models.coverage_line_to_tuple,
    ),
    "analytics.config_values": _row_binding(
        row_type=row_models.ConfigValueRow,
        to_tuple=row_models.config_value_to_tuple,
    ),
    "analytics.typedness": _row_binding(
        row_type=row_models.TypednessRow,
        to_tuple=row_models.typedness_row_to_tuple,
    ),
    "analytics.static_diagnostics": _row_binding(
        row_type=row_models.StaticDiagnosticRow,
        to_tuple=row_models.static_diagnostic_to_tuple,
    ),
    "analytics.function_validation": _row_binding(
        row_type=row_models.FunctionValidationRow,
        to_tuple=row_models.function_validation_row_to_tuple,
    ),
    "analytics.function_metrics": _row_binding(
        row_type=row_models.FunctionMetricsRow,
        to_tuple=row_models.function_metrics_row_to_tuple,
    ),
    "analytics.function_types": _row_binding(
        row_type=row_models.FunctionTypesRow,
        to_tuple=row_models.function_types_row_to_tuple,
    ),
    "analytics.graph_validation": _row_binding(
        row_type=row_models.GraphValidationRow,
        to_tuple=row_models.graph_validation_row_to_tuple,
    ),
    "analytics.hotspots": _row_binding(
        row_type=row_models.HotspotRow,
        to_tuple=row_models.hotspot_row_to_tuple,
    ),
    "analytics.test_catalog": _row_binding(
        row_type=row_models.TestCatalogRowModel,
        to_tuple=row_models.serialize_test_catalog_row,
    ),
    "analytics.test_coverage_edges": _row_binding(
        row_type=row_models.TestCoverageEdgeRow,
        to_tuple=row_models.serialize_test_coverage_edge,
    ),
    "core.docstrings": _row_binding(
        row_type=row_models.DocstringRow,
        to_tuple=row_models.docstring_row_to_tuple,
    ),
    "core.goids": _row_binding(
        row_type=row_models.GoidRow,
        to_tuple=row_models.goid_to_tuple,
    ),
    "core.goid_crosswalk": _row_binding(
        row_type=row_models.GoidCrosswalkRow,
        to_tuple=row_models.goid_crosswalk_to_tuple,
    ),
    "analytics.function_profile": _row_binding(
        row_type=row_models.FunctionProfileRowModel,
        to_tuple=row_models.function_profile_row_to_tuple,
    ),
    "analytics.function_ast_features": _row_binding(
        row_type=row_models.FunctionAstFeaturesRow,
        to_tuple=row_models.function_ast_features_row_to_tuple,
    ),
    "analytics.file_profile": _row_binding(
        row_type=row_models.FileProfileRowModel,
        to_tuple=row_models.file_profile_row_to_tuple,
    ),
    "analytics.module_profile": _row_binding(
        row_type=row_models.ModuleProfileRowModel,
        to_tuple=row_models.module_profile_row_to_tuple,
    ),
    "graph.call_graph_nodes": _row_binding(
        row_type=row_models.CallGraphNodeRow,
        to_tuple=row_models.call_graph_node_to_tuple,
    ),
    "graph.call_graph_edges": _row_binding(
        row_type=row_models.CallGraphEdgeRow,
        to_tuple=row_models.call_graph_edge_to_tuple,
    ),
    "graph.import_graph_edges": _row_binding(
        row_type=row_models.ImportEdgeRow,
        to_tuple=row_models.import_edge_to_tuple,
    ),
    "graph.import_modules": _row_binding(
        row_type=row_models.ImportModuleRow,
        to_tuple=row_models.import_module_to_tuple,
    ),
    "graph.cfg_blocks": _row_binding(
        row_type=row_models.CFGBlockRow,
        to_tuple=row_models.cfg_block_to_tuple,
    ),
    "graph.cfg_edges": _row_binding(
        row_type=row_models.CFGEdgeRow,
        to_tuple=row_models.cfg_edge_to_tuple,
    ),
    "graph.dfg_edges": _row_binding(
        row_type=row_models.DFGEdgeRow,
        to_tuple=row_models.dfg_edge_to_tuple,
    ),
    "graph.symbol_use_edges": _row_binding(
        row_type=row_models.SymbolUseRow,
        to_tuple=row_models.symbol_use_to_tuple,
    ),
    "analytics.graph_metrics_functions": _row_binding(
        row_type=row_models.GraphMetricsFunctionsRow,
        to_tuple=row_models.graph_metrics_functions_row_to_tuple,
    ),
    "analytics.graph_metrics_modules": _row_binding(
        row_type=row_models.GraphMetricsModulesRow,
        to_tuple=row_models.graph_metrics_modules_row_to_tuple,
    ),
    "analytics.graph_metrics_functions_ext": _row_binding(
        row_type=row_models.GraphMetricsFunctionsExtRow,
        to_tuple=row_models.graph_metrics_functions_ext_row_to_tuple,
    ),
    "analytics.graph_metrics_modules_ext": _row_binding(
        row_type=row_models.GraphMetricsModulesExtRow,
        to_tuple=row_models.graph_metrics_modules_ext_row_to_tuple,
    ),
    "analytics.test_profile": _row_binding(
        row_type=row_models.ProfileRowModel,
        to_tuple=row_models.serialize_test_profile_row,
    ),
    "analytics.behavioral_coverage": _row_binding(
        row_type=row_models.BehavioralCoverageRowModel,
        to_tuple=row_models.behavioral_coverage_row_to_tuple,
    ),
    "analytics.subsystem_profile_cache": _row_binding(
        row_type=row_models.SubsystemProfileCacheRow,
        to_tuple=row_models.subsystem_profile_cache_to_tuple,
    ),
    "analytics.subsystem_coverage_cache": _row_binding(
        row_type=row_models.SubsystemCoverageCacheRow,
        to_tuple=row_models.subsystem_coverage_cache_to_tuple,
    ),
    "docs.v_subsystem_profile": _row_binding(
        row_type=row_models.SubsystemProfileCacheRow,
        to_tuple=row_models.subsystem_profile_cache_to_tuple,
    ),
    "docs.v_subsystem_coverage": _row_binding(
        row_type=row_models.SubsystemCoverageCacheRow,
        to_tuple=row_models.subsystem_coverage_cache_to_tuple,
    ),
}

# Dataset-level JSON Schema metadata.
# Keys: dataset logical names (Dataset.name).
# Values: JSON Schema identifiers (filenames without .json) under
# src/codeintel/config/schemas/export/.
_JSON_SCHEMA_BY_DATASET_NAME: Final[dict[str, str]] = {
    # Profiles
    "function_profile": "function_profile",
    "file_profile": "file_profile",
    "module_profile": "module_profile",
    # Graph edges
    "call_graph_edges": "call_graph_edges",
    "symbol_use_edges": "symbol_use_edges",
    "test_coverage_edges": "test_coverage_edges",
    # Tests
    "test_profile": "test_profile",
    "behavioral_coverage": "behavioral_coverage",
    "v_subsystem_profile": "v_subsystem_profile",
    "v_subsystem_coverage": "v_subsystem_coverage",
    "subsystem_profile_cache": "subsystem_profile_cache",
    "subsystem_coverage_cache": "subsystem_coverage_cache",
    # Data models
    "data_model_fields": "data_model_fields",
    "data_model_relationships": "data_model_relationships",
}

_DESCRIPTION_BY_DATASET_NAME: Final[dict[str, str]] = {
    "function_profile": "Function-level profile combining metrics, risk, and topology.",
    "file_profile": "File-level profile with coverage, hotspots, and ownership signals.",
    "module_profile": "Module-level profile aggregating functions, imports, and risk.",
    "v_subsystem_profile": "Subsystem-level profile combining risk, connectivity, and metadata.",
    "v_subsystem_coverage": "Subsystem coverage rollup derived from test profiles.",
    "subsystem_profile_cache": "Materialized subsystem profile cache for docs views.",
    "subsystem_coverage_cache": "Materialized subsystem coverage cache for docs views.",
    "call_graph_edges": "Directed call graph edges across the codebase.",
    "symbol_use_edges": "Symbol use edges linking definitions to references.",
    "test_coverage_edges": "Test-to-target coverage edges for tracing impacts.",
    "test_profile": "Test-level profile including outcomes and runtime metadata.",
    "behavioral_coverage": "Behavioral coverage findings captured during scenario runs.",
    "data_model_fields": "Normalized data model field definitions for analytics export.",
    "data_model_relationships": "Normalized data model relationships for analytics export.",
}

_OWNER_BY_DATASET_NAME: Final[dict[str, str]] = {
    "function_profile": "analytics",
    "file_profile": "analytics",
    "module_profile": "analytics",
    "call_graph_edges": "graphs",
    "symbol_use_edges": "graphs",
    "test_coverage_edges": "analytics",
    "test_profile": "qa",
    "behavioral_coverage": "qa",
    "v_subsystem_profile": "docs",
    "v_subsystem_coverage": "docs",
    "subsystem_profile_cache": "analytics",
    "subsystem_coverage_cache": "analytics",
    "data_model_fields": "analytics",
    "data_model_relationships": "analytics",
}

_FRESHNESS_BY_DATASET_NAME: Final[dict[str, str]] = {
    "function_profile": "daily",
    "file_profile": "daily",
    "module_profile": "daily",
    "call_graph_edges": "daily",
    "symbol_use_edges": "daily",
    "test_coverage_edges": "daily",
    "test_profile": "daily",
    "behavioral_coverage": "daily",
    "v_subsystem_profile": "daily",
    "v_subsystem_coverage": "daily",
    "subsystem_profile_cache": "daily",
    "subsystem_coverage_cache": "daily",
    "data_model_fields": "daily",
    "data_model_relationships": "daily",
}

_RETENTION_BY_DATASET_NAME: Final[dict[str, str]] = {
    "function_profile": "90d",
    "file_profile": "90d",
    "module_profile": "90d",
    "call_graph_edges": "90d",
    "symbol_use_edges": "90d",
    "test_coverage_edges": "90d",
    "test_profile": "90d",
    "behavioral_coverage": "90d",
    "v_subsystem_profile": "90d",
    "v_subsystem_coverage": "90d",
    "subsystem_profile_cache": "90d",
    "subsystem_coverage_cache": "90d",
    "data_model_fields": "90d",
    "data_model_relationships": "90d",
}

_STABLE_ID_BY_DATASET_NAME: Final[dict[str, str]] = {}
_SCHEMA_VERSION_BY_DATASET_NAME: Final[dict[str, str]] = {}
_VALIDATION_PROFILE_BY_DATASET_NAME: Final[dict[str, Literal["strict", "lenient"]]] = {}

_DEPENDENCIES_BY_DATASET_NAME: Final[dict[str, tuple[str, ...]]] = {
    "function_profile": ("call_graph_edges", "symbol_use_edges"),
    "file_profile": ("call_graph_edges",),
    "module_profile": ("call_graph_edges", "symbol_use_edges"),
    "test_profile": ("test_coverage_edges",),
    "behavioral_coverage": ("test_profile",),
    "data_model_relationships": ("data_model_fields",),
}

_DEFAULT_JSONL_FILENAMES: Final[dict[str, str]] = {
    # GOIDs / crosswalk
    "core.goids": "goids.jsonl",
    "core.goid_crosswalk": "goid_crosswalk.jsonl",
    # Call graph
    "graph.call_graph_nodes": "call_graph_nodes.jsonl",
    "graph.call_graph_edges": "call_graph_edges.jsonl",
    # CFG / DFG
    "graph.cfg_blocks": "cfg_blocks.jsonl",
    "graph.cfg_edges": "cfg_edges.jsonl",
    "graph.dfg_edges": "dfg_edges.jsonl",
    # Import / symbol uses
    "graph.import_graph_edges": "import_graph_edges.jsonl",
    "graph.symbol_use_edges": "symbol_use_edges.jsonl",
    # AST / CST
    "core.ast_nodes": "ast_nodes.jsonl",
    "core.ast_metrics": "ast_metrics.jsonl",
    "core.cst_nodes": "cst_nodes.jsonl",
    "core.docstrings": "docstrings.jsonl",
    # Modules / config / diagnostics
    "core.modules": "modules.jsonl",
    "analytics.config_values": "config_values.jsonl",
    "analytics.data_models": "data_models.jsonl",
    "analytics.data_model_fields": "data_model_fields.jsonl",
    "analytics.data_model_relationships": "data_model_relationships.jsonl",
    "analytics.data_model_usage": "data_model_usage.jsonl",
    "analytics.config_data_flow": "config_data_flow.jsonl",
    "analytics.static_diagnostics": "static_diagnostics.jsonl",
    # AST analytics / typing
    "analytics.hotspots": "hotspots.jsonl",
    "analytics.typedness": "typedness.jsonl",
    # Function analytics
    "analytics.function_metrics": "function_metrics.jsonl",
    "analytics.function_types": "function_types.jsonl",
    "analytics.function_effects": "function_effects.jsonl",
    "analytics.function_contracts": "function_contracts.jsonl",
    "analytics.function_ast_features": "function_ast_features.jsonl",
    "analytics.semantic_roles_functions": "semantic_roles_functions.jsonl",
    "analytics.semantic_roles_modules": "semantic_roles_modules.jsonl",
    # Coverage + tests
    "analytics.coverage_lines": "coverage_lines.jsonl",
    "analytics.coverage_functions": "coverage_functions.jsonl",
    "analytics.test_catalog": "test_catalog.jsonl",
    "analytics.test_coverage_edges": "test_coverage_edges.jsonl",
    "analytics.entrypoints": "entrypoints.jsonl",
    "analytics.entrypoint_tests": "entrypoint_tests.jsonl",
    "analytics.external_dependencies": "external_dependencies.jsonl",
    "analytics.external_dependency_calls": "external_dependency_calls.jsonl",
    "analytics.graph_validation": "graph_validation.jsonl",
    "analytics.function_validation": "function_validation.jsonl",
    # Risk factors
    "analytics.goid_risk_factors": "goid_risk_factors.jsonl",
    "analytics.function_profile": "function_profile.jsonl",
    "analytics.function_history": "function_history.jsonl",
    "analytics.history_timeseries": "history_timeseries.jsonl",
    "analytics.file_profile": "file_profile.jsonl",
    "analytics.module_profile": "module_profile.jsonl",
    "analytics.graph_metrics_functions": "graph_metrics_functions.jsonl",
    "analytics.graph_metrics_functions_ext": "graph_metrics_functions_ext.jsonl",
    "analytics.graph_metrics_modules": "graph_metrics_modules.jsonl",
    "analytics.graph_metrics_modules_ext": "graph_metrics_modules_ext.jsonl",
    "analytics.subsystem_graph_metrics": "subsystem_graph_metrics.jsonl",
    "analytics.symbol_graph_metrics_modules": "symbol_graph_metrics_modules.jsonl",
    "analytics.symbol_graph_metrics_functions": "symbol_graph_metrics_functions.jsonl",
    "analytics.config_graph_metrics_keys": "config_graph_metrics_keys.jsonl",
    "analytics.config_graph_metrics_modules": "config_graph_metrics_modules.jsonl",
    "analytics.config_projection_key_edges": "config_projection_key_edges.jsonl",
    "analytics.config_projection_module_edges": "config_projection_module_edges.jsonl",
    "analytics.subsystem_agreement": "subsystem_agreement.jsonl",
    "analytics.graph_stats": "graph_stats.jsonl",
    "analytics.test_graph_metrics_tests": "test_graph_metrics_tests.jsonl",
    "analytics.test_graph_metrics_functions": "test_graph_metrics_functions.jsonl",
    "analytics.test_profile": "test_profile.jsonl",
    "analytics.behavioral_coverage": "behavioral_coverage.jsonl",
    "analytics.cfg_block_metrics": "cfg_block_metrics.jsonl",
    "analytics.cfg_function_metrics": "cfg_function_metrics.jsonl",
    "analytics.dfg_block_metrics": "dfg_block_metrics.jsonl",
    "analytics.dfg_function_metrics": "dfg_function_metrics.jsonl",
    "analytics.subsystems": "subsystems.jsonl",
    "analytics.subsystem_modules": "subsystem_modules.jsonl",
    # Docs views
    "docs.v_validation_summary": "validation_summary.jsonl",
}


_DEFAULT_PARQUET_FILENAMES: Final[dict[str, str]] = {
    # GOIDs / crosswalk
    "core.goids": "goids.parquet",
    "core.goid_crosswalk": "goid_crosswalk.parquet",
    # Call graph
    "graph.call_graph_nodes": "call_graph_nodes.parquet",
    "graph.call_graph_edges": "call_graph_edges.parquet",
    # CFG / DFG
    "graph.cfg_blocks": "cfg_blocks.parquet",
    "graph.cfg_edges": "cfg_edges.parquet",
    "graph.dfg_edges": "dfg_edges.parquet",
    # Import / symbol uses
    "graph.import_graph_edges": "import_graph_edges.parquet",
    "graph.symbol_use_edges": "symbol_use_edges.parquet",
    # AST / CST
    "core.ast_nodes": "ast_nodes.parquet",
    "core.ast_metrics": "ast_metrics.parquet",
    "core.cst_nodes": "cst_nodes.parquet",
    "core.docstrings": "docstrings.parquet",
    # Modules / config / diagnostics
    "core.modules": "modules.parquet",
    "analytics.config_values": "config_values.parquet",
    "analytics.data_models": "data_models.parquet",
    "analytics.data_model_fields": "data_model_fields.parquet",
    "analytics.data_model_relationships": "data_model_relationships.parquet",
    "analytics.data_model_usage": "data_model_usage.parquet",
    "analytics.config_data_flow": "config_data_flow.parquet",
    "analytics.static_diagnostics": "static_diagnostics.parquet",
    # AST analytics / typing
    "analytics.hotspots": "hotspots.parquet",
    "analytics.typedness": "typedness.parquet",
    # Function analytics
    "analytics.function_metrics": "function_metrics.parquet",
    "analytics.function_types": "function_types.parquet",
    "analytics.function_effects": "function_effects.parquet",
    "analytics.function_contracts": "function_contracts.parquet",
    "analytics.function_ast_features": "function_ast_features.parquet",
    "analytics.semantic_roles_functions": "semantic_roles_functions.parquet",
    "analytics.semantic_roles_modules": "semantic_roles_modules.parquet",
    # Coverage + tests
    "analytics.coverage_lines": "coverage_lines.parquet",
    "analytics.coverage_functions": "coverage_functions.parquet",
    "analytics.test_catalog": "test_catalog.parquet",
    "analytics.test_coverage_edges": "test_coverage_edges.parquet",
    "analytics.entrypoints": "entrypoints.parquet",
    "analytics.entrypoint_tests": "entrypoint_tests.parquet",
    "analytics.external_dependencies": "external_dependencies.parquet",
    "analytics.external_dependency_calls": "external_dependency_calls.parquet",
    "analytics.graph_validation": "graph_validation.parquet",
    "analytics.function_validation": "function_validation.parquet",
    # Risk factors
    "analytics.goid_risk_factors": "goid_risk_factors.parquet",
    "analytics.function_profile": "function_profile.parquet",
    "analytics.function_history": "function_history.parquet",
    "analytics.history_timeseries": "history_timeseries.parquet",
    "analytics.file_profile": "file_profile.parquet",
    "analytics.module_profile": "module_profile.parquet",
    "analytics.graph_metrics_functions": "graph_metrics_functions.parquet",
    "analytics.graph_metrics_functions_ext": "graph_metrics_functions_ext.parquet",
    "analytics.graph_metrics_modules": "graph_metrics_modules.parquet",
    "analytics.graph_metrics_modules_ext": "graph_metrics_modules_ext.parquet",
    "analytics.subsystem_graph_metrics": "subsystem_graph_metrics.parquet",
    "analytics.symbol_graph_metrics_modules": "symbol_graph_metrics_modules.parquet",
    "analytics.symbol_graph_metrics_functions": "symbol_graph_metrics_functions.parquet",
    "analytics.config_graph_metrics_keys": "config_graph_metrics_keys.parquet",
    "analytics.config_graph_metrics_modules": "config_graph_metrics_modules.parquet",
    "analytics.config_projection_key_edges": "config_projection_key_edges.parquet",
    "analytics.config_projection_module_edges": "config_projection_module_edges.parquet",
    "analytics.subsystem_agreement": "subsystem_agreement.parquet",
    "analytics.graph_stats": "graph_stats.parquet",
    "analytics.test_graph_metrics_tests": "test_graph_metrics_tests.parquet",
    "analytics.test_graph_metrics_functions": "test_graph_metrics_functions.parquet",
    "analytics.test_profile": "test_profile.parquet",
    "analytics.behavioral_coverage": "behavioral_coverage.parquet",
    "analytics.cfg_block_metrics": "cfg_block_metrics.parquet",
    "analytics.cfg_function_metrics": "cfg_function_metrics.parquet",
    "analytics.dfg_block_metrics": "dfg_block_metrics.parquet",
    "analytics.dfg_function_metrics": "dfg_function_metrics.parquet",
    "analytics.subsystems": "subsystems.parquet",
    "analytics.subsystem_modules": "subsystem_modules.parquet",
    # Docs views
    "docs.v_validation_summary": "validation_summary.parquet",
}


def _owner_package_for_prefix(
    prefix: str,
) -> Literal["core", "analytics", "graphs", "qa", "docs"] | None:
    if prefix == "core":
        return "core"
    if prefix == "analytics":
        return "analytics"
    if prefix in {"graph", "cfg"}:
        return "graphs"
    if prefix == "docs":
        return "docs"
    if prefix == "qa":
        return "qa"
    return None


def _build_contracts() -> dict[str, DatasetContract]:
    contracts: dict[str, DatasetContract] = {}

    for table_key, schema in TABLE_SCHEMAS.items():
        if table_key.startswith("tmp_"):
            continue
        schema_prefix, name = table_key.split(".", maxsplit=1)
        meta = _metadata_for_name(name)
        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
        json_schema_id = _JSON_SCHEMA_BY_DATASET_NAME.get(name)
        jsonl_filename = _DEFAULT_JSONL_FILENAMES.get(table_key)
        parquet_filename = _DEFAULT_PARQUET_FILENAMES.get(table_key)
        owner_package = _owner_package_for_prefix(schema_prefix)
        family = schema_prefix

        contracts[name] = DatasetContract(
            name=name,
            table_key=table_key,
            schema=schema,
            row_binding=row_binding,
            json_schema_id=json_schema_id,
            jsonl_filename=jsonl_filename,
            parquet_filename=parquet_filename,
            is_view=False,
            owner_package=owner_package,
            tags=frozenset({"base_table"}),
            description=cast("str | None", meta["description"]),
            family=family,
            owner=cast("str | None", meta["owner"]),
            freshness_sla=cast("str | None", meta["freshness_sla"]),
            retention_policy=cast("str | None", meta["retention_policy"]),
            stable_id=cast("str | None", meta["stable_id"]),
            schema_version=cast("str | None", meta["schema_version"]),
            upstream_dependencies=cast("tuple[str, ...]", meta["upstream_dependencies"]),
            validation_profile=cast("Literal['strict', 'lenient']", meta["validation_profile"]),
        )

    for view_key in DERIVED_DOCS_VIEWS:
        schema_prefix, view_name = view_key.split(".", maxsplit=1)
        meta = _metadata_for_name(view_name)
        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(view_key)
        json_schema_id = _JSON_SCHEMA_BY_DATASET_NAME.get(view_name)
        jsonl_filename = _DEFAULT_JSONL_FILENAMES.get(view_key)
        parquet_filename = _DEFAULT_PARQUET_FILENAMES.get(view_key)
        owner_package = _owner_package_for_prefix(schema_prefix)
        family = schema_prefix

        contracts[view_name] = DatasetContract(
            name=view_name,
            table_key=view_key,
            schema=None,
            row_binding=row_binding,
            json_schema_id=json_schema_id,
            jsonl_filename=jsonl_filename,
            parquet_filename=parquet_filename,
            is_view=True,
            owner_package=owner_package,
            tags=frozenset({"docs_view", "read_only"}),
            description=cast("str | None", meta["description"]),
            family=family,
            owner=cast("str | None", meta["owner"]),
            freshness_sla=cast("str | None", meta["freshness_sla"]),
            retention_policy=cast("str | None", meta["retention_policy"]),
            stable_id=cast("str | None", meta["stable_id"]),
            schema_version=cast("str | None", meta["schema_version"]),
            upstream_dependencies=cast("tuple[str, ...]", meta["upstream_dependencies"]),
            validation_profile=cast("Literal['strict', 'lenient']", meta["validation_profile"]),
        )

    return contracts


DATASET_CONTRACTS: Final[dict[str, DatasetContract]] = _build_contracts()
DATASET_CONTRACTS_BY_TABLE_KEY: Final[dict[str, DatasetContract]] = {
    contract.table_key: contract for contract in DATASET_CONTRACTS.values()
}

JSON_SCHEMA_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.json_schema_id
    for name, contract in DATASET_CONTRACTS.items()
    if contract.json_schema_id is not None
}

DEFAULT_JSONL_FILENAMES: Final[dict[str, str]] = {
    contract.table_key: contract.jsonl_filename
    for contract in DATASET_CONTRACTS.values()
    if contract.jsonl_filename is not None
}

DEFAULT_PARQUET_FILENAMES: Final[dict[str, str]] = {
    contract.table_key: contract.parquet_filename
    for contract in DATASET_CONTRACTS.values()
    if contract.parquet_filename is not None
}

DEPENDENCIES_BY_DATASET_NAME: Final[dict[str, tuple[str, ...]]] = {
    name: contract.upstream_dependencies
    for name, contract in DATASET_CONTRACTS.items()
    if contract.upstream_dependencies
}

DESCRIPTION_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.description
    for name, contract in DATASET_CONTRACTS.items()
    if contract.description is not None
}

OWNER_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.owner for name, contract in DATASET_CONTRACTS.items() if contract.owner
}

FRESHNESS_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.freshness_sla
    for name, contract in DATASET_CONTRACTS.items()
    if contract.freshness_sla is not None
}

RETENTION_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.retention_policy
    for name, contract in DATASET_CONTRACTS.items()
    if contract.retention_policy is not None
}

STABLE_ID_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.stable_id for name, contract in DATASET_CONTRACTS.items() if contract.stable_id
}

SCHEMA_VERSION_BY_DATASET_NAME: Final[dict[str, str]] = {
    name: contract.schema_version
    for name, contract in DATASET_CONTRACTS.items()
    if contract.schema_version is not None
}

VALIDATION_PROFILE_BY_DATASET_NAME: Final[dict[str, Literal["strict", "lenient"]]] = {
    name: contract.validation_profile
    for name, contract in DATASET_CONTRACTS.items()
    if contract.validation_profile is not None
}
