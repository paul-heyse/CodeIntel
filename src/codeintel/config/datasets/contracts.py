"""Dataset contracts, table schemas, and contract building logic.

This module provides:
- TABLE_SCHEMAS: All DuckDB table and view schema definitions
- COMPOSITE_SCHEMAS: Profile composition metadata
- RowBinding: Connects table keys to TypedDict row models
- DatasetContract: Full metadata for logical datasets
- Contract building functions and DATASET_CONTRACTS registry
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal, cast

from codeintel.config.datasets.rows import (
    BehavioralCoverageRowModel,
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    CFGEdgeRow,
    ConfigValueRow,
    CoverageLineRow,
    DFGEdgeRow,
    DocstringRow,
    FileProfileRowModel,
    FunctionAstFeaturesRow,
    FunctionContractsRow,
    FunctionEffectsRow,
    FunctionMetricsRow,
    FunctionProfileRowModel,
    FunctionTypesRow,
    FunctionValidationRow,
    GoidCrosswalkRow,
    GoidRow,
    GraphMetricsFunctionsExtRow,
    GraphMetricsFunctionsRow,
    GraphMetricsModulesExtRow,
    GraphMetricsModulesRow,
    GraphValidationRow,
    HotspotRow,
    ImportEdgeRow,
    ImportModuleRow,
    ModuleProfileRowModel,
    ProfileRowModel,
    StaticDiagnosticRow,
    SubsystemCoverageCacheRow,
    SubsystemProfileCacheRow,
    SymbolUseRow,
    TestCatalogRowModel,
    TestCoverageEdgeRow,
    TypednessRow,
    behavioral_coverage_row_to_tuple,
    call_graph_edge_to_tuple,
    call_graph_node_to_tuple,
    config_value_to_tuple,
    coverage_line_to_tuple,
    docstring_row_to_tuple,
    file_profile_row_to_tuple,
    function_ast_features_row_to_tuple,
    function_contracts_row_to_tuple,
    function_effects_row_to_tuple,
    function_metrics_row_to_tuple,
    function_profile_row_to_tuple,
    function_types_row_to_tuple,
    function_validation_row_to_tuple,
    graph_metrics_functions_ext_row_to_tuple,
    graph_metrics_functions_row_to_tuple,
    graph_metrics_modules_ext_row_to_tuple,
    graph_metrics_modules_row_to_tuple,
    graph_validation_row_to_tuple,
    hotspot_row_to_tuple,
    module_profile_row_to_tuple,
    serialize_test_catalog_row,
    serialize_test_coverage_edge,
    serialize_test_profile_row,
    static_diagnostic_to_tuple,
    subsystem_coverage_cache_to_tuple,
    subsystem_profile_cache_to_tuple,
    typedness_row_to_tuple,
)
from codeintel.config.datasets.schema_provider import composite_schemas, table_schemas
from codeintel.storage.view_names import DERIVED_DOCS_VIEWS

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.config.datasets.primitives import (
        CompositeSchema,
        RowDictType,
        RowToTuple,
        TableSchema,
    )


@dataclass(frozen=True)
class RowBinding:
    """Connect a DuckDB table key to a TypedDict row model and serializer.

    Parameters
    ----------
    row_type
        The TypedDict class defining the row shape.
    to_tuple
        Function to serialize a row dict to a tuple for INSERT.
    """

    row_type: RowDictType
    to_tuple: RowToTuple


@dataclass(frozen=True)
class DatasetContract:
    """Metadata describing a logical dataset backed by a DuckDB table or view.

    Parameters
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
        Optional package ownership derived from schema prefix.
    tags
        Classification tags applied to the dataset.
    description
        Optional human-readable description.
    family
        Optional dataset family inferred from schema prefix.
    owner
        Optional team or individual owner.
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
    validation_profile
        Validation strictness level ("strict" or "lenient").
    composition
        Optional CompositeSchema for profile datasets.
    deprecated
        Whether this dataset is deprecated.
    deprecation_message
        Message explaining deprecation and migration path.
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
    composition: CompositeSchema | None = None
    deprecated: bool = False
    deprecation_message: str | None = None

    def has_row_binding(self) -> bool:
        """Return True when this dataset has a TypedDict row binding.

        Returns
        -------
        bool
            True when a row binding is configured.
        """
        return self.row_binding is not None

    def require_row_binding(self) -> RowBinding:
        """Return the row binding or raise a clear error if missing.

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
        """Return capability flags derived from attached metadata.

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
            "dataset_rows_only": "dataset_rows_only" in self.tags,
        }

    def column_names(self) -> tuple[str, ...]:
        """Return column names in schema definition order.

        Returns
        -------
        tuple[str, ...]
            Ordered column names, or empty tuple for views without schema.
        """
        if self.schema is None:
            return ()
        return tuple(self.schema.column_names())


_JSON_SCHEMA_BY_DATASET_NAME: Final[dict[str, str]] = {
    "function_profile": "function_profile",
    "file_profile": "file_profile",
    "module_profile": "module_profile",
    "call_graph_edges": "call_graph_edges",
    "symbol_use_edges": "symbol_use_edges",
    "test_coverage_edges": "test_coverage_edges",
    "test_profile": "test_profile",
    "behavioral_coverage": "behavioral_coverage",
    "v_subsystem_profile": "v_subsystem_profile",
    "v_subsystem_coverage": "v_subsystem_coverage",
    "subsystem_profile_cache": "subsystem_profile_cache",
    "subsystem_coverage_cache": "subsystem_coverage_cache",
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
    "v_function_summary": (
        "function_metrics",
        "function_types",
        "coverage_functions",
        "goid_risk_factors",
    ),
    "v_function_architecture": (
        "function_profile",
        "graph_metrics_functions",
        "graph_metrics_functions_ext",
        "cfg_function_metrics",
        "dfg_function_metrics",
        "module_profile",
        "subsystems",
        "subsystem_modules",
        "subsystem_graph_metrics",
        "test_graph_metrics_functions",
    ),
    "v_function_history": ("function_profile", "function_history"),
    "v_function_history_timeseries": ("history_timeseries",),
    "v_cfg_block_architecture": ("cfg_blocks", "cfg_block_metrics", "function_profile"),
    "v_dfg_block_architecture": ("cfg_blocks", "dfg_block_metrics", "function_profile"),
    "v_module_architecture": (
        "graph_metrics_modules",
        "module_profile",
        "graph_metrics_modules_ext",
        "symbol_graph_metrics_modules",
        "config_graph_metrics_modules",
        "subsystem_modules",
        "subsystem_graph_metrics",
        "subsystem_agreement",
        "modules",
    ),
    "v_module_history_timeseries": ("history_timeseries",),
    "v_file_summary": (
        "modules",
        "ast_metrics",
        "hotspots",
        "typedness",
        "static_diagnostics",
        "goid_risk_factors",
    ),
    "v_entrypoints": ("entrypoints",),
    "v_external_dependencies": ("external_dependencies",),
    "v_external_dependency_calls": ("external_dependency_calls",),
    "v_subsystem_summary": ("subsystems", "subsystem_modules", "subsystem_agreement"),
    "v_module_with_subsystem": ("subsystem_modules", "v_module_architecture", "subsystems"),
    "v_subsystem_profile": ("subsystems", "subsystem_profile_cache", "subsystem_graph_metrics"),
    "v_subsystem_coverage": ("subsystems", "test_profile", "subsystem_coverage_cache"),
}

_DEFAULT_JSONL_FILENAMES: Final[dict[str, str]] = {
    "core.goids": "goids.jsonl",
    "core.goid_crosswalk": "goid_crosswalk.jsonl",
    "graph.call_graph_nodes": "call_graph_nodes.jsonl",
    "graph.call_graph_edges": "call_graph_edges.jsonl",
    "graph.cfg_blocks": "cfg_blocks.jsonl",
    "graph.cfg_edges": "cfg_edges.jsonl",
    "graph.dfg_edges": "dfg_edges.jsonl",
    "graph.import_graph_edges": "import_graph_edges.jsonl",
    "graph.symbol_use_edges": "symbol_use_edges.jsonl",
    "core.ast_nodes": "ast_nodes.jsonl",
    "core.ast_metrics": "ast_metrics.jsonl",
    "core.cst_nodes": "cst_nodes.jsonl",
    "core.docstrings": "docstrings.jsonl",
    "core.modules": "modules.jsonl",
    "analytics.config_values": "config_values.jsonl",
    "analytics.data_models": "data_models.jsonl",
    "analytics.data_model_fields": "data_model_fields.jsonl",
    "analytics.data_model_relationships": "data_model_relationships.jsonl",
    "analytics.data_model_usage": "data_model_usage.jsonl",
    "analytics.config_data_flow": "config_data_flow.jsonl",
    "analytics.static_diagnostics": "static_diagnostics.jsonl",
    "analytics.hotspots": "hotspots.jsonl",
    "analytics.typedness": "typedness.jsonl",
    "analytics.function_metrics": "function_metrics.jsonl",
    "analytics.function_types": "function_types.jsonl",
    "analytics.function_effects": "function_effects.jsonl",
    "analytics.function_contracts": "function_contracts.jsonl",
    "analytics.function_ast_features": "function_ast_features.jsonl",
    "analytics.semantic_roles_functions": "semantic_roles_functions.jsonl",
    "analytics.semantic_roles_modules": "semantic_roles_modules.jsonl",
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
    "docs.v_validation_summary": "validation_summary.jsonl",
}

_DEFAULT_PARQUET_FILENAMES: Final[dict[str, str]] = {
    "core.goids": "goids.parquet",
    "core.goid_crosswalk": "goid_crosswalk.parquet",
    "graph.call_graph_nodes": "call_graph_nodes.parquet",
    "graph.call_graph_edges": "call_graph_edges.parquet",
    "graph.cfg_blocks": "cfg_blocks.parquet",
    "graph.cfg_edges": "cfg_edges.parquet",
    "graph.dfg_edges": "dfg_edges.parquet",
    "graph.import_graph_edges": "import_graph_edges.parquet",
    "graph.symbol_use_edges": "symbol_use_edges.parquet",
    "core.ast_nodes": "ast_nodes.parquet",
    "core.ast_metrics": "ast_metrics.parquet",
    "core.cst_nodes": "cst_nodes.parquet",
    "core.docstrings": "docstrings.parquet",
    "core.modules": "modules.parquet",
    "analytics.config_values": "config_values.parquet",
    "analytics.data_models": "data_models.parquet",
    "analytics.data_model_fields": "data_model_fields.parquet",
    "analytics.data_model_relationships": "data_model_relationships.parquet",
    "analytics.data_model_usage": "data_model_usage.parquet",
    "analytics.config_data_flow": "config_data_flow.parquet",
    "analytics.static_diagnostics": "static_diagnostics.parquet",
    "analytics.hotspots": "hotspots.parquet",
    "analytics.typedness": "typedness.parquet",
    "analytics.function_metrics": "function_metrics.parquet",
    "analytics.function_types": "function_types.parquet",
    "analytics.function_effects": "function_effects.parquet",
    "analytics.function_contracts": "function_contracts.parquet",
    "analytics.function_ast_features": "function_ast_features.parquet",
    "analytics.semantic_roles_functions": "semantic_roles_functions.parquet",
    "analytics.semantic_roles_modules": "semantic_roles_modules.parquet",
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
    "docs.v_validation_summary": "validation_summary.parquet",
}


_DATASET_ROWS_ONLY: Final[set[str]] = {
    "analytics.config_graph_metrics_keys",
    "analytics.config_graph_metrics_modules",
    "analytics.config_projection_key_edges",
    "analytics.config_projection_module_edges",
    "analytics.config_values",
    "analytics.coverage_lines",
    "analytics.data_model_fields",
    "analytics.data_model_relationships",
    "analytics.data_models",
    "analytics.external_dependencies",
    "analytics.file_profile",
    "analytics.graph_metrics_modules",
    "analytics.graph_metrics_modules_ext",
    "analytics.graph_stats",
    "analytics.hotspots",
    "analytics.module_profile",
    "analytics.subsystem_profile_cache",
    "analytics.subsystem_coverage_cache",
    "analytics.semantic_roles_modules",
    "analytics.static_diagnostics",
    "analytics.subsystem_agreement",
    "analytics.subsystem_graph_metrics",
    "analytics.subsystem_modules",
    "analytics.subsystems",
    "analytics.symbol_graph_metrics_modules",
    "analytics.tags_index",
    "analytics.test_graph_metrics_tests",
    "analytics.typedness",
    "core.ast_metrics",
    "core.ast_nodes",
    "core.cst_nodes",
    "core.docstrings",
    "core.file_state",
    "core.goid_crosswalk",
    "core.goids",
    "core.modules",
    "core.repo_map",
    "graph.call_graph_nodes",
    "graph.import_graph_edges",
    "graph.import_modules",
}


def _metadata_for_name(name: str) -> dict[str, object]:
    """Get metadata dictionary for a dataset name.

    Parameters
    ----------
    name
        The dataset name.

    Returns
    -------
    dict[str, object]
        Metadata fields for the dataset.
    """
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


def _owner_package_for_prefix(
    prefix: str,
) -> Literal["core", "analytics", "graphs", "qa", "docs"] | None:
    """Get the owner package for a schema prefix.

    Parameters
    ----------
    prefix
        The schema prefix (e.g., "core", "analytics").

    Returns
    -------
    Literal["core", "analytics", "graphs", "qa", "docs"] | None
        The owner package, or None if not recognized.
    """
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


def _row_binding(
    row_type: RowDictType,
    to_tuple: Callable[..., tuple[object, ...]],
) -> RowBinding:
    """Create a RowBinding from a row type and serializer function.

    Parameters
    ----------
    row_type
        The TypedDict class defining the row shape.
    to_tuple
        Function to serialize a row dict to a tuple for INSERT.

    Returns
    -------
    RowBinding
        The configured row binding.
    """
    return RowBinding(row_type=row_type, to_tuple=cast("RowToTuple", to_tuple))


def _build_row_bindings() -> dict[str, RowBinding]:
    """Build the ROW_BINDINGS_BY_TABLE_KEY dictionary.

    Returns
    -------
    dict[str, RowBinding]
        Mapping from table_key to RowBinding.
    """
    return {
        "analytics.coverage_lines": _row_binding(
            row_type=CoverageLineRow,
            to_tuple=coverage_line_to_tuple,
        ),
        "analytics.config_values": _row_binding(
            row_type=ConfigValueRow,
            to_tuple=config_value_to_tuple,
        ),
        "analytics.typedness": _row_binding(
            row_type=TypednessRow,
            to_tuple=typedness_row_to_tuple,
        ),
        "analytics.static_diagnostics": _row_binding(
            row_type=StaticDiagnosticRow,
            to_tuple=static_diagnostic_to_tuple,
        ),
        "analytics.function_validation": _row_binding(
            row_type=FunctionValidationRow,
            to_tuple=function_validation_row_to_tuple,
        ),
        "analytics.function_metrics": _row_binding(
            row_type=FunctionMetricsRow,
            to_tuple=function_metrics_row_to_tuple,
        ),
        "analytics.function_types": _row_binding(
            row_type=FunctionTypesRow,
            to_tuple=function_types_row_to_tuple,
        ),
        "analytics.function_effects": _row_binding(
            row_type=FunctionEffectsRow,
            to_tuple=function_effects_row_to_tuple,
        ),
        "analytics.function_contracts": _row_binding(
            row_type=FunctionContractsRow,
            to_tuple=function_contracts_row_to_tuple,
        ),
        "analytics.graph_validation": _row_binding(
            row_type=GraphValidationRow,
            to_tuple=graph_validation_row_to_tuple,
        ),
        "analytics.hotspots": _row_binding(
            row_type=HotspotRow,
            to_tuple=hotspot_row_to_tuple,
        ),
        "analytics.test_catalog": _row_binding(
            row_type=TestCatalogRowModel,
            to_tuple=serialize_test_catalog_row,
        ),
        "analytics.test_coverage_edges": _row_binding(
            row_type=TestCoverageEdgeRow,
            to_tuple=serialize_test_coverage_edge,
        ),
        "core.docstrings": _row_binding(
            row_type=DocstringRow,
            to_tuple=docstring_row_to_tuple,
        ),
        "core.goids": _row_binding(
            row_type=GoidRow,
            to_tuple=GoidRow.to_tuple,
        ),
        "core.goid_crosswalk": _row_binding(
            row_type=GoidCrosswalkRow,
            to_tuple=GoidCrosswalkRow.to_tuple,
        ),
        "analytics.function_profile": _row_binding(
            row_type=FunctionProfileRowModel,
            to_tuple=function_profile_row_to_tuple,
        ),
        "analytics.function_ast_features": _row_binding(
            row_type=FunctionAstFeaturesRow,
            to_tuple=function_ast_features_row_to_tuple,
        ),
        "analytics.file_profile": _row_binding(
            row_type=FileProfileRowModel,
            to_tuple=file_profile_row_to_tuple,
        ),
        "analytics.module_profile": _row_binding(
            row_type=ModuleProfileRowModel,
            to_tuple=module_profile_row_to_tuple,
        ),
        "graph.call_graph_nodes": _row_binding(
            row_type=CallGraphNodeRow,
            to_tuple=call_graph_node_to_tuple,
        ),
        "graph.call_graph_edges": _row_binding(
            row_type=CallGraphEdgeRow,
            to_tuple=call_graph_edge_to_tuple,
        ),
        "graph.import_graph_edges": _row_binding(
            row_type=ImportEdgeRow,
            to_tuple=ImportEdgeRow.to_tuple,
        ),
        "graph.import_modules": _row_binding(
            row_type=ImportModuleRow,
            to_tuple=ImportModuleRow.to_tuple,
        ),
        "graph.cfg_blocks": _row_binding(
            row_type=CFGBlockRow,
            to_tuple=CFGBlockRow.to_tuple,
        ),
        "graph.cfg_edges": _row_binding(
            row_type=CFGEdgeRow,
            to_tuple=CFGEdgeRow.to_tuple,
        ),
        "graph.dfg_edges": _row_binding(
            row_type=DFGEdgeRow,
            to_tuple=DFGEdgeRow.to_tuple,
        ),
        "graph.symbol_use_edges": _row_binding(
            row_type=SymbolUseRow,
            to_tuple=SymbolUseRow.to_tuple,
        ),
        "analytics.graph_metrics_functions": _row_binding(
            row_type=GraphMetricsFunctionsRow,
            to_tuple=graph_metrics_functions_row_to_tuple,
        ),
        "analytics.graph_metrics_modules": _row_binding(
            row_type=GraphMetricsModulesRow,
            to_tuple=graph_metrics_modules_row_to_tuple,
        ),
        "analytics.graph_metrics_functions_ext": _row_binding(
            row_type=GraphMetricsFunctionsExtRow,
            to_tuple=graph_metrics_functions_ext_row_to_tuple,
        ),
        "analytics.graph_metrics_modules_ext": _row_binding(
            row_type=GraphMetricsModulesExtRow,
            to_tuple=graph_metrics_modules_ext_row_to_tuple,
        ),
        "analytics.test_profile": _row_binding(
            row_type=ProfileRowModel,
            to_tuple=serialize_test_profile_row,
        ),
        "analytics.behavioral_coverage": _row_binding(
            row_type=BehavioralCoverageRowModel,
            to_tuple=behavioral_coverage_row_to_tuple,
        ),
        "analytics.subsystem_profile_cache": _row_binding(
            row_type=SubsystemProfileCacheRow,
            to_tuple=subsystem_profile_cache_to_tuple,
        ),
        "analytics.subsystem_coverage_cache": _row_binding(
            row_type=SubsystemCoverageCacheRow,
            to_tuple=subsystem_coverage_cache_to_tuple,
        ),
        "docs.v_subsystem_profile": _row_binding(
            row_type=SubsystemProfileCacheRow,
            to_tuple=subsystem_profile_cache_to_tuple,
        ),
        "docs.v_subsystem_coverage": _row_binding(
            row_type=SubsystemCoverageCacheRow,
            to_tuple=subsystem_coverage_cache_to_tuple,
        ),
    }


def _table_contract(
    table_key: str,
    schema: TableSchema,
    row_bindings: dict[str, RowBinding],
    composites: dict[str, CompositeSchema],
) -> tuple[str, DatasetContract]:
    schema_prefix, name = table_key.split(".", maxsplit=1)
    meta = _metadata_for_name(name)
    row_binding = row_bindings.get(table_key)
    json_schema_id = _JSON_SCHEMA_BY_DATASET_NAME.get(name)
    jsonl_filename = _DEFAULT_JSONL_FILENAMES.get(table_key)
    parquet_filename = _DEFAULT_PARQUET_FILENAMES.get(table_key)
    owner_package = _owner_package_for_prefix(schema_prefix)
    family = schema_prefix

    tags = {"base_table"}
    if table_key in _DATASET_ROWS_ONLY:
        tags.add("dataset_rows_only")

    composition = composites.get(table_key)

    contract = DatasetContract(
        name=name,
        table_key=table_key,
        schema=schema,
        row_binding=row_binding,
        json_schema_id=json_schema_id,
        jsonl_filename=jsonl_filename,
        parquet_filename=parquet_filename,
        is_view=False,
        owner_package=owner_package,
        tags=frozenset(tags),
        description=cast("str | None", meta["description"]),
        family=family,
        owner=cast("str | None", meta["owner"]),
        freshness_sla=cast("str | None", meta["freshness_sla"]),
        retention_policy=cast("str | None", meta["retention_policy"]),
        stable_id=cast("str | None", meta["stable_id"]),
        schema_version=cast("str | None", meta["schema_version"]),
        upstream_dependencies=cast("tuple[str, ...]", meta["upstream_dependencies"]),
        validation_profile=cast("Literal['strict', 'lenient']", meta["validation_profile"]),
        composition=composition,
    )
    return name, contract


def _view_contract(
    view_key: str,
    schemas: dict[str, TableSchema],
    row_bindings: dict[str, RowBinding],
) -> tuple[str, DatasetContract]:
    schema_prefix, view_name = view_key.split(".", maxsplit=1)
    meta = _metadata_for_name(view_name)
    row_binding = row_bindings.get(view_key)
    json_schema_id = _JSON_SCHEMA_BY_DATASET_NAME.get(view_name)
    jsonl_filename = _DEFAULT_JSONL_FILENAMES.get(view_key)
    parquet_filename = _DEFAULT_PARQUET_FILENAMES.get(view_key)
    owner_package = _owner_package_for_prefix(schema_prefix)
    family = schema_prefix

    view_schema = schemas.get(view_key)
    contract = DatasetContract(
        name=view_name,
        table_key=view_key,
        schema=view_schema,
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
    return view_name, contract


def _build_contracts() -> dict[str, DatasetContract]:
    """Build the DATASET_CONTRACTS dictionary from schemas and metadata.

    Returns
    -------
    dict[str, DatasetContract]
        All registered dataset contracts keyed by name.
    """
    row_bindings = _build_row_bindings()
    contracts: dict[str, DatasetContract] = {}

    schemas = table_schemas()
    composites = composite_schemas()

    for table_key, schema in schemas.items():
        if table_key.startswith("tmp_"):
            continue
        name, contract = _table_contract(
            table_key=table_key,
            schema=schema,
            row_bindings=row_bindings,
            composites=composites,
        )
        contracts[name] = contract

    for view_key in DERIVED_DOCS_VIEWS:
        view_name, contract = _view_contract(view_key, schemas, row_bindings)
        contracts[view_name] = contract

    return contracts


def get_table_schemas() -> dict[str, TableSchema]:
    """Return the TABLE_SCHEMAS dictionary.

    Returns
    -------
    dict[str, TableSchema]
        All registered table schemas.
    """
    return table_schemas()


def get_composite_schemas() -> dict[str, CompositeSchema]:
    """Return the COMPOSITE_SCHEMAS dictionary.

    Returns
    -------
    dict[str, CompositeSchema]
        All registered composite schemas.
    """
    return composite_schemas()


def get_row_bindings() -> dict[str, RowBinding]:
    """Return the ROW_BINDINGS_BY_TABLE_KEY dictionary.

    Returns
    -------
    dict[str, RowBinding]
        Mapping from table_key to RowBinding.
    """
    return _row_bindings_cache()


def get_dataset_contracts() -> dict[str, DatasetContract]:
    """Return the DATASET_CONTRACTS dictionary.

    Returns
    -------
    dict[str, DatasetContract]
        All registered dataset contracts keyed by name.
    """
    return _dataset_contracts_cache()


def get_dataset_contracts_by_table_key() -> dict[str, DatasetContract]:
    """Return the DATASET_CONTRACTS_BY_TABLE_KEY dictionary.

    Returns
    -------
    dict[str, DatasetContract]
        All registered dataset contracts keyed by table_key.
    """
    return _dataset_contracts_by_table_key_cache()


@lru_cache(maxsize=1)
def _row_bindings_cache() -> dict[str, RowBinding]:
    return _build_row_bindings()


@lru_cache(maxsize=1)
def _dataset_contracts_cache() -> dict[str, DatasetContract]:
    return _build_contracts()


@lru_cache(maxsize=1)
def _dataset_contracts_by_table_key_cache() -> dict[str, DatasetContract]:
    contracts = _dataset_contracts_cache()
    return {c.table_key: c for c in contracts.values()}


__all__ = [
    "DatasetContract",
    "RowBinding",
    "get_composite_schemas",
    "get_dataset_contracts",
    "get_dataset_contracts_by_table_key",
    "get_row_bindings",
    "get_table_schemas",
]
