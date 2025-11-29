"""Dataset metadata registry backed by DuckDB's metadata.datasets table."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
from codeintel.storage import rows as row_models

RowToTuple = Callable[[Mapping[str, object]], tuple[object, ...]]
RowDictType = type[object]


@dataclass(frozen=True)
class RowBinding:
    """Connect a DuckDB table key to a TypedDict row model and serializer."""

    row_type: RowDictType
    to_tuple: RowToTuple


@dataclass(frozen=True)
class Dataset:
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
    jsonl_filename
        Default filename for JSONL exports (may be None when not exported).
    parquet_filename
        Default filename for Parquet exports (may be None when not exported).
    is_view
        True when this dataset is a docs.* view instead of a base table.
    description
        Optional human-readable description of the dataset's purpose.
    json_schema_id
        Optional JSON Schema identifier (without .json) used for export validation.
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
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    is_view: bool = False
    description: str | None = None
    json_schema_id: str | None = None
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    stable_id: str | None = None
    schema_version: str | None = None
    upstream_dependencies: tuple[str, ...] = ()

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
        return {
            "can_validate": self.json_schema_id is not None,
            "can_export_jsonl": self.jsonl_filename is not None,
            "can_export_parquet": self.parquet_filename is not None,
            "has_row_binding": self.row_binding is not None,
            "is_view": self.is_view,
        }


# Backwards-compatible alias for the canonical dataset metadata contract type.
DatasetSpec = Dataset


@dataclass(frozen=True)
class DatasetRegistry:
    """In-memory view of metadata.datasets plus Python row bindings."""

    by_name: Mapping[str, Dataset]
    by_table_key: Mapping[str, Dataset]
    jsonl_datasets: Mapping[str, str]
    parquet_datasets: Mapping[str, str]

    @property
    def all_datasets(self) -> tuple[str, ...]:
        """Return all dataset names."""
        return tuple(self.by_name.keys())

    def datasets_with_json_schema(self) -> tuple[str, ...]:
        """
        Return dataset names that have JSON Schema validation configured.

        Returns
        -------
        tuple[str, ...]
            Dataset names with attached JSON Schema identifiers.
        """
        return tuple(name for name, ds in self.by_name.items() if ds.json_schema_id is not None)

    def dataset_dependencies(self) -> dict[str, tuple[str, ...]]:
        """
        Return upstream dependencies for each dataset.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Mapping of dataset name to its upstream dependencies.
        """
        return {
            name: ds.upstream_dependencies
            for name, ds in self.by_name.items()
            if ds.upstream_dependencies
        }

    def resolve_table_key(self, name: str) -> str:
        """
        Resolve dataset name into fully qualified table or view key.

        Returns
        -------
        str
            Fully qualified table or view identifier.

        Raises
        ------
        KeyError
            If the dataset name is unknown.
        """
        ds = self.by_name.get(name)
        if ds is None:
            message = f"Unknown dataset: {name}"
            raise KeyError(message)
        return ds.table_key


def _row_binding(
    row_type: RowDictType,
    to_tuple: Callable[..., tuple[object, ...]],
) -> RowBinding:
    return RowBinding(row_type=row_type, to_tuple=cast("RowToTuple", to_tuple))


def _metadata_for_name(name: str) -> dict[str, object]:
    return {
        "description": DESCRIPTION_BY_DATASET_NAME.get(name),
        "owner": OWNER_BY_DATASET_NAME.get(name),
        "freshness_sla": FRESHNESS_BY_DATASET_NAME.get(name),
        "retention_policy": RETENTION_BY_DATASET_NAME.get(name),
        "upstream_dependencies": DEPENDENCIES_BY_DATASET_NAME.get(name, ()),
        "stable_id": STABLE_ID_BY_DATASET_NAME.get(name, name),
        "schema_version": SCHEMA_VERSION_BY_DATASET_NAME.get(name, "1"),
    }


ROW_BINDINGS_BY_TABLE_KEY: dict[str, RowBinding] = {
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
    "analytics.test_profile": _row_binding(
        row_type=row_models.ProfileRowModel,
        to_tuple=row_models.serialize_test_profile_row,
    ),
    "analytics.behavioral_coverage": _row_binding(
        row_type=row_models.BehavioralCoverageRowModel,
        to_tuple=row_models.behavioral_coverage_row_to_tuple,
    ),
}

# Dataset-level JSON Schema metadata.
# Keys: dataset logical names (Dataset.name).
# Values: JSON Schema identifiers (filenames without .json) under
# src/codeintel/config/schemas/export/.
JSON_SCHEMA_BY_DATASET_NAME: dict[str, str] = {
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
    # Data models
    "data_model_fields": "data_model_fields",
    "data_model_relationships": "data_model_relationships",
}

DESCRIPTION_BY_DATASET_NAME: dict[str, str] = {
    "function_profile": "Function-level profile combining metrics, risk, and topology.",
    "file_profile": "File-level profile with coverage, hotspots, and ownership signals.",
    "module_profile": "Module-level profile aggregating functions, imports, and risk.",
    "call_graph_edges": "Directed call graph edges across the codebase.",
    "symbol_use_edges": "Symbol use edges linking definitions to references.",
    "test_coverage_edges": "Test-to-target coverage edges for tracing impacts.",
    "test_profile": "Test-level profile including outcomes and runtime metadata.",
    "behavioral_coverage": "Behavioral coverage findings captured during scenario runs.",
    "data_model_fields": "Normalized data model field definitions for analytics export.",
    "data_model_relationships": "Normalized data model relationships for analytics export.",
}

OWNER_BY_DATASET_NAME: dict[str, str] = {
    "function_profile": "analytics",
    "file_profile": "analytics",
    "module_profile": "analytics",
    "call_graph_edges": "graphs",
    "symbol_use_edges": "graphs",
    "test_coverage_edges": "analytics",
    "test_profile": "qa",
    "behavioral_coverage": "qa",
    "data_model_fields": "analytics",
    "data_model_relationships": "analytics",
}

FRESHNESS_BY_DATASET_NAME: dict[str, str] = {
    "function_profile": "daily",
    "file_profile": "daily",
    "module_profile": "daily",
    "call_graph_edges": "daily",
    "symbol_use_edges": "daily",
    "test_coverage_edges": "daily",
    "test_profile": "daily",
    "behavioral_coverage": "daily",
    "data_model_fields": "daily",
    "data_model_relationships": "daily",
}

RETENTION_BY_DATASET_NAME: dict[str, str] = {
    "function_profile": "90d",
    "file_profile": "90d",
    "module_profile": "90d",
    "call_graph_edges": "90d",
    "symbol_use_edges": "90d",
    "test_coverage_edges": "90d",
    "test_profile": "90d",
    "behavioral_coverage": "90d",
    "data_model_fields": "90d",
    "data_model_relationships": "90d",
}

STABLE_ID_BY_DATASET_NAME: dict[str, str] = {}
SCHEMA_VERSION_BY_DATASET_NAME: dict[str, str] = {}

DEPENDENCIES_BY_DATASET_NAME: dict[str, tuple[str, ...]] = {
    "function_profile": ("call_graph_edges", "symbol_use_edges"),
    "file_profile": ("call_graph_edges",),
    "module_profile": ("call_graph_edges", "symbol_use_edges"),
    "test_profile": ("test_coverage_edges",),
    "behavioral_coverage": ("test_profile",),
    "data_model_relationships": ("data_model_fields",),
}

DEFAULT_JSONL_FILENAMES: dict[str, str] = {
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


DEFAULT_PARQUET_FILENAMES: dict[str, str] = {
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


def load_dataset_registry(con: DuckDBPyConnection) -> DatasetRegistry:
    """
    Load dataset metadata from DuckDB's metadata.datasets table.

    Assumes metadata_bootstrap.bootstrap_metadata_datasets() has run on this database.

    Returns
    -------
    DatasetRegistry
        Registry containing dataset metadata mirrored from DuckDB.
    """
    rows = con.execute(
        """
        SELECT table_key, name, is_view, jsonl_filename, parquet_filename
        FROM metadata.datasets
        ORDER BY table_key
        """
    ).fetchall()

    by_name: dict[str, Dataset] = {}
    by_table: dict[str, Dataset] = {}
    jsonl_map: dict[str, str] = {}
    parquet_map: dict[str, str] = {}

    for table_key, name, is_view, jsonl_filename, parquet_filename in rows:
        schema: TableSchema | None = None if is_view else TABLE_SCHEMAS.get(table_key)
        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
        json_schema_id = JSON_SCHEMA_BY_DATASET_NAME.get(name)
        meta = _metadata_for_name(name)
        ds = Dataset(
            table_key=table_key,
            name=name,
            schema=schema,
            row_binding=row_binding,
            jsonl_filename=jsonl_filename,
            parquet_filename=parquet_filename,
            is_view=bool(is_view),
            json_schema_id=json_schema_id,
            description=cast("str | None", meta["description"]),
            owner=cast("str | None", meta["owner"]),
            freshness_sla=cast("str | None", meta["freshness_sla"]),
            retention_policy=cast("str | None", meta["retention_policy"]),
            stable_id=cast("str | None", meta["stable_id"]),
            schema_version=cast("str | None", meta["schema_version"]),
            upstream_dependencies=cast("tuple[str, ...]", meta["upstream_dependencies"]),
        )
        by_name[name] = ds
        by_table[table_key] = ds
        if jsonl_filename:
            jsonl_map[table_key] = jsonl_filename
        if parquet_filename:
            parquet_map[table_key] = parquet_filename

    return DatasetRegistry(
        by_name=by_name,
        by_table_key=by_table,
        jsonl_datasets=jsonl_map,
        parquet_datasets=parquet_map,
    )


def dataset_for_name(registry: DatasetRegistry, name: str) -> Dataset:
    """
    Return dataset metadata for a dataset name.

    Returns
    -------
    Dataset
        Dataset metadata resolved from the registry.

    Raises
    ------
    KeyError
        If the dataset name is not present.
    """
    ds = registry.by_name.get(name)
    if ds is None:
        message = f"Unknown dataset name: {name}"
        raise KeyError(message)
    return ds


def dataset_for_table(registry: DatasetRegistry, table_key: str) -> Dataset:
    """
    Return dataset metadata for a fully qualified table or view key.

    Returns
    -------
    Dataset
        Dataset metadata resolved from the registry.

    Raises
    ------
    KeyError
        If the table key is not present.
    """
    ds = registry.by_table_key.get(table_key)
    if ds is None:
        message = f"Unknown dataset table key: {table_key}"
        raise KeyError(message)
    return ds


def describe_dataset(ds: Dataset) -> dict[str, object]:
    """
    Return a JSON-serializable description of a dataset spec.

    Returns
    -------
    dict[str, object]
        JSON-friendly representation of the dataset contract.
    """
    return {
        "name": ds.name,
        "table_key": ds.table_key,
        "is_view": ds.is_view,
        "schema_columns": (
            [col.name for col in ds.schema.columns] if ds.schema is not None else []
        ),
        "jsonl_filename": ds.jsonl_filename,
        "parquet_filename": ds.parquet_filename,
        "has_row_binding": ds.row_binding is not None,
        "json_schema_id": ds.json_schema_id,
        "description": ds.description,
        "owner": ds.owner,
        "freshness_sla": ds.freshness_sla,
        "retention_policy": ds.retention_policy,
        "stable_id": ds.stable_id,
        "schema_version": ds.schema_version,
        "upstream_dependencies": list(ds.upstream_dependencies),
        "capabilities": ds.capabilities(),
    }


def list_dataset_specs(registry: DatasetRegistry) -> list[dict[str, object]]:
    """
    Serialize all dataset specs from a DatasetRegistry.

    Returns
    -------
    list[dict[str, object]]
        List of dataset descriptions derived from the registry.
    """
    return [describe_dataset(ds) for ds in registry.by_name.values()]


def build_dataset_dependency_graph(registry: DatasetRegistry) -> dict[str, tuple[str, ...]]:
    """
    Construct a dependency graph mapping dataset -> upstream datasets.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Mapping of dataset names to their upstream dependencies.
    """
    return registry.dataset_dependencies()
