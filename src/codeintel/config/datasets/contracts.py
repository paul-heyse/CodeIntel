"""Dataset contract shims for backwards compatibility.

This module is deprecated in favor of build-owned schema and contract providers:

- Table schemas: `codeintel.build.schemas.get_schema_provider()`
- Row bindings: `codeintel.build.schemas.get_row_binding()`
- Dataset contracts: `codeintel.build.schemas.get_contract_for_table_key()`
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Final

from codeintel.build.schemas import iter_contracts, iter_contracts_by_table_key
from codeintel.build.schemas.declared_schemas import TABLE_SCHEMAS
from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.core.schemas.contract_primitives import DatasetContract, RowBinding
from codeintel.core.schemas.row_models import row_binding_for_table_schema

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema


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


def get_table_schemas() -> dict[str, TableSchema]:
    """Return declared table schemas.

    Returns
    -------
    dict[str, TableSchema]
        Mapping from table_key to declared TableSchema.
    """
    return TABLE_SCHEMAS


@lru_cache(maxsize=1)
def get_row_bindings() -> dict[str, RowBinding]:
    """Return schema-generated row bindings for all declared table schemas.

    Returns
    -------
    dict[str, RowBinding]
        Mapping from table_key to a schema-generated RowBinding.
    """
    bindings: dict[str, RowBinding] = {}
    for table_key, schema in TABLE_SCHEMAS.items():
        generated = row_binding_for_table_schema(table_schema=schema)
        bindings[table_key] = RowBinding(
            row_type=generated.row_model,
            to_tuple=generated.serializer,
        )
    return bindings


def get_dataset_contracts() -> dict[str, DatasetContract]:
    """Return dataset contracts keyed by dataset name.

    .. deprecated::
        Use ``codeintel.build.schemas.iter_contracts()``.

    Returns
    -------
    dict[str, DatasetContract]
        Mapping from dataset name to the derived DatasetContract.
    """
    warnings.warn(
        "get_dataset_contracts() is deprecated. Use codeintel.build.schemas.iter_contracts().",
        DeprecationWarning,
        stacklevel=2,
    )
    return {contract.name: contract for contract in iter_contracts()}


def get_dataset_contracts_by_table_key() -> dict[str, DatasetContract]:
    """Return dataset contracts keyed by table_key.

    .. deprecated::
        Use ``codeintel.build.schemas.iter_contracts_by_table_key()``.

    Returns
    -------
    dict[str, DatasetContract]
        Mapping from table_key to the derived DatasetContract.
    """
    warnings.warn(
        "get_dataset_contracts_by_table_key() is deprecated. "
        "Use codeintel.build.schemas.iter_contracts_by_table_key().",
        DeprecationWarning,
        stacklevel=2,
    )
    return dict(iter_contracts_by_table_key())


__all__ = [
    "_DEFAULT_JSONL_FILENAMES",
    "_DEFAULT_PARQUET_FILENAMES",
    "_JSON_SCHEMA_BY_DATASET_NAME",
    "DatasetContract",
    "RowBinding",
    "get_composite_schemas",
    "get_dataset_contracts",
    "get_dataset_contracts_by_table_key",
    "get_row_bindings",
    "get_table_schemas",
]
