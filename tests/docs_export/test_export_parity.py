"""Parity checks for Document Output dataset mappings."""

from __future__ import annotations

from codeintel.pipeline.export.datasets import JSONL_DATASETS, PARQUET_DATASETS
from tests._helpers.fixtures import ProvisionedGateway


def test_export_mappings_cover_required_tables() -> None:
    """
    Ensure export mappings include core, graph, analytics tables per architecture.

    This acts as a guardrail if new tables are added without updating exports.

    Raises
    ------
    AssertionError
        If either Parquet or JSONL mappings omit required tables.
    """
    required_tables = {
        # core
        "core.goids",
        "core.goid_crosswalk",
        "core.modules",
        "core.ast_nodes",
        "core.ast_metrics",
        "core.cst_nodes",
        "core.docstrings",
        # graph
        "graph.call_graph_nodes",
        "graph.call_graph_edges",
        "graph.cfg_blocks",
        "graph.cfg_edges",
        "graph.dfg_edges",
        "graph.import_graph_edges",
        "graph.symbol_use_edges",
        # analytics
        "analytics.function_metrics",
        "analytics.function_types",
        "analytics.coverage_lines",
        "analytics.coverage_functions",
        "analytics.test_catalog",
        "analytics.test_coverage_edges",
        "analytics.hotspots",
        "analytics.typedness",
        "analytics.static_diagnostics",
        "analytics.config_values",
        "analytics.goid_risk_factors",
    }

    missing_parquet = required_tables - set(PARQUET_DATASETS)
    missing_jsonl = required_tables - set(JSONL_DATASETS)

    if missing_parquet:
        message = f"Parquet mapping missing: {sorted(missing_parquet)}"
        raise AssertionError(message)
    if missing_jsonl:
        message = f"JSONL mapping missing: {sorted(missing_jsonl)}"
        raise AssertionError(message)


def test_export_mappings_registered_with_dataset_registry(
    docs_export_gateway: ProvisionedGateway,
) -> None:
    """
    Export mappings should reference tables registered in the dataset registry.

    Raises
    ------
    AssertionError
        If any export mapping references an unregistered table.
    """
    mapping_tables = set(docs_export_gateway.gateway.datasets.mapping.values())
    missing_parquet = set(PARQUET_DATASETS) - mapping_tables
    missing_jsonl = set(JSONL_DATASETS) - mapping_tables
    if missing_parquet:
        message = f"Parquet export tables not registered: {sorted(missing_parquet)}"
        raise AssertionError(message)
    if missing_jsonl:
        message = f"JSONL export tables not registered: {sorted(missing_jsonl)}"
        raise AssertionError(message)
