"""Parity checks for Document Output dataset mappings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests._helpers.assertions import MissingExtraOptions, format_missing_extra

if TYPE_CHECKING:
    from tests._helpers.context import TestContext


def test_export_mappings_cover_required_tables(
    docs_export_gateway: TestContext,
) -> None:
    """
    Ensure export mappings include core, graph, analytics tables per architecture.

    This acts as a guardrail if new tables are added without updating exports.

    Raises
    ------
    AssertionError
        If either Parquet or JSONL mappings omit required tables.
    """
    gateway = docs_export_gateway.gateway
    jsonl_mapping = gateway.datasets.jsonl_datasets
    parquet_mapping = gateway.datasets.parquet_datasets

    required_tables = {
        "core.goids",
        "core.goid_crosswalk",
        "core.modules",
        "core.ast_nodes",
        "core.ast_metrics",
        "core.cst_nodes",
        "core.docstrings",
        "graph.call_graph_nodes",
        "graph.call_graph_edges",
        "graph.cfg_blocks",
        "graph.cfg_edges",
        "graph.dfg_edges",
        "graph.import_graph_edges",
        "graph.symbol_use_edges",
        "analytics.function_types",
        "analytics.test_catalog",
        "analytics.static_diagnostics",
        "analytics.config_values",
    }

    parquet_tables = set(parquet_mapping)
    jsonl_tables = set(jsonl_mapping)

    missing_parquet = required_tables - parquet_tables
    missing_jsonl = required_tables - jsonl_tables

    if missing_parquet:
        raise AssertionError(
            format_missing_extra(
                required_tables,
                parquet_tables,
                options=MissingExtraOptions(
                    noun="export tables",
                    context="parquet mapping",
                ),
            )
        )
    if missing_jsonl:
        raise AssertionError(
            format_missing_extra(
                required_tables,
                jsonl_tables,
                options=MissingExtraOptions(
                    noun="export tables",
                    context="jsonl mapping",
                ),
            )
        )


def test_export_mappings_registered_with_dataset_registry(
    docs_export_gateway: TestContext,
) -> None:
    """
    Export mappings should reference tables registered in the dataset registry.

    Raises
    ------
    AssertionError
        If any export mapping references an unregistered table.
    """
    mapping_tables = set(docs_export_gateway.gateway.datasets.by_table_key)
    jsonl_mapping = docs_export_gateway.gateway.datasets.jsonl_datasets
    parquet_mapping = docs_export_gateway.gateway.datasets.parquet_datasets
    parquet_tables = set(parquet_mapping)
    jsonl_tables = set(jsonl_mapping)
    if not parquet_tables.issubset(mapping_tables):
        raise AssertionError(
            format_missing_extra(
                parquet_tables,
                mapping_tables,
                options=MissingExtraOptions(
                    noun="export tables",
                    context="parquet registry",
                ),
            )
        )
    if not jsonl_tables.issubset(mapping_tables):
        raise AssertionError(
            format_missing_extra(
                jsonl_tables,
                mapping_tables,
                options=MissingExtraOptions(
                    noun="export tables",
                    context="jsonl registry",
                ),
            )
        )
