"""Tests for datasets.py module."""

from __future__ import annotations

import pytest

from codeintel.config.datasets import DatasetContract, get_dataset_contracts_by_table_key
from codeintel.storage.datasets import (
    DatasetRegistry,
    build_dataset_dependency_graph,
    dataset_for_name,
    dataset_for_table,
    describe_all_datasets,
    describe_dataset,
    list_dataset_specs,
    load_dataset_registry,
)
from codeintel.storage.gateway import StorageGateway


def _sample_registry() -> DatasetRegistry:
    """
    Create a sample dataset registry for testing.

    Returns
    -------
    DatasetRegistry
        A test registry with sample datasets.
    """
    table_key = "core.ast_nodes"
    contract = get_dataset_contracts_by_table_key()[table_key]
    dataset = DatasetContract(
        table_key=table_key,
        name="ast_nodes",
        schema=contract.schema,
        json_schema_id="ast_nodes",
        jsonl_filename="ast_nodes.jsonl",
        parquet_filename="ast_nodes.parquet",
        owner="team-data",
        freshness_sla="daily",
        retention_policy="90d",
        schema_version="1",
        stable_id="ast_nodes",
        upstream_dependencies=("core.modules",),
        validation_profile="strict",
    )
    view_key = "docs.v_function_summary"
    view_dataset = DatasetContract(
        table_key=view_key,
        name="v_function_summary",
        schema=None,
        is_view=True,
        family="docs",
    )
    return DatasetRegistry(
        by_name={"ast_nodes": dataset, "v_function_summary": view_dataset},
        by_table_key={table_key: dataset, view_key: view_dataset},
        jsonl_datasets={table_key: "ast_nodes.jsonl"},
        parquet_datasets={table_key: "ast_nodes.parquet"},
    )


def test_dataset_registry_all_datasets_returns_tuple() -> None:
    """Verify all_datasets property returns combined names."""
    registry = _sample_registry()

    all_datasets = registry.all_datasets

    assert isinstance(all_datasets, tuple)
    assert "ast_nodes" in all_datasets


def test_dataset_registry_datasets_with_json_schema() -> None:
    """Verify datasets_with_json_schema returns filtered names."""
    registry = _sample_registry()

    with_schema = registry.datasets_with_json_schema()

    assert isinstance(with_schema, tuple)
    assert "ast_nodes" in with_schema
    assert "v_function_summary" not in with_schema


def test_dataset_registry_dataset_dependencies() -> None:
    """Verify dataset_dependencies returns mapping."""
    registry = _sample_registry()

    deps = registry.dataset_dependencies()

    assert isinstance(deps, dict)
    assert "ast_nodes" in deps
    assert "core.modules" in deps["ast_nodes"]


def test_dataset_registry_docs_dataset_names() -> None:
    """Verify docs_dataset_names returns docs-prefixed views."""
    registry = _sample_registry()

    docs_names = registry.docs_dataset_names()

    assert isinstance(docs_names, tuple)
    assert "v_function_summary" in docs_names
    assert "ast_nodes" not in docs_names


def test_dataset_registry_resolve_table_key_returns_qualified() -> None:
    """Verify resolve_table_key returns fully qualified key."""
    registry = _sample_registry()

    key = registry.resolve_table_key("ast_nodes")

    assert key == "core.ast_nodes"


def test_dataset_registry_resolve_table_key_raises_on_unknown() -> None:
    """Verify resolve_table_key raises KeyError for unknown dataset."""
    registry = _sample_registry()

    with pytest.raises(KeyError, match="Unknown dataset"):
        registry.resolve_table_key("nonexistent_dataset")


def test_dataset_for_name_returns_contract() -> None:
    """Verify dataset_for_name returns contract for valid name."""
    registry = _sample_registry()

    ds = dataset_for_name(registry, "ast_nodes")

    assert ds.table_key == "core.ast_nodes"


def test_dataset_for_name_raises_on_unknown() -> None:
    """Verify dataset_for_name raises KeyError for unknown name."""
    registry = _sample_registry()

    with pytest.raises(KeyError, match="Unknown dataset name"):
        dataset_for_name(registry, "nonexistent")


def test_dataset_for_table_returns_contract() -> None:
    """Verify dataset_for_table returns contract for valid table key."""
    registry = _sample_registry()

    ds = dataset_for_table(registry, "core.ast_nodes")

    assert ds.name == "ast_nodes"


def test_dataset_for_table_raises_on_unknown() -> None:
    """Verify dataset_for_table raises KeyError for unknown table."""
    registry = _sample_registry()

    with pytest.raises(KeyError, match="Unknown dataset table key"):
        dataset_for_table(registry, "unknown.table")


def test_describe_dataset_returns_serializable_dict() -> None:
    """Verify describe_dataset returns JSON-serializable dict."""
    registry = _sample_registry()
    ds = registry.by_name["ast_nodes"]

    desc = describe_dataset(ds)

    assert isinstance(desc, dict)
    assert desc["name"] == "ast_nodes"
    assert desc["table_key"] == "core.ast_nodes"
    assert "schema_columns" in desc
    assert "capabilities" in desc
    assert isinstance(desc["upstream_dependencies"], list)


def test_list_dataset_specs_returns_list() -> None:
    """Verify list_dataset_specs returns list of dicts."""
    registry = _sample_registry()

    specs = list_dataset_specs(registry)

    assert isinstance(specs, list)
    expected_count = 2
    assert len(specs) == expected_count
    assert all(isinstance(s, dict) for s in specs)


def test_build_dataset_dependency_graph_returns_mapping() -> None:
    """Verify build_dataset_dependency_graph returns dict."""
    registry = _sample_registry()

    graph = build_dataset_dependency_graph(registry)

    assert isinstance(graph, dict)
    assert "ast_nodes" in graph


def test_load_dataset_registry_from_db(fresh_gateway: StorageGateway) -> None:
    """Verify load_dataset_registry loads from DuckDB."""
    con = fresh_gateway.con

    registry = load_dataset_registry(con)

    assert isinstance(registry, DatasetRegistry)
    assert len(registry.by_name) > 0
    assert len(registry.by_table_key) > 0


# -------------------------------------------------------------------------
# Compatibility property tests (migrated from test_registry_helpers.py)
# -------------------------------------------------------------------------


def test_dataset_registry_mapping_property() -> None:
    """Verify mapping returns name -> table_key dict."""
    registry = _sample_registry()

    mapping = registry.mapping

    assert isinstance(mapping, dict)
    assert mapping["ast_nodes"] == "core.ast_nodes"


def test_dataset_registry_tables_property() -> None:
    """Verify tables returns non-view dataset names."""
    registry = _sample_registry()

    tables = registry.tables

    assert isinstance(tables, tuple)
    assert "ast_nodes" in tables
    assert "v_function_summary" not in tables


def test_dataset_registry_views_property() -> None:
    """Verify views returns view dataset names."""
    registry = _sample_registry()

    views = registry.views

    assert isinstance(views, tuple)
    assert "v_function_summary" in views
    assert "ast_nodes" not in views


def test_dataset_registry_meta_property() -> None:
    """Verify meta returns by_name alias."""
    registry = _sample_registry()

    meta = registry.meta

    assert meta is registry.by_name
    assert "ast_nodes" in meta


def test_dataset_registry_jsonl_mapping_property() -> None:
    """Verify jsonl_mapping returns jsonl_datasets alias."""
    registry = _sample_registry()

    jsonl_mapping = registry.jsonl_mapping

    assert jsonl_mapping is registry.jsonl_datasets


def test_dataset_registry_parquet_mapping_property() -> None:
    """Verify parquet_mapping returns parquet_datasets alias."""
    registry = _sample_registry()

    parquet_mapping = registry.parquet_mapping

    assert parquet_mapping is registry.parquet_datasets


def test_dataset_registry_table_for_name() -> None:
    """Verify table_for_name is alias for resolve_table_key."""
    registry = _sample_registry()

    table_key = registry.table_for_name("ast_nodes")

    assert table_key == "core.ast_nodes"


def test_dataset_registry_table_for_name_raises_on_unknown() -> None:
    """Verify table_for_name raises KeyError for unknown dataset."""
    registry = _sample_registry()

    with pytest.raises(KeyError, match="Unknown dataset"):
        registry.table_for_name("nonexistent_dataset")


def test_describe_all_datasets_returns_serializable_list(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify describe_all_datasets returns JSON-serializable list."""
    con = fresh_gateway.con

    descriptions = describe_all_datasets(con)

    assert isinstance(descriptions, list)
    assert len(descriptions) > 0

    first_desc = descriptions[0]
    assert isinstance(first_desc, dict)
    assert "name" in first_desc
    assert "table_key" in first_desc
