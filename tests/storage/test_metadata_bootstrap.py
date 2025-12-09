"""Tests for DuckDB-backed dataset metadata bootstrap."""

from __future__ import annotations

import pytest

from codeintel.storage.datasets import load_dataset_registry
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.metadata import (
    bootstrap_metadata_datasets,
    dataset_rows_only_entries,
    ingest_macro_coverage,
    load_dataset_schema_registry,
    load_macro_registry,
    validate_dataset_schema_registry,
    validate_macro_registry,
    validate_normalized_macro_schemas,
)
from codeintel.storage.repositories import DataflowRepository
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_not_empty,
    expect_true,
)
from tests._helpers.gateway import GatewayFactory


def _require(condition: object, message: str) -> None:
    """Raise a pytest failure when the condition is false."""
    if not condition:
        pytest.fail(message)


def test_metadata_bootstrap_populates_catalog() -> None:
    """Bootstrap should create catalog rows and expose registry mappings."""
    gateway = GatewayFactory().without_validation().open()
    con = gateway.con

    count_row = con.execute("SELECT COUNT(*) FROM metadata.datasets").fetchone()
    if count_row is None:
        pytest.fail("metadata.datasets count missing")
        return
    expect_true(int(count_row[0]) > 0, message="metadata.datasets is empty")

    registry = load_dataset_registry(con)
    dataset = registry.by_name.get("function_validation")
    if dataset is None:
        pytest.fail("function_validation dataset missing from registry")
        return
    _require(
        dataset.table_key == "analytics.function_validation",
        f"Unexpected table key: {dataset.table_key}",
    )
    _require(not dataset.is_view, "function_validation should not be a view")
    _require(dataset.schema is not None, "function_validation schema missing")
    _require(dataset.family == "analytics", f"Unexpected family: {dataset.family}")
    filename = registry.jsonl_datasets.get("analytics.function_validation")
    _require(filename == "function_validation.jsonl", f"Unexpected JSONL filename: {filename}")

    view_dataset = registry.by_name.get("v_function_summary")
    if view_dataset is None:
        pytest.fail("v_function_summary missing from registry")
        return
    _require(view_dataset.is_view, "v_function_summary should be a view")
    _require(view_dataset.schema is None, "v_function_summary should not include a TableSchema")
    _require(view_dataset.family == "docs", f"Unexpected docs family: {view_dataset.family}")

    gateway.close()


def test_dataflow_metadata_populated() -> None:
    """bootstrap_metadata_datasets should populate dataset_dataflow_* tables and repositories."""
    gateway = (
        GatewayFactory()
        .without_validation()
        .with_snapshot("test/repo", "deadbeef")
        .open()
    )
    try:
        bootstrap_metadata_datasets(gateway.con, include_views=True)
        repo = DataflowRepository(gateway, "test/repo", "deadbeef")

        nodes = repo.list_nodes()
        edges = repo.list_edges()

        _require(nodes, "Expected at least one dataflow node")
        _require(edges, "Expected at least one dataflow edge")
    finally:
        gateway.close()


def test_load_dataset_schema_registry(fresh_gateway: StorageGateway) -> None:
    """Verify load_dataset_schema_registry returns mapping of table keys to hashes."""
    con = fresh_gateway.con

    registry = load_dataset_schema_registry(con)

    expect_is_instance(registry, dict)
    expect_not_empty(registry)

    for key, value in registry.items():
        expect_in(".", key)
        expect_is_instance(value, str)


def test_load_macro_registry(fresh_gateway: StorageGateway) -> None:
    """Verify load_macro_registry returns mapping of macro names to metadata."""
    con = fresh_gateway.con

    con.execute(
        """
        INSERT INTO metadata.macro_registry (macro_name, dataset_table_key, ddl_hash, schema_hash)
        VALUES ('test_macro', 'core.test', 'hash123', 'schemahash456')
        """
    )

    registry = load_macro_registry(con)

    expect_is_instance(registry, dict)
    expect_not_empty(registry)
    expect_in("test_macro", registry)

    table_key, ddl_hash, schema_hash = registry["test_macro"]
    expect_equal(table_key, "core.test")
    expect_equal(ddl_hash, "hash123")
    expect_equal(schema_hash, "schemahash456")


def test_dataset_rows_only_entries_returns_list() -> None:
    """Verify dataset_rows_only_entries returns sorted list of allowed datasets."""
    entries = dataset_rows_only_entries()

    expect_is_instance(entries, list)
    expect_equal(entries, sorted(entries))


def test_validate_macro_registry_success(fresh_gateway: StorageGateway) -> None:
    """Verify validate_macro_registry passes on properly bootstrapped database."""
    con = fresh_gateway.con

    # Should not raise
    validate_macro_registry(con)


def test_validate_macro_registry_with_missing_entry(fresh_gateway: StorageGateway) -> None:
    """Verify validate_macro_registry handles empty registry gracefully."""
    con = fresh_gateway.con

    # Clear any existing entries
    con.execute("DELETE FROM metadata.macro_registry")

    # Should not raise (nothing to validate against)
    validate_macro_registry(con)


def test_ingest_macro_coverage_returns_present_and_missing(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify ingest_macro_coverage returns tuple of missing and present lists."""
    con = fresh_gateway.con

    missing, present = ingest_macro_coverage(con)

    # Both should be lists
    expect_is_instance(missing, list)
    expect_is_instance(present, list)

    # On a bootstrapped DB, most/all macros should be present
    expect_not_empty(present)


def test_validate_dataset_schema_registry_success(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify validate_dataset_schema_registry passes on properly bootstrapped database."""
    con = fresh_gateway.con

    # Should not raise
    validate_dataset_schema_registry(con)


def test_validate_dataset_schema_registry_detects_drift(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify validate_dataset_schema_registry raises on schema hash drift."""
    con = fresh_gateway.con

    # Manually corrupt a schema hash
    con.execute(
        """
        UPDATE metadata.dataset_schema_registry
        SET schema_hash = 'corrupted_hash'
        WHERE table_key = (SELECT table_key FROM metadata.dataset_schema_registry LIMIT 1)
        """
    )

    with pytest.raises(RuntimeError, match="schema drift"):
        validate_dataset_schema_registry(con)


def test_validate_normalized_macro_schemas_success(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify validate_normalized_macro_schemas passes on properly bootstrapped database."""
    con = fresh_gateway.con

    # Should not raise
    validate_normalized_macro_schemas(con)
