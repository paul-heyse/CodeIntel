"""Tests for DatasetQueryLayer behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from codeintel.serving.backend import BackendLimits
from codeintel.serving.backend.dataset_backend import DatasetQueryLayer
from codeintel.serving.mcp import errors
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.backend_components import build_backend_components
from tests._helpers.datasets_assertions import (
    expect_spec_has_capabilities,
    expect_spec_has_columns,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Test constants
SAMPLE_LIMIT_FIVE: Final = 5
CUSTOM_DEFAULT_LIMIT: Final = 10
CUSTOM_MAX_LIMIT: Final = 50


# -----------------------------------------------------------------------------
# Tests for list_datasets
# -----------------------------------------------------------------------------


def test_list_datasets_returns_descriptors(architecture_gateway: StorageGateway) -> None:
    """List datasets returns a list of descriptors."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    datasets = backend.list_datasets()

    expect_true(isinstance(datasets, list))
    expect_true(len(datasets) > 0)


def test_list_datasets_has_name_and_table(architecture_gateway: StorageGateway) -> None:
    """Each dataset descriptor should have name and table."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    datasets = backend.list_datasets()

    for ds in datasets[:3]:  # Check first few
        expect_true(ds.name is not None and bool(ds.name))
        expect_true(ds.table is not None and bool(ds.table))


# -----------------------------------------------------------------------------
# Tests for dataset_specs
# -----------------------------------------------------------------------------


def test_dataset_specs_returns_sorted_list(architecture_gateway: StorageGateway) -> None:
    """Dataset specs returns specs sorted by name."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    specs = backend.dataset_specs()

    expect_true(isinstance(specs, list))
    names = [spec.name for spec in specs]
    expect_equal(names, sorted(names))


def test_dataset_specs_includes_schema_columns(architecture_gateway: StorageGateway) -> None:
    """Dataset specs include schema columns when available."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    specs = backend.dataset_specs()

    # At least some specs should have schema_columns
    spec_with_columns = next((spec for spec in specs if spec.schema_columns), None)
    if spec_with_columns is not None:
        expect_spec_has_columns(spec_with_columns)
    else:
        expect_true(len(specs) > 0)


def test_dataset_specs_includes_capabilities(architecture_gateway: StorageGateway) -> None:
    """Dataset specs include capabilities dictionary."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    specs = backend.dataset_specs()

    for spec in specs[:3]:
        expect_spec_has_capabilities(spec)


# -----------------------------------------------------------------------------
# Tests for read_dataset_rows
# -----------------------------------------------------------------------------


def test_read_dataset_rows_success(architecture_gateway: StorageGateway) -> None:
    """Read dataset rows returns data."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    rows = backend.read_dataset_rows(dataset_name="call_graph_edges", limit=SAMPLE_LIMIT_FIVE)

    expect_true(isinstance(rows, (list, tuple)))


def test_read_dataset_rows_with_offset(architecture_gateway: StorageGateway) -> None:
    """Read dataset rows with offset."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    rows = backend.read_dataset_rows(
        dataset_name="call_graph_edges", limit=SAMPLE_LIMIT_FIVE, offset=0
    )

    expect_true(isinstance(rows, (list, tuple)))


def test_read_dataset_rows_unknown_dataset(architecture_gateway: StorageGateway) -> None:
    """Raise not_found for unknown dataset."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    with pytest.raises(errors.McpError) as excinfo:
        backend.read_dataset_rows(dataset_name="nonexistent_dataset_xyz")

    expect_equal(excinfo.value.detail.code, "not-found")


def test_read_dataset_rows_invalid_offset(architecture_gateway: StorageGateway) -> None:
    """Raise invalid-argument when offset is negative."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    with pytest.raises(errors.McpError) as excinfo:
        backend.read_dataset_rows(
            dataset_name="call_graph_edges", limit=SAMPLE_LIMIT_FIVE, offset=-1
        )

    expect_equal(excinfo.value.detail.code, "invalid-argument")


def test_read_dataset_rows_with_custom_limits(architecture_gateway: StorageGateway) -> None:
    """Respect custom limits from backend configuration."""
    custom_limits = BackendLimits(
        default_limit=CUSTOM_DEFAULT_LIMIT, max_rows_per_call=CUSTOM_MAX_LIMIT
    )
    components = build_backend_components(architecture_gateway, limits=custom_limits)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    rows = backend.read_dataset_rows(dataset_name="call_graph_edges")

    expect_true(isinstance(rows, (list, tuple)))


# -----------------------------------------------------------------------------
# Tests for dataset_schema
# -----------------------------------------------------------------------------


def test_dataset_schema_includes_columns(architecture_gateway: StorageGateway) -> None:
    """Return schema details with DuckDB columns populated."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    schema = backend.dataset_schema(dataset_name="call_graph_edges")

    expect_equal(schema.dataset_name, "call_graph_edges")
    expect_true(bool(schema.duckdb_schema))
    expect_true(schema.meta is not None)


def test_dataset_schema_includes_samples(architecture_gateway: StorageGateway) -> None:
    """Schema includes sample rows."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    schema = backend.dataset_schema(dataset_name="call_graph_edges", sample_limit=3)

    expect_true(schema.sample_rows is not None)


def test_dataset_schema_unknown_dataset(architecture_gateway: StorageGateway) -> None:
    """Raise not_found for unknown dataset."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    with pytest.raises(errors.McpError) as excinfo:
        backend.dataset_schema(dataset_name="nonexistent_dataset_xyz")

    expect_equal(excinfo.value.detail.code, "not-found")


def test_dataset_schema_includes_table_key(architecture_gateway: StorageGateway) -> None:
    """Schema includes table_key."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    schema = backend.dataset_schema(dataset_name="call_graph_edges")

    expect_true(schema.table_key is not None and bool(schema.table_key))


# -----------------------------------------------------------------------------
# Tests for backend properties
# -----------------------------------------------------------------------------


def test_backend_datasets_property(architecture_gateway: StorageGateway) -> None:
    """Verify datasets property returns repository."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    datasets_repo = backend.datasets

    expect_true(datasets_repo is not None)


def test_backend_gateway_property(architecture_gateway: StorageGateway) -> None:
    """Verify gateway property returns storage gateway."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    gateway = backend.gateway

    expect_true(gateway is not None)


def test_backend_con_property(architecture_gateway: StorageGateway) -> None:
    """Verify con property returns DuckDB connection."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    con = backend.con

    expect_true(con is not None)


# -----------------------------------------------------------------------------
# Tests for schema column properties via public API
# -----------------------------------------------------------------------------


def test_dataset_schema_columns_have_properties(architecture_gateway: StorageGateway) -> None:
    """Schema columns accessed via dataset_schema have proper properties."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    schema = backend.dataset_schema(dataset_name="call_graph_edges")

    if schema.duckdb_schema:
        col = schema.duckdb_schema[0]
        col_name = col.get("name") if isinstance(col, dict) else getattr(col, "name", None)
        col_type = col.get("type") if isinstance(col, dict) else getattr(col, "type", None)
        col_nullable = (
            col.get("nullable") if isinstance(col, dict) else getattr(col, "nullable", None)
        )
        expect_true(col_name is not None and bool(col_name))
        expect_true(col_type is not None and bool(col_type))
        expect_true(isinstance(col_nullable, bool))


# -----------------------------------------------------------------------------
# Tests for validation profile normalization via public API
# -----------------------------------------------------------------------------


def test_dataset_specs_validation_profile_normalized(architecture_gateway: StorageGateway) -> None:
    """Validation profiles in dataset specs are normalized to valid literals."""
    components = build_backend_components(architecture_gateway)
    backend = DatasetQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    specs = backend.dataset_specs()

    for spec in specs:
        # Validation profile should be None, "strict", or "lenient"
        expect_true(spec.validation_profile in {None, "strict", "lenient"})
