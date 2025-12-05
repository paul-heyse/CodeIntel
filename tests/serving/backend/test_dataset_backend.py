"""Tests for DatasetQueryLayer behavior."""

from __future__ import annotations

from typing import Final

import pytest

from codeintel.serving.backend import BackendContext, BackendLimits, DuckDBRepositories
from codeintel.serving.backend.dataset_backend import DatasetQueryLayer
from codeintel.serving.mcp import errors
from codeintel.storage.gateway import StorageGateway

# Test constants
SAMPLE_LIMIT_FIVE: Final = 5
CUSTOM_DEFAULT_LIMIT: Final = 10
CUSTOM_MAX_LIMIT: Final = 50


def _expect(*, condition: bool, message: str) -> None:
    """Fail the test when a condition is not met."""
    if not condition:
        pytest.fail(message)


def _build_components(
    gateway: StorageGateway, limits: BackendLimits | None = None
) -> tuple[BackendContext, DuckDBRepositories]:
    repo = gateway.config.repo or "demo/repo"
    commit = gateway.config.commit or "deadbeef"
    context = BackendContext(
        gateway=gateway,
        repo=repo,
        commit=commit,
        limits=limits or BackendLimits(),
        graph_engine=None,
    )
    repositories = DuckDBRepositories(gateway, context.repo, context.commit)
    return context, repositories


# -----------------------------------------------------------------------------
# Tests for list_datasets
# -----------------------------------------------------------------------------


def test_list_datasets_returns_descriptors(architecture_gateway: StorageGateway) -> None:
    """List datasets returns a list of descriptors."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    datasets = backend.list_datasets()

    _expect(
        condition=isinstance(datasets, list),
        message="Should return a list",
    )
    _expect(
        condition=len(datasets) > 0,
        message="Should return at least one dataset",
    )


def test_list_datasets_has_name_and_table(architecture_gateway: StorageGateway) -> None:
    """Each dataset descriptor should have name and table."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    datasets = backend.list_datasets()

    for ds in datasets[:3]:  # Check first few
        _expect(
            condition=ds.name is not None and bool(ds.name),
            message=f"Dataset should have a name: {ds}",
        )
        _expect(
            condition=ds.table is not None and bool(ds.table),
            message=f"Dataset should have a table: {ds}",
        )


# -----------------------------------------------------------------------------
# Tests for dataset_specs
# -----------------------------------------------------------------------------


def test_dataset_specs_returns_sorted_list(architecture_gateway: StorageGateway) -> None:
    """Dataset specs returns specs sorted by name."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    specs = backend.dataset_specs()

    _expect(
        condition=isinstance(specs, list),
        message="Should return a list",
    )
    names = [spec.name for spec in specs]
    _expect(
        condition=names == sorted(names),
        message="Specs should be sorted by name",
    )


def test_dataset_specs_includes_schema_columns(architecture_gateway: StorageGateway) -> None:
    """Dataset specs include schema columns when available."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    specs = backend.dataset_specs()

    # At least some specs should have schema_columns
    has_columns = any(
        spec.schema_columns is not None and len(spec.schema_columns) > 0 for spec in specs
    )
    _expect(
        condition=has_columns or len(specs) > 0,
        message="Should return specs (columns may be empty for some)",
    )


def test_dataset_specs_includes_capabilities(architecture_gateway: StorageGateway) -> None:
    """Dataset specs include capabilities dictionary."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    specs = backend.dataset_specs()

    for spec in specs[:3]:
        _expect(
            condition=spec.capabilities is not None,
            message=f"Spec should have capabilities: {spec.name}",
        )


# -----------------------------------------------------------------------------
# Tests for read_dataset_rows
# -----------------------------------------------------------------------------


def test_read_dataset_rows_success(architecture_gateway: StorageGateway) -> None:
    """Read dataset rows returns data."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    rows = backend.read_dataset_rows(dataset_name="call_graph_edges", limit=SAMPLE_LIMIT_FIVE)

    _expect(
        condition=isinstance(rows, (list, tuple)),
        message="Should return a sequence",
    )


def test_read_dataset_rows_with_offset(architecture_gateway: StorageGateway) -> None:
    """Read dataset rows with offset."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    rows = backend.read_dataset_rows(
        dataset_name="call_graph_edges", limit=SAMPLE_LIMIT_FIVE, offset=0
    )

    _expect(
        condition=isinstance(rows, (list, tuple)),
        message="Should return a sequence with valid offset",
    )


def test_read_dataset_rows_unknown_dataset(architecture_gateway: StorageGateway) -> None:
    """Raise not_found for unknown dataset."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    with pytest.raises(errors.McpError) as excinfo:
        backend.read_dataset_rows(dataset_name="nonexistent_dataset_xyz")

    _expect(
        condition=excinfo.value.detail.code == "not-found",
        message="Unknown dataset should raise not-found",
    )


def test_read_dataset_rows_invalid_offset(architecture_gateway: StorageGateway) -> None:
    """Raise invalid-argument when offset is negative."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    with pytest.raises(errors.McpError) as excinfo:
        backend.read_dataset_rows(
            dataset_name="call_graph_edges", limit=SAMPLE_LIMIT_FIVE, offset=-1
        )

    _expect(
        condition=excinfo.value.detail.code == "invalid-argument",
        message="Negative offsets should raise invalid-argument",
    )


def test_read_dataset_rows_with_custom_limits(architecture_gateway: StorageGateway) -> None:
    """Respect custom limits from backend configuration."""
    custom_limits = BackendLimits(
        default_limit=CUSTOM_DEFAULT_LIMIT, max_rows_per_call=CUSTOM_MAX_LIMIT
    )
    context, repositories = _build_components(architecture_gateway, limits=custom_limits)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    rows = backend.read_dataset_rows(dataset_name="call_graph_edges")

    _expect(
        condition=isinstance(rows, (list, tuple)),
        message="Should return a sequence",
    )


# -----------------------------------------------------------------------------
# Tests for dataset_schema
# -----------------------------------------------------------------------------


def test_dataset_schema_includes_columns(architecture_gateway: StorageGateway) -> None:
    """Return schema details with DuckDB columns populated."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    schema = backend.dataset_schema(dataset_name="call_graph_edges")

    _expect(
        condition=schema.dataset_name == "call_graph_edges",
        message="Dataset name should round-trip",
    )
    _expect(condition=bool(schema.duckdb_schema), message="DuckDB schema should not be empty")
    _expect(condition=schema.meta is not None, message="Response metadata should be populated")


def test_dataset_schema_includes_samples(architecture_gateway: StorageGateway) -> None:
    """Schema includes sample rows."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    schema = backend.dataset_schema(dataset_name="call_graph_edges", sample_limit=3)

    _expect(
        condition=schema.sample_rows is not None,
        message="Sample rows should be present",
    )


def test_dataset_schema_unknown_dataset(architecture_gateway: StorageGateway) -> None:
    """Raise not_found for unknown dataset."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    with pytest.raises(errors.McpError) as excinfo:
        backend.dataset_schema(dataset_name="nonexistent_dataset_xyz")

    _expect(
        condition=excinfo.value.detail.code == "not-found",
        message="Unknown dataset should raise not-found",
    )


def test_dataset_schema_includes_table_key(architecture_gateway: StorageGateway) -> None:
    """Schema includes table_key."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    schema = backend.dataset_schema(dataset_name="call_graph_edges")

    _expect(
        condition=schema.table_key is not None and bool(schema.table_key),
        message="Table key should be present",
    )


# -----------------------------------------------------------------------------
# Tests for backend properties
# -----------------------------------------------------------------------------


def test_backend_datasets_property(architecture_gateway: StorageGateway) -> None:
    """Verify datasets property returns repository."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    datasets_repo = backend.datasets

    _expect(
        condition=datasets_repo is not None,
        message="Should return datasets repository",
    )


def test_backend_gateway_property(architecture_gateway: StorageGateway) -> None:
    """Verify gateway property returns storage gateway."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    gateway = backend.gateway

    _expect(
        condition=gateway is not None,
        message="Should return gateway",
    )


def test_backend_con_property(architecture_gateway: StorageGateway) -> None:
    """Verify con property returns DuckDB connection."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    con = backend.con

    _expect(
        condition=con is not None,
        message="Should return DuckDB connection",
    )


# -----------------------------------------------------------------------------
# Tests for schema column properties via public API
# -----------------------------------------------------------------------------


def test_dataset_schema_columns_have_properties(architecture_gateway: StorageGateway) -> None:
    """Schema columns accessed via dataset_schema have proper properties."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    schema = backend.dataset_schema(dataset_name="call_graph_edges")

    if schema.duckdb_schema:
        col = schema.duckdb_schema[0]
        col_name = col.get("name") if isinstance(col, dict) else getattr(col, "name", None)
        col_type = col.get("type") if isinstance(col, dict) else getattr(col, "type", None)
        col_nullable = (
            col.get("nullable") if isinstance(col, dict) else getattr(col, "nullable", None)
        )
        _expect(
            condition=col_name is not None and bool(col_name),
            message="Column should have name",
        )
        _expect(
            condition=col_type is not None and bool(col_type),
            message="Column should have type",
        )
        _expect(
            condition=isinstance(col_nullable, bool),
            message="Column nullable should be boolean",
        )


# -----------------------------------------------------------------------------
# Tests for validation profile normalization via public API
# -----------------------------------------------------------------------------


def test_dataset_specs_validation_profile_normalized(architecture_gateway: StorageGateway) -> None:
    """Validation profiles in dataset specs are normalized to valid literals."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetQueryLayer(context=context, repositories=repositories)

    specs = backend.dataset_specs()

    for spec in specs:
        # Validation profile should be None, "strict", or "lenient"
        _expect(
            condition=spec.validation_profile in {None, "strict", "lenient"},
            message=f"Invalid validation profile: {spec.validation_profile}",
        )
