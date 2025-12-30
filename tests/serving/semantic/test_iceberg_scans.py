"""Tests for Iceberg scan helpers."""

from __future__ import annotations

from pyiceberg.expressions import And, StartsWith

from codeintel.core.config.settings import IcebergSettings
from codeintel.serving.semantic.iceberg_scans import (
    iceberg_row_filter_from_filters,
    required_scan_fields,
    resolve_iceberg_ref_for_identity,
)
from codeintel.serving.semantic.models import FilterSpec


def test_resolve_iceberg_ref_for_identity_prefers_read_ref() -> None:
    """Prefer the read ref when read is disabled."""
    settings = IcebergSettings(read_enabled=False, read_ref="main")
    ref = resolve_iceberg_ref_for_identity(run_id="run-1", commit="abc", settings=settings)
    assert ref == "main"


def test_resolve_iceberg_ref_for_identity_prefers_run_id() -> None:
    """Prefer the run id ref when read is enabled."""
    settings = IcebergSettings(read_enabled=True)
    ref = resolve_iceberg_ref_for_identity(run_id="run-1", commit="abc", settings=settings)
    assert ref == "run/run-1"


def test_resolve_iceberg_ref_for_identity_falls_back_to_commit() -> None:
    """Fall back to commit ref when no run id is provided."""
    settings = IcebergSettings(read_enabled=True)
    ref = resolve_iceberg_ref_for_identity(run_id=None, commit="abc", settings=settings)
    assert ref == "commit/abc"


def test_required_scan_fields_include_filters_and_order() -> None:
    """Required scan fields should include filter and order columns."""
    filters = [FilterSpec(column="name", op="eq", value="demo")]
    fields = required_scan_fields(columns=("id",), filters=filters, order_by=("-created_at",))
    assert fields == ("id", "name", "created_at")


def test_iceberg_row_filter_reports_pushdown_for_supported_ops() -> None:
    """Supported Iceberg filters should report pushdown coverage."""
    filters = [
        FilterSpec(column="name", op="startswith", value="mod_"),
        FilterSpec(column="name", op="contains", value="x"),
    ]
    result = iceberg_row_filter_from_filters(filters=filters, column_types={"name": "VARCHAR"})
    assert result.supported == 1
    assert result.total == 2
    assert result.coverage == 0.5
    assert isinstance(result.row_filter, StartsWith)


def test_iceberg_row_filter_combines_supported_filters() -> None:
    """Supported filters should combine into a single Iceberg expression."""
    filters = [
        FilterSpec(column="id", op="gte", value=10),
        FilterSpec(column="id", op="lt", value=20),
    ]
    result = iceberg_row_filter_from_filters(filters=filters, column_types={"id": "INTEGER"})
    assert result.supported == 2
    assert result.total == 2
    assert result.coverage == 1.0
    assert isinstance(result.row_filter, And)
