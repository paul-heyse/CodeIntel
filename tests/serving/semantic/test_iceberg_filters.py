"""Tests for Iceberg filter translation helpers."""

from __future__ import annotations

import pytest

from codeintel.serving.semantic.iceberg_scans import iceberg_row_filter_from_filters
from codeintel.serving.semantic.models import FilterSpec

pytestmark = pytest.mark.no_runtime_env

EXPECTED_TOTAL = 2
EXPECTED_COVERAGE = 0.5


def test_iceberg_row_filter_reports_coverage() -> None:
    """Iceberg filter translation reports supported coverage."""
    filters = [
        FilterSpec(column="id", op="eq", value=1),
        FilterSpec(column="name", op="contains", value="foo"),
    ]
    result = iceberg_row_filter_from_filters(filters=filters, column_types={"id": "INTEGER"})
    assert result.total == EXPECTED_TOTAL
    assert result.supported == 1
    assert result.coverage == EXPECTED_COVERAGE
    assert result.row_filter is not None
