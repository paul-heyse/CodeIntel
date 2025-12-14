"""PR51: Tests for data_model_usage native Hamilton migration.

This module tests the migration of compute_data_model_usage to
Hamilton-compatible patterns. It verifies:
1. build_data_model_usage_rows returns correct tuple format
2. compute_data_model_usage emits DeprecationWarning
3. Exports are correct
4. Column count matches schema
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.compute.data_models import (
    DATA_MODEL_USAGE_COLS,
    build_data_model_usage_rows,
    compute_data_model_usage,
)
from tests.build.hamilton.test_pr50_architecture_guardrails import ALLOWLIST_IBIS_WRITE_FILES

if TYPE_CHECKING:
    from tests._helpers import TestContext

EXPECTED_DATA_MODEL_USAGE_COLS = 8


# =============================================================================
# Tests for build_data_model_usage_rows
# =============================================================================


def test_build_data_model_usage_rows_returns_tuple(test_ctx: TestContext) -> None:
    """Verify build_data_model_usage_rows returns tuple of tuples."""
    # Empty inputs should return empty tuple
    rows = build_data_model_usage_rows(
        test_ctx.gateway,
        test_ctx.snapshot,
        module_map={},
        ast_by_goid={},
    )

    if not isinstance(rows, tuple):
        pytest.fail(f"Expected tuple, got {type(rows).__name__}")


def test_build_data_model_usage_rows_empty_when_no_models(test_ctx: TestContext) -> None:
    """Verify build_data_model_usage_rows returns empty tuple when no models exist."""
    # No models in database means empty result
    rows = build_data_model_usage_rows(
        test_ctx.gateway,
        test_ctx.snapshot,
        module_map={"test.py": "test_module"},
        ast_by_goid={},
    )

    if len(rows) != 0:
        pytest.fail(f"Expected 0 rows when no models exist, got {len(rows)}")


# =============================================================================
# Tests for deprecation warnings
# =============================================================================


def test_compute_data_model_usage_deprecation(test_ctx: TestContext) -> None:
    """Verify compute_data_model_usage emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="compute_data_model_usage is deprecated"):
        compute_data_model_usage(
            test_ctx.gateway,
            test_ctx.snapshot,
            module_map={},
            ast_by_goid={},
        )


# =============================================================================
# Architecture guardrail tests
# =============================================================================


def test_data_model_usage_in_allowlist() -> None:
    """Verify analytics/compute/data_models/usage.py is in allowlist for backward compat.

    The deprecated compute_data_model_usage function still has direct DB writes
    for backward compatibility. Once the function is removed, the file
    should be removed from the allowlist.

    New code should use build_data_model_usage_rows with materialize_rows.
    """
    if "src/codeintel/analytics/compute/data_models/usage.py" not in ALLOWLIST_IBIS_WRITE_FILES:
        pytest.fail(
            "analytics/compute/data_models/usage.py should remain in "
            "ALLOWLIST_IBIS_WRITE_FILES until deprecated function is removed"
        )


# =============================================================================
# Column count tests
# =============================================================================


def test_data_model_usage_cols_count() -> None:
    """Verify DATA_MODEL_USAGE_COLS has expected number of columns."""
    col_count = len(DATA_MODEL_USAGE_COLS)
    if col_count != EXPECTED_DATA_MODEL_USAGE_COLS:
        pytest.fail(
            f"Expected {EXPECTED_DATA_MODEL_USAGE_COLS} columns, got {col_count}: {DATA_MODEL_USAGE_COLS}"
        )


# =============================================================================
# Export tests
# =============================================================================


def test_module_exports_new_function() -> None:
    """Verify analytics/compute/data_models exports build_data_model_usage_rows."""
    from codeintel.analytics.compute import data_models  # noqa: PLC0415

    if not hasattr(data_models, "build_data_model_usage_rows"):
        pytest.fail("build_data_model_usage_rows not exported from data_models module")


def test_module_exports_cols_constant() -> None:
    """Verify analytics/compute/data_models exports DATA_MODEL_USAGE_COLS."""
    from codeintel.analytics.compute import data_models  # noqa: PLC0415

    if not hasattr(data_models, "DATA_MODEL_USAGE_COLS"):
        pytest.fail("DATA_MODEL_USAGE_COLS not exported from data_models module")
