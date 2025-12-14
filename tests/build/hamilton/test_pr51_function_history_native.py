"""PR51: Tests for function_history native Hamilton module.

This module verifies:
1. build_function_history_rows returns tuple without writing
2. compute_function_history emits deprecation warning
3. File is in allowlist for architecture guardrails
4. Native module exports expected Hamilton nodes
5. Hamilton nodes have proper tags
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.functions import (
    FUNCTION_HISTORY_COLS,
    build_function_history_rows,
    compute_function_history,
)
from codeintel.build.hamilton.native.analytics import function_history as native_module
from tests.build.hamilton.test_pr50_architecture_guardrails import (
    ALLOWLIST_IBIS_WRITE_FILES,
)

if TYPE_CHECKING:
    from tests._helpers import TestContext


# Constants to avoid magic numbers
EXPECTED_FUNCTION_HISTORY_COLS = 21
EXPECTED_ROW_COUNT_EMPTY = 0


# =============================================================================
# Tests for build_function_history_rows
# =============================================================================


def test_build_function_history_rows_returns_tuple(test_ctx: TestContext) -> None:
    """Verify build_function_history_rows returns tuple type."""
    result = build_function_history_rows(test_ctx.gateway, test_ctx.snapshot)

    assert isinstance(result, tuple)  # noqa: S101


def test_build_function_history_rows_empty_without_spans(test_ctx: TestContext) -> None:
    """Verify build_function_history_rows returns empty tuple when no function spans exist."""
    result = build_function_history_rows(test_ctx.gateway, test_ctx.snapshot)

    assert result == ()  # noqa: S101


# =============================================================================
# Tests for column count
# =============================================================================


def test_function_history_cols_count() -> None:
    """Verify FUNCTION_HISTORY_COLS has expected column count."""
    assert len(FUNCTION_HISTORY_COLS) == EXPECTED_FUNCTION_HISTORY_COLS  # noqa: S101


# =============================================================================
# Deprecation warning tests
# =============================================================================


def test_compute_function_history_deprecation(test_ctx: TestContext) -> None:
    """Verify compute_function_history emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="compute_function_history is deprecated"):
        compute_function_history(test_ctx.gateway, test_ctx.snapshot)


# =============================================================================
# Architecture guardrail tests
# =============================================================================


def test_function_history_in_allowlist() -> None:
    """Verify function_history.py is in allowlist for backward compatibility."""
    assert "src/codeintel/analytics/functions/function_history.py" in ALLOWLIST_IBIS_WRITE_FILES  # noqa: S101


# =============================================================================
# Native module export tests
# =============================================================================


def test_native_module_exports() -> None:
    """Verify native module exports expected Hamilton nodes."""
    expected = {"t__function_history", "t__function_history__compute"}
    actual = set(native_module.__all__)

    assert actual == expected  # noqa: S101


def test_hamilton_nodes_have_tags() -> None:
    """Verify Hamilton nodes have proper tag decorators."""
    compute_node = native_module.t__function_history__compute
    materialize_node = native_module.t__function_history

    # Hamilton stores tag decorators in decorate_nodes attribute
    for node, name in [
        (compute_node, "compute"),
        (materialize_node, "materialize"),
    ]:
        if not hasattr(node, "decorate_nodes"):
            pytest.fail(f"{name} node missing decorate_nodes (no @tag decorator)")


# =============================================================================
# Module exports tests
# =============================================================================


def test_module_exports_build_function_history_rows() -> None:
    """Verify analytics.functions exports build_function_history_rows."""
    from codeintel.analytics import functions as fns  # noqa: PLC0415

    assert callable(fns.build_function_history_rows)  # noqa: S101


def test_module_exports_function_history_cols() -> None:
    """Verify analytics.functions exports FUNCTION_HISTORY_COLS."""
    from codeintel.analytics import functions as fns  # noqa: PLC0415

    assert isinstance(fns.FUNCTION_HISTORY_COLS, list)  # noqa: S101
    assert len(fns.FUNCTION_HISTORY_COLS) == EXPECTED_FUNCTION_HISTORY_COLS  # noqa: S101
