"""PR51: Tests for history_timeseries native Hamilton module.

This module verifies:
1. build_history_timeseries_rows returns tuple without writing
2. compute_history_timeseries emits deprecation warning
3. compute_history_timeseries_gateways emits deprecation warning
4. File is in allowlist for architecture guardrails
5. Native module exports expected Hamilton nodes
6. Hamilton nodes have proper tags
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.history import (
    HISTORY_TIMESERIES_COLS,
    build_history_timeseries_rows,
    compute_history_timeseries,
    compute_history_timeseries_gateways,
)
from codeintel.analytics.history.history_timeseries import HistoryTimeseriesOptions
from codeintel.build.hamilton.native.analytics import (
    history_timeseries as native_module,
)
from tests.build.hamilton.test_pr50_architecture_guardrails import (
    ALLOWLIST_IBIS_WRITE_FILES,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import DuckDBConnection, StorageGateway
    from tests._helpers import TestContext


# Constants to avoid magic numbers
EXPECTED_HISTORY_TIMESERIES_COLS = 19
EXPECTED_ROW_COUNT_EMPTY = 0


# =============================================================================
# Tests for build_history_timeseries_rows
# =============================================================================


def test_build_history_timeseries_rows_returns_tuple(test_ctx: TestContext) -> None:
    """Verify build_history_timeseries_rows returns tuple type."""

    def mock_resolver(_commit: str) -> DuckDBConnection:
        return test_ctx.gateway.con

    options = HistoryTimeseriesOptions(commits=())
    result = build_history_timeseries_rows(
        test_ctx.gateway,
        test_ctx.snapshot,
        mock_resolver,
        options=options,
    )

    assert isinstance(result, tuple)  # noqa: S101


def test_build_history_timeseries_rows_empty_without_commits(test_ctx: TestContext) -> None:
    """Verify build_history_timeseries_rows returns empty tuple when no commits provided."""

    def mock_resolver(_commit: str) -> DuckDBConnection:
        return test_ctx.gateway.con

    options = HistoryTimeseriesOptions(commits=())
    result = build_history_timeseries_rows(
        test_ctx.gateway,
        test_ctx.snapshot,
        mock_resolver,
        options=options,
    )

    assert result == ()  # noqa: S101


# =============================================================================
# Tests for column count
# =============================================================================


def test_history_timeseries_cols_count() -> None:
    """Verify HISTORY_TIMESERIES_COLS has expected column count."""
    assert len(HISTORY_TIMESERIES_COLS) == EXPECTED_HISTORY_TIMESERIES_COLS  # noqa: S101


# =============================================================================
# Deprecation warning tests
# =============================================================================


def test_compute_history_timeseries_deprecation(test_ctx: TestContext) -> None:
    """Verify compute_history_timeseries emits DeprecationWarning."""

    def mock_resolver(_commit: str) -> DuckDBConnection:
        return test_ctx.gateway.con

    options = HistoryTimeseriesOptions(commits=())

    with pytest.warns(DeprecationWarning, match="compute_history_timeseries is deprecated"):
        compute_history_timeseries(
            test_ctx.gateway,
            test_ctx.snapshot,
            mock_resolver,
            options=options,
        )


def test_compute_history_timeseries_gateways_deprecation(test_ctx: TestContext) -> None:
    """Verify compute_history_timeseries_gateways emits DeprecationWarning."""

    def mock_resolver(_commit: str) -> StorageGateway:
        return test_ctx.gateway

    options = HistoryTimeseriesOptions(commits=())

    with pytest.warns(DeprecationWarning, match="compute_history_timeseries_gateways is deprecated"):
        compute_history_timeseries_gateways(
            test_ctx.gateway,
            test_ctx.snapshot,
            mock_resolver,
            options=options,
        )


# =============================================================================
# Architecture guardrail tests
# =============================================================================


def test_history_timeseries_in_allowlist() -> None:
    """Verify history_timeseries.py is in allowlist for backward compatibility."""
    assert "src/codeintel/analytics/history/history_timeseries.py" in ALLOWLIST_IBIS_WRITE_FILES  # noqa: S101


# =============================================================================
# Native module export tests
# =============================================================================


def test_native_module_exports() -> None:
    """Verify native module exports expected Hamilton nodes."""
    expected = {
        "t__history_timeseries",
        "t__history_timeseries__compute",
        "HISTORY_TIMESERIES_COLS",
        "build_history_timeseries_rows",
    }
    actual = set(native_module.__all__)

    assert actual == expected  # noqa: S101


def test_hamilton_nodes_have_tags() -> None:
    """Verify Hamilton nodes have proper tag decorators."""
    compute_node = native_module.t__history_timeseries__compute
    materialize_node = native_module.t__history_timeseries

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


def test_module_exports_build_history_timeseries_rows() -> None:
    """Verify analytics.history exports build_history_timeseries_rows."""
    from codeintel.analytics import history as hist  # noqa: PLC0415

    assert callable(hist.build_history_timeseries_rows)  # noqa: S101


def test_module_exports_history_timeseries_cols() -> None:
    """Verify analytics.history exports HISTORY_TIMESERIES_COLS."""
    from codeintel.analytics import history as hist  # noqa: PLC0415

    assert isinstance(hist.HISTORY_TIMESERIES_COLS, list)  # noqa: S101
    assert len(hist.HISTORY_TIMESERIES_COLS) == EXPECTED_HISTORY_TIMESERIES_COLS  # noqa: S101
