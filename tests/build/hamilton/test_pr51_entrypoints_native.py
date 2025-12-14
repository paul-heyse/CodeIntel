"""PR51: Tests for entrypoints native Hamilton module.

This module tests the migration from plugin-based entrypoints to
Hamilton native nodes. It verifies:
1. Pure compute function returns correct result type
2. Column counts match schema
3. Native Hamilton nodes integrate properly
4. Both tables are populated with correct schemas
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.entrypoints import (
    EntrypointBuildInputs,
    EntrypointsResult,
    build_entrypoints,
    compute_entrypoints_pure,
)
from codeintel.analytics.entrypoints.core import (
    ENTRYPOINT_TESTS_COLS,
    ENTRYPOINTS_COLS,
)
from codeintel.build.hamilton.native.analytics import entrypoints as ep_module
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.config.primitives import SnapshotRef
from tests.build.hamilton.test_pr50_architecture_guardrails import ALLOWLIST_IBIS_WRITE_FILES

if TYPE_CHECKING:
    from tests._helpers import TestContext


EXPECTED_ENTRYPOINTS_COLS = 30
EXPECTED_ENTRYPOINT_TESTS_COLS = 9
EXPECTED_ROW_COUNT_SINGLE = 1
EXPECTED_ROW_COUNT_EMPTY = 0


class _MockCatalog:
    """Mock catalog for testing."""

    module_by_path: dict[str, str]

    def __init__(self) -> None:
        """Initialize mock catalog."""
        self.module_by_path = {}


class _MockCatalogProvider:
    """Mock catalog provider for testing."""

    @staticmethod
    def catalog() -> _MockCatalog:
        """Return a mock catalog.

        Returns
        -------
        _MockCatalog
            Empty mock catalog.
        """
        return _MockCatalog()

    @staticmethod
    def lookup_goid(
        rel_path: str,
        lineno: int,
        end_lineno: int,
        qualname: str,
    ) -> int | None:
        """Return None for any lookup.

        Returns
        -------
        int | None
            Always None.
        """
        del rel_path, lineno, end_lineno, qualname
        return None

    @staticmethod
    def urn_for_goid(goid: int) -> str | None:
        """Return None for any GOID.

        Returns
        -------
        str | None
            Always None.
        """
        del goid
        return None


def _create_empty_inputs() -> EntrypointBuildInputs:
    """Create empty inputs for testing.

    Returns
    -------
    EntrypointBuildInputs
        Empty inputs structure for testing.
    """
    return EntrypointBuildInputs(
        catalog_provider=_MockCatalogProvider(),  # type: ignore[arg-type]
        module_map={},
        features_map={},
    )


# =============================================================================
# Tests for compute_entrypoints_pure
# =============================================================================


def test_entrypoints_pure_returns_correct_type(test_ctx: TestContext) -> None:
    """Verify compute_entrypoints_pure returns EntrypointsResult type."""
    inputs = _create_empty_inputs()
    result = compute_entrypoints_pure(
        test_ctx.gateway,
        SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(test_ctx.repo),
        ),
        inputs,
    )

    if not isinstance(result, EntrypointsResult):
        pytest.fail(f"Expected EntrypointsResult, got {type(result)}")


def test_entrypoints_pure_empty_returns_empty(test_ctx: TestContext) -> None:
    """Verify empty inputs return empty result without error."""
    inputs = _create_empty_inputs()
    result = compute_entrypoints_pure(
        test_ctx.gateway,
        SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(test_ctx.repo),
        ),
        inputs,
    )

    if result.entrypoint_rows:
        pytest.fail(f"Expected empty entrypoint_rows, got {len(result.entrypoint_rows)}")
    if result.test_rows:
        pytest.fail(f"Expected empty test_rows, got {len(result.test_rows)}")


# =============================================================================
# Tests for materialize_rows with entrypoints
# =============================================================================


def test_materialize_rows_writes_entrypoints(test_ctx: TestContext) -> None:
    """Verify materialize_rows writes entrypoints rows to database."""
    test_ctx.gateway.policy.ensure_table("analytics.entrypoints")

    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    # Create a minimal row matching ENTRYPOINTS_COLS (30 columns)
    now = datetime.now(UTC)
    rows = [
        (
            test_ctx.repo,  # repo
            test_ctx.commit,  # commit
            "ep_id_123",  # entrypoint_id
            "http",  # kind
            "fastapi",  # framework
            Decimal(12345),  # handler_goid_h128
            "urn:test:handler",  # handler_urn
            "pkg/api.py",  # handler_rel_path
            "pkg.api",  # handler_module
            "handle_request",  # handler_qualname
            "GET",  # http_method
            "/api/v1/users",  # route_path
            [200, 404],  # status_codes
            True,  # auth_required
            None,  # command_name
            None,  # arguments_schema
            None,  # schedule
            None,  # trigger
            {"version": "1.0"},  # extra
            "subsys_001",  # subsystem_id
            "UserService",  # subsystem_name
            ["api", "users"],  # tags
            ["team-backend"],  # owners
            5,  # tests_touching
            0,  # failing_tests
            1,  # slow_tests
            0,  # flaky_tests
            0.85,  # entrypoint_coverage_ratio
            "all_passing",  # last_test_status
            now,  # created_at
        )
    ]

    ref = materialize_rows(
        ctx,
        "analytics.entrypoints",
        rows,
        ENTRYPOINTS_COLS,
    )

    if ref.row_count != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_SINGLE}, got {ref.row_count}")

    count = test_ctx.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.entrypoints
        WHERE repo = ? AND commit = ?
        """,
        [test_ctx.repo, test_ctx.commit],
    ).fetchone()
    if count is None or count[0] != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row in DB, got {count}")


def test_materialize_rows_writes_entrypoint_tests(test_ctx: TestContext) -> None:
    """Verify materialize_rows writes entrypoint_tests rows to database."""
    test_ctx.gateway.policy.ensure_table("analytics.entrypoint_tests")

    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    # Create a minimal row matching ENTRYPOINT_TESTS_COLS (9 columns)
    now = datetime.now(UTC)
    rows = [
        (
            test_ctx.repo,  # repo
            test_ctx.commit,  # commit
            "ep_id_123",  # entrypoint_id
            "test_001",  # test_id
            Decimal(54321),  # test_goid_h128
            0.75,  # coverage_ratio
            "passed",  # status
            150.5,  # duration_ms
            now,  # created_at
        )
    ]

    ref = materialize_rows(
        ctx,
        "analytics.entrypoint_tests",
        rows,
        ENTRYPOINT_TESTS_COLS,
    )

    if ref.row_count != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_SINGLE}, got {ref.row_count}")

    count = test_ctx.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.entrypoint_tests
        WHERE repo = ? AND commit = ?
        """,
        [test_ctx.repo, test_ctx.commit],
    ).fetchone()
    if count is None or count[0] != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row in DB, got {count}")


def test_materialize_rows_handles_empty_entrypoints(test_ctx: TestContext) -> None:
    """Verify materialize_rows handles empty row list gracefully."""
    test_ctx.gateway.policy.ensure_table("analytics.entrypoints")

    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    ref = materialize_rows(
        ctx,
        "analytics.entrypoints",
        [],
        ENTRYPOINTS_COLS,
    )

    if ref.row_count != EXPECTED_ROW_COUNT_EMPTY:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_EMPTY}, got {ref.row_count}")


# =============================================================================
# Architecture guardrail tests
# =============================================================================


def test_entrypoints_core_in_allowlist() -> None:
    """Verify analytics/entrypoints/core.py is in allowlist for backward compat.

    The deprecated function build_entrypoints still has direct DB writes
    for backward compatibility. Once the function is removed, the file
    should be removed from the allowlist.

    New code should use the Hamilton native module instead:
    `codeintel.build.hamilton.native.analytics.entrypoints`
    """
    if "src/codeintel/analytics/entrypoints/core.py" not in ALLOWLIST_IBIS_WRITE_FILES:
        pytest.fail(
            "analytics/entrypoints/core.py should remain in "
            "ALLOWLIST_IBIS_WRITE_FILES until deprecated function is removed"
        )


# =============================================================================
# Deprecation warning tests
# =============================================================================


def test_build_entrypoints_deprecation(test_ctx: TestContext) -> None:
    """Verify build_entrypoints emits DeprecationWarning."""
    inputs = _create_empty_inputs()
    with pytest.warns(DeprecationWarning, match="build_entrypoints is deprecated"):
        build_entrypoints(
            test_ctx.gateway,
            SnapshotRef(
                repo=test_ctx.repo,
                commit=test_ctx.commit,
                repo_root=Path(test_ctx.repo),
            ),
            inputs,
        )


# =============================================================================
# Native module export tests
# =============================================================================


def test_native_module_exports() -> None:
    """Verify native module exports expected Hamilton nodes."""
    expected = {
        "t__entrypoints",
        "t__entrypoints__compute",
    }
    actual = set(ep_module.__all__)
    if actual != expected:
        pytest.fail(f"Expected exports {expected}, got {actual}")


def test_hamilton_nodes_have_tags() -> None:
    """Verify Hamilton nodes have proper tag decorators."""
    compute_node = ep_module.t__entrypoints__compute
    materialize_node = ep_module.t__entrypoints

    # Hamilton stores tag decorators in decorate_nodes attribute
    for node, name in [
        (compute_node, "compute"),
        (materialize_node, "materialize"),
    ]:
        if not hasattr(node, "decorate_nodes"):
            pytest.fail(f"{name} missing decorate_nodes attribute from @tag decorator")


# =============================================================================
# Column count tests
# =============================================================================


def test_entrypoints_cols_count() -> None:
    """Verify ENTRYPOINTS_COLS has expected column count."""
    actual_count = len(ENTRYPOINTS_COLS)
    if actual_count != EXPECTED_ENTRYPOINTS_COLS:
        pytest.fail(
            f"Expected {EXPECTED_ENTRYPOINTS_COLS} columns in ENTRYPOINTS_COLS, got {actual_count}"
        )


def test_entrypoint_tests_cols_count() -> None:
    """Verify ENTRYPOINT_TESTS_COLS has expected column count."""
    actual_count = len(ENTRYPOINT_TESTS_COLS)
    if actual_count != EXPECTED_ENTRYPOINT_TESTS_COLS:
        pytest.fail(
            f"Expected {EXPECTED_ENTRYPOINT_TESTS_COLS} columns in ENTRYPOINT_TESTS_COLS, "
            f"got {actual_count}"
        )
