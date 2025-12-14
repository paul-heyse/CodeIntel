"""PR51: Tests for external_deps native Hamilton module.

This module tests the migration from plugin-based external_deps to
Hamilton native nodes. It verifies:
1. Pure compute functions return correct result types
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

from codeintel.analytics.dependencies import (
    DependencyCallsResult,
    ExternalDependenciesResult,
    build_external_dependencies,
    build_external_dependency_calls,
    compute_dependency_calls_pure,
    compute_external_dependencies_pure,
)
from codeintel.analytics.dependencies.core import (
    EXTERNAL_DEPENDENCIES_COLS,
    EXTERNAL_DEPENDENCY_CALLS_COLS,
    ExternalDependencyInputs,
)
from codeintel.build.hamilton.native.analytics import dependencies as deps_module
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.config.primitives import SnapshotRef
from tests.build.hamilton.test_pr50_architecture_guardrails import ALLOWLIST_IBIS_WRITE_FILES

if TYPE_CHECKING:
    from tests._helpers import TestContext


EXPECTED_CALLS_COLS = len(EXTERNAL_DEPENDENCY_CALLS_COLS)
EXPECTED_DEPS_COLS = len(EXTERNAL_DEPENDENCIES_COLS)
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
    def urn_for_goid(goid: int) -> str | None:
        """Return None for any GOID.

        Returns
        -------
        str | None
            Always None.
        """
        del goid
        return None

    @staticmethod
    def module_for_path(rel_path: str) -> str | None:
        """Return None for any path.

        Returns
        -------
        str | None
            Always None.
        """
        del rel_path
        return None


def _create_empty_inputs() -> ExternalDependencyInputs:
    """Create empty inputs for testing.

    Returns
    -------
    ExternalDependencyInputs
        Empty inputs structure for testing.
    """
    return ExternalDependencyInputs(
        catalog_provider=_MockCatalogProvider(),  # type: ignore[arg-type]
        module_map={},
        ast_by_goid={},
        features_map={},
        missing_goids=None,
    )


# =============================================================================
# Tests for compute_dependency_calls_pure
# =============================================================================


def test_dependency_calls_pure_returns_correct_type(test_ctx: TestContext) -> None:
    """Verify compute_dependency_calls_pure returns DependencyCallsResult type."""
    inputs = _create_empty_inputs()
    result = compute_dependency_calls_pure(
        test_ctx.gateway,
        SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(test_ctx.repo),
        ),
        inputs,
    )

    if not isinstance(result, DependencyCallsResult):
        pytest.fail(f"Expected DependencyCallsResult, got {type(result)}")


def test_dependency_calls_pure_empty_returns_empty(test_ctx: TestContext) -> None:
    """Verify empty inputs return empty result without error."""
    inputs = _create_empty_inputs()
    result = compute_dependency_calls_pure(
        test_ctx.gateway,
        SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(test_ctx.repo),
        ),
        inputs,
    )

    if result.rows:
        pytest.fail(f"Expected empty rows, got {len(result.rows)}")


# =============================================================================
# Tests for compute_external_dependencies_pure
# =============================================================================


def test_external_dependencies_pure_returns_correct_type(test_ctx: TestContext) -> None:
    """Verify compute_external_dependencies_pure returns ExternalDependenciesResult type."""
    # Ensure the table exists first
    test_ctx.gateway.policy.ensure_table("analytics.external_dependency_calls")

    result = compute_external_dependencies_pure(
        test_ctx.gateway,
        SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(test_ctx.repo),
        ),
    )

    if not isinstance(result, ExternalDependenciesResult):
        pytest.fail(f"Expected ExternalDependenciesResult, got {type(result)}")


def test_external_dependencies_pure_empty_returns_empty(test_ctx: TestContext) -> None:
    """Verify empty dependency calls table returns empty result."""
    # Ensure the table exists first
    test_ctx.gateway.policy.ensure_table("analytics.external_dependency_calls")

    result = compute_external_dependencies_pure(
        test_ctx.gateway,
        SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(test_ctx.repo),
        ),
    )

    if result.rows:
        pytest.fail(f"Expected empty rows, got {len(result.rows)}")


# =============================================================================
# Tests for materialize_rows with dependencies
# =============================================================================


def test_materialize_rows_writes_dependency_calls(test_ctx: TestContext) -> None:
    """Verify materialize_rows writes dependency_calls rows to database."""
    test_ctx.gateway.policy.ensure_table("analytics.external_dependency_calls")

    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    # Create a minimal row matching EXTERNAL_DEPENDENCY_CALLS_COLS
    now = datetime.now(UTC)
    rows = [
        (
            test_ctx.repo,  # repo
            test_ctx.commit,  # commit
            "dep_id_123",  # dep_id
            "redis",  # library
            "Redis",  # service_name
            Decimal(12345),  # function_goid_h128
            "urn:test:func",  # function_urn
            "pkg/service.py",  # rel_path
            "pkg.service",  # module
            "process_data",  # qualname
            3,  # callsite_count
            ["read", "write"],  # modes
            None,  # evidence_json
            now,  # created_at
        )
    ]

    ref = materialize_rows(
        ctx,
        "analytics.external_dependency_calls",
        rows,
        EXTERNAL_DEPENDENCY_CALLS_COLS,
    )

    if ref.row_count != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_SINGLE}, got {ref.row_count}")

    count = test_ctx.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.external_dependency_calls
        WHERE repo = ? AND commit = ?
        """,
        [test_ctx.repo, test_ctx.commit],
    ).fetchone()
    if count is None or count[0] != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row in DB, got {count}")


def test_materialize_rows_writes_external_dependencies(test_ctx: TestContext) -> None:
    """Verify materialize_rows writes external_dependencies rows to database."""
    test_ctx.gateway.policy.ensure_table("analytics.external_dependencies")

    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    # Create a minimal row matching EXTERNAL_DEPENDENCIES_COLS
    now = datetime.now(UTC)
    rows = [
        (
            test_ctx.repo,  # repo
            test_ctx.commit,  # commit
            "dep_id_123",  # dep_id
            "redis",  # library
            "Redis",  # service_name
            "cache",  # category
            "python",  # language
            "medium",  # severity
            1.5,  # criticality
            3.0,  # risk_score
            5,  # function_count
            10,  # callsite_count
            ["pkg.service", "pkg.cache"],  # modules_json
            ["read", "write"],  # usage_modes
            ["REDIS_URL"],  # config_keys
            "medium",  # risk_level
            now,  # created_at
        )
    ]

    ref = materialize_rows(
        ctx,
        "analytics.external_dependencies",
        rows,
        EXTERNAL_DEPENDENCIES_COLS,
    )

    if ref.row_count != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_SINGLE}, got {ref.row_count}")

    count = test_ctx.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.external_dependencies
        WHERE repo = ? AND commit = ?
        """,
        [test_ctx.repo, test_ctx.commit],
    ).fetchone()
    if count is None or count[0] != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row in DB, got {count}")


def test_materialize_rows_handles_empty_dependencies(test_ctx: TestContext) -> None:
    """Verify materialize_rows handles empty row list gracefully."""
    test_ctx.gateway.policy.ensure_table("analytics.external_dependency_calls")

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
        "analytics.external_dependency_calls",
        [],
        EXTERNAL_DEPENDENCY_CALLS_COLS,
    )

    if ref.row_count != EXPECTED_ROW_COUNT_EMPTY:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_EMPTY}, got {ref.row_count}")


# =============================================================================
# Architecture guardrail tests
# =============================================================================


def test_dependencies_core_in_allowlist() -> None:
    """Verify analytics/dependencies/core.py is in allowlist for backward compat.

    The deprecated functions build_external_dependency_calls and
    build_external_dependencies still have direct DB writes for backward
    compatibility. Once the functions are removed, the file should be
    removed from the allowlist.

    New code should use the Hamilton native module instead:
    `codeintel.build.hamilton.native.analytics.dependencies`
    """
    if "src/codeintel/analytics/dependencies/core.py" not in ALLOWLIST_IBIS_WRITE_FILES:
        pytest.fail(
            "analytics/dependencies/core.py should remain in "
            "ALLOWLIST_IBIS_WRITE_FILES until deprecated functions are removed"
        )


# =============================================================================
# Deprecation warning tests
# =============================================================================


def test_build_external_dependency_calls_deprecation(test_ctx: TestContext) -> None:
    """Verify build_external_dependency_calls emits DeprecationWarning."""
    inputs = _create_empty_inputs()
    with pytest.warns(DeprecationWarning, match="build_external_dependency_calls is deprecated"):
        build_external_dependency_calls(
            test_ctx.gateway,
            SnapshotRef(
                repo=test_ctx.repo,
                commit=test_ctx.commit,
                repo_root=Path(test_ctx.repo),
            ),
            inputs=inputs,
        )


def test_build_external_dependencies_deprecation(test_ctx: TestContext) -> None:
    """Verify build_external_dependencies emits DeprecationWarning."""
    # Ensure the dependency_calls table exists for the aggregation function
    test_ctx.gateway.policy.ensure_table("analytics.external_dependency_calls")

    with pytest.warns(DeprecationWarning, match="build_external_dependencies is deprecated"):
        build_external_dependencies(
            test_ctx.gateway,
            SnapshotRef(
                repo=test_ctx.repo,
                commit=test_ctx.commit,
                repo_root=Path(test_ctx.repo),
            ),
        )


# =============================================================================
# Native module export tests
# =============================================================================


def test_native_module_exports() -> None:
    """Verify native module exports expected Hamilton nodes."""
    expected = {
        "t__external_deps",
        "t__external_deps__compute_calls",
    }
    actual = set(deps_module.__all__)
    if actual != expected:
        pytest.fail(f"Expected exports {expected}, got {actual}")


def test_hamilton_nodes_have_tags() -> None:
    """Verify Hamilton nodes have proper tag decorators."""
    compute_node = deps_module.t__external_deps__compute_calls
    materialize_node = deps_module.t__external_deps

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


def test_external_dependency_calls_cols_count() -> None:
    """Verify EXTERNAL_DEPENDENCY_CALLS_COLS has expected column count."""
    expected_count = 14
    actual_count = len(EXTERNAL_DEPENDENCY_CALLS_COLS)
    if actual_count != expected_count:
        pytest.fail(
            f"Expected {expected_count} columns in EXTERNAL_DEPENDENCY_CALLS_COLS, "
            f"got {actual_count}"
        )


def test_external_dependencies_cols_count() -> None:
    """Verify EXTERNAL_DEPENDENCIES_COLS has expected column count."""
    expected_count = 17
    actual_count = len(EXTERNAL_DEPENDENCIES_COLS)
    if actual_count != expected_count:
        pytest.fail(
            f"Expected {expected_count} columns in EXTERNAL_DEPENDENCIES_COLS, got {actual_count}"
        )
