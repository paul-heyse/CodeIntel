"""PR51: Tests for data_models native Hamilton module.

This module tests the migration from plugin-based data_models to
Hamilton native nodes. It verifies:
1. Pure compute function returns correct result type
2. Column counts match schema
3. Native Hamilton nodes integrate properly
4. All 3 tables are populated with correct schemas
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.data_models import (
    DataModelsResult,
    compute_data_models,
    compute_data_models_pure,
)
from codeintel.analytics.data_models.core import (
    DATA_MODEL_FIELDS_COLS,
    DATA_MODEL_RELATIONSHIPS_COLS,
    DATA_MODELS_COLS,
)
from codeintel.build.hamilton.native.analytics import data_models as dm_module
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.config.primitives import SnapshotRef
from tests._helpers.builders import (
    GoidRow,
    ModuleRow,
    insert_rows,
)
from tests.build.hamilton.test_pr50_architecture_guardrails import ALLOWLIST_IBIS_WRITE_FILES

if TYPE_CHECKING:
    from tests._helpers import TestContext


REL_PATH = "pkg/models.py"
GOID_TEST_CLASS = 1
EXPECTED_MODEL_COLS = len(DATA_MODELS_COLS)
EXPECTED_FIELD_COLS = len(DATA_MODEL_FIELDS_COLS)
EXPECTED_RELATIONSHIP_COLS = len(DATA_MODEL_RELATIONSHIPS_COLS)
EXPECTED_ROW_COUNT_SINGLE = 1
EXPECTED_ROW_COUNT_EMPTY = 0


def _seed_class(ctx: TestContext) -> None:
    """Seed a test class with module and GOID.

    Parameters
    ----------
    ctx
        Test context with gateway.
    """
    now = datetime.now(UTC)
    insert_rows(
        ctx.gateway,
        [ModuleRow(module="pkg.models", path=REL_PATH, repo=ctx.repo, commit=ctx.commit)],
    )
    insert_rows(
        ctx.gateway,
        [
            GoidRow(
                goid_h128=GOID_TEST_CLASS,
                urn="urn:pkg.models:TestModel",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=REL_PATH,
                kind="class",
                qualname="pkg.models.TestModel",
                start_line=1,
                end_line=10,
                language="python",
                created_at=now,
            )
        ],
    )


# =============================================================================
# Tests for compute_data_models_pure
# =============================================================================


def test_data_models_pure_returns_correct_type(test_ctx: TestContext) -> None:
    """Verify compute_data_models_pure returns DataModelsResult type."""
    result = compute_data_models_pure(
        test_ctx.gateway,
        SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(test_ctx.repo),
        ),
    )

    if not isinstance(result, DataModelsResult):
        pytest.fail(f"Expected DataModelsResult, got {type(result)}")


def test_data_models_pure_empty_returns_empty(test_ctx: TestContext) -> None:
    """Verify empty class metadata returns empty result without error."""
    result = compute_data_models_pure(
        test_ctx.gateway,
        SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(test_ctx.repo),
        ),
    )

    if result.model_rows:
        pytest.fail(f"Expected empty model_rows, got {len(result.model_rows)}")
    if result.field_rows:
        pytest.fail(f"Expected empty field_rows, got {len(result.field_rows)}")
    if result.relationship_rows:
        pytest.fail(f"Expected empty relationship_rows, got {len(result.relationship_rows)}")


def test_data_models_pure_with_class_produces_rows(test_ctx: TestContext, tmp_path: Path) -> None:
    """Verify compute_data_models_pure produces rows when classes exist."""
    # Create a test file with a dataclass
    test_file = tmp_path / "pkg" / "models.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(
        '''"""Test models module."""
from dataclasses import dataclass

@dataclass
class TestModel:
    """A test data model."""
    name: str
    value: int = 0
'''
    )

    now = datetime.now(UTC)
    insert_rows(
        test_ctx.gateway,
        [
            ModuleRow(
                module="pkg.models",
                path="pkg/models.py",
                repo=test_ctx.repo,
                commit=test_ctx.commit,
            )
        ],
    )
    insert_rows(
        test_ctx.gateway,
        [
            GoidRow(
                goid_h128=GOID_TEST_CLASS,
                urn="urn:pkg.models:TestModel",
                repo=test_ctx.repo,
                commit=test_ctx.commit,
                rel_path="pkg/models.py",
                kind="class",
                qualname="pkg.models.TestModel",
                start_line=5,
                end_line=9,
                language="python",
                created_at=now,
            )
        ],
    )

    result = compute_data_models_pure(
        test_ctx.gateway,
        SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=tmp_path,
        ),
    )

    if not result.model_rows:
        pytest.fail("Expected at least one model row")

    # Verify column count matches schema
    actual_cols = len(result.model_rows[0])
    if actual_cols != EXPECTED_MODEL_COLS:
        pytest.fail(f"Expected {EXPECTED_MODEL_COLS} model columns, got {actual_cols}")


# =============================================================================
# Tests for materialize_rows with data_models
# =============================================================================


def test_materialize_rows_writes_data_models(test_ctx: TestContext) -> None:
    """Verify materialize_rows writes data_models rows to database."""
    test_ctx.gateway.policy.ensure_table("analytics.data_models")

    ctx = MaterializationContext(
        gateway=test_ctx.gateway,
        snapshot=SnapshotRef(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            repo_root=Path(),
        ),
        validate=False,
    )

    now = datetime.now(UTC)
    rows = [
        (
            test_ctx.repo,
            test_ctx.commit,
            "model_id_123",
            GOID_TEST_CLASS,
            "TestModel",
            "pkg.models",
            REL_PATH,
            "dataclass",
            "[]",
            "Short description",
            "Long description",
            now,
        )
    ]

    ref = materialize_rows(
        ctx,
        "analytics.data_models",
        rows,
        DATA_MODELS_COLS,
    )

    if ref.row_count != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_SINGLE}, got {ref.row_count}")

    count = test_ctx.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.data_models
        WHERE repo = ? AND commit = ?
        """,
        [test_ctx.repo, test_ctx.commit],
    ).fetchone()
    if count is None or count[0] != EXPECTED_ROW_COUNT_SINGLE:
        pytest.fail(f"Expected {EXPECTED_ROW_COUNT_SINGLE} row in DB, got {count}")


def test_materialize_rows_handles_empty_data_models(test_ctx: TestContext) -> None:
    """Verify materialize_rows handles empty row list gracefully."""
    test_ctx.gateway.policy.ensure_table("analytics.data_models")

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
        "analytics.data_models",
        [],
        DATA_MODELS_COLS,
    )

    if ref.row_count != EXPECTED_ROW_COUNT_EMPTY:
        pytest.fail(f"Expected row_count={EXPECTED_ROW_COUNT_EMPTY}, got {ref.row_count}")


# =============================================================================
# Architecture guardrail tests
# =============================================================================


def test_data_models_core_in_allowlist() -> None:
    """Verify analytics/data_models/core.py is in allowlist for backward compat.

    The deprecated function compute_data_models still has direct DB writes
    for backward compatibility. Once the function is removed, the file
    should be removed from the allowlist.

    New code should use the Hamilton native module instead:
    `codeintel.build.hamilton.native.analytics.data_models`
    """
    if "src/codeintel/analytics/data_models/core.py" not in ALLOWLIST_IBIS_WRITE_FILES:
        pytest.fail(
            "analytics/data_models/core.py should remain in "
            "ALLOWLIST_IBIS_WRITE_FILES until deprecated function is removed"
        )


# =============================================================================
# Deprecation warning tests
# =============================================================================


def test_compute_data_models_deprecation(test_ctx: TestContext) -> None:
    """Verify compute_data_models emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="compute_data_models is deprecated"):
        compute_data_models(
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
        "t__data_models",
        "t__data_models__compute",
    }
    actual = set(dm_module.__all__)
    if actual != expected:
        pytest.fail(f"Expected exports {expected}, got {actual}")


def test_hamilton_nodes_have_tags() -> None:
    """Verify Hamilton nodes have proper tag decorators."""
    compute_node = dm_module.t__data_models__compute
    materialize_node = dm_module.t__data_models

    # Hamilton stores tag decorators in decorate_nodes attribute
    for node, name in [
        (compute_node, "compute"),
        (materialize_node, "materialize"),
    ]:
        if not hasattr(node, "decorate_nodes"):
            pytest.fail(f"{name} missing decorate_nodes attribute from @tag decorator")
