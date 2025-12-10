"""Test data model adapter classes.

Test the data model adapter for persisting data model usage patterns
using real DuckDB instances.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterator

import pytest

from codeintel.analytics.adapters.data_models import DataModelUsageAdapter
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions import (
    expect_equal,
    expect_is_not_none,
)
from tests._helpers.contracts import count_rows
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.env_options import EnvOptions
from tests._helpers.rows import (
    DataModelUsagePayloadSeed,
    data_model_usage_payload,
)

# =============================================================================
# Constants
# =============================================================================

DEMO_REPO = "demo/repo"
DEMO_COMMIT = "abc123def456"
EXPECTED_COUNT_0 = 0
EXPECTED_COUNT_1 = 1
EXPECTED_COUNT_2 = 2
EXPECTED_COUNT_3 = 3
TEST_GOID_12345 = Decimal(12345)
TEST_GOID_67890 = Decimal(67890)
TEST_GOID_11111 = Decimal(11111)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[TestContext]:
    """
    Create a test context aligned with the demo repo/commit identifiers.

    Parameters
    ----------
    tmp_path
        Temporary directory for the test artifacts.

    Yields
    ------
    TestContext
        Configured context with schemas applied.
    """
    options = EnvOptions(repo=DEMO_REPO, commit=DEMO_COMMIT)
    context = create_test_context(tmp_path, options=options)
    try:
        yield context
    finally:
        context.close()


@pytest.fixture
def snapshot(ctx: TestContext) -> SnapshotRef:
    """Expose the snapshot from the shared test context."""
    return ctx.snapshot


# =============================================================================
# DataModelUsageAdapter Tests
# =============================================================================


def test_adapter_table_name(
    ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = DataModelUsageAdapter(ctx.gateway, ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.data_model_usage")


def test_adapter_load_raises(
    ctx: TestContext,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = DataModelUsageAdapter(ctx.gateway, ctx.snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_adapter_persist_empty(
    ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = DataModelUsageAdapter(ctx.gateway, ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, EXPECTED_COUNT_0)


def test_adapter_persist_single(
    ctx: TestContext,
) -> None:
    """Persist single data model usage row."""
    adapter = DataModelUsageAdapter(ctx.gateway, ctx.snapshot)
    row = data_model_usage_payload(
        DataModelUsagePayloadSeed(
            model_id="model_user",
            goid=TEST_GOID_12345,
            usage_kinds=["field_access", "method_call"],
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
            evidence_json=[
                {"type": "attribute_access", "attr": "name", "line": 42},
                {"type": "method_call", "method": "save", "line": 45},
            ],
            context_json={
                "file_path": "src/services/user_service.py",
                "function_name": "get_user",
            },
            created_at=datetime.now(tz=UTC),
        )
    )

    count = adapter.persist([row])
    expect_equal(count, EXPECTED_COUNT_1)

    # Verify row was inserted
    total = count_rows(
        ctx.gateway.con,
        "SELECT COUNT(*) FROM analytics.data_model_usage WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    )
    expect_equal(total, EXPECTED_COUNT_1)


def test_adapter_persist_multiple(
    ctx: TestContext,
) -> None:
    """Persist multiple data model usage rows."""
    adapter = DataModelUsageAdapter(ctx.gateway, ctx.snapshot)

    rows = [
        data_model_usage_payload(
            DataModelUsagePayloadSeed(
                model_id="model_user",
                goid=TEST_GOID_12345,
                usage_kinds=["field_access", "method_call"],
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        data_model_usage_payload(
            DataModelUsagePayloadSeed(
                model_id="model_order",
                goid=TEST_GOID_67890,
                usage_kinds=["field_access", "method_call"],
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        data_model_usage_payload(
            DataModelUsagePayloadSeed(
                model_id="model_product",
                goid=TEST_GOID_11111,
                usage_kinds=["field_access", "method_call"],
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_3)


def test_adapter_persist_same_model_multiple_functions(
    ctx: TestContext,
) -> None:
    """Persist same model used by multiple functions."""
    adapter = DataModelUsageAdapter(ctx.gateway, ctx.snapshot)

    rows = [
        data_model_usage_payload(
            DataModelUsagePayloadSeed(
                model_id="model_user",
                goid=TEST_GOID_12345,
                usage_kinds=["field_access"],
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
        data_model_usage_payload(
            DataModelUsagePayloadSeed(
                model_id="model_user",
                goid=TEST_GOID_67890,
                usage_kinds=["method_call", "instantiation"],
                repo=DEMO_REPO,
                commit=DEMO_COMMIT,
            )
        ),
    ]

    count = adapter.persist(rows)
    expect_equal(count, EXPECTED_COUNT_2)


def test_adapter_persist_verifies_data(
    ctx: TestContext,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = DataModelUsageAdapter(ctx.gateway, ctx.snapshot)
    row = data_model_usage_payload(
        DataModelUsagePayloadSeed(
            model_id="model_account",
            goid=TEST_GOID_12345,
            usage_kinds=["instantiation", "serialization"],
            repo=DEMO_REPO,
            commit=DEMO_COMMIT,
        )
    )
    adapter.persist([row])

    # Query and verify
    result = ctx.gateway.con.execute(
        """
        SELECT model_id
        FROM analytics.data_model_usage
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    row = expect_is_not_none(result)
    expect_equal(row[0], "model_account")
