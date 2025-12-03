"""Test data model adapter classes.

Test the data model adapter for persisting data model usage patterns
using real DuckDB instances.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from codeintel.analytics.adapters.data_models import DataModelUsageAdapter
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway, open_memory_gateway

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
# Test Data Factories
# =============================================================================


def _make_data_model_usage_row(
    model_id: str = "model_user",
    goid: Decimal = TEST_GOID_12345,
    usage_kinds: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create a data model usage row for testing.

    Parameters
    ----------
    model_id
        Unique model identifier.
    goid
        Function global object ID using the model.
    usage_kinds
        List of usage kinds.

    Returns
    -------
    dict[str, Any]
        Data model usage row dict.
    """
    if usage_kinds is None:
        usage_kinds = ["field_access", "method_call"]

    return {
        "repo": DEMO_REPO,
        "commit": DEMO_COMMIT,
        "model_id": model_id,
        "function_goid_h128": goid,
        "usage_kinds_json": usage_kinds,
        "evidence_json": [
            {"type": "attribute_access", "attr": "name", "line": 42},
            {"type": "method_call", "method": "save", "line": 45},
        ],
        "context_json": {
            "file_path": "src/services/user_service.py",
            "function_name": "get_user",
        },
        "created_at": datetime.now(tz=UTC),
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def analytics_gateway() -> Iterator[StorageGateway]:
    """
    Create gateway with analytics schema.

    Yields
    ------
    StorageGateway
        Gateway with analytics tables available.
    """
    gateway = open_memory_gateway(
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
    )
    try:
        yield gateway
    finally:
        gateway.close()


@pytest.fixture
def snapshot() -> SnapshotRef:
    """
    Create snapshot reference.

    Returns
    -------
    SnapshotRef
        Snapshot reference for testing.
    """
    return SnapshotRef(
        repo=DEMO_REPO,
        commit=DEMO_COMMIT,
        repo_root=Path("/workspace/demo"),
    )


# =============================================================================
# DataModelUsageAdapter Tests
# =============================================================================


def test_adapter_table_name(
    analytics_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = DataModelUsageAdapter(analytics_gateway, snapshot)
    assert adapter.table_name == "analytics.data_model_usage"


def test_adapter_load_raises(
    analytics_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = DataModelUsageAdapter(analytics_gateway, snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_adapter_persist_empty(
    analytics_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = DataModelUsageAdapter(analytics_gateway, snapshot)
    count = adapter.persist([])
    assert count == EXPECTED_COUNT_0


def test_adapter_persist_single(
    analytics_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single data model usage row."""
    adapter = DataModelUsageAdapter(analytics_gateway, snapshot)
    row = _make_data_model_usage_row()

    count = adapter.persist([row])
    assert count == EXPECTED_COUNT_1

    # Verify row was inserted
    result = analytics_gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.data_model_usage WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()
    assert result is not None
    assert result[0] == EXPECTED_COUNT_1


def test_adapter_persist_multiple(
    analytics_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple data model usage rows."""
    adapter = DataModelUsageAdapter(analytics_gateway, snapshot)

    rows = [
        _make_data_model_usage_row(model_id="model_user", goid=TEST_GOID_12345),
        _make_data_model_usage_row(model_id="model_order", goid=TEST_GOID_67890),
        _make_data_model_usage_row(model_id="model_product", goid=TEST_GOID_11111),
    ]

    count = adapter.persist(rows)
    assert count == EXPECTED_COUNT_3


def test_adapter_persist_same_model_multiple_functions(
    analytics_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist same model used by multiple functions."""
    adapter = DataModelUsageAdapter(analytics_gateway, snapshot)

    rows = [
        _make_data_model_usage_row(
            model_id="model_user",
            goid=TEST_GOID_12345,
            usage_kinds=["field_access"],
        ),
        _make_data_model_usage_row(
            model_id="model_user",
            goid=TEST_GOID_67890,
            usage_kinds=["method_call", "instantiation"],
        ),
    ]

    count = adapter.persist(rows)
    assert count == EXPECTED_COUNT_2


def test_adapter_persist_verifies_data(
    analytics_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = DataModelUsageAdapter(analytics_gateway, snapshot)
    row = _make_data_model_usage_row(
        model_id="model_account",
        goid=TEST_GOID_12345,
        usage_kinds=["instantiation", "serialization"],
    )
    adapter.persist([row])

    # Query and verify
    result = analytics_gateway.con.execute(
        """
        SELECT model_id
        FROM analytics.data_model_usage
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == "model_account"
