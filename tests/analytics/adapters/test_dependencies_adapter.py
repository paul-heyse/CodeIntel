"""Test dependency adapter classes.

Test the dependency-specific adapters for persisting external dependency
call and aggregate data using real DuckDB instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from codeintel.analytics.adapters.dependencies import (
    DependencyAggregateAdapter,
    DependencyAggregateRow,
    DependencyCallAdapter,
    DependencyCallRow,
    compute_dep_id,
    to_decimal,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers import assert_frozen

# =============================================================================
# Constants
# =============================================================================

DEMO_REPO = "demo/repo"
DEMO_COMMIT = "abc123def456"
EXPECTED_COUNT_0 = 0
EXPECTED_COUNT_1 = 1
EXPECTED_COUNT_2 = 2
EXPECTED_COUNT_3 = 3
TEST_GOID_12345 = 12345
TEST_GOID_67890 = 67890
CALLSITE_COUNT_5 = 5
CALLSITE_COUNT_10 = 10
FUNCTION_COUNT_3 = 3
RISK_SCORE_0_75 = 0.75
CRITICALITY_0_5 = 0.5
DEP_ID_LENGTH = 16


# =============================================================================
# Test Data
# =============================================================================


@dataclass(frozen=True)
class DependencyCallSeedData:
    """Parameters for seeding a DependencyCallRow."""

    library: str
    service_name: str
    qualname: str
    rel_path: str = "src/services/api.py"
    module: str = "services.api"
    callsite_count: int = 1
    modes: tuple[str, ...] = ("read",)


@dataclass(frozen=True)
class DependencyAggregateSeedData:
    """Parameters for seeding a DependencyAggregateRow."""

    library: str
    service_name: str
    category: str | None = "database"
    language: str = "python"
    severity: str | None = "medium"
    criticality: float | None = 0.5
    risk_score: float | None = 0.75
    function_count: int = 3
    callsite_count: int = 10
    risk_level: str = "moderate"


def _make_dependency_call_row(
    seed: DependencyCallSeedData,
    goid: int = TEST_GOID_12345,
) -> DependencyCallRow:
    """
    Create a DependencyCallRow for testing.

    Parameters
    ----------
    seed
        Seed data for the row.
    goid
        Global object ID.

    Returns
    -------
    DependencyCallRow
        A test DependencyCallRow.
    """
    dep_id = compute_dep_id(DEMO_REPO, DEMO_COMMIT, seed.library)
    return DependencyCallRow(
        repo=DEMO_REPO,
        commit=DEMO_COMMIT,
        dep_id=dep_id,
        library=seed.library,
        service_name=seed.service_name,
        function_goid_h128=to_decimal(goid),
        function_urn=f"urn:demo:repo::{seed.qualname}",
        rel_path=seed.rel_path,
        module=seed.module,
        qualname=seed.qualname,
        callsite_count=seed.callsite_count,
        modes=list(seed.modes),
        evidence_json=[{"type": "call", "line": 42}],
        created_at=datetime.now(tz=UTC),
    )


def _make_dependency_aggregate_row(
    seed: DependencyAggregateSeedData,
) -> DependencyAggregateRow:
    """
    Create a DependencyAggregateRow for testing.

    Parameters
    ----------
    seed
        Seed data for the row.

    Returns
    -------
    DependencyAggregateRow
        A test DependencyAggregateRow.
    """
    dep_id = compute_dep_id(DEMO_REPO, DEMO_COMMIT, seed.library)
    return DependencyAggregateRow(
        repo=DEMO_REPO,
        commit=DEMO_COMMIT,
        dep_id=dep_id,
        library=seed.library,
        service_name=seed.service_name,
        category=seed.category,
        language=seed.language,
        severity=seed.severity,
        criticality=seed.criticality,
        risk_score=seed.risk_score,
        function_count=seed.function_count,
        callsite_count=seed.callsite_count,
        modules_json=["services.api", "services.db"],
        usage_modes=["read", "write"],
        config_keys=["DATABASE_URL"],
        risk_level=seed.risk_level,
        created_at=datetime.now(tz=UTC),
    )


# =============================================================================
# Fixtures
# =============================================================================


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
# Helper Function Tests
# =============================================================================


def test_compute_dep_id_deterministic() -> None:
    """Compute dependency ID is deterministic for same inputs."""
    dep_id_1 = compute_dep_id(DEMO_REPO, DEMO_COMMIT, "requests")
    dep_id_2 = compute_dep_id(DEMO_REPO, DEMO_COMMIT, "requests")
    assert dep_id_1 == dep_id_2


def test_compute_dep_id_length() -> None:
    """Compute dependency ID returns 16-character hex string."""
    dep_id = compute_dep_id(DEMO_REPO, DEMO_COMMIT, "sqlalchemy")
    assert len(dep_id) == DEP_ID_LENGTH
    # Verify it's a valid hex string
    int(dep_id, 16)


def test_compute_dep_id_unique_per_library() -> None:
    """Compute dependency ID is unique per library."""
    dep_id_requests = compute_dep_id(DEMO_REPO, DEMO_COMMIT, "requests")
    dep_id_sqlalchemy = compute_dep_id(DEMO_REPO, DEMO_COMMIT, "sqlalchemy")
    assert dep_id_requests != dep_id_sqlalchemy


def test_compute_dep_id_unique_per_repo() -> None:
    """Compute dependency ID is unique per repository."""
    dep_id_1 = compute_dep_id("repo/a", DEMO_COMMIT, "requests")
    dep_id_2 = compute_dep_id("repo/b", DEMO_COMMIT, "requests")
    assert dep_id_1 != dep_id_2


def test_compute_dep_id_unique_per_commit() -> None:
    """Compute dependency ID is unique per commit."""
    dep_id_1 = compute_dep_id(DEMO_REPO, "commit1", "requests")
    dep_id_2 = compute_dep_id(DEMO_REPO, "commit2", "requests")
    assert dep_id_1 != dep_id_2


def test_to_decimal_converts_int() -> None:
    """Convert integer to Decimal."""
    result = to_decimal(TEST_GOID_12345)
    assert isinstance(result, Decimal)
    assert result == Decimal(TEST_GOID_12345)


def test_to_decimal_handles_zero() -> None:
    """Convert zero to Decimal."""
    result = to_decimal(0)
    assert result == Decimal(0)


def test_to_decimal_handles_large_int() -> None:
    """Convert large integer to Decimal (for hugeint support)."""
    large_int = 2**127 - 1
    result = to_decimal(large_int)
    assert result == Decimal(large_int)


# =============================================================================
# DependencyCallRow Tests
# =============================================================================


def test_dependency_call_row_creation() -> None:
    """Create DependencyCallRow with all required fields."""
    seed = DependencyCallSeedData(
        library="requests",
        service_name="HTTP Client",
        qualname="services.api.fetch_data",
    )
    row = _make_dependency_call_row(seed)

    assert row.repo == DEMO_REPO
    assert row.commit == DEMO_COMMIT
    assert row.library == "requests"
    assert row.service_name == "HTTP Client"
    assert row.qualname == "services.api.fetch_data"


def test_dependency_call_row_goid_is_decimal() -> None:
    """DependencyCallRow goid is stored as Decimal."""
    seed = DependencyCallSeedData(
        library="requests",
        service_name="HTTP Client",
        qualname="services.api.fetch_data",
    )
    row = _make_dependency_call_row(seed, goid=TEST_GOID_67890)

    assert isinstance(row.function_goid_h128, Decimal)
    assert row.function_goid_h128 == Decimal(TEST_GOID_67890)


def test_dependency_call_row_modes_is_list() -> None:
    """DependencyCallRow modes is a list of strings."""
    seed = DependencyCallSeedData(
        library="redis",
        service_name="Redis Cache",
        qualname="cache.client.get",
        modes=("read", "write", "delete"),
    )
    row = _make_dependency_call_row(seed)

    assert isinstance(row.modes, list)
    assert "read" in row.modes
    assert "write" in row.modes
    assert "delete" in row.modes


def test_dependency_call_row_is_frozen() -> None:
    """DependencyCallRow is immutable."""
    seed = DependencyCallSeedData(
        library="requests",
        service_name="HTTP Client",
        qualname="api.fetch",
    )
    row = _make_dependency_call_row(seed)

    assert_frozen(row, "library", "other")


# =============================================================================
# DependencyAggregateRow Tests
# =============================================================================


def test_dependency_aggregate_row_creation() -> None:
    """Create DependencyAggregateRow with all required fields."""
    seed = DependencyAggregateSeedData(
        library="sqlalchemy",
        service_name="SQL Database",
        category="database",
        risk_level="high",
    )
    row = _make_dependency_aggregate_row(seed)

    assert row.repo == DEMO_REPO
    assert row.commit == DEMO_COMMIT
    assert row.library == "sqlalchemy"
    assert row.service_name == "SQL Database"
    assert row.category == "database"
    assert row.risk_level == "high"


def test_dependency_aggregate_row_optional_fields() -> None:
    """DependencyAggregateRow handles optional fields."""
    seed = DependencyAggregateSeedData(
        library="custom_lib",
        service_name="Custom Service",
        category=None,
        severity=None,
        criticality=None,
        risk_score=None,
    )
    row = _make_dependency_aggregate_row(seed)

    assert row.category is None
    assert row.severity is None
    assert row.criticality is None
    assert row.risk_score is None


def test_dependency_aggregate_row_numeric_fields() -> None:
    """DependencyAggregateRow numeric fields have correct types."""
    seed = DependencyAggregateSeedData(
        library="redis",
        service_name="Redis Cache",
        criticality=CRITICALITY_0_5,
        risk_score=RISK_SCORE_0_75,
        function_count=FUNCTION_COUNT_3,
        callsite_count=CALLSITE_COUNT_10,
    )
    row = _make_dependency_aggregate_row(seed)

    assert row.criticality == CRITICALITY_0_5
    assert row.risk_score == RISK_SCORE_0_75
    assert row.function_count == FUNCTION_COUNT_3
    assert row.callsite_count == CALLSITE_COUNT_10


def test_dependency_aggregate_row_list_fields() -> None:
    """DependencyAggregateRow list fields are populated."""
    seed = DependencyAggregateSeedData(
        library="requests",
        service_name="HTTP Client",
    )
    row = _make_dependency_aggregate_row(seed)

    assert isinstance(row.modules_json, list)
    assert isinstance(row.usage_modes, list)
    assert isinstance(row.config_keys, list)


def test_dependency_aggregate_row_is_frozen() -> None:
    """DependencyAggregateRow is immutable."""
    seed = DependencyAggregateSeedData(
        library="requests",
        service_name="HTTP Client",
    )
    row = _make_dependency_aggregate_row(seed)

    assert_frozen(row, "library", "other")


# =============================================================================
# DependencyCallAdapter Tests
# =============================================================================


def test_call_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)
    assert adapter.table_name == "analytics.external_dependency_calls"


def test_call_adapter_load_returns_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load returns empty iterator (write-only adapter)."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)
    rows = list(adapter.load())
    assert not rows


def test_call_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    assert count == EXPECTED_COUNT_0


def test_call_adapter_persist_single_row(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single row inserts to database."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)
    seed = DependencyCallSeedData(
        library="requests",
        service_name="HTTP Client",
        qualname="api.fetch_data",
        callsite_count=CALLSITE_COUNT_5,
    )
    row = _make_dependency_call_row(seed)

    count = adapter.persist([row])
    assert count == EXPECTED_COUNT_1

    # Verify row was inserted
    result = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.external_dependency_calls WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()
    assert result is not None
    assert result[0] == EXPECTED_COUNT_1


def test_call_adapter_persist_multiple_rows(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple rows inserts all to database."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)

    rows = [
        _make_dependency_call_row(
            DependencyCallSeedData(
                library="requests",
                service_name="HTTP Client",
                qualname="api.get_user",
            ),
            goid=1001,
        ),
        _make_dependency_call_row(
            DependencyCallSeedData(
                library="requests",
                service_name="HTTP Client",
                qualname="api.get_orders",
            ),
            goid=1002,
        ),
        _make_dependency_call_row(
            DependencyCallSeedData(
                library="sqlalchemy",
                service_name="SQL Database",
                qualname="db.query_users",
            ),
            goid=1003,
        ),
    ]

    count = adapter.persist(rows)
    assert count == EXPECTED_COUNT_3

    # Verify rows were inserted
    result = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.external_dependency_calls WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()
    assert result is not None
    assert result[0] == EXPECTED_COUNT_3


def test_call_adapter_persist_verifies_data(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = DependencyCallAdapter(fresh_gateway, snapshot)
    seed = DependencyCallSeedData(
        library="redis",
        service_name="Redis Cache",
        qualname="cache.client.get_value",
        callsite_count=CALLSITE_COUNT_5,
        modes=("read",),
    )
    row = _make_dependency_call_row(seed)
    adapter.persist([row])

    # Query and verify
    result = fresh_gateway.con.execute(
        """
        SELECT library, service_name, qualname, callsite_count
        FROM analytics.external_dependency_calls
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == "redis"
    assert result[1] == "Redis Cache"
    assert result[2] == "cache.client.get_value"
    assert result[3] == CALLSITE_COUNT_5


# =============================================================================
# DependencyAggregateAdapter Tests
# =============================================================================


def test_aggregate_adapter_table_name(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    assert adapter.table_name == "analytics.external_dependencies"


def test_aggregate_adapter_load_returns_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load returns empty iterator (write-only adapter)."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    rows = list(adapter.load())
    assert not rows


def test_aggregate_adapter_persist_empty(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    count = adapter.persist([])
    assert count == EXPECTED_COUNT_0


def test_aggregate_adapter_persist_single_row(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist single row inserts to database."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    seed = DependencyAggregateSeedData(
        library="requests",
        service_name="HTTP Client",
        category="http",
        risk_level="low",
    )
    row = _make_dependency_aggregate_row(seed)

    count = adapter.persist([row])
    assert count == EXPECTED_COUNT_1

    # Verify row was inserted
    result = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.external_dependencies WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()
    assert result is not None
    assert result[0] == EXPECTED_COUNT_1


def test_aggregate_adapter_persist_multiple_rows(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist multiple rows inserts all to database."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)

    rows = [
        _make_dependency_aggregate_row(
            DependencyAggregateSeedData(
                library="requests",
                service_name="HTTP Client",
                category="http",
            )
        ),
        _make_dependency_aggregate_row(
            DependencyAggregateSeedData(
                library="sqlalchemy",
                service_name="SQL Database",
                category="database",
            )
        ),
    ]

    count = adapter.persist(rows)
    assert count == EXPECTED_COUNT_2

    # Verify rows were inserted
    result = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.external_dependencies WHERE repo = ? AND commit = ?",
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()
    assert result is not None
    assert result[0] == EXPECTED_COUNT_2


def test_aggregate_adapter_persist_verifies_data(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persisted data can be retrieved and verified."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    seed = DependencyAggregateSeedData(
        library="sqlalchemy",
        service_name="SQL Database",
        category="database",
        severity="high",
        risk_level="critical",
        function_count=FUNCTION_COUNT_3,
        callsite_count=CALLSITE_COUNT_10,
    )
    row = _make_dependency_aggregate_row(seed)
    adapter.persist([row])

    # Query and verify
    result = fresh_gateway.con.execute(
        """
        SELECT library, category, severity, risk_level, function_count, callsite_count
        FROM analytics.external_dependencies
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == "sqlalchemy"
    assert result[1] == "database"
    assert result[2] == "high"
    assert result[3] == "critical"
    assert result[4] == FUNCTION_COUNT_3
    assert result[5] == CALLSITE_COUNT_10


def test_aggregate_adapter_persist_with_null_fields(
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist row with null optional fields."""
    adapter = DependencyAggregateAdapter(fresh_gateway, snapshot)
    seed = DependencyAggregateSeedData(
        library="custom_lib",
        service_name="Custom Service",
        category=None,
        severity=None,
        criticality=None,
        risk_score=None,
    )
    row = _make_dependency_aggregate_row(seed)

    count = adapter.persist([row])
    assert count == EXPECTED_COUNT_1

    # Verify null fields
    result = fresh_gateway.con.execute(
        """
        SELECT category, severity, criticality, risk_score
        FROM analytics.external_dependencies
        WHERE repo = ? AND commit = ?
        """,
        [DEMO_REPO, DEMO_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] is None  # category
    assert result[1] is None  # severity
    assert result[2] is None  # criticality
    assert result[3] is None  # risk_score
