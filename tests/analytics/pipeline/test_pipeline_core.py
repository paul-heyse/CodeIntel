"""Tests for analytics pipeline core modules.

This module provides comprehensive tests for the pipeline subsystem:
- contracts.py: Dataset contract validation
- lineage.py: Dataset lineage tracking
- protocol.py: Pipeline protocol definitions
- scheduler.py: DAG-based execution scheduling
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from codeintel.analytics.pipeline.contracts import (
    ColumnRule,
    ContractValidationResult,
    ContractViolation,
    DatasetContract,
    DatasetContractValidator,
)
from codeintel.analytics.pipeline.lineage import (
    DatasetLineage,
    LineageStore,
    compute_table_hash,
)
from codeintel.analytics.pipeline.protocol import (
    DatasetResult,
    DatasetSpec,
    PipelineContext,
    TableSchema,
)
from codeintel.analytics.pipeline.scheduler import (
    ExecutionPlan,
    ExecutionStep,
    PipelineReport,
    PipelineScheduler,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway, open_memory_gateway

TEST_TABLE_NAME = "analytics.test_metrics"
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
EXPECTED_COLUMN_COUNT = 3
SAMPLE_ROW_COUNT = 5
HASH_LENGTH = 16
MIN_ROWS_THRESHOLD = 100
MAX_ROWS_LIMIT = 1000
MAX_ROWS_LOW = 2
EXPECTED_INPUT_COUNT = 2
EXPECTED_ROW_COUNT_100 = 100
EXPECTED_ROW_COUNT_50 = 50
EXPECTED_ROW_COUNT_10 = 10
EXPECTED_ROW_COUNT_15 = 15
EXPECTED_ROW_COUNT_25 = 25
EXPECTED_ROW_COUNT_42 = 42
EXPECTED_DURATION_MS = 100.0
VIOLATION_ROW_COUNT = 5
EXPECTED_LINEAGE_COUNT = 3
SCHEMA_COLUMN_COUNT = 3
PLAN_LEVEL_COUNT = 3
PLAN_MAX_LEVEL = 2
TOTAL_PIPELINE_ROWS = 45


@pytest.fixture
def memory_gateway() -> Iterator[StorageGateway]:
    """Provide an in-memory DuckDB gateway for testing.

    Yields
    ------
    StorageGateway
        Configured gateway with schema applied.
    """
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    try:
        yield gateway
    finally:
        gateway.con.close()


@pytest.fixture
def sample_table(memory_gateway: StorageGateway) -> str:
    """Create a sample table with test data.

    Parameters
    ----------
    memory_gateway
        Gateway fixture.

    Returns
    -------
    str
        The created table name.
    """
    memory_gateway.con.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics.test_metrics (
            repo VARCHAR NOT NULL,
            commit VARCHAR NOT NULL,
            goid_h128 DECIMAL(38,0) NOT NULL,
            metric_value INTEGER,
            is_valid BOOLEAN DEFAULT TRUE,
            PRIMARY KEY (repo, commit, goid_h128)
        )
        """
    )

    for i in range(SAMPLE_ROW_COUNT):
        memory_gateway.con.execute(
            """
            INSERT INTO analytics.test_metrics (repo, commit, goid_h128, metric_value, is_valid)
            VALUES (?, ?, ?, ?, ?)
            """,
            [TEST_REPO, TEST_COMMIT, i + 1, (i + 1) * 10, True],
        )

    return TEST_TABLE_NAME


@pytest.fixture
def pipeline_context(memory_gateway: StorageGateway, tmp_path: Path) -> PipelineContext:
    """Create a pipeline context for testing.

    Parameters
    ----------
    memory_gateway
        Gateway fixture.
    tmp_path
        Temporary directory.

    Returns
    -------
    PipelineContext
        Configured context.
    """
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=tmp_path)
    return PipelineContext(
        gateway=memory_gateway,
        snapshot=snapshot,
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(tz=UTC),
    )


def test_column_rule_creation_minimal() -> None:
    """Create a column rule with just the column name."""
    rule = ColumnRule(column="goid_h128")
    assert rule.column == "goid_h128"
    assert rule.not_null is False
    assert rule.unique is False
    assert rule.min_value is None
    assert rule.max_value is None


def test_column_rule_creation_full() -> None:
    """Create a column rule with all options."""
    max_value = 100.0
    rule = ColumnRule(
        column="metric_value",
        not_null=True,
        unique=False,
        min_value=0.0,
        max_value=max_value,
        pattern=r"^\d+$",
        allowed_values=frozenset({1, 2, 3}),
    )
    assert rule.not_null is True
    assert rule.min_value == 0.0
    assert rule.max_value == max_value
    assert rule.allowed_values == frozenset({1, 2, 3})


def test_column_rule_is_frozen() -> None:
    """Column rules should be immutable."""
    rule = ColumnRule(column="test")
    with pytest.raises(AttributeError):
        rule.column = "other"  # type: ignore[misc]


def test_contract_creation_minimal() -> None:
    """Create a contract with minimal fields."""
    contract = DatasetContract(table="analytics.metrics")
    assert contract.table == "analytics.metrics"
    assert contract.min_rows == 0
    assert contract.max_rows is None
    assert contract.required_columns == ()


def test_contract_creation_full() -> None:
    """Create a contract with all fields."""
    rule = ColumnRule(column="goid", not_null=True)
    contract = DatasetContract(
        table="analytics.metrics",
        min_rows=1,
        max_rows=MAX_ROWS_LIMIT,
        required_columns=("repo", "commit", "goid"),
        column_rules=(rule,),
        custom_checks=("SELECT 1",),
        description="Test contract",
    )
    assert contract.min_rows == 1
    assert contract.max_rows == MAX_ROWS_LIMIT
    assert len(contract.required_columns) == EXPECTED_COLUMN_COUNT
    assert len(contract.column_rules) == 1


def test_contract_is_frozen() -> None:
    """Contracts should be immutable."""
    contract = DatasetContract(table="test")
    with pytest.raises(AttributeError):
        contract.table = "other"  # type: ignore[misc]


def test_violation_creation() -> None:
    """Create a violation with all fields."""
    violation = ContractViolation(
        table="analytics.test",
        rule="min_rows",
        message="Expected 10 rows, found 5",
        severity="error",
        row_count=VIOLATION_ROW_COUNT,
    )
    assert violation.table == "analytics.test"
    assert violation.rule == "min_rows"
    assert violation.severity == "error"
    assert violation.row_count == VIOLATION_ROW_COUNT


def test_violation_default_severity() -> None:
    """Violations default to error severity."""
    violation = ContractViolation(
        table="test",
        rule="test_rule",
        message="Test message",
    )
    assert violation.severity == "error"


def test_result_creation_valid() -> None:
    """Create a valid result."""
    contract = DatasetContract(table="test")
    result = ContractValidationResult(
        contract=contract,
        valid=True,
        violations=[],
        row_count=EXPECTED_ROW_COUNT_10,
        duration_ms=5.5,
    )
    assert result.valid is True
    assert result.row_count == EXPECTED_ROW_COUNT_10
    assert len(result.violations) == 0


def test_result_creation_with_violations() -> None:
    """Create a result with violations."""
    contract = DatasetContract(table="test")
    violation = ContractViolation(
        table="test",
        rule="min_rows",
        message="Not enough rows",
    )
    result = ContractValidationResult(
        contract=contract,
        valid=False,
        violations=[violation],
        row_count=0,
    )
    assert result.valid is False
    assert len(result.violations) == 1


def test_validate_passes_with_sufficient_rows(
    memory_gateway: StorageGateway, sample_table: str
) -> None:
    """Validation passes when table has enough rows."""
    contract = DatasetContract(
        table=sample_table,
        min_rows=1,
        max_rows=MIN_ROWS_THRESHOLD,
    )
    validator = DatasetContractValidator(memory_gateway)
    result = validator.validate(contract, repo=TEST_REPO, commit=TEST_COMMIT)

    assert result.valid is True
    assert result.row_count == SAMPLE_ROW_COUNT
    assert len(result.violations) == 0


def test_validate_fails_min_rows(memory_gateway: StorageGateway, sample_table: str) -> None:
    """Validation fails when below min_rows."""
    contract = DatasetContract(
        table=sample_table,
        min_rows=MIN_ROWS_THRESHOLD,
    )
    validator = DatasetContractValidator(memory_gateway)
    result = validator.validate(contract, repo=TEST_REPO, commit=TEST_COMMIT)

    assert result.valid is False
    assert any(v.rule == "min_rows" for v in result.violations)


def test_validate_fails_max_rows(memory_gateway: StorageGateway, sample_table: str) -> None:
    """Validation fails when above max_rows."""
    contract = DatasetContract(
        table=sample_table,
        max_rows=MAX_ROWS_LOW,
    )
    validator = DatasetContractValidator(memory_gateway)
    result = validator.validate(contract, repo=TEST_REPO, commit=TEST_COMMIT)

    assert result.valid is False
    assert any(v.rule == "max_rows" for v in result.violations)


def test_validate_checks_required_columns(
    memory_gateway: StorageGateway, sample_table: str
) -> None:
    """Validation checks for required columns."""
    contract = DatasetContract(
        table=sample_table,
        required_columns=("repo", "commit", "nonexistent_column"),
    )
    validator = DatasetContractValidator(memory_gateway)
    result = validator.validate(contract)

    assert result.valid is False
    assert any(
        v.rule == "required_column" and "nonexistent_column" in v.message for v in result.violations
    )


def test_validate_checks_not_null_rule(memory_gateway: StorageGateway, sample_table: str) -> None:
    """Validation checks not_null column rule."""
    memory_gateway.con.execute(
        """
        INSERT INTO analytics.test_metrics (repo, commit, goid_h128, metric_value, is_valid)
        VALUES (?, ?, ?, NULL, ?)
        """,
        [TEST_REPO, TEST_COMMIT, 999, True],
    )

    contract = DatasetContract(
        table=sample_table,
        column_rules=(ColumnRule(column="metric_value", not_null=True),),
    )
    validator = DatasetContractValidator(memory_gateway)
    result = validator.validate(contract, repo=TEST_REPO, commit=TEST_COMMIT)

    assert result.valid is False
    assert any(v.rule == "not_null" for v in result.violations)


def test_validate_records_duration(memory_gateway: StorageGateway, sample_table: str) -> None:
    """Validation records execution duration."""
    contract = DatasetContract(table=sample_table)
    validator = DatasetContractValidator(memory_gateway)
    result = validator.validate(contract)

    assert result.duration_ms > 0


def test_validate_custom_check_passes(memory_gateway: StorageGateway, sample_table: str) -> None:
    """Validate custom SQL check that passes."""
    contract = DatasetContract(
        table=sample_table,
        custom_checks=("SELECT 1",),
    )
    validator = DatasetContractValidator(memory_gateway)
    result = validator.validate(contract)

    assert result.valid is True


def test_validate_custom_check_fails(memory_gateway: StorageGateway, sample_table: str) -> None:
    """Validate custom SQL check that fails."""
    contract = DatasetContract(
        table=sample_table,
        custom_checks=("SELECT NULL",),
    )
    validator = DatasetContractValidator(memory_gateway)
    result = validator.validate(contract)

    assert result.valid is False
    assert any(v.rule == "custom_check" for v in result.violations)


def test_lineage_creation() -> None:
    """Create a lineage record with all fields."""
    now = datetime.now(tz=UTC)
    lineage = DatasetLineage(
        dataset="analytics.function_metrics",
        run_id="run-123",
        input_datasets=("core.goids", "graph.call_graph_edges"),
        input_hashes=("hash1", "hash2"),
        output_hash="output_hash",
        row_count=EXPECTED_ROW_COUNT_100,
        computed_at=now,
        duration_ms=500.0,
        version="1.0.0",
        metadata={"key": "value"},
    )
    assert lineage.dataset == "analytics.function_metrics"
    assert len(lineage.input_datasets) == EXPECTED_INPUT_COUNT
    assert lineage.row_count == EXPECTED_ROW_COUNT_100
    assert lineage.computed_at == now


def test_lineage_to_dict() -> None:
    """Convert lineage to dictionary."""
    now = datetime.now(tz=UTC)
    lineage = DatasetLineage(
        dataset="test.dataset",
        run_id="run-456",
        input_datasets=("input1",),
        input_hashes=("hash1",),
        output_hash="output",
        row_count=EXPECTED_ROW_COUNT_50,
        computed_at=now,
    )
    data = lineage.to_dict()

    assert data["dataset"] == "test.dataset"
    assert data["run_id"] == "run-456"
    assert data["row_count"] == EXPECTED_ROW_COUNT_50
    assert isinstance(data["computed_at"], str)


def test_lineage_from_dict() -> None:
    """Create lineage from dictionary."""
    data: dict[str, object] = {
        "dataset": "test.dataset",
        "run_id": "run-789",
        "input_datasets": ["input1", "input2"],
        "input_hashes": ["h1", "h2"],
        "output_hash": "out",
        "row_count": 25,
        "computed_at": "2024-01-01T12:00:00+00:00",
        "duration_ms": 100.0,
        "version": "2.0.0",
        "metadata": {"extra": "data"},
    }
    lineage = DatasetLineage.from_dict(data)

    assert lineage.dataset == "test.dataset"
    assert lineage.input_datasets == ("input1", "input2")
    assert lineage.version == "2.0.0"


def test_lineage_round_trip() -> None:
    """Ensure lineage survives to_dict and from_dict round trip."""
    original = DatasetLineage(
        dataset="round.trip",
        run_id="run-trip",
        input_datasets=("a", "b"),
        input_hashes=("ha", "hb"),
        output_hash="hout",
        row_count=EXPECTED_ROW_COUNT_42,
        computed_at=datetime.now(tz=UTC),
        duration_ms=123.4,
        version="3.0.0",
        metadata={"test": True},
    )
    restored = DatasetLineage.from_dict(original.to_dict())

    assert restored.dataset == original.dataset
    assert restored.run_id == original.run_id
    assert restored.row_count == original.row_count
    assert restored.version == original.version


def test_hash_returns_string(memory_gateway: StorageGateway, sample_table: str) -> None:
    """Verify hash computation returns a hex string."""
    result = compute_table_hash(
        memory_gateway,
        sample_table,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
    )
    assert isinstance(result, str)
    assert len(result) == HASH_LENGTH


def test_hash_stable_for_same_data(memory_gateway: StorageGateway, sample_table: str) -> None:
    """Verify same data produces same hash."""
    hash1 = compute_table_hash(memory_gateway, sample_table, repo=TEST_REPO, commit=TEST_COMMIT)
    hash2 = compute_table_hash(memory_gateway, sample_table, repo=TEST_REPO, commit=TEST_COMMIT)
    assert hash1 == hash2


def test_hash_changes_with_data(memory_gateway: StorageGateway, sample_table: str) -> None:
    """Verify different data produces different hash."""
    hash_before = compute_table_hash(
        memory_gateway, sample_table, repo=TEST_REPO, commit=TEST_COMMIT
    )

    memory_gateway.con.execute(
        """
        INSERT INTO analytics.test_metrics (repo, commit, goid_h128, metric_value)
        VALUES (?, ?, ?, ?)
        """,
        [TEST_REPO, TEST_COMMIT, 1000, 9999],
    )

    hash_after = compute_table_hash(
        memory_gateway, sample_table, repo=TEST_REPO, commit=TEST_COMMIT
    )
    assert hash_before != hash_after


def test_hash_handles_nonexistent_table(memory_gateway: StorageGateway) -> None:
    """Verify hash returns 'error' for nonexistent table."""
    result = compute_table_hash(memory_gateway, "nonexistent.table")
    assert result == "error"


def test_store_initialization(memory_gateway: StorageGateway) -> None:
    """Verify store initializes and creates table."""
    store = LineageStore(memory_gateway)
    assert store is not None

    result = memory_gateway.con.execute("SELECT COUNT(*) FROM analytics.dataset_lineage").fetchone()
    assert result is not None


def test_store_record_and_get_latest(memory_gateway: StorageGateway) -> None:
    """Verify store can record and retrieve lineage."""
    store = LineageStore(memory_gateway)
    lineage = DatasetLineage(
        dataset="test.dataset",
        run_id="run-store-test",
        input_datasets=("input1",),
        input_hashes=("hash1",),
        output_hash="output_hash",
        row_count=EXPECTED_ROW_COUNT_10,
        computed_at=datetime.now(tz=UTC),
    )

    store.record(lineage)
    retrieved = store.get_latest("test.dataset")

    assert retrieved is not None
    assert retrieved.dataset == "test.dataset"
    assert retrieved.run_id == "run-store-test"
    assert retrieved.row_count == EXPECTED_ROW_COUNT_10


def test_store_get_latest_returns_newest(memory_gateway: StorageGateway) -> None:
    """Verify get_latest returns most recent record."""
    store = LineageStore(memory_gateway)

    older = DatasetLineage(
        dataset="version.test",
        run_id="run-1",
        input_datasets=(),
        input_hashes=(),
        output_hash="old",
        row_count=VIOLATION_ROW_COUNT,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    store.record(older)

    newer = DatasetLineage(
        dataset="version.test",
        run_id="run-2",
        input_datasets=(),
        input_hashes=(),
        output_hash="new",
        row_count=EXPECTED_ROW_COUNT_15,
        computed_at=datetime(2024, 6, 1, tzinfo=UTC),
    )
    store.record(newer)

    latest = store.get_latest("version.test")
    assert latest is not None
    assert latest.run_id == "run-2"
    assert latest.row_count == EXPECTED_ROW_COUNT_15


def test_store_get_latest_returns_none_for_missing(memory_gateway: StorageGateway) -> None:
    """Verify get_latest returns None for non-existent dataset."""
    store = LineageStore(memory_gateway)
    result = store.get_latest("nonexistent.dataset")
    assert result is None


def test_store_get_by_run(memory_gateway: StorageGateway) -> None:
    """Verify get_by_run returns all lineage for a run."""
    store = LineageStore(memory_gateway)
    run_id = "run-multi-dataset"

    for i in range(EXPECTED_LINEAGE_COUNT):
        lineage = DatasetLineage(
            dataset=f"dataset.{i}",
            run_id=run_id,
            input_datasets=(),
            input_hashes=(),
            output_hash=f"hash_{i}",
            row_count=i * 10,
            computed_at=datetime.now(tz=UTC),
        )
        store.record(lineage)

    results = store.get_by_run(run_id)
    assert len(results) == EXPECTED_LINEAGE_COUNT
    datasets = {r.dataset for r in results}
    assert datasets == {"dataset.0", "dataset.1", "dataset.2"}


def test_store_needs_recompute_true_when_missing(memory_gateway: StorageGateway) -> None:
    """Verify needs_recompute returns True for missing dataset."""
    store = LineageStore(memory_gateway)
    result = store.needs_recompute("missing.dataset", {"input": "hash"})
    assert result is True


def test_store_needs_recompute_false_when_unchanged(memory_gateway: StorageGateway) -> None:
    """Verify needs_recompute returns False when hashes match."""
    store = LineageStore(memory_gateway)

    lineage = DatasetLineage(
        dataset="unchanged.dataset",
        run_id="run-unchanged",
        input_datasets=("input1", "input2"),
        input_hashes=("hash1", "hash2"),
        output_hash="out",
        row_count=VIOLATION_ROW_COUNT,
        computed_at=datetime.now(tz=UTC),
    )
    store.record(lineage)

    result = store.needs_recompute(
        "unchanged.dataset",
        {"input1": "hash1", "input2": "hash2"},
    )
    assert result is False


def test_store_needs_recompute_true_when_hash_changed(memory_gateway: StorageGateway) -> None:
    """Verify needs_recompute returns True when input hash changed."""
    store = LineageStore(memory_gateway)

    lineage = DatasetLineage(
        dataset="changed.dataset",
        run_id="run-changed",
        input_datasets=("input1",),
        input_hashes=("old_hash",),
        output_hash="out",
        row_count=VIOLATION_ROW_COUNT,
        computed_at=datetime.now(tz=UTC),
    )
    store.record(lineage)

    result = store.needs_recompute("changed.dataset", {"input1": "new_hash"})
    assert result is True


def test_schema_creation() -> None:
    """Create a table schema."""
    schema = TableSchema(
        name="analytics.metrics",
        columns=(
            ("repo", "VARCHAR", False),
            ("commit", "VARCHAR", False),
            ("value", "INTEGER", True),
        ),
        primary_key=("repo", "commit"),
        indexes=(("value",),),
    )
    assert schema.name == "analytics.metrics"
    assert len(schema.columns) == SCHEMA_COLUMN_COUNT
    assert schema.primary_key == ("repo", "commit")


def test_schema_column_names_property() -> None:
    """Verify column_names property returns column names."""
    schema = TableSchema(
        name="test",
        columns=(
            ("col1", "INT", False),
            ("col2", "VARCHAR", True),
        ),
    )
    assert schema.column_names == ("col1", "col2")


def test_schema_is_frozen() -> None:
    """Verify schema is immutable."""
    schema = TableSchema(name="test")
    with pytest.raises(AttributeError):
        schema.name = "other"  # type: ignore[misc]


def test_spec_creation_minimal() -> None:
    """Create a spec with minimal fields."""
    spec: DatasetSpec[dict[str, Any]] = DatasetSpec(name="test.dataset")
    assert spec.name == "test.dataset"
    assert spec.inputs == ()
    assert spec.outputs == ()
    assert spec.version == "1.0.0"


def test_spec_creation_full() -> None:
    """Create a spec with all fields."""
    schema = TableSchema(name="analytics.test")
    contract = DatasetContract(table="analytics.test")

    spec: DatasetSpec[dict[str, Any]] = DatasetSpec(
        name="analytics.test",
        description="A test dataset",
        row_type=dict,
        schema=schema,
        inputs=("core.goids",),
        outputs=("analytics.test",),
        contract=contract,
        version="2.0.0",
        tags=("analytics", "test"),
    )
    assert spec.description == "A test dataset"
    assert spec.inputs == ("core.goids",)
    assert spec.version == "2.0.0"
    assert "analytics" in spec.tags


def test_spec_primary_output_property() -> None:
    """Verify primary_output returns first output or name."""
    spec_with_outputs: DatasetSpec[dict[str, Any]] = DatasetSpec(
        name="test",
        outputs=("output1", "output2"),
    )
    assert spec_with_outputs.primary_output == "output1"

    spec_without_outputs: DatasetSpec[dict[str, Any]] = DatasetSpec(name="test")
    assert spec_without_outputs.primary_output == "test"


def test_context_creation(memory_gateway: StorageGateway, tmp_path: Path) -> None:
    """Create a pipeline context."""
    snapshot = SnapshotRef(repo="test/repo", commit="abc", repo_root=tmp_path)
    ctx = PipelineContext(
        gateway=memory_gateway,
        snapshot=snapshot,
        run_id="run-123",
    )
    assert ctx.gateway is memory_gateway
    assert ctx.run_id == "run-123"


def test_context_repo_property(memory_gateway: StorageGateway, tmp_path: Path) -> None:
    """Verify repo property returns snapshot repo."""
    snapshot = SnapshotRef(repo="my/repo", commit="xyz", repo_root=tmp_path)
    ctx = PipelineContext(
        gateway=memory_gateway,
        snapshot=snapshot,
        run_id="run",
    )
    assert ctx.repo == "my/repo"


def test_context_commit_property(memory_gateway: StorageGateway, tmp_path: Path) -> None:
    """Verify commit property returns snapshot commit."""
    snapshot = SnapshotRef(repo="repo", commit="commit123", repo_root=tmp_path)
    ctx = PipelineContext(
        gateway=memory_gateway,
        snapshot=snapshot,
        run_id="run",
    )
    assert ctx.commit == "commit123"


def test_context_extra_field(memory_gateway: StorageGateway, tmp_path: Path) -> None:
    """Verify context can store extra data."""
    snapshot = SnapshotRef(repo="repo", commit="commit", repo_root=tmp_path)
    ctx = PipelineContext(
        gateway=memory_gateway,
        snapshot=snapshot,
        run_id="run",
        extra={"custom": "data"},
    )
    assert ctx.extra["custom"] == "data"


def test_result_success() -> None:
    """Create a successful result."""
    spec: DatasetSpec[dict[str, Any]] = DatasetSpec(name="test")
    result: DatasetResult[dict[str, Any]] = DatasetResult(
        spec=spec,
        row_count=EXPECTED_ROW_COUNT_100,
        duration_ms=50.0,
        success=True,
    )
    assert result.success is True
    assert result.row_count == EXPECTED_ROW_COUNT_100
    assert result.error is None


def test_result_failure() -> None:
    """Create a failed result."""
    spec: DatasetSpec[dict[str, Any]] = DatasetSpec(name="test")
    result: DatasetResult[dict[str, Any]] = DatasetResult(
        spec=spec,
        duration_ms=10.0,
        success=False,
        error="Computation failed",
    )
    assert result.success is False
    assert result.error == "Computation failed"


@dataclass
class SimpleRow:
    """Simple row type for test computations."""

    id: int
    value: str


class SimpleComputation:
    """A simple computation for testing."""

    def __init__(self, name: str, inputs: tuple[str, ...] = (), row_count: int = 10) -> None:
        """Initialize the computation.

        Parameters
        ----------
        name
            Dataset name.
        inputs
            Input dataset names.
        row_count
            Number of rows to generate.
        """
        self._spec: DatasetSpec[SimpleRow] = DatasetSpec(
            name=name,
            inputs=inputs,
            outputs=(name,),
        )
        self._row_count = row_count

    @property
    def spec(self) -> DatasetSpec[SimpleRow]:
        """Return the dataset specification."""
        return self._spec

    def compute(
        self,
        ctx: PipelineContext,
        inputs: dict[str, Any],
    ) -> Iterator[SimpleRow]:
        """Compute dataset rows.

        Parameters
        ----------
        ctx
            Pipeline context.
        inputs
            Input datasets.

        Yields
        ------
        SimpleRow
            Generated rows.
        """
        _ = ctx.run_id
        _ = inputs
        for i in range(self._row_count):
            yield SimpleRow(id=i, value=f"value_{i}")


class FailingComputation:
    """A computation that always fails."""

    def __init__(self, name: str) -> None:
        """Initialize the failing computation.

        Parameters
        ----------
        name
            Dataset name.
        """
        self._spec: DatasetSpec[SimpleRow] = DatasetSpec(
            name=name,
            outputs=(name,),
        )

    @property
    def spec(self) -> DatasetSpec[SimpleRow]:
        """Return the dataset specification."""
        return self._spec

    def compute(
        self,
        ctx: PipelineContext,
        inputs: dict[str, Any],
    ) -> Iterator[SimpleRow]:
        """Fail immediately.

        Parameters
        ----------
        ctx
            Pipeline context.
        inputs
            Input datasets.

        Raises
        ------
        RuntimeError
            Always raised.
        """
        _ = ctx.run_id
        _ = inputs
        message = f"Intentional failure for {self._spec.name}"
        raise RuntimeError(message)


def test_step_creation() -> None:
    """Create an execution step."""
    spec: DatasetSpec[SimpleRow] = DatasetSpec(name="test")
    computation = SimpleComputation("test")

    step = ExecutionStep(
        dataset="test",
        spec=spec,
        computation=computation,
        level=0,
        dependencies=(),
    )
    assert step.dataset == "test"
    assert step.level == 0


def test_plan_creation() -> None:
    """Create an execution plan."""
    plan = ExecutionPlan(target_datasets=("target1", "target2"))
    assert plan.target_datasets == ("target1", "target2")
    assert plan.total_datasets == 0


def test_plan_add_step() -> None:
    """Add steps to a plan."""
    plan = ExecutionPlan(target_datasets=("test",))
    spec: DatasetSpec[SimpleRow] = DatasetSpec(name="test")
    computation = SimpleComputation("test")

    step = ExecutionStep(
        dataset="test",
        spec=spec,
        computation=computation,
        level=0,
        dependencies=(),
    )
    plan.add_step(step)

    assert plan.total_datasets == 1
    assert len(plan.steps) == 1
    assert 0 in plan.levels


def test_plan_max_level() -> None:
    """Verify max_level returns highest level."""
    plan = ExecutionPlan(target_datasets=())

    for level in range(PLAN_LEVEL_COUNT):
        spec: DatasetSpec[SimpleRow] = DatasetSpec(name=f"ds_{level}")
        computation = SimpleComputation(f"ds_{level}")
        step = ExecutionStep(
            dataset=f"ds_{level}",
            spec=spec,
            computation=computation,
            level=level,
            dependencies=(),
        )
        plan.add_step(step)

    assert plan.max_level == PLAN_MAX_LEVEL


def test_plan_iter_by_level() -> None:
    """Verify iter_by_level returns steps in order."""
    plan = ExecutionPlan(target_datasets=())

    for level in [2, 0, 1]:
        spec: DatasetSpec[SimpleRow] = DatasetSpec(name=f"ds_{level}")
        computation = SimpleComputation(f"ds_{level}")
        step = ExecutionStep(
            dataset=f"ds_{level}",
            spec=spec,
            computation=computation,
            level=level,
            dependencies=(),
        )
        plan.add_step(step)

    levels = plan.iter_by_level()
    assert len(levels) == PLAN_LEVEL_COUNT
    assert levels[0][0].dataset == "ds_0"
    assert levels[1][0].dataset == "ds_1"
    assert levels[2][0].dataset == "ds_2"


def test_report_creation() -> None:
    """Create a pipeline report."""
    plan = ExecutionPlan(target_datasets=("test",))
    report = PipelineReport(run_id="run-123", plan=plan)

    assert report.run_id == "run-123"
    assert report.success is True
    assert report.total_rows == 0


def test_report_record_result() -> None:
    """Record results in report."""
    plan = ExecutionPlan(target_datasets=())
    report = PipelineReport(run_id="run", plan=plan)

    spec: DatasetSpec[SimpleRow] = DatasetSpec(name="test")
    result: DatasetResult[SimpleRow] = DatasetResult(
        spec=spec,
        row_count=EXPECTED_ROW_COUNT_50,
        duration_ms=EXPECTED_DURATION_MS,
        success=True,
    )

    report.record_result(result)

    assert report.total_rows == EXPECTED_ROW_COUNT_50
    assert report.total_duration_ms == EXPECTED_DURATION_MS
    assert "test" in report.results


def test_report_records_failure() -> None:
    """Verify report tracks failures."""
    plan = ExecutionPlan(target_datasets=())
    report = PipelineReport(run_id="run", plan=plan)

    spec: DatasetSpec[SimpleRow] = DatasetSpec(name="failed")
    result: DatasetResult[SimpleRow] = DatasetResult(
        spec=spec,
        success=False,
        error="Something went wrong",
    )

    report.record_result(result)

    assert report.success is False
    assert len(report.errors) == 1
    assert "failed" in report.errors[0]


def test_scheduler_creation() -> None:
    """Create a scheduler."""
    scheduler = PipelineScheduler()
    assert scheduler is not None


def test_scheduler_register() -> None:
    """Register a computation."""
    scheduler = PipelineScheduler()
    computation = SimpleComputation("test.dataset")

    scheduler.register(computation)
    plan = scheduler.plan(["test.dataset"])
    assert plan.total_datasets == 1


def test_scheduler_unregister() -> None:
    """Unregister a computation."""
    scheduler = PipelineScheduler()
    computation = SimpleComputation("test.dataset")

    scheduler.register(computation)
    scheduler.unregister("test.dataset")

    with pytest.raises(ValueError, match="not registered"):
        scheduler.plan(["test.dataset"])


def test_scheduler_plan_simple() -> None:
    """Plan execution for single dataset."""
    scheduler = PipelineScheduler()
    scheduler.register(SimpleComputation("dataset.a"))

    plan = scheduler.plan(["dataset.a"])

    assert plan.total_datasets == 1
    assert plan.target_datasets == ("dataset.a",)


def test_scheduler_plan_with_dependencies() -> None:
    """Verify plan includes transitive dependencies."""
    scheduler = PipelineScheduler()

    scheduler.register(SimpleComputation("dataset.a"))
    scheduler.register(SimpleComputation("dataset.b", inputs=("dataset.a",)))
    scheduler.register(SimpleComputation("dataset.c", inputs=("dataset.b",)))

    plan = scheduler.plan(["dataset.c"])

    assert plan.total_datasets == PLAN_LEVEL_COUNT
    assert plan.max_level == PLAN_MAX_LEVEL

    level_0 = [s.dataset for s in plan.levels[0]]
    level_1 = [s.dataset for s in plan.levels[1]]
    level_2 = [s.dataset for s in plan.levels[2]]

    assert "dataset.a" in level_0
    assert "dataset.b" in level_1
    assert "dataset.c" in level_2


def test_scheduler_plan_without_dependencies() -> None:
    """Verify plan can exclude dependencies."""
    scheduler = PipelineScheduler()
    scheduler.register(SimpleComputation("dataset.a"))
    scheduler.register(SimpleComputation("dataset.b", inputs=("dataset.a",)))

    plan = scheduler.plan(["dataset.b"], include_dependencies=False)

    assert plan.total_datasets == 1


def test_scheduler_plan_missing_dataset_raises() -> None:
    """Verify planning for unregistered dataset raises error."""
    scheduler = PipelineScheduler()

    with pytest.raises(ValueError, match="not registered"):
        scheduler.plan(["nonexistent.dataset"])


def test_scheduler_execute_simple(pipeline_context: PipelineContext) -> None:
    """Execute a simple computation."""
    scheduler = PipelineScheduler()
    scheduler.register(SimpleComputation("simple.dataset", row_count=EXPECTED_ROW_COUNT_25))

    plan = scheduler.plan(["simple.dataset"])
    report = scheduler.execute(plan, pipeline_context)

    assert report.success is True
    assert report.total_rows == EXPECTED_ROW_COUNT_25
    assert "simple.dataset" in report.results
    assert report.results["simple.dataset"].row_count == EXPECTED_ROW_COUNT_25


def test_scheduler_execute_with_dependencies(pipeline_context: PipelineContext) -> None:
    """Execute datasets with dependencies."""
    scheduler = PipelineScheduler()
    scheduler.register(SimpleComputation("base", row_count=EXPECTED_ROW_COUNT_10))
    scheduler.register(
        SimpleComputation("derived", inputs=("base",), row_count=VIOLATION_ROW_COUNT)
    )

    plan = scheduler.plan(["derived"])
    report = scheduler.execute(plan, pipeline_context)

    assert report.success is True
    assert report.total_rows == EXPECTED_ROW_COUNT_15
    assert len(report.results) == EXPECTED_INPUT_COUNT


def test_scheduler_execute_fail_fast(pipeline_context: PipelineContext) -> None:
    """Verify execute stops on first failure with fail_fast=True."""
    scheduler = PipelineScheduler()
    scheduler.register(FailingComputation("failing"))
    scheduler.register(SimpleComputation("after_fail", inputs=("failing",)))

    plan = scheduler.plan(["after_fail"])
    report = scheduler.execute(plan, pipeline_context, fail_fast=True)

    assert report.success is False
    assert len(report.errors) >= 1
    assert report.results["after_fail"].success is False


def test_scheduler_execute_continue_on_failure(pipeline_context: PipelineContext) -> None:
    """Verify execute continues with fail_fast=False."""
    scheduler = PipelineScheduler()
    scheduler.register(SimpleComputation("good1", row_count=VIOLATION_ROW_COUNT))
    scheduler.register(FailingComputation("bad"))
    scheduler.register(SimpleComputation("good2", row_count=VIOLATION_ROW_COUNT))

    plan = scheduler.plan(["good1", "bad", "good2"], include_dependencies=False)
    report = scheduler.execute(plan, pipeline_context, fail_fast=False)

    assert report.success is False
    assert report.results["good1"].success is True
    assert report.results["good2"].success is True
    assert report.results["bad"].success is False


def test_scheduler_records_lineage(pipeline_context: PipelineContext) -> None:
    """Verify scheduler records lineage for successful computations."""
    scheduler = PipelineScheduler()
    scheduler.register(SimpleComputation("lineage.test", row_count=EXPECTED_ROW_COUNT_10))

    plan = scheduler.plan(["lineage.test"])
    scheduler.execute(plan, pipeline_context)

    store = LineageStore(pipeline_context.gateway)
    lineage = store.get_latest("lineage.test")

    assert lineage is not None
    assert lineage.dataset == "lineage.test"
    assert lineage.run_id == pipeline_context.run_id
    assert lineage.row_count == EXPECTED_ROW_COUNT_10


def test_scheduler_completed_at_set(pipeline_context: PipelineContext) -> None:
    """Verify report completed_at is set after execution."""
    scheduler = PipelineScheduler()
    scheduler.register(SimpleComputation("timing.test"))

    plan = scheduler.plan(["timing.test"])
    report = scheduler.execute(plan, pipeline_context)

    assert report.completed_at is not None
    assert report.completed_at > report.started_at


def test_full_pipeline_flow(pipeline_context: PipelineContext) -> None:
    """Test complete pipeline: register, plan, execute, validate."""
    scheduler = PipelineScheduler()

    scheduler.register(SimpleComputation("step1", row_count=20))
    scheduler.register(SimpleComputation("step2", inputs=("step1",), row_count=15))
    scheduler.register(SimpleComputation("step3", inputs=("step2",), row_count=10))

    plan = scheduler.plan(["step3"])
    assert plan.total_datasets == PLAN_LEVEL_COUNT
    assert plan.max_level == PLAN_MAX_LEVEL

    report = scheduler.execute(plan, pipeline_context)
    assert report.success is True
    assert report.total_rows == TOTAL_PIPELINE_ROWS

    store = LineageStore(pipeline_context.gateway)
    for dataset_name in ["step1", "step2", "step3"]:
        lineage = store.get_latest(dataset_name)
        assert lineage is not None
        assert lineage.run_id == pipeline_context.run_id


def test_contract_validation_after_execution(
    memory_gateway: StorageGateway,
    sample_table: str,
) -> None:
    """Verify contract validation works after pipeline execution."""
    contract = DatasetContract(
        table=sample_table,
        min_rows=1,
        required_columns=("repo", "commit", "goid_h128"),
    )

    validator = DatasetContractValidator(memory_gateway)
    result = validator.validate(contract, repo=TEST_REPO, commit=TEST_COMMIT)

    assert result.valid is True
    assert result.row_count == SAMPLE_ROW_COUNT
