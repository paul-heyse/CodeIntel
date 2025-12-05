"""Unit tests for BasePluginExecutor infrastructure.

Test the unified base executor infrastructure, verifying:
- BaseExecutionPolicy functionality
- BaseExecutorContext properties
- BaseExecutionReport metrics
- Plugin plan and scratch utilities
- Telemetry span lifecycle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeintel.analytics.core.executor import PluginExecutor
from codeintel.config.primitives import SnapshotRef
from codeintel.core.execution.context import RunContext
from codeintel.core.execution.telemetry import RuntimeTelemetry, TelemetryConfig
from codeintel.core.plugins.execution.context import PluginExecutionContext, PluginScratch
from codeintel.core.plugins.execution.executor_context import BaseExecutorContext
from codeintel.core.plugins.execution.policy import BaseExecutionPolicy
from codeintel.core.plugins.types.protocol import (
    PluginKind,
    PluginMetadata,
    PluginSeverity,
    PluginStage,
    ValidationResult,
)
from codeintel.core.plugins.types.report import BaseExecutionReport
from codeintel.core.plugins.types.result import PluginExecutionRecord, PluginResult
from codeintel.graphs.runtime.executor import GraphExecutorContext
from tests._helpers.factories import make_snapshot
from tests._helpers.gateway import gateway_with_macros

# ---------------------------------------------------------------------------
# Constants for test assertions
# ---------------------------------------------------------------------------

EXPECTED_DEFAULT_RETRY_BACKOFF_MS = 100
EXPECTED_DEFAULT_MAX_WORKERS = 4
EXPECTED_CUSTOM_MAX_RETRIES = 3
EXPECTED_CUSTOM_BACKOFF_MS = 500
EXPECTED_POLICY_RETRIES = 5
EXPECTED_TWO_PLUGINS = 2
EXPECTED_ROW_COUNT = 10
EXPECTED_BACKOFF_MS = 200

# Error messages for mock exceptions
_SIMULATED_ERROR_MSG = "Simulated error"


class SimulatedPluginError(RuntimeError):
    """Error raised by mock plugin to simulate failures."""

    def __init__(self) -> None:
        """Initialize with standard error message."""
        super().__init__(_SIMULATED_ERROR_MSG)


# ---------------------------------------------------------------------------
# Mock implementations for testing
# ---------------------------------------------------------------------------


@dataclass
class MockPluginMetadata:
    """Mock plugin metadata for testing.

    Attributes
    ----------
    name
        Plugin identifier.
    description
        Human-readable description.
    kind
        Plugin kind classification.
    stage
        Pipeline stage.
    severity
        Failure severity.
    """

    name: str
    description: str = "Test plugin"
    kind: PluginKind = "analytics"
    stage: PluginStage = "function"
    severity: PluginSeverity = "fatal"


@dataclass
class MockPlugin:
    """Mock plugin implementation for testing.

    Attributes
    ----------
    _metadata
        Plugin metadata.
    should_succeed
        Whether execute should return success.
    should_skip
        Whether execute should return skipped.
    should_raise
        Whether execute should raise an exception.
    validate_valid
        Whether validate_inputs should return valid.
    execute_count
        Counter for execute calls.
    """

    _metadata: MockPluginMetadata
    should_succeed: bool = True
    should_skip: bool = False
    should_raise: bool = False
    validate_valid: bool = True
    execute_count: int = field(default=0, init=False)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        PluginMetadata
            The plugin's metadata.
        """
        return PluginMetadata(
            name=self._metadata.name,
            description=self._metadata.description,
            kind=self._metadata.kind,
            stage=self._metadata.stage,
            severity=self._metadata.severity,
        )

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        """Execute the mock plugin.

        Parameters
        ----------
        ctx
            Execution context (unused in mock).

        Returns
        -------
        PluginResult
            Mock execution result.

        Raises
        ------
        SimulatedPluginError
            When should_raise is True.
        """
        _ = ctx  # Unused in mock
        self.execute_count += 1
        if self.should_raise:
            raise SimulatedPluginError
        if self.should_skip:
            return PluginResult(success=True, skipped=True)
        return PluginResult(
            success=self.should_succeed,
            error="Simulated failure" if not self.should_succeed else None,
            row_counts={"test_table": EXPECTED_ROW_COUNT} if self.should_succeed else {},
        )

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate plugin inputs.

        Parameters
        ----------
        ctx
            Execution context (unused in mock).

        Returns
        -------
        ValidationResult
            Validation result based on validate_valid flag.
        """
        _ = ctx  # Unused in mock
        if self.validate_valid:
            return ValidationResult.success()
        return ValidationResult.failure(("Validation failed",))


@dataclass
class MockExecutorContext(BaseExecutorContext):
    """Mock executor context for testing.

    Attributes
    ----------
    extra_field
        Additional test field.
    """

    extra_field: str = "test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_snapshot(tmp_path: Path) -> SnapshotRef:
    """Create a mock snapshot reference.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.

    Returns
    -------
    SnapshotRef
        Test snapshot reference.
    """
    return make_snapshot(repo="test/repo", commit="abc123", repo_root=tmp_path)


@pytest.fixture
def mock_executor_ctx(mock_snapshot: SnapshotRef) -> MockExecutorContext:
    """Create a mock executor context.

    Parameters
    ----------
    mock_snapshot
        Test snapshot.

    Returns
    -------
    MockExecutorContext
        Test executor context.
    """
    gw = gateway_with_macros()
    return MockExecutorContext(
        gateway=gw,
        snapshot=mock_snapshot,
        extra_field="test-context",
    )


@pytest.fixture
def mock_telemetry() -> RuntimeTelemetry:
    """Create a test telemetry instance with metrics disabled.

    Returns
    -------
    RuntimeTelemetry
        Test telemetry instance.
    """
    config = TelemetryConfig(
        service_name="test",
        enable_tracing=False,
        enable_metrics=False,
    )
    return RuntimeTelemetry(config)


# ---------------------------------------------------------------------------
# Test BaseExecutionPolicy
# ---------------------------------------------------------------------------


def test_base_execution_policy_default_values() -> None:
    """Test default policy values."""
    policy = BaseExecutionPolicy()
    assert policy.fail_fast is True
    assert policy.max_retries == 0
    assert policy.retry_backoff_ms == EXPECTED_DEFAULT_RETRY_BACKOFF_MS
    assert policy.skip_on_unchanged is False
    assert policy.dry_run is False
    assert policy.enable_parallel is False
    assert policy.max_workers == EXPECTED_DEFAULT_MAX_WORKERS
    assert policy.timeout_ms is None
    assert policy.validate_contracts is False


def test_base_execution_policy_custom_values() -> None:
    """Test custom policy values."""
    policy = BaseExecutionPolicy(
        fail_fast=False,
        max_retries=EXPECTED_CUSTOM_MAX_RETRIES,
        retry_backoff_ms=EXPECTED_CUSTOM_BACKOFF_MS,
        dry_run=True,
    )
    assert policy.fail_fast is False
    assert policy.max_retries == EXPECTED_CUSTOM_MAX_RETRIES
    assert policy.retry_backoff_ms == EXPECTED_CUSTOM_BACKOFF_MS
    assert policy.dry_run is True


def test_base_execution_policy_to_retry_policy() -> None:
    """Test conversion to tenacity RetryPolicy."""
    policy = BaseExecutionPolicy(
        max_retries=EXPECTED_CUSTOM_MAX_RETRIES,
        retry_backoff_ms=EXPECTED_BACKOFF_MS,
    )
    retry_policy = policy.to_retry_policy()
    # max_attempts = max_retries + 1 (initial attempt + retries)
    expected_attempts = EXPECTED_CUSTOM_MAX_RETRIES + 1
    assert retry_policy.max_attempts == expected_attempts


def test_base_execution_policy_to_retry_policy_no_retries() -> None:
    """Test conversion with no retries configured."""
    policy = BaseExecutionPolicy(max_retries=0)
    retry_policy = policy.to_retry_policy()
    # With 0 retries, max_attempts = 1 (just the initial attempt)
    assert retry_policy.max_attempts == 1


# ---------------------------------------------------------------------------
# Test BaseExecutorContext
# ---------------------------------------------------------------------------


def test_base_executor_context_effective_run_id_from_run_context(
    mock_snapshot: SnapshotRef,
) -> None:
    """Test effective_run_id returns run_context.run_id when available.

    Parameters
    ----------
    mock_snapshot
        Test snapshot.
    """
    mock_gw = MagicMock()
    run_ctx = RunContext(
        run_id="run-from-context",
        kind="full",
        snapshot=mock_snapshot,
        trigger="cli",
    )
    ctx = BaseExecutorContext(
        gateway=mock_gw,
        snapshot=mock_snapshot,
        run_context=run_ctx,
    )
    assert ctx.effective_run_id == "run-from-context"


def test_base_executor_context_effective_run_id_empty_when_no_context(
    mock_snapshot: SnapshotRef,
) -> None:
    """Test effective_run_id returns empty string when no run_context.

    Parameters
    ----------
    mock_snapshot
        Test snapshot.
    """
    mock_gw = MagicMock()
    ctx = BaseExecutorContext(
        gateway=mock_gw,
        snapshot=mock_snapshot,
    )
    assert not ctx.effective_run_id


# ---------------------------------------------------------------------------
# Test BaseExecutionReport
# ---------------------------------------------------------------------------


def test_base_execution_report_success_count() -> None:
    """Test success_count property."""
    now = datetime.now(tz=UTC)
    duration_ms = 100.0
    total_duration_ms = 300.0
    records = [
        PluginExecutionRecord(
            plugin_name="p1",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=duration_ms,
        ),
        PluginExecutionRecord(
            plugin_name="p2",
            status="failed",
            started_at=now,
            ended_at=now,
            duration_ms=duration_ms,
        ),
        PluginExecutionRecord(
            plugin_name="p3",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=duration_ms,
        ),
    ]
    report = BaseExecutionReport(
        run_id="test",
        started_at=now,
        ended_at=now,
        duration_ms=total_duration_ms,
        records=tuple(records),
    )
    assert report.success_count == EXPECTED_TWO_PLUGINS
    assert report.failure_count == 1
    assert report.skip_count == 0


def test_base_execution_report_status_succeeded() -> None:
    """Test status property returns 'succeeded' when all succeed."""
    now = datetime.now(tz=UTC)
    duration_ms = 100.0
    records = [
        PluginExecutionRecord(
            plugin_name="p1",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=duration_ms,
        ),
    ]
    report = BaseExecutionReport(
        run_id="test",
        started_at=now,
        ended_at=now,
        duration_ms=duration_ms,
        records=tuple(records),
    )
    assert report.status == "succeeded"


def test_base_execution_report_status_failed() -> None:
    """Test status property returns 'failed' when fatal_error is True."""
    now = datetime.now(tz=UTC)
    duration_ms = 100.0
    report = BaseExecutionReport(
        run_id="test",
        started_at=now,
        ended_at=now,
        duration_ms=duration_ms,
        records=(),
        fatal_error=True,
    )
    assert report.status == "failed"


def test_base_execution_report_status_partial() -> None:
    """Test status property returns 'partial' with skipped plugins."""
    now = datetime.now(tz=UTC)
    duration_ms = 100.0
    total_duration_ms = 200.0
    # "partial" status is for runs with skips (not failures)
    records = [
        PluginExecutionRecord(
            plugin_name="p1",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=duration_ms,
        ),
        PluginExecutionRecord(
            plugin_name="p2",
            status="skipped",
            started_at=now,
            ended_at=now,
            duration_ms=duration_ms,
        ),
    ]
    report = BaseExecutionReport(
        run_id="test",
        started_at=now,
        ended_at=now,
        duration_ms=total_duration_ms,
        records=tuple(records),
    )
    assert report.status == "partial"


def test_base_execution_report_status_failed_with_failures() -> None:
    """Test status property returns 'failed' when there are failures."""
    now = datetime.now(tz=UTC)
    duration_ms = 100.0
    total_duration_ms = 200.0
    records = [
        PluginExecutionRecord(
            plugin_name="p1",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=duration_ms,
        ),
        PluginExecutionRecord(
            plugin_name="p2",
            status="failed",
            started_at=now,
            ended_at=now,
            duration_ms=duration_ms,
        ),
    ]
    report = BaseExecutionReport(
        run_id="test",
        started_at=now,
        ended_at=now,
        duration_ms=total_duration_ms,
        records=tuple(records),
    )
    assert report.status == "failed"


# ---------------------------------------------------------------------------
# Test BasePluginExecutor via real implementations
#
# Since BasePluginExecutor is an ABC with complex abstract methods,
# we test it through the real executor implementations to ensure
# the infrastructure works correctly end-to-end.
# ---------------------------------------------------------------------------


def test_analytics_executor_uses_base_infrastructure() -> None:
    """Verify analytics executor properly extends base infrastructure."""
    executor = PluginExecutor()
    # Verify base properties are accessible
    assert executor.policy is not None
    assert executor.telemetry is not None


def test_graphs_executor_uses_base_infrastructure() -> None:
    """Verify graphs execution uses base infrastructure."""
    # Verify the executor context extends base
    assert issubclass(GraphExecutorContext, BaseExecutorContext)


# ---------------------------------------------------------------------------
# Test plugin scratch utilities
# ---------------------------------------------------------------------------


def test_plugin_scratch_declare_and_consume() -> None:
    """Test that plugin scratch stores and retrieves values."""
    scratch = PluginScratch()
    scratch.declare("test_key", {"data": 123})

    result = scratch.consume("test_key")
    assert result == {"data": 123}


def test_plugin_scratch_consume_missing_returns_default() -> None:
    """Test that consuming missing key returns default."""
    scratch = PluginScratch()
    result = scratch.consume("missing_key", default="default_value")
    assert result == "default_value"


def test_plugin_scratch_cleanup_runs_callbacks() -> None:
    """Test that cleanup runs registered callbacks."""
    callback_ran = {"value": False}

    def cleanup_callback() -> None:
        callback_ran["value"] = True

    scratch = PluginScratch()
    scratch.register_cleanup(cleanup_callback)
    scratch.cleanup()

    assert callback_ran["value"] is True


# ---------------------------------------------------------------------------
# Test telemetry integration
# ---------------------------------------------------------------------------


def test_telemetry_span_lifecycle(mock_telemetry: RuntimeTelemetry) -> None:
    """Test telemetry span start and end.

    Parameters
    ----------
    mock_telemetry
        Test telemetry instance.
    """
    span = mock_telemetry.start_span("test.plugin", "run-123")
    assert span.plugin_name == "test.plugin"
    assert span.run_id == "run-123"

    duration = mock_telemetry.end_span(span, success=True, rows_written=100)
    assert duration >= 0


def test_telemetry_record_run_metrics(mock_telemetry: RuntimeTelemetry) -> None:
    """Test telemetry run metrics recording.

    Parameters
    ----------
    mock_telemetry
        Test telemetry instance.
    """
    # This should not raise
    mock_telemetry.record_run_metrics(
        run_id="test-run",
        success_count=5,
        failure_count=1,
        skip_count=0,
        duration_s=1.5,
    )


__all__ = [
    "MockExecutorContext",
    "MockPlugin",
    "MockPluginMetadata",
    "SimulatedPluginError",
]
