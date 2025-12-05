"""Coverage tests for ingestion runtime executor.

This module provides comprehensive tests for the executor infrastructure,
including timeout handling, error classification, and batch execution.
All tests use the standard test helpers without monkeypatching.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

import pytest

from codeintel.core.execution.timing import utc_now
from codeintel.ingestion.core.base import BaseIngestPlugin
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.plugins.protocol import IngestPluginResult, IngestStage
from codeintel.ingestion.runtime.executor import (
    IngestExecutorConfig,
    IngestPluginExecutionRecord,
    IngestRunReport,
    PluginExecutionSettings,
    execute_plugin,
    execute_plugin_batch,
    execute_plugin_with_timeout,
)
from codeintel.ingestion.runtime.telemetry import get_ingest_telemetry
from tests._helpers.harnesses import IngestTestSetup

# =============================================================================
# Test Plugins
# =============================================================================


@dataclass
class SuccessPlugin(BaseIngestPlugin):
    """Plugin that always succeeds with row counts."""

    plugin_name: ClassVar[str] = "success_plugin"
    plugin_description: ClassVar[str] = "Plugin that succeeds"
    plugin_stage: ClassVar[IngestStage] = "parse"

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int]:
        """Return test row counts.

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).

        Returns
        -------
        Mapping[str, int]
            Row counts for test tables.
        """
        _ = self, ctx
        return {"core.test": 10, "core.other": 5}


@dataclass
class FailingPlugin(BaseIngestPlugin):
    """Plugin that always fails with a ValueError."""

    plugin_name: ClassVar[str] = "failing_plugin"
    plugin_description: ClassVar[str] = "Plugin that fails"
    plugin_stage: ClassVar[IngestStage] = "parse"

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int]:
        """Raise an error to test failure handling.

        Parameters
        ----------
        ctx
            Execution context (unused - exception raised immediately).

        Raises
        ------
        ValueError
            Always raised to test failure handling.
        """
        _ = self, ctx
        msg = "Intentional test failure"
        raise ValueError(msg)


@dataclass
class SlowPlugin(BaseIngestPlugin):
    """Plugin that takes time to execute for timeout testing."""

    plugin_name: ClassVar[str] = "slow_plugin"
    plugin_description: ClassVar[str] = "Plugin that is slow"
    plugin_stage: ClassVar[IngestStage] = "parse"
    delay_ms: int = 100

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int]:
        """Sleep briefly then return.

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).

        Returns
        -------
        Mapping[str, int]
            Row counts after sleeping.
        """
        _ = ctx
        time.sleep(self.delay_ms / 1000)
        return {"core.slow": 1}


@dataclass
class NoRowsPlugin(BaseIngestPlugin):
    """Plugin that succeeds but returns no rows."""

    plugin_name: ClassVar[str] = "no_rows_plugin"
    plugin_description: ClassVar[str] = "Plugin with no rows"
    plugin_stage: ClassVar[IngestStage] = "parse"

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        """Return None for no rows.

        Returns
        -------
        Mapping[str, int] | None
            None to indicate no rows produced.
        """
        _ = self, ctx
        return None


# =============================================================================
# IngestPluginExecutionRecord Tests
# =============================================================================


class TestIngestPluginExecutionRecordCoverage:
    """Coverage tests for IngestPluginExecutionRecord."""

    def test_success_with_result(self) -> None:
        """Record is successful when result present and no error."""
        record = IngestPluginExecutionRecord(
            plugin_name="test",
            started_at=utc_now(),
            result=IngestPluginResult.ok(row_counts={"t": 1}),
        )
        assert record.success is True
        assert record.status == "succeeded"

    def test_success_false_with_error(self) -> None:
        """Record is not successful when error present."""
        record = IngestPluginExecutionRecord(
            plugin_name="test",
            started_at=utc_now(),
            error=ValueError("test error"),
        )
        assert record.success is False
        assert record.status == "failed"

    def test_status_skipped_when_no_result_no_error(self) -> None:
        """Record status is skipped when no result and no error."""
        record = IngestPluginExecutionRecord(
            plugin_name="test",
            started_at=utc_now(),
        )
        assert record.success is False
        assert record.status == "skipped"

    def test_rows_written_and_table_counts(self) -> None:
        """Record tracks rows written and table counts."""
        record = IngestPluginExecutionRecord(
            plugin_name="test",
            started_at=utc_now(),
            result=IngestPluginResult.ok(),
            rows_written=15,
            table_counts={"core.a": 10, "core.b": 5},
        )
        assert record.rows_written == 15
        assert record.table_counts == {"core.a": 10, "core.b": 5}

    def test_duration_s_computed(self) -> None:
        """Duration is computed from timestamps."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 1, 12, 0, 5, tzinfo=UTC)
        record = IngestPluginExecutionRecord(
            plugin_name="test",
            started_at=start,
            ended_at=end,
            result=IngestPluginResult.ok(),
        )
        assert record.duration_s == pytest.approx(5.0, abs=0.01)


# =============================================================================
# IngestRunReport Tests
# =============================================================================


class TestIngestRunReportCoverage:
    """Coverage tests for IngestRunReport."""

    def test_duration_s_zero_when_not_ended(self) -> None:
        """Duration is zero when ended_at is None."""
        report = IngestRunReport(run_id="test", started_at=utc_now())
        assert report.duration_s == 0.0

    def test_duration_s_computed(self) -> None:
        """Duration is computed when ended_at is set."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 1, 12, 0, 10, tzinfo=UTC)
        report = IngestRunReport(run_id="test", started_at=start, ended_at=end)
        assert report.duration_s == pytest.approx(10.0, abs=0.01)

    def test_success_true_when_all_records_succeed(self) -> None:
        """Report is successful when all records succeed."""
        report = IngestRunReport(
            run_id="test",
            records=[
                IngestPluginExecutionRecord(
                    plugin_name="a",
                    started_at=utc_now(),
                    result=IngestPluginResult.ok(),
                ),
                IngestPluginExecutionRecord(
                    plugin_name="b",
                    started_at=utc_now(),
                    result=IngestPluginResult.ok(),
                ),
            ],
        )
        assert report.success is True

    def test_success_false_when_any_record_fails(self) -> None:
        """Report is not successful when any record fails."""
        report = IngestRunReport(
            run_id="test",
            records=[
                IngestPluginExecutionRecord(
                    plugin_name="a",
                    started_at=utc_now(),
                    result=IngestPluginResult.ok(),
                ),
                IngestPluginExecutionRecord(
                    plugin_name="b",
                    started_at=utc_now(),
                    error=ValueError("failed"),
                ),
            ],
        )
        assert report.success is False

    def test_total_rows_written(self) -> None:
        """Report sums rows written from all records."""
        report = IngestRunReport(
            run_id="test",
            records=[
                IngestPluginExecutionRecord(plugin_name="a", started_at=utc_now(), rows_written=10),
                IngestPluginExecutionRecord(plugin_name="b", started_at=utc_now(), rows_written=25),
            ],
        )
        assert report.total_rows_written == 35

    def test_get_table_counts_aggregates(self) -> None:
        """Report aggregates table counts from all records."""
        report = IngestRunReport(
            run_id="test",
            records=[
                IngestPluginExecutionRecord(
                    plugin_name="a",
                    started_at=utc_now(),
                    table_counts={"core.x": 5, "core.y": 3},
                ),
                IngestPluginExecutionRecord(
                    plugin_name="b",
                    started_at=utc_now(),
                    table_counts={"core.x": 2, "core.z": 7},
                ),
            ],
        )
        counts = report.get_table_counts()
        assert counts == {"core.x": 7, "core.y": 3, "core.z": 7}

    def test_empty_records(self) -> None:
        """Report handles empty records list."""
        report = IngestRunReport(run_id="test")
        assert report.success is True
        assert report.total_rows_written == 0
        assert report.get_table_counts() == {}


# =============================================================================
# PluginExecutionSettings Tests
# =============================================================================


class TestPluginExecutionSettings:
    """Tests for PluginExecutionSettings defaults and values."""

    def test_default_values(self) -> None:
        """Settings have sensible defaults."""
        settings = PluginExecutionSettings(name="test")
        assert settings.name == "test"
        assert settings.severity == "soft_fail"
        assert settings.timeout_s is None
        assert settings.fail_fast is True
        assert settings.max_retries == 0

    def test_custom_values(self) -> None:
        """Settings accept custom values."""
        settings = PluginExecutionSettings(
            name="custom",
            severity="fatal",
            timeout_s=30,
            fail_fast=False,
            max_retries=3,
        )
        assert settings.name == "custom"
        assert settings.severity == "fatal"
        assert settings.timeout_s == 30
        assert settings.fail_fast is False
        assert settings.max_retries == 3


# =============================================================================
# IngestExecutorConfig Tests
# =============================================================================


class TestIngestExecutorConfig:
    """Tests for IngestExecutorConfig defaults and values."""

    def test_default_values(self) -> None:
        """Config has sensible defaults."""
        config = IngestExecutorConfig()
        # Verify run_id is exactly an empty string (not None)
        assert isinstance(config.run_id, str)
        assert len(config.run_id) == 0
        assert config.enable_parallel is True
        assert config.max_workers == 4
        assert config.default_timeout_s is None
        assert config.telemetry is not None

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        telemetry = get_ingest_telemetry()
        config = IngestExecutorConfig(
            run_id="my-run",
            enable_parallel=False,
            max_workers=2,
            default_timeout_s=60,
            telemetry=telemetry,
        )
        assert config.run_id == "my-run"
        assert config.enable_parallel is False
        assert config.max_workers == 2
        assert config.default_timeout_s == 60
        assert config.telemetry is telemetry


# =============================================================================
# execute_plugin Tests
# =============================================================================


class TestExecutePlugin:
    """Tests for execute_plugin function."""

    def test_successful_execution(self, ingest_setup: IngestTestSetup) -> None:
        """Execute plugin that succeeds returns success record."""
        plugin = SuccessPlugin()
        ctx = ingest_setup.build_context("success_plugin")

        record = execute_plugin(plugin, ctx)

        assert record.success is True
        assert record.error is None
        assert record.result is not None
        assert record.rows_written == 15
        assert record.table_counts == {"core.test": 10, "core.other": 5}
        assert record.plugin_name == "success_plugin"
        assert record.ended_at is not None

    def test_failed_execution(self, ingest_setup: IngestTestSetup) -> None:
        """Execute plugin that fails returns record with failed result.

        Note: BaseIngestPlugin.execute() catches exceptions internally and
        converts them to IngestPluginResult.fail(), so record.error is None
        but result.success is False.
        """
        plugin = FailingPlugin()
        ctx = ingest_setup.build_context("failing_plugin")

        record = execute_plugin(plugin, ctx)

        # Plugin exceptions are caught by execute() and stored in result
        assert record.result is not None
        assert record.result.success is False
        assert record.result.error is not None
        assert "Intentional test failure" in record.result.error
        # record.error is None because the exception was handled
        assert record.error is None
        assert record.ended_at is not None

    def test_execution_with_custom_settings(self, ingest_setup: IngestTestSetup) -> None:
        """Execute plugin with custom settings."""
        plugin = SuccessPlugin()
        ctx = ingest_setup.build_context("success_plugin")
        settings = PluginExecutionSettings(
            name="custom_name",
            severity="fatal",
        )

        record = execute_plugin(plugin, ctx, settings=settings)

        assert record.success is True
        assert record.plugin_name == "success_plugin"

    def test_execution_with_no_rows(self, ingest_setup: IngestTestSetup) -> None:
        """Execute plugin that returns None for row counts."""
        plugin = NoRowsPlugin()
        ctx = ingest_setup.build_context("no_rows_plugin")

        record = execute_plugin(plugin, ctx)

        assert record.success is True
        assert record.rows_written == 0
        assert record.table_counts == {}


# =============================================================================
# execute_plugin_with_timeout Tests
# =============================================================================


class TestExecutePluginWithTimeout:
    """Tests for execute_plugin_with_timeout function."""

    def test_without_timeout(self, ingest_setup: IngestTestSetup) -> None:
        """Execute plugin without timeout delegates to execute_plugin."""
        plugin = SuccessPlugin()
        ctx = ingest_setup.build_context("success_plugin")

        record = execute_plugin_with_timeout(plugin, ctx, timeout_s=None)

        assert record.success is True
        assert record.rows_written == 15

    def test_with_timeout_completes_in_time(self, ingest_setup: IngestTestSetup) -> None:
        """Execute plugin that completes within timeout succeeds."""
        plugin = SlowPlugin(delay_ms=50)
        ctx = ingest_setup.build_context("slow_plugin")

        record = execute_plugin_with_timeout(plugin, ctx, timeout_s=5)

        assert record.success is True
        assert record.rows_written == 1

    def test_with_timeout_exceeds_limit(self, ingest_setup: IngestTestSetup) -> None:
        """Execute plugin that exceeds timeout returns timeout error."""
        plugin = SlowPlugin(delay_ms=2000)
        ctx = ingest_setup.build_context("slow_plugin")

        record = execute_plugin_with_timeout(plugin, ctx, timeout_s=1)

        assert record.success is False
        assert record.error is not None
        assert isinstance(record.error, TimeoutError)
        assert "timed out" in str(record.error).lower()

    def test_failing_plugin_with_timeout(self, ingest_setup: IngestTestSetup) -> None:
        """Execute failing plugin with timeout returns record with failed result.

        Note: BaseIngestPlugin.execute() catches exceptions internally and
        converts them to IngestPluginResult.fail().
        """
        plugin = FailingPlugin()
        ctx = ingest_setup.build_context("failing_plugin")

        record = execute_plugin_with_timeout(plugin, ctx, timeout_s=5)

        # Plugin exceptions are caught by execute() and stored in result
        assert record.result is not None
        assert record.result.success is False
        assert record.result.error is not None


# =============================================================================
# execute_plugin_batch Tests
# =============================================================================


class TestExecutePluginBatch:
    """Tests for execute_plugin_batch function."""

    def test_empty_batch(self, ingest_setup: IngestTestSetup) -> None:
        """Execute empty batch returns success report."""
        ctx = ingest_setup.build_context("test")

        report = execute_plugin_batch([], ctx)

        assert report.success is True
        assert report.status == "succeeded"
        assert len(report.records) == 0
        assert report.ended_at is not None

    def test_single_plugin_success(self, ingest_setup: IngestTestSetup) -> None:
        """Execute single successful plugin."""
        plugin = SuccessPlugin()
        ctx = ingest_setup.build_context("success_plugin")

        report = execute_plugin_batch([plugin], ctx)

        assert report.success is True
        assert report.status == "succeeded"
        assert len(report.records) == 1
        assert report.total_rows_written == 15

    def test_sequential_batch_multiple_plugins(self, ingest_setup: IngestTestSetup) -> None:
        """Execute multiple plugins sequentially."""
        plugins = [SuccessPlugin(), NoRowsPlugin()]
        ctx = ingest_setup.build_context("test")

        report = execute_plugin_batch(plugins, ctx, parallel=False)

        assert report.success is True
        assert len(report.records) == 2
        assert report.records[0].plugin_name == "success_plugin"
        assert report.records[1].plugin_name == "no_rows_plugin"

    def test_batch_with_failure(self, ingest_setup: IngestTestSetup) -> None:
        """Execute batch where one plugin fails.

        Note: Plugin failures are caught by execute() and returned as
        IngestPluginResult.fail(), so record.success is still True but
        result.success is False.
        """
        plugins = [SuccessPlugin(), FailingPlugin()]
        ctx = ingest_setup.build_context("test")

        report = execute_plugin_batch(plugins, ctx, parallel=False)

        # report.success checks all record.success (no uncaught exceptions)
        # so it's True even when plugins return failed results
        assert len(report.records) == 2
        assert report.records[0].success is True
        assert report.records[0].result is not None
        assert report.records[0].result.success is True
        # FailingPlugin returns a failed result but record.success is True
        assert report.records[1].success is True
        assert report.records[1].result is not None
        assert report.records[1].result.success is False

    def test_parallel_batch_execution(self, ingest_setup: IngestTestSetup) -> None:
        """Execute batch in parallel mode."""
        plugins = [SuccessPlugin(), NoRowsPlugin()]
        ctx = ingest_setup.build_context("test")
        config = IngestExecutorConfig(enable_parallel=True, max_workers=2)

        report = execute_plugin_batch(plugins, ctx, config=config, parallel=True)

        assert report.success is True
        assert len(report.records) == 2
        # Order may vary in parallel execution
        names = {r.plugin_name for r in report.records}
        assert names == {"success_plugin", "no_rows_plugin"}

    def test_parallel_disabled_falls_back_to_sequential(
        self, ingest_setup: IngestTestSetup
    ) -> None:
        """Parallel disabled in config falls back to sequential."""
        plugins = [SuccessPlugin(), NoRowsPlugin()]
        ctx = ingest_setup.build_context("test")
        config = IngestExecutorConfig(enable_parallel=False)

        report = execute_plugin_batch(plugins, ctx, config=config, parallel=True)

        assert report.success is True
        assert len(report.records) == 2
        # Sequential execution maintains order
        assert report.records[0].plugin_name == "success_plugin"

    def test_batch_with_timeout(self, ingest_setup: IngestTestSetup) -> None:
        """Execute batch with timeout configured."""
        plugins = [SuccessPlugin()]
        ctx = ingest_setup.build_context("test")
        config = IngestExecutorConfig(default_timeout_s=30)

        report = execute_plugin_batch(plugins, ctx, config=config)

        assert report.success is True
        assert len(report.records) == 1

    def test_batch_with_run_id(self, ingest_setup: IngestTestSetup) -> None:
        """Execute batch with custom run ID."""
        plugins = [SuccessPlugin()]
        ctx = ingest_setup.build_context("test")
        config = IngestExecutorConfig(run_id="custom-run-123")

        report = execute_plugin_batch(plugins, ctx, config=config)

        assert report.run_id == "custom-run-123"

    def test_batch_table_counts_aggregated(self, ingest_setup: IngestTestSetup) -> None:
        """Execute batch and verify table counts are aggregated correctly."""
        plugins = [SuccessPlugin(), SuccessPlugin()]
        ctx = ingest_setup.build_context("test")

        report = execute_plugin_batch(plugins, ctx, parallel=False)

        assert report.success is True
        counts = report.get_table_counts()
        # Each SuccessPlugin writes 10 to core.test and 5 to core.other
        assert counts["core.test"] == 20
        assert counts["core.other"] == 10
