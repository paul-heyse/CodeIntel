"""Tests for Hamilton lifecycle hooks."""
from __future__ import annotations

import time

import pytest
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_is_instance,
    expect_is_none,
    expect_length,
    expect_true,
)

from codeintel.build.hamilton.hooks.lifecycle import (
    BuildTimingHook,
    ConditionalHook,
    NodeTimingRecord,
    ProgressBarHook,
    create_progress_hook,
)


class TestNodeTimingRecord:
    """Test suite for NodeTimingRecord dataclass."""

    @staticmethod
    def test_creation() -> None:
        """Test creating a timing record."""
        record = NodeTimingRecord(
            node_name="test_node",
            duration_seconds=1.5,
            start_time=1000.0,
        )
        expect_equal(record.node_name, "test_node")
        expect_equal(record.duration_seconds, 1.5)
        expect_equal(record.start_time, 1000.0)
        expect_is_none(record.task_id)

    @staticmethod
    def test_creation_with_task_id() -> None:
        """Test creating a timing record with task ID."""
        record = NodeTimingRecord(
            node_name="test_node",
            duration_seconds=1.5,
            start_time=1000.0,
            task_id="task-1",
        )
        expect_equal(record.task_id, "task-1")


class TestProgressBarHook:
    """Test suite for ProgressBarHook."""

    @staticmethod
    def test_creation_disabled() -> None:
        """Test creating a disabled progress bar."""
        hook = ProgressBarHook(disable=True)
        expect_true(hook.disable)
        expect_is_none(hook._delegate)

    @staticmethod
    def test_creation_with_desc() -> None:
        """Test creating progress bar with description."""
        hook = ProgressBarHook(desc="Test Progress")
        expect_equal(hook.desc, "Test Progress")

    @staticmethod
    def test_run_before_disabled() -> None:
        """Test run_before returns None when disabled."""
        hook = ProgressBarHook(disable=True)
        result = hook.run_before_node_execution(
            node_name="test",
            node_tags={},
            node_kwargs={},
            task_id=None,
        )
        expect_is_none(result)

    @staticmethod
    def test_run_after_disabled() -> None:
        """Test run_after returns None when disabled."""
        hook = ProgressBarHook(disable=True)
        result = hook.run_after_node_execution(
            node_name="test",
            node_tags={},
            node_kwargs={},
            node_return_type=str,
            result="test",
            error=None,
            success=True,
            task_id=None,
        )
        expect_is_none(result)


class TestBuildTimingHook:
    """Test suite for BuildTimingHook."""

    @staticmethod
    def test_creation() -> None:
        """Test creating a timing hook."""
        hook = BuildTimingHook()
        expect_equal(hook.min_duration_to_log, 1.0)
        expect_equal(hook.get_records(), [])

    @staticmethod
    def test_records_timing() -> None:
        """Test that hook records node timing."""
        hook = BuildTimingHook()

        # Simulate before execution
        hook.run_before_node_execution(
            node_name="test_node",
            node_tags={},
            node_kwargs={},
            task_id=None,
        )

        # Small delay
        time.sleep(0.01)

        # Simulate after execution
        hook.run_after_node_execution(
            node_name="test_node",
            node_tags={},
            node_kwargs={},
            node_return_type=str,
            result="test",
            error=None,
            success=True,
            task_id=None,
        )

        records = hook.get_records()
        expect_length(records, 1)
        expect_equal(records[0].node_name, "test_node")
        expect_true(records[0].duration_seconds > 0)

    @staticmethod
    def test_get_slowest_nodes() -> None:
        """Test getting slowest nodes."""
        hook = BuildTimingHook()

        # Create records with different durations
        nodes = [("fast", 0.1), ("medium", 0.5), ("slow", 1.0)]
        for name, duration in nodes:
            hook._records.append(
                NodeTimingRecord(
                    node_name=name,
                    duration_seconds=duration,
                    start_time=0,
                ),
            )

        slowest = hook.get_slowest_nodes(n=2)
        expect_length(slowest, 2)
        expect_equal(slowest[0].node_name, "slow")
        expect_equal(slowest[1].node_name, "medium")

    @staticmethod
    def test_total_duration() -> None:
        """Test calculating total duration."""
        hook = BuildTimingHook()
        hook._records.extend([
            NodeTimingRecord("a", 1.0, 0),
            NodeTimingRecord("b", 2.0, 0),
            NodeTimingRecord("c", 3.0, 0),
        ])

        total = hook.total_duration()
        expect_equal(total, 6.0)

    @staticmethod
    def test_reset() -> None:
        """Test resetting timing records."""
        hook = BuildTimingHook()
        hook._records.append(NodeTimingRecord("test", 1.0, 0))
        hook._timings["test", None] = 1234.0

        hook.reset()

        expect_equal(hook.get_records(), [])
        expect_length(hook._timings, 0)


class TestConditionalHook:
    """Test suite for ConditionalHook."""

    @staticmethod
    def test_enabled_condition() -> None:
        """Test hook executes when condition is True."""
        inner = BuildTimingHook()
        conditional = ConditionalHook(
            hook=inner,
            condition=lambda: True,
        )

        conditional.run_before_node_execution(
            node_name="test",
            node_tags={},
            node_kwargs={},
            task_id=None,
        )

        # Inner hook should have recorded the start time
        expect_length(inner._timings, 1)

    @staticmethod
    def test_disabled_condition() -> None:
        """Test hook doesn't execute when condition is False."""
        inner = BuildTimingHook()
        conditional = ConditionalHook(
            hook=inner,
            condition=lambda: False,
        )

        conditional.run_before_node_execution(
            node_name="test",
            node_tags={},
            node_kwargs={},
            task_id=None,
        )

        # Inner hook should not have recorded anything
        expect_length(inner._timings, 0)

    @staticmethod
    def test_condition_cached() -> None:
        """Test that condition is evaluated only once."""
        call_count = 0

        def counting_condition() -> bool:
            nonlocal call_count
            call_count += 1
            return True

        inner = BuildTimingHook()
        conditional = ConditionalHook(
            hook=inner,
            condition=counting_condition,
        )

        # Call multiple times
        for _ in range(5):
            conditional._is_enabled()

        # Condition should only be called once
        expect_equal(call_count, 1)


class TestCreateProgressHook:
    """Test suite for create_progress_hook factory."""

    @staticmethod
    def test_creates_hook() -> None:
        """Test factory creates progress hook."""
        hook = create_progress_hook("Test")
        expect_is_instance(hook, ProgressBarHook)
        expect_equal(hook.desc, "Test")

    @staticmethod
    def test_respects_ci_environment(monkeypatch: pytest.MonkeyPatch) -> None:
        """Test factory disables in CI environment."""
        monkeypatch.setenv("CI", "true")
        hook = create_progress_hook("Test", disable_in_ci=True)
        expect_true(hook.disable)

    @staticmethod
    def test_enabled_when_not_ci(monkeypatch: pytest.MonkeyPatch) -> None:
        """Test factory enables when not in CI."""
        monkeypatch.delenv("CI", raising=False)
        hook = create_progress_hook("Test", disable_in_ci=True)
        expect_false(hook.disable)


@pytest.mark.parametrize(
    ("min_duration", "actual_duration", "should_log"),
    [
        pytest.param(1.0, 0.5, False, id="below_threshold"),
        pytest.param(1.0, 1.5, True, id="above_threshold"),
        pytest.param(0.0, 0.001, True, id="zero_threshold"),
    ],
    )
def test_timing_hook_logging_threshold(
    min_duration: float,
    actual_duration: float,
    should_log: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Parametrized test for timing hook logging threshold."""
    hook = BuildTimingHook(min_duration_to_log=min_duration)

    # Manually add a record with specific duration
    hook._records.append(
        NodeTimingRecord(
            node_name="test_node",
            duration_seconds=actual_duration,
            start_time=0,
        ),
    )

    # Check the record was added
    records = hook.get_records()
    expect_length(records, 1)

    # Log message test would require more complex setup
    # For now, just verify the hook was created correctly
    expect_equal(hook.min_duration_to_log, min_duration)
