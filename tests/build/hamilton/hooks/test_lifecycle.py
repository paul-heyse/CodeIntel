"""Tests for Hamilton lifecycle hooks."""

from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING

import pytest
from hamilton.node import Node

from codeintel.build.hamilton.hooks.lifecycle import (
    BuildTimingHook,
    ConditionalHook,
    NodeTimingRecord,
    ProgressBarHook,
    create_progress_hook,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_is_instance,
    expect_is_none,
    expect_length,
    expect_true,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def _make_test_node(name: str) -> Node:
    """Create a minimal Hamilton Node for hook unit tests.

    Notes
    -----
    Hamilton Node construction requires a non-None callable in recent
    Hamilton releases. These unit tests only need a ``Node`` instance with a
    stable name, so we construct it via ``Node.from_fn`` using a trivial
    function.

    Parameters
    ----------
    name
        Node name.

    Returns
    -------
    Node
        Hamilton Node instance with the given name.
    """

    def _fn() -> str:
        return ""

    return Node.from_fn(_fn, name=name)


@contextlib.contextmanager
def _temporary_env(values: dict[str, str | None]) -> Iterator[None]:
    saved: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield None
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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

    @staticmethod
    def test_creation_with_desc() -> None:
        """Test creating progress bar with description."""
        hook = ProgressBarHook(desc="Test Progress")
        expect_equal(hook.desc, "Test Progress")

    @staticmethod
    def test_run_before_disabled() -> None:
        """Test run_before returns None when disabled."""
        hook = ProgressBarHook(disable=True)
        node = _make_test_node("test")
        result = hook.pre_node_execute(node_=node)
        expect_is_none(result)

    @staticmethod
    def test_run_after_disabled() -> None:
        """Test run_after returns None when disabled."""
        hook = ProgressBarHook(disable=True)
        node = _make_test_node("test")
        result = hook.post_node_execute(node_=node, success=True, error=None)
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
        node = _make_test_node("test_node")

        # Simulate before execution
        hook.pre_node_execute(node_=node)

        # Small delay
        time.sleep(0.01)

        # Simulate after execution
        hook.post_node_execute(node_=node, success=True, error=None)

        records = hook.get_records()
        expect_length(records, 1)
        expect_equal(records[0].node_name, "test_node")
        expect_true(records[0].duration_seconds > 0)

    @staticmethod
    def test_get_slowest_nodes() -> None:
        """Test getting slowest nodes."""
        perf_values = iter([0.0, 0.1, 1.0, 1.5, 2.0, 4.0])
        hook = BuildTimingHook(clock=lambda: next(perf_values))
        for node_name in ("fast", "medium", "slow"):
            node = _make_test_node(node_name)
            hook.pre_node_execute(node_=node)
            hook.post_node_execute(node_=node, success=True, error=None)

        slowest = hook.get_slowest_nodes(n=2)
        expect_length(slowest, 2)
        expect_equal(slowest[0].node_name, "slow")
        expect_equal(slowest[1].node_name, "medium")

    @staticmethod
    def test_total_duration() -> None:
        """Test calculating total duration."""
        perf_values = iter([0.0, 1.0, 10.0, 12.0, 20.0, 23.0])
        hook = BuildTimingHook(clock=lambda: next(perf_values))
        for node_name in ("a", "b", "c"):
            node = _make_test_node(node_name)
            hook.pre_node_execute(node_=node)
            hook.post_node_execute(node_=node, success=True, error=None)

        total = hook.total_duration()
        expect_equal(total, 6.0)

    @staticmethod
    def test_reset() -> None:
        """Test resetting timing records."""
        hook = BuildTimingHook()
        node = _make_test_node("test")
        hook.pre_node_execute(node_=node)
        hook.post_node_execute(node_=node, success=True, error=None)
        hook.reset()

        expect_equal(hook.get_records(), [])
        hook.post_node_execute(node_=node, success=True, error=None)
        expect_equal(hook.get_records(), [])


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

        node = _make_test_node("test")
        conditional.pre_node_execute(node_=node)
        conditional.post_node_execute(node_=node, success=True, error=None)

        expect_length(inner.get_records(), 1)

    @staticmethod
    def test_disabled_condition() -> None:
        """Test hook doesn't execute when condition is False."""
        inner = BuildTimingHook()
        conditional = ConditionalHook(
            hook=inner,
            condition=lambda: False,
        )

        node = _make_test_node("test")
        conditional.pre_node_execute(node_=node)
        conditional.post_node_execute(node_=node, success=True, error=None)

        expect_length(inner.get_records(), 0)

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

        node = _make_test_node("test")
        for _ in range(5):
            conditional.pre_node_execute(node_=node)

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
    def test_respects_ci_environment() -> None:
        """Test factory disables in CI environment."""
        with _temporary_env({"CI": "true"}):
            hook = create_progress_hook("Test", disable_in_ci=True)
            expect_true(hook.disable)

    @staticmethod
    def test_enabled_when_not_ci() -> None:
        """Test factory enables when not in CI."""
        with _temporary_env({"CI": None}):
            hook = create_progress_hook("Test", disable_in_ci=True)
            expect_false(hook.disable)


@pytest.mark.parametrize(
    ("min_duration", "actual_duration"),
    [
        pytest.param(1.0, 0.5, id="below_threshold"),
        pytest.param(1.0, 1.5, id="above_threshold"),
        pytest.param(0.0, 0.001, id="zero_threshold"),
    ],
)
def test_timing_hook_logging_threshold(
    min_duration: float,
    actual_duration: float,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Parametrized test for timing hook logging threshold."""
    caplog.set_level("INFO", logger="codeintel.build.hamilton.hooks.lifecycle")
    perf_values = iter([0.0, actual_duration])
    hook = BuildTimingHook(min_duration_to_log=min_duration, clock=lambda: next(perf_values))

    node = _make_test_node("test_node")
    hook.pre_node_execute(node_=node)
    hook.post_node_execute(node_=node, success=True, error=None)

    records = hook.get_records()
    expect_length(records, 1)
    expect_equal(hook.min_duration_to_log, min_duration)
    expect_true(records[0].duration_seconds >= 0)

    should_log = actual_duration >= min_duration
    logged = any("Node test_node took" in record.getMessage() for record in caplog.records)
    expect_equal(logged, should_log)
