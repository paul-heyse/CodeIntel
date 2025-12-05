"""Test error types from codeintel.core.execution.errors.

This module tests:
- PLUGIN_CATCHABLE_ERRORS tuple contents
- PluginFatalError with record and original exception
- PluginTimeoutError message formatting
- PluginSkippedError message formatting
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.core.execution.errors import (
    PLUGIN_CATCHABLE_ERRORS,
    PluginFatalError,
    PluginSkippedError,
    PluginTimeoutError,
)
from codeintel.core.plugins.types.result import PluginExecutionRecord

# =============================================================================
# PLUGIN_CATCHABLE_ERRORS Tests
# =============================================================================


def test_plugin_catchable_errors_is_tuple() -> None:
    """Verify PLUGIN_CATCHABLE_ERRORS is a tuple of exception types."""
    assert isinstance(PLUGIN_CATCHABLE_ERRORS, tuple)


def test_plugin_catchable_errors_contains_exception_types() -> None:
    """Verify PLUGIN_CATCHABLE_ERRORS contains only exception classes."""
    for exc_type in PLUGIN_CATCHABLE_ERRORS:
        assert issubclass(exc_type, Exception)


def test_plugin_catchable_errors_includes_common_exceptions() -> None:
    """Verify PLUGIN_CATCHABLE_ERRORS includes common exception types."""
    # These are the exceptions explicitly listed in errors.py
    expected = {
        AttributeError,
        LookupError,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
    }
    actual = set(PLUGIN_CATCHABLE_ERRORS)
    assert expected.issubset(actual)


def test_plugin_catchable_errors_can_catch() -> None:
    """Verify PLUGIN_CATCHABLE_ERRORS can be used in try/except."""
    caught = False
    try:
        msg = "test"
        raise ValueError(msg)
    except PLUGIN_CATCHABLE_ERRORS:
        caught = True

    assert caught


# =============================================================================
# PluginFatalError Tests
# =============================================================================


@pytest.fixture
def sample_record() -> PluginExecutionRecord:
    """Create a sample execution record for testing.

    Returns
    -------
    PluginExecutionRecord
        Record with failed status for testing.
    """
    now = datetime.now(UTC)
    return PluginExecutionRecord(
        plugin_name="test.plugin",
        status="failed",
        started_at=now,
        ended_at=now,
        duration_ms=100.0,
        error="Original error message",
    )


def test_plugin_fatal_error_construction(sample_record: PluginExecutionRecord) -> None:
    """Verify PluginFatalError can be constructed."""
    original = ValueError("Something went wrong")
    error = PluginFatalError(sample_record, original)

    assert error.record is sample_record
    assert error.original is original


def test_plugin_fatal_error_message(sample_record: PluginExecutionRecord) -> None:
    """Verify PluginFatalError message comes from original exception."""
    original = ValueError("Detailed error message")
    error = PluginFatalError(sample_record, original)

    assert str(error) == "Detailed error message"


def test_plugin_fatal_error_inherits_from_exception(
    sample_record: PluginExecutionRecord,
) -> None:
    """Verify PluginFatalError is an Exception subclass."""
    error = PluginFatalError(sample_record, ValueError("test"))
    assert isinstance(error, Exception)


def test_plugin_fatal_error_can_be_raised(sample_record: PluginExecutionRecord) -> None:
    """Verify PluginFatalError can be raised and caught."""
    original = RuntimeError("Test failure")
    error = PluginFatalError(sample_record, original)

    with pytest.raises(PluginFatalError) as exc_info:
        raise error

    assert exc_info.value.record is sample_record
    assert exc_info.value.original is original


def test_plugin_fatal_error_with_chained_exception(
    sample_record: PluginExecutionRecord,
) -> None:
    """Verify PluginFatalError can be chained from original."""
    original = ValueError("Root cause")

    with pytest.raises(PluginFatalError) as exc_info:
        try:
            raise original
        except ValueError as e:
            raise PluginFatalError(sample_record, e) from e

    assert exc_info.value.__cause__ is original


def test_plugin_fatal_error_preserves_record(
    sample_record: PluginExecutionRecord,
) -> None:
    """Verify PluginFatalError preserves execution record details."""
    error = PluginFatalError(sample_record, ValueError("test"))

    assert error.record.plugin_name == "test.plugin"
    assert error.record.status == "failed"
    assert error.record.duration_ms == 100.0


# =============================================================================
# PluginTimeoutError Tests
# =============================================================================


def test_plugin_timeout_error_basic() -> None:
    """Verify PluginTimeoutError basic construction."""
    error = PluginTimeoutError("my.plugin", 5000)

    assert error.plugin_name == "my.plugin"
    assert error.timeout_ms == 5000
    assert error.elapsed_ms is None


def test_plugin_timeout_error_message_basic() -> None:
    """Verify PluginTimeoutError message without elapsed time."""
    error = PluginTimeoutError("my.plugin", 5000)

    assert str(error) == "Plugin 'my.plugin' exceeded timeout of 5000ms"


def test_plugin_timeout_error_with_elapsed() -> None:
    """Verify PluginTimeoutError with elapsed time."""
    error = PluginTimeoutError("my.plugin", 5000, elapsed_ms=5123.45)

    assert error.elapsed_ms == 5123.45


def test_plugin_timeout_error_message_with_elapsed() -> None:
    """Verify PluginTimeoutError message includes elapsed time."""
    error = PluginTimeoutError("my.plugin", 5000, elapsed_ms=5123.45)

    message = str(error)
    assert "Plugin 'my.plugin' exceeded timeout of 5000ms" in message
    assert "(elapsed: 5123.45ms)" in message


def test_plugin_timeout_error_inherits_from_exception() -> None:
    """Verify PluginTimeoutError is an Exception subclass."""
    error = PluginTimeoutError("test", 1000)
    assert isinstance(error, Exception)


def test_plugin_timeout_error_can_be_raised() -> None:
    """Verify PluginTimeoutError can be raised and caught."""
    with pytest.raises(PluginTimeoutError) as exc_info:
        msg = "slow.plugin"
        raise PluginTimeoutError(msg, 10000, elapsed_ms=15000.0)

    assert exc_info.value.plugin_name == "slow.plugin"
    assert exc_info.value.timeout_ms == 10000


# =============================================================================
# PluginSkippedError Tests
# =============================================================================


def test_plugin_skipped_error_basic() -> None:
    """Verify PluginSkippedError basic construction."""
    error = PluginSkippedError("my.plugin", "missing dependency")

    assert error.plugin_name == "my.plugin"
    assert error.reason == "missing dependency"


def test_plugin_skipped_error_message() -> None:
    """Verify PluginSkippedError message format."""
    error = PluginSkippedError("my.plugin", "feature disabled")

    assert str(error) == "Plugin 'my.plugin' skipped: feature disabled"


def test_plugin_skipped_error_inherits_from_exception() -> None:
    """Verify PluginSkippedError is an Exception subclass."""
    error = PluginSkippedError("test", "reason")
    assert isinstance(error, Exception)


def test_plugin_skipped_error_can_be_raised() -> None:
    """Verify PluginSkippedError can be raised and caught."""
    with pytest.raises(PluginSkippedError) as exc_info:
        msg = "disabled.plugin"
        raise PluginSkippedError(msg, "explicitly disabled")

    assert exc_info.value.plugin_name == "disabled.plugin"
    assert exc_info.value.reason == "explicitly disabled"


def test_plugin_skipped_error_various_reasons() -> None:
    """Verify PluginSkippedError with various skip reasons."""
    reasons = [
        "missing dependency",
        "feature disabled",
        "configuration error",
        "dependency cycle",
        "not applicable",
    ]

    for reason in reasons:
        error = PluginSkippedError("test.plugin", reason)
        assert reason in str(error)
        assert error.reason == reason
