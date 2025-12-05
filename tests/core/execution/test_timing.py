"""Test timing utilities from codeintel.core.execution.timing.

This module tests:
- TimingResult start/stop/elapsed properties
- timed() context manager
- measure_duration() function wrapper
- measure_duration_ms() convenience wrapper
"""

from __future__ import annotations

import time

import pytest

from codeintel.core.execution.timing import (
    TimingResult,
    measure_duration,
    measure_duration_ms,
    timed,
)

# Test constants
MS_10_IN_NS = 10_000_000  # 10ms in nanoseconds
MS_10 = 10.0  # 10ms
MS_15 = 15.0  # 15ms
MS_20 = 20.0  # 20ms
MS_30 = 30.0  # 30ms
MS_50 = 50.0  # 50ms upper bound
S_0_02 = 0.02  # 20ms in seconds
S_0_05 = 0.05  # 50ms in seconds
MS_TOLERANCE = 0.001  # Tolerance for ms comparisons
S_TOLERANCE = 0.000001  # Tolerance for second comparisons
EXPECTED_5 = 5  # Expected result for 2 + 3
EXPECTED_20 = 20  # Expected result for 4 * 5
EXPECTED_27 = 27  # Expected result for 3 ** 3
EXPECTED_42 = 42  # Expected result for timed_work
TIMING_5_MS = 5.0  # First timing threshold


# =============================================================================
# TimingResult Tests
# =============================================================================


def test_timing_result_construction() -> None:
    """Verify TimingResult can be constructed."""
    result = TimingResult()

    assert result.start_ns > 0
    assert result.end_ns is None


def test_timing_result_custom_start() -> None:
    """Verify TimingResult accepts custom start time."""
    custom_start = 1000000000
    result = TimingResult(start_ns=custom_start)

    assert result.start_ns == custom_start


def test_timing_result_stop() -> None:
    """Verify TimingResult.stop() records end time."""
    result = TimingResult()
    assert result.end_ns is None

    result.stop()

    assert result.end_ns is not None
    assert result.end_ns >= result.start_ns


def test_timing_result_stop_idempotent() -> None:
    """Verify TimingResult.stop() is idempotent."""
    result = TimingResult()
    result.stop()
    first_end = result.end_ns

    time.sleep(0.01)
    result.stop()  # Second call

    assert result.end_ns == first_end  # Should not change


def test_timing_result_is_stopped_false() -> None:
    """Verify is_stopped returns False before stop()."""
    result = TimingResult()
    assert result.is_stopped is False


def test_timing_result_is_stopped_true() -> None:
    """Verify is_stopped returns True after stop()."""
    result = TimingResult()
    result.stop()
    assert result.is_stopped is True


def test_timing_result_elapsed_ns() -> None:
    """Verify elapsed_ns returns nanoseconds."""
    result = TimingResult()
    time.sleep(0.01)  # 10ms
    result.stop()

    # Should be at least 10ms = 10,000,000 ns
    assert result.elapsed_ns >= MS_10_IN_NS


def test_timing_result_elapsed_ns_before_stop() -> None:
    """Verify elapsed_ns works before stop() is called."""
    result = TimingResult()
    time.sleep(0.01)  # 10ms

    # Should still return elapsed time (up to now)
    assert result.elapsed_ns >= MS_10_IN_NS


def test_timing_result_elapsed_ms() -> None:
    """Verify elapsed_ms returns milliseconds."""
    result = TimingResult()
    time.sleep(0.01)  # 10ms
    result.stop()

    # Should be at least 10ms
    assert result.elapsed_ms >= MS_10


def test_timing_result_elapsed_s() -> None:
    """Verify elapsed_s returns seconds."""
    result = TimingResult()
    time.sleep(0.05)  # 50ms
    result.stop()

    # Should be at least 0.05s
    assert result.elapsed_s >= S_0_05


def test_timing_result_elapsed_consistency() -> None:
    """Verify elapsed properties are consistent with each other."""
    result = TimingResult()
    time.sleep(0.01)
    result.stop()

    ns = result.elapsed_ns
    ms = result.elapsed_ms
    s = result.elapsed_s

    # Conversions should be accurate
    assert abs(ms - ns / 1_000_000) < MS_TOLERANCE
    assert abs(s - ns / 1_000_000_000) < S_TOLERANCE


def test_timing_result_elapsed_frozen_after_stop() -> None:
    """Verify elapsed values are frozen after stop()."""
    result = TimingResult()
    result.stop()
    elapsed_after_stop = result.elapsed_ns

    time.sleep(0.01)  # Wait a bit

    # Should be the same (frozen at stop time)
    assert result.elapsed_ns == elapsed_after_stop


# =============================================================================
# timed() Context Manager Tests
# =============================================================================


def test_timed_yields_timing_result() -> None:
    """Verify timed() yields a TimingResult."""
    with timed() as t:
        assert isinstance(t, TimingResult)


def test_timed_starts_timing() -> None:
    """Verify timed() starts timing on entry."""
    with timed() as t:
        assert t.start_ns > 0
        assert t.is_stopped is False


def test_timed_stops_on_exit() -> None:
    """Verify timed() stops timing on exit."""
    with timed() as t:
        time.sleep(0.01)

    assert t.is_stopped is True
    assert t.elapsed_ms >= MS_10


def test_timed_stops_on_exception() -> None:
    """Verify timed() stops timing even on exception."""
    timing_result = None
    err_msg = "test_error"

    def raise_after_timing() -> None:
        nonlocal timing_result
        with timed() as t:
            timing_result = t
            time.sleep(0.01)
            raise ValueError(err_msg)

    with pytest.raises(ValueError, match=err_msg):
        raise_after_timing()

    assert timing_result is not None
    assert timing_result.is_stopped is True


def test_timed_measures_work() -> None:
    """Verify timed() accurately measures work duration."""
    with timed() as t:
        time.sleep(0.02)  # 20ms of "work"

    assert t.elapsed_ms >= MS_20
    assert t.elapsed_s >= S_0_02


def test_timed_nested() -> None:
    """Verify timed() works in nested contexts."""
    with timed() as outer:
        time.sleep(0.01)
        with timed() as inner:
            time.sleep(0.01)
        time.sleep(0.01)

    assert outer.elapsed_ms >= MS_30  # All three sleeps
    assert inner.elapsed_ms >= MS_10  # Just inner sleep


# =============================================================================
# measure_duration() Tests
# =============================================================================


def test_measure_duration_returns_result() -> None:
    """Verify measure_duration returns function result."""

    def simple_fn() -> str:
        return "result"

    result, _ = measure_duration(simple_fn)
    assert result == "result"


def test_measure_duration_returns_timing() -> None:
    """Verify measure_duration returns timing result."""

    def simple_fn() -> str:
        time.sleep(0.01)
        return "result"

    _, timing = measure_duration(simple_fn)

    assert isinstance(timing, TimingResult)
    assert timing.is_stopped is True
    assert timing.elapsed_ms >= MS_10


def test_measure_duration_with_args() -> None:
    """Verify measure_duration passes positional arguments."""

    def add(a: int, b: int) -> int:
        return a + b

    result, _ = measure_duration(add, 2, 3)
    assert result == EXPECTED_5


def test_measure_duration_with_kwargs() -> None:
    """Verify measure_duration passes keyword arguments."""

    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    result, _ = measure_duration(greet, "World", greeting="Hi")
    assert result == "Hi, World!"


def test_measure_duration_measures_work() -> None:
    """Verify measure_duration accurately measures work time."""

    def slow_fn() -> str:
        time.sleep(0.02)
        return "done"

    result, timing = measure_duration(slow_fn)

    assert result == "done"
    assert timing.elapsed_ms >= MS_20


def test_measure_duration_with_exception() -> None:
    """Verify measure_duration propagates exceptions."""
    err_msg = "intentional_error"

    def failing_fn() -> str:
        time.sleep(0.01)
        raise ValueError(err_msg)

    with pytest.raises(ValueError, match=err_msg):
        measure_duration(failing_fn)


# =============================================================================
# measure_duration_ms() Tests
# =============================================================================


def test_measure_duration_ms_returns_result() -> None:
    """Verify measure_duration_ms returns function result."""

    def simple_fn() -> str:
        return "ms_result"

    result, _ = measure_duration_ms(simple_fn)
    assert result == "ms_result"


def test_measure_duration_ms_returns_float() -> None:
    """Verify measure_duration_ms returns duration as float."""

    def simple_fn() -> str:
        time.sleep(0.01)
        return "result"

    _, duration_ms = measure_duration_ms(simple_fn)

    assert isinstance(duration_ms, float)
    assert duration_ms >= MS_10


def test_measure_duration_ms_with_args() -> None:
    """Verify measure_duration_ms passes arguments."""

    def multiply(a: int, b: int) -> int:
        return a * b

    result, _ = measure_duration_ms(multiply, 4, 5)
    assert result == EXPECTED_20


def test_measure_duration_ms_with_kwargs() -> None:
    """Verify measure_duration_ms passes keyword arguments."""

    def power(base: int, exponent: int = 2) -> int:
        return base**exponent

    result, _ = measure_duration_ms(power, 3, exponent=3)
    assert result == EXPECTED_27


def test_measure_duration_ms_accuracy() -> None:
    """Verify measure_duration_ms is accurate."""

    def timed_work() -> int:
        time.sleep(0.015)  # 15ms
        return EXPECTED_42

    result, duration = measure_duration_ms(timed_work)

    assert result == EXPECTED_42
    assert duration >= MS_15
    # Should be close to 15ms (allowing some overhead)
    assert duration < MS_50  # Generous upper bound


# =============================================================================
# Integration Tests
# =============================================================================


def test_timing_result_in_loop() -> None:
    """Verify TimingResult works correctly when reused in a loop pattern."""
    timings = []

    for i in range(3):
        with timed() as t:
            time.sleep(0.005 * (i + 1))  # 5ms, 10ms, 15ms
        timings.append(t.elapsed_ms)

    # Each timing should be independent and increasing
    assert timings[0] >= TIMING_5_MS
    assert timings[1] >= MS_10
    assert timings[2] >= MS_15
    assert timings[2] > timings[1] > timings[0]


def test_measure_duration_vs_timed_equivalence() -> None:
    """Verify measure_duration and timed() give equivalent results."""
    work_duration_ms = 10

    def work() -> str:
        time.sleep(work_duration_ms / 1000)
        return "done"

    # Using measure_duration
    result1, timing1 = measure_duration(work)

    # Using timed()
    with timed() as timing2:
        result2 = work()

    # Both should succeed
    assert result1 == result2 == "done"

    # Both should measure approximately the same duration
    assert timing1.elapsed_ms >= work_duration_ms
    assert timing2.elapsed_ms >= work_duration_ms
