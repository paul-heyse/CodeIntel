"""Tests for observability registry helpers."""

from __future__ import annotations

from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider

from codeintel.observability.registry import get_instrument_registry, get_instrumentation_registry
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def test_instrument_registry_caches_group() -> None:
    """InstrumentRegistry should cache group instances per meter."""
    registry = get_instrument_registry()
    meter = MeterProvider().get_meter("meter-1")
    calls = {"count": 0}

    def _builder(_meter: Meter) -> object:
        calls["count"] += 1
        return {"meter": _meter}

    first = registry.get_group(meter, "group", _builder)
    second = registry.get_group(meter, "group", _builder)

    expect_equal(calls["count"], 1)
    expect_true(first is second)


def test_instrumentation_registry_summary_and_snapshot() -> None:
    """InstrumentationRegistry should track and summarize statuses."""
    registry = get_instrumentation_registry()
    registry.clear()

    registry.record_enabled("b")
    registry.record_unavailable("a")
    registry.record_error("c")

    snapshot = registry.snapshot()
    expect_equal([record.name for record in snapshot], ["a", "b", "c"])

    summary = registry.summary()
    expect_equal(summary["enabled"], 1)
    expect_equal(summary["unavailable"], 1)
    expect_equal(summary["error"], 1)
