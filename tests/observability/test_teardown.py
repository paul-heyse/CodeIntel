"""Teardown telemetry helpers tests."""

from __future__ import annotations

from codeintel.observability.teardown import (
    TeardownTelemetry,
    snapshot_active_threads,
    snapshot_pending_tasks,
)

_FLUSH_MS = 12.5


def test_snapshot_pending_tasks_no_loop() -> None:
    """Pending tasks should be empty when no loop is running."""
    count, samples = snapshot_pending_tasks()
    assert count is None
    assert samples == ()


def test_snapshot_active_threads_sample_limit() -> None:
    """Thread samples should respect the configured limit."""
    count, names = snapshot_active_threads(sample_limit=1, allowlisted_daemon_names=set())
    assert count >= 1
    assert len(names) <= 1


def test_teardown_telemetry_payload_includes_flush() -> None:
    """Flush fields should map into span and log payloads."""
    telemetry = TeardownTelemetry(
        telemetry_flush_ok=True,
        telemetry_flush_ms=_FLUSH_MS,
    )
    payload = telemetry.to_log_payload()
    assert payload["telemetry_flush_ok"] is True
    assert payload["telemetry_flush_ms"] == _FLUSH_MS
    attrs = telemetry.span_attributes()
    assert attrs["telemetry.flush.ok"] is True
    assert attrs["telemetry.flush.ms"] == _FLUSH_MS
