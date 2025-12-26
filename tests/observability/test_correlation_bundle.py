"""Correlation bundle tests."""

from __future__ import annotations

from codeintel.observability.telemetry_context import (
    RepoCommitContext,
    TelemetryContext,
    current_telemetry_context,
    telemetry_context,
)


def test_correlation_bundle_empty() -> None:
    """Empty context should yield empty bundles."""
    bundle = current_telemetry_context()
    assert bundle == TelemetryContext(
        correlation_id=None,
        run_id=None,
        domain=None,
        repo=None,
        commit=None,
        actor=None,
    )
    assert bundle.span_attributes() == {}
    assert bundle.metric_attributes() == {}


def test_correlation_bundle_populated() -> None:
    """Context managers should populate bundle attributes."""
    with (
        telemetry_context(
            correlation_id="corr-1",
            run_id="run-1",
            domain="tests",
            repo_commit=RepoCommitContext(
                repo="org/repo",
                commit="abc123",
            ),
            actor="alice",
        )
    ):
        bundle = current_telemetry_context()
        span_attrs = bundle.span_attributes()
        metric_attrs = bundle.metric_attributes()

    assert span_attrs["codeintel.correlation_id"] == "corr-1"
    assert span_attrs["codeintel.run_id"] == "run-1"
    assert span_attrs["codeintel.domain"] == "tests"
    assert span_attrs["codeintel.repo"] == "org/repo"
    assert span_attrs["codeintel.commit"] == "abc123"
    assert span_attrs["codeintel.actor"] == "alice"
    assert "codeintel.correlation_id" not in metric_attrs
    assert metric_attrs["codeintel.run_id"] == "run-1"
    assert metric_attrs["codeintel.domain"] == "tests"
    assert metric_attrs["codeintel.repo"] == "org/repo"
    assert metric_attrs["codeintel.commit"] == "abc123"
