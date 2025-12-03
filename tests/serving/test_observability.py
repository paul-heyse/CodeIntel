"""Tests for serving layer observability module.

This module tests the observability primitives for query services,
including metrics recording through the public API.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from codeintel.serving.context import RequestContext
from codeintel.serving.services.observability import (
    ServiceCallContext,
    ServiceCallMetrics,
    ServiceObservability,
)

# Constants for test values
DURATION_MS = 15.5
DURATION_PRECISE = 15.12345
DURATION_ROUNDED = 15.12
ROW_COUNT = 10
MESSAGE_COUNT_TWO = 2
ROW_COUNT_THREE = 3
ROW_COUNT_FIVE = 5


# =============================================================================
# ServiceCallMetrics Tests
# =============================================================================


def test_service_call_metrics_required_fields() -> None:
    """Verify ServiceCallMetrics with only required fields."""
    metrics = ServiceCallMetrics(
        name="get_function_summary",
        transport="local",
        duration_ms=DURATION_MS,
    )

    assert metrics.name == "get_function_summary"
    assert metrics.transport == "local"
    assert metrics.duration_ms == DURATION_MS


def test_service_call_metrics_all_fields() -> None:
    """Verify ServiceCallMetrics with all fields populated."""
    metrics = ServiceCallMetrics(
        name="list_datasets",
        transport="http",
        duration_ms=DURATION_MS,
        rows=ROW_COUNT,
        dataset="analytics.functions",
        messages=MESSAGE_COUNT_TWO,
        error=None,
        truncated=False,
        schema_version="1.0.0",
        retries=1,
        correlation_id="corr-123",
        external_transport="mcp",
        operation="datasets.rows",
        repo="demo/repo",
        commit="deadbeef",
        client_id="client-abc",
        user_agent="CodeIntel/1.0",
    )

    assert metrics.rows == ROW_COUNT
    assert metrics.dataset == "analytics.functions"
    assert metrics.messages == MESSAGE_COUNT_TWO
    assert metrics.truncated is False
    assert metrics.schema_version == "1.0.0"
    assert metrics.retries == 1
    assert metrics.correlation_id == "corr-123"


def test_service_call_metrics_with_error() -> None:
    """Verify ServiceCallMetrics records error information."""
    metrics = ServiceCallMetrics(
        name="get_function_summary",
        transport="local",
        duration_ms=DURATION_MS,
        error="ValueError",
    )

    assert metrics.error == "ValueError"


def test_service_call_metrics_optional_fields_none() -> None:
    """Verify ServiceCallMetrics optional fields default to None."""
    metrics = ServiceCallMetrics(
        name="test",
        transport="local",
        duration_ms=1.0,
    )

    assert metrics.rows is None
    assert metrics.dataset is None
    assert metrics.messages is None
    assert metrics.error is None
    assert metrics.truncated is None
    assert metrics.schema_version is None
    assert metrics.retries is None
    assert metrics.correlation_id is None
    assert metrics.external_transport is None
    assert metrics.operation is None
    assert metrics.repo is None
    assert metrics.commit is None
    assert metrics.client_id is None
    assert metrics.user_agent is None


# =============================================================================
# ServiceCallContext Tests
# =============================================================================


def test_service_call_context_defaults() -> None:
    """Verify ServiceCallContext default values."""
    ctx = ServiceCallContext()

    assert ctx.dataset is None
    assert ctx.schema_version is None
    assert ctx.retries is None


def test_service_call_context_with_values() -> None:
    """Verify ServiceCallContext with all values set."""
    ctx = ServiceCallContext(
        dataset="test.dataset",
        schema_version="2.0",
        retries=ROW_COUNT_THREE,
    )

    assert ctx.dataset == "test.dataset"
    assert ctx.schema_version == "2.0"
    assert ctx.retries == ROW_COUNT_THREE


# =============================================================================
# ServiceObservability Tests
# =============================================================================


def test_service_observability_disabled_by_default() -> None:
    """Verify ServiceObservability is disabled by default."""
    obs = ServiceObservability()

    assert obs.enabled is False


def test_service_observability_enabled() -> None:
    """Verify ServiceObservability can be enabled."""
    obs = ServiceObservability(enabled=True)

    assert obs.enabled is True


def test_service_observability_custom_logger() -> None:
    """Verify ServiceObservability accepts custom logger."""
    custom_logger = logging.getLogger("custom.test")
    obs = ServiceObservability(enabled=True, logger=custom_logger)

    assert obs.logger is custom_logger


def test_service_observability_record_when_disabled() -> None:
    """Verify record does nothing when observability is disabled."""
    mock_logger = MagicMock(spec=logging.Logger)
    obs = ServiceObservability(enabled=False, logger=mock_logger)

    metrics = ServiceCallMetrics(name="test", transport="local", duration_ms=1.0)
    obs.record(metrics)

    # Logger should not be called when disabled
    mock_logger.info.assert_not_called()


def test_service_observability_record_when_enabled() -> None:
    """Verify record logs when observability is enabled."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.isEnabledFor.return_value = True
    obs = ServiceObservability(enabled=True, logger=mock_logger)

    metrics = ServiceCallMetrics(
        name="test_call",
        transport="local",
        duration_ms=DURATION_MS,
        rows=ROW_COUNT_FIVE,
    )
    obs.record(metrics)

    # Logger.info should be called with service_call prefix
    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args
    assert call_args[0][0] == "service_call %s"


def test_service_observability_record_with_context() -> None:
    """Verify record includes RequestContext fields."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.isEnabledFor.return_value = True
    obs = ServiceObservability(enabled=True, logger=mock_logger)

    ctx = RequestContext(
        correlation_id="ctx-123",
        transport="http",
        operation="datasets.rows",
        repo="test/repo",
        commit="abc123",
    )
    metrics = ServiceCallMetrics(
        name="test_call",
        transport="local",
        duration_ms=DURATION_MS,
    )
    obs.record(metrics, context=ctx)

    mock_logger.info.assert_called_once()


def test_service_observability_record_merges_context_values() -> None:
    """Verify record prefers metric values over context values."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.isEnabledFor.return_value = True
    obs = ServiceObservability(enabled=True, logger=mock_logger)

    ctx = RequestContext(
        correlation_id="ctx-fallback",
        transport="http",
    )
    metrics = ServiceCallMetrics(
        name="test",
        transport="local",
        duration_ms=1.0,
        correlation_id="metric-override",  # Should override ctx
    )
    obs.record(metrics, context=ctx)

    # Verify the call happened
    mock_logger.info.assert_called_once()


def test_service_observability_record_all_optional_fields() -> None:
    """Verify record handles all optional metric fields."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.isEnabledFor.return_value = True
    obs = ServiceObservability(enabled=True, logger=mock_logger)

    metrics = ServiceCallMetrics(
        name="full_metrics",
        transport="http",
        duration_ms=DURATION_MS,
        rows=ROW_COUNT,
        dataset="test.dataset",
        messages=MESSAGE_COUNT_TWO,
        error=None,
        truncated=True,
        schema_version="1.0",
        retries=1,
    )
    obs.record(metrics)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args
    payload = call_args[0][1]

    assert payload["rows"] == ROW_COUNT
    assert payload["dataset"] == "test.dataset"
    assert payload["truncated"] is True


def test_service_observability_record_with_error() -> None:
    """Verify record includes error information."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.isEnabledFor.return_value = True
    obs = ServiceObservability(enabled=True, logger=mock_logger)

    metrics = ServiceCallMetrics(
        name="error_call",
        transport="local",
        duration_ms=DURATION_MS,
        error="ValueError",
    )
    obs.record(metrics)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args
    payload = call_args[0][1]
    assert payload["error"] == "ValueError"


def test_service_observability_record_logger_not_enabled() -> None:
    """Verify record does nothing when logger level not enabled."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.isEnabledFor.return_value = False
    obs = ServiceObservability(enabled=True, logger=mock_logger)

    metrics = ServiceCallMetrics(name="test", transport="local", duration_ms=1.0)
    obs.record(metrics)

    # Logger.info should not be called when INFO not enabled
    mock_logger.info.assert_not_called()


def test_service_observability_record_context_enrichment() -> None:
    """Verify record enriches payload from RequestContext."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.isEnabledFor.return_value = True
    obs = ServiceObservability(enabled=True, logger=mock_logger)

    ctx = RequestContext(
        correlation_id="enrichment-test",
        transport="mcp",
        operation="get_function_summary",
        repo="demo/repo",
        commit="abc123",
        client_id="client-001",
        user_agent="TestAgent/1.0",
    )
    metrics = ServiceCallMetrics(
        name="context_test",
        transport="local",
        duration_ms=DURATION_MS,
    )
    obs.record(metrics, context=ctx)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args
    payload = call_args[0][1]

    # Context values should be in payload
    assert payload["correlation_id"] == "enrichment-test"
    assert payload["external_transport"] == "mcp"
    assert payload["operation"] == "get_function_summary"
    assert payload["repo"] == "demo/repo"


def test_service_observability_record_rounds_duration() -> None:
    """Verify record rounds duration_ms to 2 decimal places."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.isEnabledFor.return_value = True
    obs = ServiceObservability(enabled=True, logger=mock_logger)

    metrics = ServiceCallMetrics(
        name="duration_test",
        transport="local",
        duration_ms=DURATION_PRECISE,
    )
    obs.record(metrics)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args
    payload = call_args[0][1]

    # Duration should be rounded to 2 decimal places
    assert payload["duration_ms"] == DURATION_ROUNDED


def test_service_observability_record_excludes_none_values() -> None:
    """Verify record excludes optional fields that are None."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.isEnabledFor.return_value = True
    obs = ServiceObservability(enabled=True, logger=mock_logger)

    metrics = ServiceCallMetrics(
        name="minimal",
        transport="local",
        duration_ms=1.0,
    )
    obs.record(metrics)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args
    payload = call_args[0][1]

    # None values should not be in payload
    assert "rows" not in payload
    assert "dataset" not in payload
    assert "error" not in payload
    assert "truncated" not in payload


def test_service_observability_record_metric_overrides_context() -> None:
    """Verify metric values take precedence over context values."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.isEnabledFor.return_value = True
    obs = ServiceObservability(enabled=True, logger=mock_logger)

    ctx = RequestContext(
        correlation_id="context-corr",
        transport="http",
        repo="context-repo",
    )
    metrics = ServiceCallMetrics(
        name="override_test",
        transport="local",
        duration_ms=1.0,
        correlation_id="metric-corr",  # Should override context
        repo="metric-repo",  # Should override context
    )
    obs.record(metrics, context=ctx)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args
    payload = call_args[0][1]

    # Metric values should take precedence
    assert payload["correlation_id"] == "metric-corr"
    assert payload["repo"] == "metric-repo"


def test_service_observability_default_logger() -> None:
    """Verify ServiceObservability has default logger."""
    obs = ServiceObservability(enabled=True)

    assert obs.logger is not None
    assert obs.logger.name == "codeintel.serving.services.query"


def test_service_call_metrics_repo_commit_fields() -> None:
    """Verify ServiceCallMetrics supports repo and commit fields."""
    metrics = ServiceCallMetrics(
        name="repo_test",
        transport="http",
        duration_ms=1.0,
        repo="test/repository",
        commit="sha256hash",
    )

    assert metrics.repo == "test/repository"
    assert metrics.commit == "sha256hash"


def test_service_call_metrics_external_transport() -> None:
    """Verify ServiceCallMetrics external_transport field."""
    metrics = ServiceCallMetrics(
        name="transport_test",
        transport="local",
        duration_ms=1.0,
        external_transport="cli",
    )

    assert metrics.external_transport == "cli"
