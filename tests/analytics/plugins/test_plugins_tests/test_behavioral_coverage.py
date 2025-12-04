"""Tests for BehavioralCoveragePlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from codeintel.analytics.core.protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.plugins.tests.behavioral_coverage import (
    BehavioralCoveragePlugin,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import BehavioralCoverageStepConfig

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_VERSION = "2.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_CAPABILITY_COUNT = 1


def _create_config() -> BehavioralCoverageStepConfig:
    """Create a test configuration.

    Returns
    -------
    BehavioralCoverageStepConfig
        Test configuration.
    """
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=Path("/test/repo"))
    return BehavioralCoverageStepConfig(snapshot=snapshot)


def _create_mock_context(
    *,
    has_config: bool = True,
    llm_runner: object | None = None,
) -> MagicMock:
    """Create a mock execution context.

    Parameters
    ----------
    has_config
        Whether config is available.
    llm_runner
        Optional LLM runner to include in extra.

    Returns
    -------
    MagicMock
        Mock execution context.
    """
    ctx = MagicMock()

    ctx.has_config.return_value = has_config
    if has_config:
        ctx.get_config.return_value = _create_config()
    else:
        ctx.get_config.side_effect = ValueError("Config not found")

    ctx.extra = {}
    if llm_runner is not None:
        ctx.extra["behavioral_llm_runner"] = llm_runner

    ctx.gateway = MagicMock()

    return ctx


# =============================================================================
# Metadata Tests
# =============================================================================


def test_behavioral_coverage_plugin_metadata_name() -> None:
    """Plugin metadata has correct name."""
    plugin = BehavioralCoveragePlugin()
    assert plugin.metadata.name == "tests.behavioral_coverage"


def test_behavioral_coverage_plugin_metadata_stage() -> None:
    """Plugin metadata has correct stage."""
    plugin = BehavioralCoveragePlugin()
    assert plugin.metadata.stage == "test"


def test_behavioral_coverage_plugin_metadata_version() -> None:
    """Plugin metadata has correct version."""
    plugin = BehavioralCoveragePlugin()
    assert plugin.metadata.version == TEST_VERSION


def test_behavioral_coverage_plugin_metadata_outputs() -> None:
    """Plugin metadata has correct outputs."""
    plugin = BehavioralCoveragePlugin()
    assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT

    output_names = {o.name for o in plugin.metadata.outputs}
    assert "behavioral_coverage" in output_names


def test_behavioral_coverage_plugin_metadata_capabilities_provided() -> None:
    """Plugin metadata provides correct capabilities."""
    plugin = BehavioralCoveragePlugin()
    assert len(plugin.metadata.provides) == EXPECTED_CAPABILITY_COUNT

    assert "analytics.behavioral_coverage" in plugin.metadata.provides


def test_behavioral_coverage_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = BehavioralCoveragePlugin()
    assert "tests" in plugin.metadata.tags
    assert "behavioral" in plugin.metadata.tags
    assert "classification" in plugin.metadata.tags


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_success_with_config() -> None:
    """Validation succeeds when config is present."""
    plugin = BehavioralCoveragePlugin()
    ctx = _create_mock_context(has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config() -> None:
    """Validation fails when config is missing."""
    plugin = BehavioralCoveragePlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_with_callable_llm_runner() -> None:
    """Validation succeeds with callable LLM runner."""
    plugin = BehavioralCoveragePlugin()

    def mock_llm_runner() -> None:
        pass

    ctx = _create_mock_context(has_config=True, llm_runner=mock_llm_runner)

    result = plugin.validate_inputs(ctx)

    assert result.valid is True


def test_validate_inputs_with_non_callable_llm_runner() -> None:
    """Validation fails with non-callable LLM runner."""
    plugin = BehavioralCoveragePlugin()
    ctx = _create_mock_context(has_config=True, llm_runner="not_callable")

    result = plugin.validate_inputs(ctx)

    assert result.valid is False


def test_validate_inputs_returns_error_details() -> None:
    """Validation returns specific error messages."""
    plugin = BehavioralCoveragePlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "BehavioralCoverageStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config() -> None:
    """Execute fails when config is not available."""
    plugin = BehavioralCoveragePlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_fails_with_non_callable_llm_runner() -> None:
    """Execute fails with non-callable LLM runner."""
    plugin = BehavioralCoveragePlugin()
    ctx = _create_mock_context(has_config=True, llm_runner="not_callable")

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


# Note: Integration tests for execute_succeeds_without_llm_runner and
# execute_returns_row_counts require full test profile infrastructure
# which is tested in integration test suites


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_is_dataclass() -> None:
    """Plugin is a dataclass."""
    plugin = BehavioralCoveragePlugin()
    assert hasattr(plugin, "__dataclass_fields__")


def test_plugin_metadata_is_consistent() -> None:
    """Plugin metadata is consistent between calls."""
    plugin = BehavioralCoveragePlugin()

    meta1 = plugin.metadata
    meta2 = plugin.metadata

    assert meta1.name == meta2.name
    assert meta1.version == meta2.version
    assert meta1.stage == meta2.stage
