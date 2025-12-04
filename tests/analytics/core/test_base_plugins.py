"""Tests for core base plugin classes.

This module tests:
- ResolvedConfig container
- BasePlugin metadata synthesis
- TableWriterPlugin output spec building
- Plugin execution flow
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from codeintel.analytics.core.base import (
    BasePlugin,
    ResolvedConfig,
    TableWriterPlugin,
)
from codeintel.analytics.core.protocol import (
    PluginResult,
    PluginStage,
    ValidationResult,
)

# Test constants
CONFIG_TEST_VALUE = 42
TABLE_A_ROW_COUNT = 5
TABLE_B_ROW_COUNT = 3
TEST_TABLE_ROW_COUNT = 10
EXPECTED_TWO_OUTPUT_SPECS = 2


# =============================================================================
# ResolvedConfig Tests
# =============================================================================


def test_resolved_config_initial_state() -> None:
    """ResolvedConfig starts unresolved with None value."""
    container: ResolvedConfig[str] = ResolvedConfig()

    assert container.value is None
    assert container.resolved is False


def test_resolved_config_set() -> None:
    """Setting a value marks the config as resolved."""
    container: ResolvedConfig[str] = ResolvedConfig()

    container.set("test_value")

    assert container.value == "test_value"
    assert container.resolved is True


def test_resolved_config_get() -> None:
    """Get returns the stored value after resolution."""
    container: ResolvedConfig[int] = ResolvedConfig()
    container.set(CONFIG_TEST_VALUE)

    result = container.get("test_plugin")

    assert result == CONFIG_TEST_VALUE


def test_resolved_config_get_not_resolved_raises() -> None:
    """Get raises ValueError when config not resolved."""
    container: ResolvedConfig[str] = ResolvedConfig()

    with pytest.raises(ValueError, match="Config not resolved"):
        container.get("my_plugin")


def test_resolved_config_get_or_none_resolved() -> None:
    """Get or none returns value when resolved."""
    container: ResolvedConfig[str] = ResolvedConfig()
    container.set("value")

    result = container.get_or_none()

    assert result == "value"


def test_resolved_config_get_or_none_not_resolved() -> None:
    """Get or none returns None when not resolved."""
    container: ResolvedConfig[str] = ResolvedConfig()

    result = container.get_or_none()

    assert result is None


# =============================================================================
# Mock Plugin Implementations
# =============================================================================


@dataclass
class TestPlugin(BasePlugin):
    """Test implementation of BasePlugin.

    Attributes
    ----------
    should_fail
        Whether compute should raise an exception.
    """

    plugin_name: ClassVar[str] = "test.plugin"
    plugin_description: ClassVar[str] = "A test plugin"
    plugin_stage: ClassVar[PluginStage] = "function"
    plugin_version: ClassVar[str] = "1.0.0"
    tags: ClassVar[tuple[str, ...]] = ("test", "sample")

    should_fail: bool = False

    def compute(self, ctx: MagicMock) -> Mapping[str, int] | None:
        """Execute test computation.

        Parameters
        ----------
        ctx
            Execution context (used for interface compliance).

        Returns
        -------
        Mapping[str, int] | None
            Row counts or None.

        Raises
        ------
        ValueError
            If should_fail is True.
        """
        _ = ctx  # Required by interface
        if self.should_fail:
            message = "Intentional test failure"
            raise ValueError(message)
        return {"analytics.test_table": TEST_TABLE_ROW_COUNT}


@dataclass
class TestTableWriter(TableWriterPlugin):
    """Test implementation of TableWriterPlugin.

    Attributes
    ----------
    should_fail
        Whether compute should raise an exception.
    """

    plugin_name: ClassVar[str] = "test.table_writer"
    plugin_description: ClassVar[str] = "A test table writer plugin"
    plugin_stage: ClassVar[PluginStage] = "function"
    output_tables: ClassVar[tuple[str, ...]] = (
        "analytics.table_a",
        "analytics.table_b",
    )

    should_fail: bool = False

    def compute(self, ctx: MagicMock) -> Mapping[str, int] | None:
        """Execute test computation.

        Parameters
        ----------
        ctx
            Execution context (used for interface compliance).

        Returns
        -------
        Mapping[str, int] | None
            Row counts or None.

        Raises
        ------
        RuntimeError
            If should_fail is True.
        """
        _ = ctx  # Required by interface
        if self.should_fail:
            message = "Intentional table writer failure"
            raise RuntimeError(message)
        return {
            "analytics.table_a": TABLE_A_ROW_COUNT,
            "analytics.table_b": TABLE_B_ROW_COUNT,
        }


# =============================================================================
# BasePlugin Tests
# =============================================================================


def test_base_plugin_metadata() -> None:
    """Plugin metadata is synthesized from class attributes."""
    plugin = TestPlugin()

    meta = plugin.metadata

    assert meta.name == "test.plugin"
    assert meta.description == "A test plugin"
    assert meta.stage == "function"
    assert meta.version == "1.0.0"
    assert "test" in meta.tags
    assert "sample" in meta.tags


def test_base_plugin_metadata_defaults_name_to_class() -> None:
    """Plugin name defaults to class name if not set."""

    @dataclass
    class UnnamedPlugin(BasePlugin):
        """Plugin without explicit name."""

        def compute(self, ctx: MagicMock) -> Mapping[str, int] | None:
            _ = (self, ctx)  # Required by interface
            return None

    plugin = UnnamedPlugin()
    meta = plugin.metadata

    assert meta.name == "UnnamedPlugin"


def test_base_plugin_validate_inputs_default() -> None:
    """Default validate_inputs returns success."""
    plugin = TestPlugin()
    mock_ctx = MagicMock()

    result = plugin.validate_inputs(mock_ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_base_plugin_execute_success() -> None:
    """Execute returns success result on successful compute."""
    plugin = TestPlugin(should_fail=False)
    mock_ctx = MagicMock()
    mock_ctx.repo = "test-repo"
    mock_ctx.commit = "abc123"
    mock_ctx.run_id = "run-001"

    result = plugin.execute(mock_ctx)

    assert isinstance(result, PluginResult)
    assert result.success is True
    assert result.row_counts is not None
    assert result.row_counts.get("analytics.test_table") == TEST_TABLE_ROW_COUNT


def test_base_plugin_execute_failure() -> None:
    """Execute returns failure result when compute raises."""
    plugin = TestPlugin(should_fail=True)
    mock_ctx = MagicMock()

    result = plugin.execute(mock_ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False
    assert result.error is not None
    assert "failed" in result.error.lower()


def test_base_plugin_build_input_specs_default() -> None:
    """Default build_input_specs returns empty tuple."""
    specs = TestPlugin.build_input_specs()

    assert specs == ()


# =============================================================================
# TableWriterPlugin Tests
# =============================================================================


def test_table_writer_output_specs() -> None:
    """TableWriterPlugin builds output specs from output_tables."""
    plugin = TestTableWriter()

    meta = plugin.metadata

    assert len(meta.outputs) == EXPECTED_TWO_OUTPUT_SPECS
    output_names = {spec.name for spec in meta.outputs}
    # Output spec names are the table names (possibly without schema prefix)
    assert "table_a" in output_names or "analytics.table_a" in output_names
    assert "table_b" in output_names or "analytics.table_b" in output_names


def test_table_writer_execute_success() -> None:
    """TableWriterPlugin execute returns row counts."""
    plugin = TestTableWriter(should_fail=False)
    mock_ctx = MagicMock()
    mock_ctx.repo = "test-repo"
    mock_ctx.commit = "abc123"
    mock_ctx.run_id = None

    result = plugin.execute(mock_ctx)

    assert result.success is True
    assert result.row_counts is not None
    assert result.row_counts.get("analytics.table_a") == TABLE_A_ROW_COUNT
    assert result.row_counts.get("analytics.table_b") == TABLE_B_ROW_COUNT


def test_table_writer_execute_failure() -> None:
    """TableWriterPlugin execute handles compute failures."""
    plugin = TestTableWriter(should_fail=True)
    mock_ctx = MagicMock()

    result = plugin.execute(mock_ctx)

    assert result.success is False
    assert result.error is not None
    assert "failed" in result.error.lower()


def test_table_writer_with_contract_specs() -> None:
    """TableWriterPlugin with output contracts includes specs in metadata."""

    @dataclass
    class PluginWithContracts(TableWriterPlugin):
        """Plugin with output contracts."""

        plugin_name: ClassVar[str] = "test.with_contracts"
        plugin_stage: ClassVar[PluginStage] = "function"
        output_tables: ClassVar[tuple[str, ...]] = ("analytics.contracted_table",)

        def compute(self, ctx: MagicMock) -> Mapping[str, int] | None:
            _ = (self, ctx)  # Required by interface
            return {"analytics.contracted_table": 1}

    plugin = PluginWithContracts()
    meta = plugin.metadata

    assert len(meta.outputs) == 1
    # Output spec name may not have the schema prefix
    assert "contracted_table" in meta.outputs[0].name


def test_table_writer_empty_output_tables() -> None:
    """TableWriterPlugin with no output tables has no output specs."""

    @dataclass
    class EmptyOutputPlugin(TableWriterPlugin):
        """Plugin with no outputs."""

        plugin_name: ClassVar[str] = "test.empty_outputs"
        plugin_stage: ClassVar[PluginStage] = "other"
        output_tables: ClassVar[tuple[str, ...]] = ()

        def compute(self, ctx: MagicMock) -> Mapping[str, int] | None:
            _ = (self, ctx)  # Required by interface
            return None

    plugin = EmptyOutputPlugin()
    meta = plugin.metadata

    assert len(meta.outputs) == 0
