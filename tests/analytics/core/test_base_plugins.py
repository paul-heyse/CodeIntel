"""Tests for core base plugin classes.

This module tests:
- ResolvedConfig container
- BasePlugin metadata synthesis
- TableWriterPlugin output spec building
- Plugin execution flow
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest

from codeintel.analytics.core.base import (
    BasePlugin,
    ResolvedConfig,
    TableWriterPlugin,
)
from codeintel.analytics.core.context import PluginExecutionContext
from codeintel.analytics.core.protocol import (
    PluginResult,
    PluginStage,
    ValidationResult,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers.fakes.graph_contexts import create_graph_gateway

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext

# Test constants
CONFIG_TEST_VALUE = 42
TABLE_A_ROW_COUNT = 5
TABLE_B_ROW_COUNT = 3
TEST_TABLE_ROW_COUNT = 10
EXPECTED_TWO_OUTPUT_SPECS = 2
DEFAULT_REPO = "test/repo"
DEFAULT_COMMIT = "abc123"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_gateway() -> Generator[StorageGateway, None, None]:
    """Provide a test gateway that auto-closes.

    Yields
    ------
    StorageGateway
        In-memory gateway with schema applied.
    """
    gateway = create_graph_gateway()
    yield gateway
    gateway.close()


@pytest.fixture
def test_snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide a test snapshot.

    Parameters
    ----------
    tmp_path
        Pytest temporary path fixture.

    Returns
    -------
    SnapshotRef
        Test snapshot reference.
    """
    return SnapshotRef(repo=DEFAULT_REPO, commit=DEFAULT_COMMIT, repo_root=tmp_path)


@pytest.fixture
def test_context(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> PluginExecutionContext:
    """Provide a test execution context.

    Parameters
    ----------
    test_gateway
        Storage gateway fixture.
    test_snapshot
        Snapshot reference fixture.

    Returns
    -------
    PluginExecutionContext
        Test execution context.
    """
    return PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
        run_id="test-run-001",
    )


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
# Sample Plugin Implementations
# =============================================================================


@dataclass
class SamplePlugin(BasePlugin):
    """Sample implementation of BasePlugin for testing.

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

    def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
        """Execute test computation.

        Parameters
        ----------
        ctx
            Execution context.

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
class SampleTableWriter(TableWriterPlugin):
    """Sample implementation of TableWriterPlugin for testing.

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

    def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
        """Execute test computation.

        Parameters
        ----------
        ctx
            Execution context.

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
    plugin = SamplePlugin()

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

        def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
            _ = (self, ctx)  # Required by interface
            return None

    plugin = UnnamedPlugin()
    meta = plugin.metadata

    assert meta.name == "UnnamedPlugin"


def test_base_plugin_validate_inputs_default(
    test_context: PluginExecutionContext,
) -> None:
    """Default validate_inputs returns success."""
    plugin = SamplePlugin()

    result = plugin.validate_inputs(test_context)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_base_plugin_execute_success(
    test_context: PluginExecutionContext,
) -> None:
    """Execute returns success result on successful compute."""
    plugin = SamplePlugin(should_fail=False)

    result = plugin.execute(test_context)

    assert isinstance(result, PluginResult)
    assert result.success is True
    assert result.row_counts is not None
    assert result.row_counts.get("analytics.test_table") == TEST_TABLE_ROW_COUNT


def test_base_plugin_execute_failure(
    test_context: PluginExecutionContext,
) -> None:
    """Execute returns failure result when compute raises."""
    plugin = SamplePlugin(should_fail=True)

    result = plugin.execute(test_context)

    assert isinstance(result, PluginResult)
    assert result.success is False
    assert result.error is not None
    assert "failed" in result.error.lower()


def test_base_plugin_build_input_specs_default() -> None:
    """Default build_input_specs returns empty tuple."""
    specs = SamplePlugin.build_input_specs()

    assert specs == ()


# =============================================================================
# TableWriterPlugin Tests
# =============================================================================


def test_table_writer_output_specs() -> None:
    """TableWriterPlugin builds output specs from output_tables."""
    plugin = SampleTableWriter()

    meta = plugin.metadata

    assert len(meta.outputs) == EXPECTED_TWO_OUTPUT_SPECS
    output_names = {spec.name for spec in meta.outputs}
    # Output spec names are the table names (possibly without schema prefix)
    assert "table_a" in output_names or "analytics.table_a" in output_names
    assert "table_b" in output_names or "analytics.table_b" in output_names


def test_table_writer_execute_success(
    test_context: PluginExecutionContext,
) -> None:
    """TableWriterPlugin execute returns row counts."""
    plugin = SampleTableWriter(should_fail=False)

    result = plugin.execute(test_context)

    assert result.success is True
    assert result.row_counts is not None
    assert result.row_counts.get("analytics.table_a") == TABLE_A_ROW_COUNT
    assert result.row_counts.get("analytics.table_b") == TABLE_B_ROW_COUNT


def test_table_writer_execute_failure(
    test_context: PluginExecutionContext,
) -> None:
    """TableWriterPlugin execute handles compute failures."""
    plugin = SampleTableWriter(should_fail=True)

    result = plugin.execute(test_context)

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

        def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
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

        def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
            _ = (self, ctx)  # Required by interface
            return None

    plugin = EmptyOutputPlugin()
    meta = plugin.metadata

    assert len(meta.outputs) == 0
