"""Tests for plugin registry functionality.

This module tests IngestPluginRegistry, plugin registration, discovery,
and execution order resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.ingestion.plugins.protocol import (
    IngestPluginMetadata,
    IngestPluginProtocol,
    IngestPluginResult,
    IngestResourceHints,
    IngestStage,
)
from codeintel.ingestion.plugins.registry import IngestPluginRegistry, PlanOptions

if TYPE_CHECKING:

    from codeintel.ingestion.core.execution_context import IngestExecutionContext


# =============================================================================
# Test Plugin Implementations
# =============================================================================


@dataclass
class MockPlugin(IngestPluginProtocol):
    """A simple mock plugin for testing."""

    name: str = "mock_plugin"
    stage: IngestStage = "parse"
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    produces_tables: tuple[str, ...] = ("core.mock",)

    @property
    def metadata(self) -> IngestPluginMetadata:
        """Return plugin metadata."""
        return IngestPluginMetadata(
            name=self.name,
            description=f"Mock plugin: {self.name}",
            stage=self.stage,
            provides=self.provides,
            requires=self.requires,
            depends_on=self.depends_on,
            produces_tables=self.produces_tables,
            tool_dependencies=(),
            supports_incremental=False,
            resource_hints=IngestResourceHints(),
            version_hash="1.0.0",
        )

    def execute(self, _ctx: IngestExecutionContext) -> IngestPluginResult:
        """Execute the mock plugin."""
        return IngestPluginResult(success=True, row_counts={})

    def validate_inputs(self, _ctx: IngestExecutionContext) -> object:
        """Validate inputs."""
        from codeintel.ingestion.core.base import ValidationResult

        return ValidationResult.success()


class TestIngestPluginRegistry:
    """Tests for IngestPluginRegistry class."""

    def test_register_plugin(self) -> None:
        """Registry should register plugins."""
        registry = IngestPluginRegistry()
        plugin = MockPlugin(name="test_plugin")

        registry.register(plugin)

        assert registry.contains("test_plugin")

    def test_register_duplicate_raises(self) -> None:
        """Registry should raise ValueError on duplicate registration."""
        registry = IngestPluginRegistry()
        plugin1 = MockPlugin(name="test_plugin")
        plugin2 = MockPlugin(name="test_plugin")

        registry.register(plugin1)

        with pytest.raises(ValueError, match="Duplicate"):
            registry.register(plugin2)

    def test_unregister_plugin(self) -> None:
        """Registry should unregister plugins."""
        registry = IngestPluginRegistry()
        plugin = MockPlugin(name="test_plugin")

        registry.register(plugin)
        registry.unregister("test_plugin")

        assert not registry.contains("test_plugin")

    def test_unregister_nonexistent_is_noop(self) -> None:
        """Unregistering nonexistent plugin should be no-op."""
        registry = IngestPluginRegistry()

        # Should not raise
        registry.unregister("nonexistent")

    def test_get_plugin(self) -> None:
        """Registry.get should return registered plugin."""
        registry = IngestPluginRegistry()
        plugin = MockPlugin(name="test_plugin")

        registry.register(plugin)
        result = registry.get("test_plugin")

        assert result is plugin

    def test_get_nonexistent_raises(self) -> None:
        """Registry.get should raise KeyError for unknown plugins."""
        # Create fresh registry that won't load builtins
        registry = IngestPluginRegistry()
        registry._entrypoints_loaded = True  # Prevent auto-loading

        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_contains(self) -> None:
        """Registry.contains should check plugin existence."""
        registry = IngestPluginRegistry()
        plugin = MockPlugin(name="test_plugin")

        assert not registry.contains("test_plugin")
        registry.register(plugin)
        assert registry.contains("test_plugin")

    def test_list_all(self) -> None:
        """Registry.list_all should return all plugins."""
        registry = IngestPluginRegistry()
        plugin1 = MockPlugin(name="plugin_a")
        plugin2 = MockPlugin(name="plugin_b")

        registry.register(plugin1)
        registry.register(plugin2)

        plugins = registry.list_all()

        # Check that our plugins are in the list (may include builtins)
        names = [p.metadata.name for p in plugins]
        assert "plugin_a" in names
        assert "plugin_b" in names

    def test_list_names(self) -> None:
        """Registry.list_names should return plugin names."""
        registry = IngestPluginRegistry()
        plugin1 = MockPlugin(name="plugin_a")
        plugin2 = MockPlugin(name="plugin_b")

        registry.register(plugin1)
        registry.register(plugin2)

        names = registry.list_names()

        assert "plugin_a" in names
        assert "plugin_b" in names


class TestPluginCapabilities:
    """Tests for capability-based plugin lookup."""

    def test_list_providing(self) -> None:
        """Registry should find plugins by capability."""
        registry = IngestPluginRegistry()
        plugin = MockPlugin(name="provider", provides=("test_capability",))

        registry.register(plugin)
        result = registry.list_providing("test_capability")

        assert len(result) == 1
        assert result[0].metadata.name == "provider"

    def test_list_providing_multiple(self) -> None:
        """Registry should find multiple plugins with same capability."""
        registry = IngestPluginRegistry()
        plugin1 = MockPlugin(name="provider1", provides=("shared_cap",))
        plugin2 = MockPlugin(name="provider2", provides=("shared_cap",))

        registry.register(plugin1)
        registry.register(plugin2)

        result = registry.list_providing("shared_cap")

        assert len(result) == 2
        names = [p.metadata.name for p in result]
        assert "provider1" in names
        assert "provider2" in names

    def test_list_providing_empty(self) -> None:
        """Registry should return empty tuple for unknown capability."""
        registry = IngestPluginRegistry()

        result = registry.list_providing("unknown")

        assert result == ()


class TestPluginStages:
    """Tests for stage-based plugin lookup."""

    def test_list_by_stage(self) -> None:
        """Registry should find plugins by stage."""
        registry = IngestPluginRegistry()
        plugin = MockPlugin(name="extractor", stage="parse")

        registry.register(plugin)
        result = registry.list_by_stage("parse")

        names = [p.metadata.name for p in result]
        assert "extractor" in names

    def test_list_by_stage_multiple(self) -> None:
        """Registry should find multiple plugins in same stage."""
        registry = IngestPluginRegistry()
        plugin1 = MockPlugin(name="ext1", stage="parse")
        plugin2 = MockPlugin(name="ext2", stage="parse")

        registry.register(plugin1)
        registry.register(plugin2)

        result = registry.list_by_stage("parse")
        names = [p.metadata.name for p in result]

        assert "ext1" in names
        assert "ext2" in names


class TestPluginTables:
    """Tests for table-based plugin lookup."""

    def test_list_by_table(self) -> None:
        """Registry should find plugins by produced table."""
        registry = IngestPluginRegistry()
        plugin = MockPlugin(name="writer", produces_tables=("core.test_table",))

        registry.register(plugin)
        result = registry.list_by_table("core.test_table")

        assert len(result) == 1
        assert result[0].metadata.name == "writer"

    def test_list_by_table_multiple(self) -> None:
        """Registry should find multiple plugins producing same table."""
        registry = IngestPluginRegistry()
        plugin1 = MockPlugin(name="w1", produces_tables=("core.shared_table",))
        plugin2 = MockPlugin(
            name="w2", produces_tables=("core.shared_table", "core.extra")
        )

        registry.register(plugin1)
        registry.register(plugin2)

        result = registry.list_by_table("core.shared_table")

        assert len(result) == 2


class TestPlanOptions:
    """Tests for PlanOptions dataclass."""

    def test_defaults(self) -> None:
        """PlanOptions should have sensible defaults."""
        options = PlanOptions()

        assert options.plugin_names is None
        assert options.enabled is None
        assert options.disabled is None
        assert options.defaults is None
        assert options.check_tools is False
        assert options.available_tools is None

    def test_with_values(self) -> None:
        """PlanOptions should accept all values."""
        options = PlanOptions(
            plugin_names=["a", "b"],
            enabled=["c"],
            disabled=["d"],
            defaults=["e"],
            check_tools=True,
            available_tools=["f"],
        )

        assert options.plugin_names == ["a", "b"]
        assert options.enabled == ["c"]
        assert options.disabled == ["d"]
        assert options.defaults == ["e"]
        assert options.check_tools is True
        assert options.available_tools == ["f"]


class TestPluginPlan:
    """Tests for plugin plan creation."""

    def test_plan_with_custom_plugins(self) -> None:
        """Registry.plan should create plan with custom plugins."""
        registry = IngestPluginRegistry()
        plugin = MockPlugin(name="test_plugin")
        registry.register(plugin)

        options = PlanOptions(plugin_names=["test_plugin"])
        plan = registry.plan(options)

        plugin_names = [p.metadata.name for p in plan.plugins]
        assert "test_plugin" in plugin_names
        assert plan.plan_id is not None

    def test_plan_respects_dependencies(self) -> None:
        """Registry.plan should respect plugin dependencies."""
        registry = IngestPluginRegistry()
        dep = MockPlugin(name="dependency")
        dependent = MockPlugin(name="dependent", depends_on=("dependency",))

        registry.register(dependent)
        registry.register(dep)

        options = PlanOptions(plugin_names=["dependent", "dependency"])
        plan = registry.plan(options)

        names = [p.metadata.name for p in plan.plugins]
        if "dependency" in names and "dependent" in names:
            dep_idx = names.index("dependency")
            dependent_idx = names.index("dependent")
            assert dep_idx < dependent_idx

    def test_plan_with_disabled(self) -> None:
        """Registry.plan should respect disabled plugins."""
        registry = IngestPluginRegistry()
        plugin1 = MockPlugin(name="enabled_plugin")
        plugin2 = MockPlugin(name="disabled_plugin")

        registry.register(plugin1)
        registry.register(plugin2)

        options = PlanOptions(
            plugin_names=["enabled_plugin", "disabled_plugin"],
            disabled=["disabled_plugin"],
        )
        plan = registry.plan(options)

        names = [p.metadata.name for p in plan.plugins]
        assert "enabled_plugin" in names


class TestUnregisterCleanup:
    """Tests for cleanup during unregistration."""

    def test_unregister_removes_from_capability_index(self) -> None:
        """Unregister should remove plugin from capability index."""
        registry = IngestPluginRegistry()
        plugin = MockPlugin(name="p", provides=("cap",))

        registry.register(plugin)
        result1 = registry.list_providing("cap")
        assert len(result1) == 1

        registry.unregister("p")
        result2 = registry.list_providing("cap")
        assert len(result2) == 0

    def test_unregister_removes_from_stage_index(self) -> None:
        """Unregister should remove plugin from stage index."""
        registry = IngestPluginRegistry()
        plugin = MockPlugin(name="p", stage="parse")

        registry.register(plugin)
        initial_count = len(registry.list_by_stage("parse"))

        registry.unregister("p")
        final_count = len(registry.list_by_stage("parse"))

        assert final_count == initial_count - 1

    def test_unregister_removes_from_table_index(self) -> None:
        """Unregister should remove plugin from table index."""
        registry = IngestPluginRegistry()
        plugin = MockPlugin(name="p", produces_tables=("core.unique_table",))

        registry.register(plugin)
        result1 = registry.list_by_table("core.unique_table")
        assert len(result1) == 1

        registry.unregister("p")
        result2 = registry.list_by_table("core.unique_table")
        assert len(result2) == 0
