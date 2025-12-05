"""Coverage tests for ingestion runtime planning.

This module provides comprehensive tests for the planning infrastructure,
including dependency resolution, topological ordering, and plan creation.
All tests use the standard test helpers without monkeypatching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.ingestion.core.base import BaseIngestPlugin
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.plugins.protocol import IngestPluginProtocol, IngestPluginSkip, IngestStage
from codeintel.ingestion.plugins.registry import IngestPluginRegistry, PlanOptions
from codeintel.ingestion.runtime.planning import (
    IngestPlanContext,
    PluginExecutionPlan,
    plan_ingest_plugins,
    resolve_plugin_order,
)


class IsolatedIngestRegistry(IngestPluginRegistry):
    """Registry subclass for isolated testing.

    Extend IngestPluginRegistry to provide an isolated testing
    environment that doesn't load plugins from entry points.
    This allows tests to control exactly which plugins are available.
    """

    def __init__(self, plugins: tuple[IngestPluginProtocol, ...] = ()) -> None:
        """Initialize an isolated registry.

        Parameters
        ----------
        plugins
            Plugins to register in the isolated registry.
        """
        super().__init__()
        # Mark entry points as loaded to prevent auto-discovery
        self._entrypoints_loaded = True
        # Register provided plugins only
        for plugin in plugins:
            self.register(plugin)

# =============================================================================
# Test Plugins
# =============================================================================


@dataclass
class AlphaPlugin(BaseIngestPlugin):
    """Plugin with no dependencies, provides 'alpha_cap' capability."""

    plugin_name: ClassVar[str] = "alpha"
    plugin_description: ClassVar[str] = "Alpha plugin"
    plugin_stage: ClassVar[IngestStage] = "scan"
    depends_on: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("alpha_cap",)

    def compute(self, ctx: IngestExecutionContext) -> None:
        """Perform computation (no-op for testing).

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).
        """
        _ = self, ctx


@dataclass
class BravoPlugin(BaseIngestPlugin):
    """Plugin that depends on alpha, provides 'bravo_cap' capability."""

    plugin_name: ClassVar[str] = "bravo"
    plugin_description: ClassVar[str] = "Bravo plugin"
    plugin_stage: ClassVar[IngestStage] = "parse"
    depends_on: ClassVar[tuple[str, ...]] = ("alpha",)
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("bravo_cap",)

    def compute(self, ctx: IngestExecutionContext) -> None:
        """Perform computation (no-op for testing).

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).
        """
        _ = self, ctx


@dataclass
class CharliePlugin(BaseIngestPlugin):
    """Plugin that depends on bravo."""

    plugin_name: ClassVar[str] = "charlie"
    plugin_description: ClassVar[str] = "Charlie plugin"
    plugin_stage: ClassVar[IngestStage] = "enrich"
    depends_on: ClassVar[tuple[str, ...]] = ("bravo",)
    requires: ClassVar[tuple[str, ...]] = ()

    def compute(self, ctx: IngestExecutionContext) -> None:
        """Perform computation (no-op for testing).

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).
        """
        _ = self, ctx


@dataclass
class DeltaPlugin(BaseIngestPlugin):
    """Plugin that depends on alpha and bravo."""

    plugin_name: ClassVar[str] = "delta"
    plugin_description: ClassVar[str] = "Delta plugin"
    plugin_stage: ClassVar[IngestStage] = "validate"
    depends_on: ClassVar[tuple[str, ...]] = ("alpha", "bravo")
    requires: ClassVar[tuple[str, ...]] = ()

    def compute(self, ctx: IngestExecutionContext) -> None:
        """Perform computation (no-op for testing).

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).
        """
        _ = self, ctx


# Plugins for resolve_plugin_order tests (use 'requires' for dependencies)


@dataclass
class OrderAlpha(BaseIngestPlugin):
    """Plugin with no dependencies for ordering tests."""

    plugin_name: ClassVar[str] = "order_alpha"
    plugin_description: ClassVar[str] = "Order Alpha"
    plugin_stage: ClassVar[IngestStage] = "scan"
    depends_on: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[tuple[str, ...]] = ()

    def compute(self, ctx: IngestExecutionContext) -> None:
        """Perform computation (no-op for testing).

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).
        """
        _ = self, ctx


@dataclass
class OrderBravo(BaseIngestPlugin):
    """Plugin that requires order_alpha for ordering tests."""

    plugin_name: ClassVar[str] = "order_bravo"
    plugin_description: ClassVar[str] = "Order Bravo"
    plugin_stage: ClassVar[IngestStage] = "parse"
    depends_on: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[tuple[str, ...]] = ("order_alpha",)

    def compute(self, ctx: IngestExecutionContext) -> None:
        """Perform computation (no-op for testing).

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).
        """
        _ = self, ctx


@dataclass
class OrderCharlie(BaseIngestPlugin):
    """Plugin that requires order_bravo for ordering tests."""

    plugin_name: ClassVar[str] = "order_charlie"
    plugin_description: ClassVar[str] = "Order Charlie"
    plugin_stage: ClassVar[IngestStage] = "enrich"
    depends_on: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[tuple[str, ...]] = ("order_bravo",)

    def compute(self, ctx: IngestExecutionContext) -> None:
        """Perform computation (no-op for testing).

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).
        """
        _ = self, ctx


@dataclass
class OrderDelta(BaseIngestPlugin):
    """Plugin that requires order_alpha and order_bravo for ordering tests."""

    plugin_name: ClassVar[str] = "order_delta"
    plugin_description: ClassVar[str] = "Order Delta"
    plugin_stage: ClassVar[IngestStage] = "validate"
    depends_on: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[tuple[str, ...]] = ("order_alpha", "order_bravo")

    def compute(self, ctx: IngestExecutionContext) -> None:
        """Perform computation (no-op for testing).

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).
        """
        _ = self, ctx


@dataclass
class CyclicA(BaseIngestPlugin):
    """Plugin that creates a cycle with CyclicB (via requires)."""

    plugin_name: ClassVar[str] = "cyclic_a"
    plugin_description: ClassVar[str] = "Cyclic A"
    plugin_stage: ClassVar[IngestStage] = "parse"
    depends_on: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[tuple[str, ...]] = ("cyclic_b",)

    def compute(self, ctx: IngestExecutionContext) -> None:
        """Perform computation (no-op for testing).

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).
        """
        _ = self, ctx


@dataclass
class CyclicB(BaseIngestPlugin):
    """Plugin that creates a cycle with CyclicA (via requires)."""

    plugin_name: ClassVar[str] = "cyclic_b"
    plugin_description: ClassVar[str] = "Cyclic B"
    plugin_stage: ClassVar[IngestStage] = "parse"
    depends_on: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[tuple[str, ...]] = ("cyclic_a",)

    def compute(self, ctx: IngestExecutionContext) -> None:
        """Perform computation (no-op for testing).

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).
        """
        _ = self, ctx


# =============================================================================
# PluginExecutionPlan Tests
# =============================================================================


class TestPluginExecutionPlan:
    """Tests for PluginExecutionPlan dataclass."""

    def test_default_values(self) -> None:
        """Plan has sensible defaults."""
        plan = PluginExecutionPlan()
        assert plan.plugins == ()
        assert plan.skipped == ()
        assert plan.snapshot is None
        # plan_id is generated
        assert plan.plan_id.startswith("plan-")

    def test_plugin_names_property(self) -> None:
        """Plugin names property returns correct names."""
        plugins = (AlphaPlugin(), BravoPlugin())
        plan = PluginExecutionPlan(plugins=plugins)

        assert plan.plugin_names == ("alpha", "bravo")

    def test_plugin_names_empty(self) -> None:
        """Plugin names is empty tuple for no plugins."""
        plan = PluginExecutionPlan()
        assert plan.plugin_names == ()

    def test_skipped_names_property(self) -> None:
        """Skipped names property returns correct names."""
        skipped = (
            IngestPluginSkip(name="disabled_a", reason="disabled"),
            IngestPluginSkip(name="disabled_b", reason="missing_dependency"),
        )
        plan = PluginExecutionPlan(skipped=skipped)

        assert plan.skipped_names == ("disabled_a", "disabled_b")

    def test_skipped_names_empty(self) -> None:
        """Skipped names is empty tuple for no skipped plugins."""
        plan = PluginExecutionPlan()
        assert plan.skipped_names == ()

    def test_with_snapshot(self, tmp_path: Path) -> None:
        """Plan can include snapshot reference."""
        snapshot = SnapshotRef(
            repo="test/repo",
            commit="abc123",
            repo_root=tmp_path,
        )
        plan = PluginExecutionPlan(snapshot=snapshot)

        assert plan.snapshot is snapshot
        assert plan.snapshot is not None
        assert plan.snapshot.repo == "test/repo"

    def test_custom_plan_id(self) -> None:
        """Plan accepts custom plan_id."""
        plan = PluginExecutionPlan(plan_id="custom-plan-123")
        assert plan.plan_id == "custom-plan-123"


# =============================================================================
# IngestPlanContext Tests
# =============================================================================


class TestIngestPlanContext:
    """Tests for IngestPlanContext dataclass."""

    def test_construction(self, tmp_path: Path) -> None:
        """Context is constructed correctly."""
        snapshot = SnapshotRef(
            repo="test/repo",
            commit="abc123",
            repo_root=tmp_path,
        )
        registry = IngestPluginRegistry()
        options = PlanOptions(
            plugin_names=("alpha", "bravo"),
            defaults=("alpha", "bravo"),
        )

        context = IngestPlanContext(
            snapshot=snapshot,
            registry=registry,
            options=options,
        )

        assert context.snapshot is snapshot
        assert context.registry is registry
        assert context.options is options

    def test_default_options(self, tmp_path: Path) -> None:
        """Context uses default PlanOptions when not specified."""
        snapshot = SnapshotRef(
            repo="test/repo",
            commit="abc123",
            repo_root=tmp_path,
        )
        registry = IngestPluginRegistry()

        context = IngestPlanContext(
            snapshot=snapshot,
            registry=registry,
        )

        assert context.options is not None
        assert isinstance(context.options, PlanOptions)


# =============================================================================
# plan_ingest_plugins Tests
# =============================================================================


class TestPlanIngestPlugins:
    """Tests for plan_ingest_plugins function."""

    def test_empty_plugin_names(self, tmp_path: Path) -> None:
        """Plan with no plugin names uses defaults from registry."""
        snapshot = SnapshotRef(
            repo="test/repo",
            commit="abc123",
            repo_root=tmp_path,
        )
        # Create registry with only our test plugin
        registry = IsolatedIngestRegistry(plugins=(AlphaPlugin(),))

        options = PlanOptions(
            plugin_names=(),
            defaults=(),
        )
        context = IngestPlanContext(
            snapshot=snapshot,
            registry=registry,
            options=options,
        )

        plan = plan_ingest_plugins(context)

        # Empty plugin_names and defaults -> no plugins in plan
        assert plan.plugins == ()
        assert plan.snapshot is snapshot

    def test_single_plugin(self, tmp_path: Path) -> None:
        """Plan with single plugin."""
        snapshot = SnapshotRef(
            repo="test/repo",
            commit="abc123",
            repo_root=tmp_path,
        )
        registry = IsolatedIngestRegistry(plugins=(AlphaPlugin(),))

        options = PlanOptions(
            plugin_names=("alpha",),
            defaults=("alpha",),
        )
        context = IngestPlanContext(
            snapshot=snapshot,
            registry=registry,
            options=options,
        )

        plan = plan_ingest_plugins(context)

        assert plan.plugin_names == ("alpha",)
        assert plan.snapshot is snapshot

    def test_multiple_plugins_with_dependencies(self, tmp_path: Path) -> None:
        """Plan orders plugins by dependencies."""
        snapshot = SnapshotRef(
            repo="test/repo",
            commit="abc123",
            repo_root=tmp_path,
        )
        registry = IsolatedIngestRegistry(
            plugins=(AlphaPlugin(), BravoPlugin(), CharliePlugin())
        )

        options = PlanOptions(
            plugin_names=("charlie", "alpha", "bravo"),
            defaults=("alpha", "bravo", "charlie"),
        )
        context = IngestPlanContext(
            snapshot=snapshot,
            registry=registry,
            options=options,
        )

        plan = plan_ingest_plugins(context)

        # Should be ordered by dependencies
        names = plan.plugin_names
        assert "alpha" in names
        assert names.index("alpha") < names.index("bravo")
        assert names.index("bravo") < names.index("charlie")

    def test_disabled_plugins_are_skipped(self, tmp_path: Path) -> None:
        """Disabled plugins appear in skipped list."""
        snapshot = SnapshotRef(
            repo="test/repo",
            commit="abc123",
            repo_root=tmp_path,
        )
        registry = IsolatedIngestRegistry(plugins=(AlphaPlugin(), BravoPlugin()))

        options = PlanOptions(
            plugin_names=("alpha", "bravo"),
            defaults=("alpha", "bravo"),
            disabled=("bravo",),
        )
        context = IngestPlanContext(
            snapshot=snapshot,
            registry=registry,
            options=options,
        )

        plan = plan_ingest_plugins(context)

        assert "bravo" in plan.skipped_names
        assert "bravo" not in plan.plugin_names


# =============================================================================
# resolve_plugin_order Tests
# =============================================================================


class TestResolvePluginOrder:
    """Tests for resolve_plugin_order function."""

    def test_empty_plugins(self) -> None:
        """Empty plugins list returns empty list."""
        result = resolve_plugin_order([])
        assert result == []

    def test_single_plugin(self) -> None:
        """Single plugin returns list with that plugin."""
        plugins = [OrderAlpha()]
        result = resolve_plugin_order(plugins)

        assert len(result) == 1
        assert result[0].metadata.name == "order_alpha"

    def test_independent_plugins(self) -> None:
        """Independent plugins can be in any order."""
        # Create plugins with no dependencies
        plugins = [OrderAlpha()]
        result = resolve_plugin_order(plugins)

        assert len(result) == 1
        names = [p.metadata.name for p in result]
        assert "order_alpha" in names

    def test_linear_dependencies(self) -> None:
        """Linear dependency chain is ordered correctly."""
        plugins = [OrderCharlie(), OrderBravo(), OrderAlpha()]
        result = resolve_plugin_order(plugins)

        names = [p.metadata.name for p in result]
        assert names.index("order_alpha") < names.index("order_bravo")
        assert names.index("order_bravo") < names.index("order_charlie")

    def test_diamond_dependencies(self) -> None:
        """Diamond dependency pattern is resolved correctly."""
        plugins = [OrderDelta(), OrderCharlie(), OrderBravo(), OrderAlpha()]
        result = resolve_plugin_order(plugins)

        names = [p.metadata.name for p in result]
        # Alpha must come before bravo and delta
        assert names.index("order_alpha") < names.index("order_bravo")
        assert names.index("order_alpha") < names.index("order_delta")
        # Bravo must come before charlie and delta
        assert names.index("order_bravo") < names.index("order_charlie")
        assert names.index("order_bravo") < names.index("order_delta")

    def test_circular_dependency_raises_error(self) -> None:
        """Circular dependencies raise ValueError."""
        plugins = [CyclicA(), CyclicB()]

        with pytest.raises(ValueError, match="Circular dependency"):
            resolve_plugin_order(plugins)

    def test_missing_dependency_handled(self) -> None:
        """Plugins with missing dependencies are still processed.

        Dependencies outside the given plugin list are ignored.
        """
        # OrderBravo requires order_alpha, but order_alpha is not in the list
        plugins = [OrderBravo()]
        result = resolve_plugin_order(plugins)

        # Should still return bravo since the dependency is external
        assert len(result) == 1
        assert result[0].metadata.name == "order_bravo"

    def test_preserves_plugin_instances(self) -> None:
        """Returned plugins are the same instances as input."""
        alpha = OrderAlpha()
        bravo = OrderBravo()
        plugins = [bravo, alpha]

        result = resolve_plugin_order(plugins)

        # Same instances, not copies
        assert alpha in result
        assert bravo in result
