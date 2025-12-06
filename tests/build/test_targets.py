"""Unit tests for OutputTarget and TargetGraph."""

from __future__ import annotations

import pytest

from codeintel.build.targets import OutputTarget, TargetGraph
from tests._helpers import assert_frozen


class TestOutputTarget:
    """Tests for OutputTarget dataclass."""

    def test_create_target_with_required_fields(self) -> None:
        """Create a target with only required fields."""
        target = OutputTarget(
            name="test_target",
            module="ingestion",
            plugin="test_plugin",
            tables=("core.test_table",),
        )
        assert target.name == "test_target"
        assert target.module == "ingestion"
        assert target.plugin == "test_plugin"
        assert target.tables == ("core.test_table",)
        assert target.dependencies == ()
        assert not target.description
        # estimated_duration_ms is now computed from TargetExecution (default: 5000ms)
        assert target.estimated_duration_ms == 5000

    def test_create_target_with_all_fields(self) -> None:
        """Create a target with all optional fields."""
        target = OutputTarget(
            name="test_target",
            module="analytics",
            plugin="test_plugin",
            tables=("analytics.test_table",),
            dependencies=("dep1", "dep2"),
            description="Test target description",
        )
        assert target.name == "test_target"
        assert target.dependencies == ("dep1", "dep2")
        assert target.description == "Test target description"
        # estimated_duration_ms is computed from default TargetExecution
        assert target.estimated_duration_ms == 5000

    def test_target_is_frozen(self) -> None:
        """Verify target is immutable."""
        target = OutputTarget(
            name="test_target",
            module="graphs",
            plugin="test_plugin",
            tables=("graph.test_table",),
        )
        assert_frozen(target, "name", "new_name")


class TestTargetGraph:
    """Tests for TargetGraph class."""

    def test_register_target(self) -> None:
        """Register a target in the graph."""
        graph = TargetGraph()
        target = OutputTarget(
            name="test_target",
            module="ingestion",
            plugin="test_plugin",
            tables=("core.test_table",),
        )
        graph.register(target)

        assert "test_target" in graph
        assert len(graph) == 1

    def test_register_duplicate_raises(self) -> None:
        """Registering the same target twice raises ValueError."""
        graph = TargetGraph()
        target = OutputTarget(
            name="test_target",
            module="ingestion",
            plugin="test_plugin",
            tables=("core.test_table",),
        )
        graph.register(target)

        with pytest.raises(ValueError, match="already registered"):
            graph.register(target)

    def test_get_target(self) -> None:
        """Get a registered target by name."""
        graph = TargetGraph()
        target = OutputTarget(
            name="test_target",
            module="ingestion",
            plugin="test_plugin",
            tables=("core.test_table",),
        )
        graph.register(target)

        retrieved = graph.get("test_target")
        assert retrieved is target

    def test_get_nonexistent_raises(self) -> None:
        """Getting a non-existent target raises KeyError."""
        graph = TargetGraph()

        with pytest.raises(KeyError, match="not found"):
            graph.get("nonexistent")

    def test_all_targets(self) -> None:
        """Get all registered targets."""
        graph = TargetGraph()
        t1 = OutputTarget(
            name="target1",
            module="ingestion",
            plugin="plugin1",
            tables=("core.t1",),
        )
        t2 = OutputTarget(
            name="target2",
            module="graphs",
            plugin="plugin2",
            tables=("graph.t2",),
        )
        graph.register(t1)
        graph.register(t2)

        all_targets = graph.all_targets
        assert len(all_targets) == 2
        assert t1 in all_targets
        assert t2 in all_targets

    def test_dependencies_of(self) -> None:
        """Get direct dependencies of a target."""
        graph = TargetGraph()
        t1 = OutputTarget(
            name="target1",
            module="ingestion",
            plugin="plugin1",
            tables=("core.t1",),
        )
        t2 = OutputTarget(
            name="target2",
            module="graphs",
            plugin="plugin2",
            tables=("graph.t2",),
            dependencies=("target1",),
        )
        graph.register(t1)
        graph.register(t2)

        deps = graph.dependencies_of("target2")
        assert deps == ("target1",)

    def test_transitive_deps(self) -> None:
        """Get transitive dependencies of a target."""
        graph = TargetGraph()
        t1 = OutputTarget(
            name="target1",
            module="ingestion",
            plugin="plugin1",
            tables=("core.t1",),
        )
        t2 = OutputTarget(
            name="target2",
            module="graphs",
            plugin="plugin2",
            tables=("graph.t2",),
            dependencies=("target1",),
        )
        t3 = OutputTarget(
            name="target3",
            module="analytics",
            plugin="plugin3",
            tables=("analytics.t3",),
            dependencies=("target2",),
        )
        graph.register(t1)
        graph.register(t2)
        graph.register(t3)

        trans_deps = graph.transitive_deps("target3")
        assert trans_deps == frozenset({"target1", "target2"})

    def test_dependents_of(self) -> None:
        """Get dependents of a target."""
        graph = TargetGraph()
        t1 = OutputTarget(
            name="target1",
            module="ingestion",
            plugin="plugin1",
            tables=("core.t1",),
        )
        t2 = OutputTarget(
            name="target2",
            module="graphs",
            plugin="plugin2",
            tables=("graph.t2",),
            dependencies=("target1",),
        )
        t3 = OutputTarget(
            name="target3",
            module="analytics",
            plugin="plugin3",
            tables=("analytics.t3",),
            dependencies=("target1",),
        )
        graph.register(t1)
        graph.register(t2)
        graph.register(t3)

        dependents = graph.dependents_of("target1")
        assert set(dependents) == {"target2", "target3"}

    def test_topological_order_simple(self) -> None:
        """Topological sort of targets."""
        graph = TargetGraph()
        t1 = OutputTarget(
            name="target1",
            module="ingestion",
            plugin="plugin1",
            tables=("core.t1",),
        )
        t2 = OutputTarget(
            name="target2",
            module="graphs",
            plugin="plugin2",
            tables=("graph.t2",),
            dependencies=("target1",),
        )
        t3 = OutputTarget(
            name="target3",
            module="analytics",
            plugin="plugin3",
            tables=("analytics.t3",),
            dependencies=("target2",),
        )
        graph.register(t1)
        graph.register(t2)
        graph.register(t3)

        order = graph.topological_order(["target3"])
        # target1 must come before target2, target2 before target3
        assert order.index("target1") < order.index("target2")
        assert order.index("target2") < order.index("target3")

    def test_topological_order_multiple_roots(self) -> None:
        """Topological sort with multiple independent roots."""
        graph = TargetGraph()
        t1 = OutputTarget(
            name="target1",
            module="ingestion",
            plugin="plugin1",
            tables=("core.t1",),
        )
        t2 = OutputTarget(
            name="target2",
            module="ingestion",
            plugin="plugin2",
            tables=("core.t2",),
        )
        t3 = OutputTarget(
            name="target3",
            module="analytics",
            plugin="plugin3",
            tables=("analytics.t3",),
            dependencies=("target1", "target2"),
        )
        graph.register(t1)
        graph.register(t2)
        graph.register(t3)

        order = graph.topological_order(["target3"])
        # Both target1 and target2 must come before target3
        assert order.index("target1") < order.index("target3")
        assert order.index("target2") < order.index("target3")

    def test_topological_order_cycle_raises(self) -> None:
        """Topological sort with cycle raises ValueError."""
        graph = TargetGraph()
        # Create a cycle: t1 -> t2 -> t3 -> t1
        t1 = OutputTarget(
            name="target1",
            module="ingestion",
            plugin="plugin1",
            tables=("core.t1",),
            dependencies=("target3",),
        )
        t2 = OutputTarget(
            name="target2",
            module="graphs",
            plugin="plugin2",
            tables=("graph.t2",),
            dependencies=("target1",),
        )
        t3 = OutputTarget(
            name="target3",
            module="analytics",
            plugin="plugin3",
            tables=("analytics.t3",),
            dependencies=("target2",),
        )
        graph.register(t1)
        graph.register(t2)
        graph.register(t3)

        with pytest.raises(ValueError, match="Cycle detected"):
            graph.topological_order(["target1"])

    def test_targets_for_module(self) -> None:
        """Filter targets by module."""
        graph = TargetGraph()
        t1 = OutputTarget(
            name="target1",
            module="ingestion",
            plugin="plugin1",
            tables=("core.t1",),
        )
        t2 = OutputTarget(
            name="target2",
            module="graphs",
            plugin="plugin2",
            tables=("graph.t2",),
        )
        t3 = OutputTarget(
            name="target3",
            module="ingestion",
            plugin="plugin3",
            tables=("core.t3",),
        )
        graph.register(t1)
        graph.register(t2)
        graph.register(t3)

        ingestion_targets = graph.targets_for_module("ingestion")
        assert len(ingestion_targets) == 2
        assert t1 in ingestion_targets
        assert t3 in ingestion_targets

    def test_validate_valid_graph(self) -> None:
        """Validate a valid graph returns no errors."""
        graph = TargetGraph()
        t1 = OutputTarget(
            name="target1",
            module="ingestion",
            plugin="plugin1",
            tables=("core.t1",),
        )
        t2 = OutputTarget(
            name="target2",
            module="graphs",
            plugin="plugin2",
            tables=("graph.t2",),
            dependencies=("target1",),
        )
        graph.register(t1)
        graph.register(t2)

        errors = graph.validate()
        assert errors == ()

    def test_validate_missing_dependency(self) -> None:
        """Validate graph with missing dependency returns error."""
        graph = TargetGraph()
        t1 = OutputTarget(
            name="target1",
            module="graphs",
            plugin="plugin1",
            tables=("graph.t1",),
            dependencies=("nonexistent",),
        )
        graph.register(t1)

        errors = graph.validate()
        assert len(errors) == 1
        assert "nonexistent" in errors[0]

    def test_iterate_over_graph(self) -> None:
        """Iterate over target names in graph."""
        graph = TargetGraph()
        t1 = OutputTarget(
            name="target1",
            module="ingestion",
            plugin="plugin1",
            tables=("core.t1",),
        )
        t2 = OutputTarget(
            name="target2",
            module="graphs",
            plugin="plugin2",
            tables=("graph.t2",),
        )
        graph.register(t1)
        graph.register(t2)

        names = list(graph)
        assert set(names) == {"target1", "target2"}
