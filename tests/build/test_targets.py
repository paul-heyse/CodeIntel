"""Unit tests for OutputTarget and TargetGraph."""

from __future__ import annotations

import pytest

from codeintel.build.targets import OutputTarget, TargetGraph
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_length,
    expect_true,
)


class TestOutputTarget:
    """Tests for OutputTarget dataclass."""

    @staticmethod
    def test_create_target_with_required_fields() -> None:
        """Create a target with only required fields."""
        target = OutputTarget(
            name="test_target",
            module="ingestion",
            plugin="test_plugin",
            tables=("core.test_table",),
        )
        expect_equal(target.name, "test_target")
        expect_equal(target.module, "ingestion")
        expect_equal(target.plugin, "test_plugin")
        expect_equal(target.tables, ("core.test_table",))
        expect_equal(target.dependencies, ())
        expect_true(not target.description)
        # estimated_duration_ms is now computed from TargetExecution (default: 5000ms)
        expect_equal(target.estimated_duration_ms, 5000)

    @staticmethod
    def test_create_target_with_all_fields() -> None:
        """Create a target with all optional fields."""
        target = OutputTarget(
            name="test_target",
            module="analytics",
            plugin="test_plugin",
            tables=("analytics.test_table",),
            dependencies=("dep1", "dep2"),
            description="Test target description",
        )
        expect_equal(target.name, "test_target")
        expect_equal(target.dependencies, ("dep1", "dep2"))
        expect_equal(target.description, "Test target description")
        # estimated_duration_ms is computed from default TargetExecution
        expect_equal(target.estimated_duration_ms, 5000)

    @staticmethod
    def test_target_is_frozen() -> None:
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

    @staticmethod
    def test_register_target() -> None:
        """Register a target in the graph."""
        graph = TargetGraph()
        target = OutputTarget(
            name="test_target",
            module="ingestion",
            plugin="test_plugin",
            tables=("core.test_table",),
        )
        graph.register(target)

        expect_in("test_target", graph)
        expect_length(graph, 1)

    @staticmethod
    def test_register_duplicate_raises() -> None:
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

    @staticmethod
    def test_get_target() -> None:
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
        expect_true(retrieved is target)

    @staticmethod
    def test_get_nonexistent_raises() -> None:
        """Getting a non-existent target raises KeyError."""
        graph = TargetGraph()

        with pytest.raises(KeyError, match="not found"):
            graph.get("nonexistent")

    @staticmethod
    def test_all_targets() -> None:
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
        expect_length(all_targets, 2)
        expect_true(t1 in all_targets)
        expect_true(t2 in all_targets)

    @staticmethod
    def test_dependencies_of() -> None:
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
        expect_equal(deps, ("target1",))

    @staticmethod
    def test_transitive_deps() -> None:
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
        expect_equal(trans_deps, frozenset({"target1", "target2"}))

    @staticmethod
    def test_dependents_of() -> None:
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
        expect_equal(set(dependents), {"target2", "target3"})

    @staticmethod
    def test_topological_order_simple() -> None:
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
        expect_true(order.index("target1") < order.index("target2"))
        expect_true(order.index("target2") < order.index("target3"))

    @staticmethod
    def test_topological_order_multiple_roots() -> None:
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
        expect_true(order.index("target1") < order.index("target3"))
        expect_true(order.index("target2") < order.index("target3"))

    @staticmethod
    def test_topological_order_cycle_raises() -> None:
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

    @staticmethod
    def test_targets_for_module() -> None:
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
        expect_length(ingestion_targets, 2)
        expect_true(t1 in ingestion_targets)
        expect_true(t3 in ingestion_targets)

    @staticmethod
    def test_validate_valid_graph() -> None:
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
        expect_equal(errors, ())

    @staticmethod
    def test_validate_missing_dependency() -> None:
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
        expect_length(errors, 1)
        expect_in("nonexistent", errors[0])

    @staticmethod
    def test_iterate_over_graph() -> None:
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
        expect_equal(set(names), {"target1", "target2"})
