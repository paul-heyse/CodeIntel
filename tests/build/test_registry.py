"""Integration tests for the build target registry."""

from __future__ import annotations

import pytest

from codeintel.build.registry import (
    ALL_TARGETS,
    MODULES_TARGET,
    PROFILES_TARGET,
    build_target_graph,
    get_target_graph,
)


class TestTargetRegistry:
    """Tests for the target registry."""

    def test_all_targets_not_empty(self) -> None:
        """Verify ALL_TARGETS contains targets."""
        assert len(ALL_TARGETS) > 0

    def test_modules_target_has_no_dependencies(self) -> None:
        """Modules target is a root with no dependencies."""
        assert MODULES_TARGET.dependencies == ()
        assert MODULES_TARGET.module == "ingestion"

    def test_profiles_target_has_dependencies(self) -> None:
        """Profiles target depends on multiple other targets."""
        assert len(PROFILES_TARGET.dependencies) > 0
        assert PROFILES_TARGET.module == "analytics"

    def test_build_target_graph_succeeds(self) -> None:
        """Build target graph without validation errors."""
        graph = build_target_graph()
        assert len(graph) == len(ALL_TARGETS)

    def test_get_target_graph_is_cached(self) -> None:
        """get_target_graph returns the same instance."""
        graph1 = get_target_graph()
        graph2 = get_target_graph()
        assert graph1 is graph2

    def test_all_targets_have_valid_modules(self) -> None:
        """All targets have valid module assignments."""
        valid_modules = {"ingestion", "graphs", "analytics", "export"}
        for target in ALL_TARGETS:
            assert target.module in valid_modules, f"Invalid module for {target.name}"

    def test_all_targets_have_tables(self) -> None:
        """All non-export targets specify at least one output table."""
        for target in ALL_TARGETS:
            # Export targets produce files, not tables
            if target.module == "export":
                continue
            assert len(target.tables) > 0, f"No tables for {target.name}"

    def test_target_names_are_unique(self) -> None:
        """All target names are unique."""
        names = [t.name for t in ALL_TARGETS]
        assert len(names) == len(set(names))

    def test_topological_order_includes_all_deps(self) -> None:
        """Topological sort includes all transitive dependencies."""
        graph = get_target_graph()

        # Get order for profiles (has many deps)
        order = graph.topological_order(["profiles"])

        # All transitive deps should be included
        trans_deps = graph.transitive_deps("profiles")
        for dep in trans_deps:
            assert dep in order

    def test_ingestion_targets_come_before_graphs(self) -> None:
        """Ingestion targets precede graph targets in topological order."""
        graph = get_target_graph()

        # Get order for call_graph (depends on ingestion)
        order = graph.topological_order(["call_graph"])

        # Ingestion deps should come before call_graph
        call_graph_idx = order.index("call_graph")
        for dep in graph.transitive_deps("call_graph"):
            dep_target = graph.get(dep)
            if dep_target.module == "ingestion":
                assert order.index(dep) < call_graph_idx

    def test_no_cycles_in_registry(self) -> None:
        """Registry has no cyclic dependencies."""
        graph = get_target_graph()

        # If we can get topological order of all, no cycles
        order = graph.topological_order(list(graph))
        assert len(order) == len(ALL_TARGETS)


class TestTargetsByModule:
    """Tests for module-specific targets."""

    def test_ingestion_targets_exist(self) -> None:
        """At least some ingestion targets are registered."""
        graph = get_target_graph()
        ingestion = graph.targets_for_module("ingestion")
        assert len(ingestion) > 0

    def test_graphs_targets_exist(self) -> None:
        """At least some graphs targets are registered."""
        graph = get_target_graph()
        graphs = graph.targets_for_module("graphs")
        assert len(graphs) > 0

    def test_analytics_targets_exist(self) -> None:
        """At least some analytics targets are registered."""
        graph = get_target_graph()
        analytics = graph.targets_for_module("analytics")
        assert len(analytics) > 0

    def test_module_distribution_reasonable(self) -> None:
        """Check target distribution across modules."""
        graph = get_target_graph()

        ingestion = graph.targets_for_module("ingestion")
        graphs = graph.targets_for_module("graphs")
        analytics = graph.targets_for_module("analytics")

        # Ingestion should have several targets
        assert len(ingestion) >= 5

        # Graphs should have several targets
        assert len(graphs) >= 5

        # Analytics should have the most targets
        assert len(analytics) >= 10


@pytest.mark.parametrize(
    "target_name",
    ["modules", "ast", "scip", "goids", "call_graph", "function_metrics", "profiles"],
)
def test_key_targets_are_registered(target_name: str) -> None:
    """Key targets are available in the registry."""
    graph = get_target_graph()
    assert target_name in graph
