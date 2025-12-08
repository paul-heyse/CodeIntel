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
from tests._helpers.assertions import expect_equal, expect_in, expect_true

MIN_INGESTION_TARGETS = 5
MIN_GRAPHS_TARGETS = 5
MIN_ANALYTICS_TARGETS = 10


class TestTargetRegistry:
    """Tests for the target registry."""

    @staticmethod
    def test_all_targets_not_empty() -> None:
        """Verify ALL_TARGETS contains targets."""
        expect_true(len(ALL_TARGETS) > 0)

    @staticmethod
    def test_modules_target_has_no_dependencies() -> None:
        """Modules target is a root with no dependencies."""
        expect_equal(MODULES_TARGET.dependencies, ())
        expect_equal(MODULES_TARGET.module, "ingestion")

    @staticmethod
    def test_profiles_target_has_dependencies() -> None:
        """Profiles target depends on multiple other targets."""
        expect_true(len(PROFILES_TARGET.dependencies) > 0)
        expect_equal(PROFILES_TARGET.module, "analytics")

    @staticmethod
    def test_build_target_graph_succeeds() -> None:
        """Build target graph without validation errors."""
        graph = build_target_graph()
        expect_equal(len(graph), len(ALL_TARGETS))

    @staticmethod
    def test_get_target_graph_is_cached() -> None:
        """get_target_graph returns the same instance."""
        graph1 = get_target_graph()
        graph2 = get_target_graph()
        expect_true(graph1 is graph2)

    @staticmethod
    def test_all_targets_have_valid_modules() -> None:
        """All targets have valid module assignments."""
        valid_modules = {"ingestion", "graphs", "analytics", "export"}
        for target in ALL_TARGETS:
            expect_true(target.module in valid_modules, message=f"Invalid module for {target.name}")

    @staticmethod
    def test_all_targets_have_tables() -> None:
        """All non-export targets specify at least one output table."""
        for target in ALL_TARGETS:
            # Export targets produce files, not tables
            if target.module == "export":
                continue
            expect_true(len(target.tables) > 0, message=f"No tables for {target.name}")

    @staticmethod
    def test_target_names_are_unique() -> None:
        """All target names are unique."""
        names = [t.name for t in ALL_TARGETS]
        expect_equal(len(names), len(set(names)))

    @staticmethod
    def test_topological_order_includes_all_deps() -> None:
        """Topological sort includes all transitive dependencies."""
        graph = get_target_graph()

        # Get order for profiles (has many deps)
        order = graph.topological_order(["profiles"])

        # All transitive deps should be included
        trans_deps = graph.transitive_deps("profiles")
        for dep in trans_deps:
            expect_in(dep, order)

    @staticmethod
    def test_ingestion_targets_come_before_graphs() -> None:
        """Ingestion targets precede graph targets in topological order."""
        graph = get_target_graph()

        # Get order for call_graph (depends on ingestion)
        order = graph.topological_order(["call_graph"])

        # Ingestion deps should come before call_graph
        call_graph_idx = order.index("call_graph")
        for dep in graph.transitive_deps("call_graph"):
            dep_target = graph.get(dep)
            if dep_target.module == "ingestion":
                expect_true(order.index(dep) < call_graph_idx)

    @staticmethod
    def test_no_cycles_in_registry() -> None:
        """Registry has no cyclic dependencies."""
        graph = get_target_graph()

        # If we can get topological order of all, no cycles
        order = graph.topological_order(list(graph))
        expect_equal(len(order), len(ALL_TARGETS))


class TestTargetsByModule:
    """Tests for module-specific targets."""

    @staticmethod
    def test_ingestion_targets_exist() -> None:
        """At least some ingestion targets are registered."""
        graph = get_target_graph()
        ingestion = graph.targets_for_module("ingestion")
        expect_true(len(ingestion) > 0)

    @staticmethod
    def test_graphs_targets_exist() -> None:
        """At least some graphs targets are registered."""
        graph = get_target_graph()
        graphs = graph.targets_for_module("graphs")
        expect_true(len(graphs) > 0)

    @staticmethod
    def test_analytics_targets_exist() -> None:
        """At least some analytics targets are registered."""
        graph = get_target_graph()
        analytics = graph.targets_for_module("analytics")
        expect_true(len(analytics) > 0)

    @staticmethod
    def test_module_distribution_reasonable() -> None:
        """Check target distribution across modules."""
        graph = get_target_graph()

        ingestion = graph.targets_for_module("ingestion")
        graphs = graph.targets_for_module("graphs")
        analytics = graph.targets_for_module("analytics")

        # Ingestion should have several targets
        expect_true(len(ingestion) >= MIN_INGESTION_TARGETS)

        # Graphs should have several targets
        expect_true(len(graphs) >= MIN_GRAPHS_TARGETS)

        # Analytics should have the most targets
        expect_true(len(analytics) >= MIN_ANALYTICS_TARGETS)


@pytest.mark.parametrize(
    "target_name",
    ["modules", "ast", "scip", "goids", "call_graph", "function_metrics", "profiles"],
)
def test_key_targets_are_registered(target_name: str) -> None:
    """Key targets are available in the registry."""
    graph = get_target_graph()
    expect_in(target_name, graph)
