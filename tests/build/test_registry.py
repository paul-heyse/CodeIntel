"""Integration tests for the build target registry."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.target_metadata import get_target_metadata_service
from tests._helpers.assertions import expect_equal, expect_in, expect_true

MIN_INGESTION_TARGETS = 5
MIN_GRAPHS_TARGETS = 5
MIN_ANALYTICS_TARGETS = 10


class TestTargetRegistry:
    """Tests for the target registry."""

    @staticmethod
    def test_registry_not_empty() -> None:
        """Verify the target registry contains targets."""
        catalog = get_target_metadata_service().system.catalog
        expect_true(len(catalog) > 0)

    @staticmethod
    def test_modules_target_has_no_dependencies() -> None:
        """Modules target is a root with no dependencies."""
        catalog = get_target_metadata_service().system.catalog
        modules_target = catalog.get("modules")
        expect_equal(modules_target.dependencies, ())
        expect_equal(modules_target.module, "ingestion")

    @staticmethod
    def test_profiles_target_has_dependencies() -> None:
        """Profiles target depends on multiple other targets."""
        catalog = get_target_metadata_service().system.catalog
        profiles_target = catalog.get("profiles")
        expect_true(len(profiles_target.dependencies) > 0)
        expect_equal(profiles_target.module, "analytics")

    @staticmethod
    def test_get_target_graph_succeeds() -> None:
        """Get target graph succeeds with Hamilton-derived dependencies."""
        catalog = get_target_metadata_service().system.catalog
        expect_true(len(catalog) > 0)

    @staticmethod
    def test_get_target_graph_is_cached() -> None:
        """Target graph resolution returns the same instance."""
        catalog1 = get_target_metadata_service().system.catalog
        catalog2 = get_target_metadata_service().system.catalog
        expect_true(catalog1 is catalog2)

    @staticmethod
    def test_all_targets_have_valid_modules() -> None:
        """All targets have valid module assignments."""
        catalog = get_target_metadata_service().system.catalog
        valid_modules = {"ingestion", "graphs", "analytics", "export"}
        for target in catalog.all_targets:
            expect_true(target.module in valid_modules, message=f"Invalid module for {target.name}")

    @staticmethod
    def test_all_targets_have_tables() -> None:
        """All targets specify at least one output table or artifact."""
        catalog = get_target_metadata_service().system.catalog
        for target in catalog.all_targets:
            has_tables = len(catalog.table_outputs_by_target.get(target.name, ())) > 0
            has_artifacts = len(catalog.artifact_outputs_by_target.get(target.name, ())) > 0
            expect_true(
                has_tables or has_artifacts,
                message=f"No tables or artifacts for {target.name}",
            )

    @staticmethod
    def test_target_names_are_unique() -> None:
        """All target names are unique."""
        catalog = get_target_metadata_service().system.catalog
        names = [t.name for t in catalog.all_targets]
        expect_equal(len(names), len(set(names)))

    @staticmethod
    def test_topological_order_includes_all_deps() -> None:
        """Topological sort includes all transitive dependencies."""
        catalog = get_target_metadata_service().system.catalog
        order = catalog.closure(("profiles",))
        trans_deps = _transitive_deps(catalog, "profiles")
        for dep in trans_deps:
            expect_in(dep, order)

    @staticmethod
    def test_ingestion_targets_come_before_graphs() -> None:
        """Ingestion targets precede graph targets in topological order."""
        catalog = get_target_metadata_service().system.catalog
        order = catalog.closure(("call_graph",))

        call_graph_idx = order.index("call_graph")
        for dep in _transitive_deps(catalog, "call_graph"):
            dep_target = catalog.get(dep)
            if dep_target.module == "ingestion":
                expect_true(order.index(dep) < call_graph_idx)

    @staticmethod
    def test_no_cycles_in_registry() -> None:
        """Registry has no cyclic dependencies."""
        catalog = get_target_metadata_service().system.catalog
        order = catalog.closure(tuple(catalog))
        expect_equal(len(order), len(catalog))


class TestTargetsByModule:
    """Tests for module-specific targets."""

    @staticmethod
    def test_ingestion_targets_exist() -> None:
        """At least some ingestion targets are registered."""
        catalog = get_target_metadata_service().system.catalog
        ingestion = catalog.targets_for_module("ingestion")
        expect_true(len(ingestion) > 0)

    @staticmethod
    def test_graphs_targets_exist() -> None:
        """At least some graphs targets are registered."""
        catalog = get_target_metadata_service().system.catalog
        graphs = catalog.targets_for_module("graphs")
        expect_true(len(graphs) > 0)

    @staticmethod
    def test_analytics_targets_exist() -> None:
        """At least some analytics targets are registered."""
        catalog = get_target_metadata_service().system.catalog
        analytics = catalog.targets_for_module("analytics")
        expect_true(len(analytics) > 0)

    @staticmethod
    def test_module_distribution_reasonable() -> None:
        """Check target distribution across modules."""
        catalog = get_target_metadata_service().system.catalog

        ingestion = catalog.targets_for_module("ingestion")
        graphs = catalog.targets_for_module("graphs")
        analytics = catalog.targets_for_module("analytics")

        expect_true(len(ingestion) >= MIN_INGESTION_TARGETS)

        expect_true(len(graphs) >= MIN_GRAPHS_TARGETS)

        expect_true(len(analytics) >= MIN_ANALYTICS_TARGETS)


@pytest.mark.parametrize(
    "target_name",
    ["modules", "ast", "scip", "goids", "call_graph", "function_metrics", "profiles"],
)
def test_key_targets_are_registered(target_name: str) -> None:
    """Key targets are available in the registry."""
    catalog = get_target_metadata_service().system.catalog
    expect_in(target_name, catalog)


def _transitive_deps(catalog: DagCatalog, target_name: str) -> frozenset[str]:
    result: set[str] = set()
    stack = list(catalog.target_dependencies.get(target_name, ()))
    while stack:
        dep = stack.pop()
        if dep in result:
            continue
        result.add(dep)
        stack.extend(catalog.target_dependencies.get(dep, ()))
    return frozenset(result)
