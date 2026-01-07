"""Integration tests for the build target registry."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle
from tests._helpers.assertions import expect_equal, expect_in, expect_true

MIN_INGESTION_TARGETS = 5
MIN_GRAPHS_TARGETS = 5
MIN_ANALYTICS_TARGETS = 10


class TestTargetRegistry:
    """Tests for the target registry."""

    @staticmethod
    def test_registry_not_empty(catalog: DagCatalog) -> None:
        """Verify the target registry contains targets."""
        expect_true(len(catalog) > 0)

    @staticmethod
    def test_modules_target_has_no_dependencies(catalog: DagCatalog) -> None:
        """Modules target is a root with no dependencies."""
        modules_target = catalog.get("modules")
        expect_equal(modules_target.dependencies, ())
        expect_equal(modules_target.module, "ingestion")

    @staticmethod
    def test_function_types_target_has_dependencies(catalog: DagCatalog) -> None:
        """function_types target depends on multiple other targets."""
        types_target = catalog.get("function_types")
        expect_true(len(types_target.dependencies) > 0)
        expect_equal(types_target.module, "analytics")

    @staticmethod
    def test_get_target_graph_succeeds(catalog: DagCatalog) -> None:
        """Get target graph succeeds with Hamilton-derived dependencies."""
        expect_true(len(catalog) > 0)

    @staticmethod
    def test_get_target_graph_is_cached(hamilton_runtime: HamiltonRuntimeBundle) -> None:
        """Target graph resolution returns the same instance."""
        catalog1 = hamilton_runtime.catalog
        catalog2 = hamilton_runtime.catalog
        expect_true(catalog1 is catalog2)

    @staticmethod
    def test_all_targets_have_valid_modules(catalog: DagCatalog) -> None:
        """All targets have valid module assignments."""
        valid_modules = {"ingestion", "graphs", "analytics", "export"}
        for target in catalog.all_targets:
            expect_true(target.module in valid_modules, message=f"Invalid module for {target.name}")

    @staticmethod
    def test_all_targets_have_tables(catalog: DagCatalog) -> None:
        """All targets specify at least one output table or artifact."""
        for target in catalog.all_targets:
            has_tables = len(catalog.table_outputs_by_target.get(target.name, ())) > 0
            has_artifacts = len(catalog.artifact_outputs_by_target.get(target.name, ())) > 0
            expect_true(
                has_tables or has_artifacts,
                message=f"No tables or artifacts for {target.name}",
            )

    @staticmethod
    def test_target_names_are_unique(catalog: DagCatalog) -> None:
        """All target names are unique."""
        names = [t.name for t in catalog.all_targets]
        expect_equal(len(names), len(set(names)))

    @staticmethod
    def test_topological_order_includes_all_deps(catalog: DagCatalog) -> None:
        """Topological sort includes all transitive dependencies."""
        order = catalog.closure(("function_types",))
        trans_deps = _transitive_deps(catalog, "function_types")
        for dep in trans_deps:
            expect_in(dep, order)

    @staticmethod
    def test_ingestion_targets_come_before_graphs(catalog: DagCatalog) -> None:
        """Ingestion targets precede graph targets in topological order."""
        order = catalog.closure(("call_graph",))

        call_graph_idx = order.index("call_graph")
        for dep in _transitive_deps(catalog, "call_graph"):
            dep_target = catalog.get(dep)
            if dep_target.module == "ingestion":
                expect_true(order.index(dep) < call_graph_idx)

    @staticmethod
    def test_no_cycles_in_registry(catalog: DagCatalog) -> None:
        """Registry has no cyclic dependencies."""
        order = catalog.closure(tuple(catalog))
        expect_equal(len(order), len(catalog))


class TestTargetsByModule:
    """Tests for module-specific targets."""

    @staticmethod
    def test_ingestion_targets_exist(catalog: DagCatalog) -> None:
        """At least some ingestion targets are registered."""
        ingestion = catalog.targets_for_module("ingestion")
        expect_true(len(ingestion) > 0)

    @staticmethod
    def test_graphs_targets_exist(catalog: DagCatalog) -> None:
        """At least some graphs targets are registered."""
        graphs = catalog.targets_for_module("graphs")
        expect_true(len(graphs) > 0)

    @staticmethod
    def test_analytics_targets_exist(catalog: DagCatalog) -> None:
        """At least some analytics targets are registered."""
        analytics = catalog.targets_for_module("analytics")
        expect_true(len(analytics) > 0)

    @staticmethod
    def test_module_distribution_reasonable(catalog: DagCatalog) -> None:
        """Check target distribution across modules."""
        ingestion = catalog.targets_for_module("ingestion")
        graphs = catalog.targets_for_module("graphs")
        analytics = catalog.targets_for_module("analytics")

        expect_true(len(ingestion) >= MIN_INGESTION_TARGETS)

        expect_true(len(graphs) >= MIN_GRAPHS_TARGETS)

        expect_true(len(analytics) >= MIN_ANALYTICS_TARGETS)


@pytest.mark.parametrize(
    "target_name",
    ["modules", "ast", "scip", "goids", "call_graph", "function_types"],
)
def test_key_targets_are_registered(target_name: str, catalog: DagCatalog) -> None:
    """Key targets are available in the registry."""
    expect_in(target_name, catalog)


@pytest.fixture(scope="module")
def catalog(hamilton_runtime: HamiltonRuntimeBundle) -> DagCatalog:
    """Provide the DAG catalog for registry tests.

    Returns
    -------
    DagCatalog
        Catalog for registry tests.
    """
    return hamilton_runtime.catalog


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
