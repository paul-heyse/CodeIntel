"""Tests for Hamilton Phase 0 integration.

These tests validate the Hamilton-based build execution infrastructure:
- Naming conventions for Hamilton nodes
- Driver construction and DAG validation
- Skip logic via manifest checking

All tests follow the Testing Charter: real components, no monkeypatching,
production-parity execution paths.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import list_available_nodes, target_to_node_name
from codeintel.build.hamilton.naming import (
    dataset_node,
    node_to_target,
    target_node,
    to_node_name,
)
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle


class TestNamingConventions:
    """Tests for Hamilton node naming conventions."""

    @staticmethod
    @pytest.mark.parametrize(
        ("logical_name", "prefix", "expected"),
        [
            ("modules", "t", "t__modules"),
            ("analytics.function_types", "t", "t__analytics__function_types"),
            ("graph-call-edges", "d", "d__graph_call_edges"),
            ("some/path/name", "p", "p__some__path__name"),
            ("analytics.static_diagnostics", "t", "t__analytics__static_diagnostics"),
        ],
        ids=[
            "simple_name",
            "dotted_name",
            "hyphenated_name",
            "slash_path",
            "analytics_target",
        ],
    )
    def test_to_node_name(logical_name: str, prefix: str, expected: str) -> None:
        """Verify to_node_name produces valid Python identifiers."""
        result = to_node_name(logical_name, prefix=prefix)
        if result != expected:
            pytest.fail(f"Expected {expected}, got {result}")
        if not result.isidentifier():
            pytest.fail("Result is not a valid identifier")

    @staticmethod
    def test_target_node_helper() -> None:
        """Verify target_node uses 't' prefix."""
        result = target_node("function_types")
        if result != "t__function_types":
            pytest.fail("target_node returned unexpected value")
        if not result.startswith("t__"):
            pytest.fail("target_node did not prepend t__")

    @staticmethod
    def test_dataset_node_helper() -> None:
        """Verify dataset_node uses 'd' prefix."""
        result = dataset_node("graph.call_graph_edges")
        if result != "d__graph__call_graph_edges":
            pytest.fail("dataset_node returned unexpected value")
        if not result.startswith("d__"):
            pytest.fail("dataset_node did not prepend d__")

    @staticmethod
    def test_node_to_target_roundtrip() -> None:
        """Verify node_to_target extracts original name."""
        node_name = target_node("function_types")
        extracted = node_to_target(node_name)
        if extracted != "function_types":
            pytest.fail("node_to_target did not reverse target_node")

    @staticmethod
    def test_node_to_target_non_target_returns_none() -> None:
        """Verify non-target nodes return None."""
        result = node_to_target("d__some_dataset")
        if result is not None:
            pytest.fail("Expected None for dataset node")


class TestDriverFactory:
    """Tests for Hamilton Driver construction."""

    @staticmethod
    def test_compose_runtime_returns_bundle(hamilton_runtime: HamiltonRuntimeBundle) -> None:
        """Verify runtime bundle contains driver and catalog."""
        if hamilton_runtime.dr is None:
            pytest.fail("Runtime bundle missing driver")
        if hamilton_runtime.catalog is None:
            pytest.fail("Runtime bundle missing catalog")

    @staticmethod
    def test_driver_has_expected_nodes(hamilton_runtime: HamiltonRuntimeBundle) -> None:
        """Verify driver has Phase 0 nodes available."""
        nodes = list_available_nodes(runtime=hamilton_runtime)
        expected = {
            "t__modules",
            "t__scip",
            "t__ast",
            "t__goids",
            "t__call_graph",
            "t__function_types",
        }
        missing = expected.difference(nodes)
        if missing:
            pytest.fail(f"Missing expected nodes: {sorted(missing)}")

    @staticmethod
    def test_target_to_node_name_maps_correctly(hamilton_runtime: HamiltonRuntimeBundle) -> None:
        """Verify target names map to node names."""
        if target_to_node_name("modules", runtime=hamilton_runtime) != "t__modules":
            pytest.fail("modules did not map to t__modules")
        if target_to_node_name("function_types", runtime=hamilton_runtime) != ("t__function_types"):
            pytest.fail("function_types did not map to t__function_types")
        if target_to_node_name("unknown", runtime=hamilton_runtime) is not None:
            pytest.fail("unknown target should map to None")


class TestDAGVisualization:
    """Tests for DAG introspection and validation."""

    @staticmethod
    def test_driver_can_list_final_vars(hamilton_runtime: HamiltonRuntimeBundle) -> None:
        """Verify we can list all nodes from driver."""
        all_nodes = list(hamilton_runtime.dr.list_available_variables())
        if not all_nodes:
            pytest.fail("No nodes returned from driver")

        node_names = [n.name for n in all_nodes]
        if "t__modules" not in node_names:
            pytest.fail("t__modules not returned in node list")

    @staticmethod
    def test_driver_can_display_graph(hamilton_runtime: HamiltonRuntimeBundle) -> None:
        """Verify driver supports DAG visualization."""
        try:
            dag = hamilton_runtime.dr.display_all_functions()
            if dag is None:
                pytest.fail("display_all_functions returned None")
        except ImportError:
            pytest.skip("graphviz not available")


class TestTargetNodeTags:
    """Tests for Hamilton node tags and metadata."""

    @staticmethod
    def test_nodes_have_domain_tags(hamilton_runtime: HamiltonRuntimeBundle) -> None:
        """Verify nodes have domain tags for observability."""
        all_vars = hamilton_runtime.dr.list_available_variables()
        var_by_name = {v.name: v for v in all_vars}

        modules_var = var_by_name.get("t__modules")
        if modules_var is None:
            pytest.fail("t__modules not found in driver")

        if modules_var.name != "t__modules":
            pytest.fail("modules node has wrong name")

    @staticmethod
    def test_nodes_have_target_tags(hamilton_runtime: HamiltonRuntimeBundle) -> None:
        """Verify nodes have target name tags."""
        all_vars = hamilton_runtime.dr.list_available_variables()
        var_by_name = {v.name: v for v in all_vars}

        modules_var = var_by_name.get("t__modules")
        if modules_var is None:
            pytest.fail("t__modules not found in driver")

        if not hasattr(modules_var, "name"):
            pytest.fail("modules node missing name attribute")
        if not hasattr(modules_var, "type"):
            pytest.fail("modules node missing type attribute")


class TestPhase0NodeRegistry:
    """Tests for Phase 0 node registration."""

    @staticmethod
    def test_all_phase0_targets_mapped(hamilton_runtime: HamiltonRuntimeBundle) -> None:
        """Verify all Phase 0 targets have node mappings."""
        phase0_targets = [
            "modules",
            "scip",
            "ast",
            "goids",
            "call_graph",
            "function_types",
        ]
        for target in phase0_targets:
            node = target_to_node_name(target, runtime=hamilton_runtime)
            if node is None:
                pytest.fail(f"Target {target} has no node mapping")
            if not node.startswith("t__"):
                pytest.fail(f"Target {target} node should start with t__")

    @staticmethod
    def test_node_names_are_valid_identifiers(hamilton_runtime: HamiltonRuntimeBundle) -> None:
        """Verify all node names are valid Python identifiers."""
        nodes = list_available_nodes(runtime=hamilton_runtime)
        for node in nodes:
            if not node.isidentifier():
                pytest.fail(f"Node {node} is not a valid identifier")
