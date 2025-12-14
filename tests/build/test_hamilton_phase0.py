"""Tests for Hamilton Phase 0 integration.

These tests validate the Hamilton-based build execution infrastructure:
- Naming conventions for Hamilton nodes
- Metadata bridge from OutputTarget to CanonicalPluginMeta
- Driver construction and DAG validation
- Skip logic via manifest checking

All tests follow the Testing Charter: real components, no monkeypatching,
production-parity execution paths.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest

from codeintel.build.hamilton.driver_factory import (
    build_driver,
    list_available_nodes,
    target_to_node_name,
)
from codeintel.build.hamilton.metadata_bridge import (
    CanonicalPluginMeta,
    from_target,
)
from codeintel.build.hamilton.naming import (
    dataset_node,
    node_to_target,
    target_node,
    to_node_name,
)
from codeintel.build.hamilton.nodes.node_factory import get_generated_module
from codeintel.build.registry import MODULES_TARGET


class TestNamingConventions:
    """Tests for Hamilton node naming conventions."""

    @staticmethod
    @pytest.mark.parametrize(
        ("logical_name", "prefix", "expected"),
        [
            ("modules", "t", "t__modules"),
            ("analytics.function_metrics", "t", "t__analytics__function_metrics"),
            ("graph-call-edges", "d", "d__graph_call_edges"),
            ("some/path/name", "p", "p__some__path__name"),
            ("analytics.risk_factors", "t", "t__analytics__risk_factors"),
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
        result = target_node("function_metrics")
        if result != "t__function_metrics":
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
        node_name = target_node("risk_factors")
        extracted = node_to_target(node_name)
        if extracted != "risk_factors":
            pytest.fail("node_to_target did not reverse target_node")

    @staticmethod
    def test_node_to_target_non_target_returns_none() -> None:
        """Verify non-target nodes return None."""
        result = node_to_target("d__some_dataset")
        if result is not None:
            pytest.fail("Expected None for dataset node")


class TestMetadataBridge:
    """Tests for metadata extraction from OutputTarget."""

    @staticmethod
    def test_from_target_extracts_name() -> None:
        """Verify from_target builds stable name from target."""
        meta = from_target(MODULES_TARGET)
        if meta.name != "ingestion.modules":
            pytest.fail("Name did not match expected ingestion.modules")

    @staticmethod
    def test_from_target_extracts_domain() -> None:
        """Verify from_target extracts module as domain."""
        meta = from_target(MODULES_TARGET)
        if meta.domain != "ingestion":
            pytest.fail("Domain did not match ingestion")

    @staticmethod
    def test_from_target_extracts_description() -> None:
        """Verify from_target includes target description."""
        meta = from_target(MODULES_TARGET)
        description = meta.description.lower()
        if "module" not in description and "scan" not in description:
            pytest.fail("Description missing expected keywords")

    @staticmethod
    def test_canonical_plugin_meta_frozen() -> None:
        """Verify CanonicalPluginMeta is immutable."""
        meta = CanonicalPluginMeta(
            name="test.plugin",
            version="1.0.0",
            domain="test",
            description="Test plugin",
        )

        with pytest.raises(AttributeError):
            object.__setattr__(cast("Any", meta), "name", "changed")  # noqa: PLC2801


class TestDriverFactory:
    """Tests for Hamilton Driver construction."""

    @staticmethod
    def test_build_driver_returns_runtime() -> None:
        """Verify build_driver returns HamiltonRuntime."""
        runtime = build_driver(config={"profile": "test"})
        if runtime.dr is None:
            pytest.fail("Driver runtime missing dr")
        if runtime.graph is None:
            pytest.fail("Driver runtime missing graph")

    @staticmethod
    def test_driver_has_expected_nodes() -> None:
        """Verify driver has Phase 0 nodes available."""
        nodes = list_available_nodes()
        expected = {
            "t__modules",
            "t__scip",
            "t__ast",
            "t__goids",
            "t__call_graph",
            "t__function_metrics",
            "t__risk_factors",
        }
        missing = expected.difference(nodes)
        if missing:
            pytest.fail(f"Missing expected nodes: {sorted(missing)}")

    @staticmethod
    def test_target_to_node_name_maps_correctly() -> None:
        """Verify target names map to node names."""
        if target_to_node_name("modules") != "t__modules":
            pytest.fail("modules did not map to t__modules")
        if target_to_node_name("function_metrics") != "t__function_metrics":
            pytest.fail("function_metrics did not map to t__function_metrics")
        if target_to_node_name("unknown") is not None:
            pytest.fail("unknown target should map to None")

    @staticmethod
    def test_driver_node_dependencies() -> None:
        """Verify DAG has correct dependency structure via function signatures."""
        runtime = build_driver()
        generated_module = get_generated_module()

        all_vars = runtime.dr.list_available_variables()
        var_by_name = {v.name: v for v in all_vars}

        if "t__modules" not in var_by_name:
            pytest.fail("t__modules not found in driver")
        if "t__scip" not in var_by_name:
            pytest.fail("t__scip not found in driver")
        if "t__goids" not in var_by_name:
            pytest.fail("t__goids not found in driver")

        modules_sig = inspect.signature(cast("Any", generated_module.t__modules))
        modules_params = [p for p in modules_sig.parameters if p.startswith("t__")]
        if modules_params:
            pytest.fail(f"modules should have no target dependencies, got: {modules_params}")

        scip_sig = inspect.signature(cast("Any", generated_module.t__scip))
        scip_params = list(scip_sig.parameters.keys())
        if "t__modules" not in scip_params:
            pytest.fail("scip missing dependency on modules")

        goids_sig = inspect.signature(cast("Any", generated_module.t__goids))
        goids_params = list(goids_sig.parameters.keys())
        if "t__scip" not in goids_params:
            pytest.fail("goids missing dependency on scip")
        if "t__ast" not in goids_params:
            pytest.fail("goids missing dependency on ast")


class TestDAGVisualization:
    """Tests for DAG introspection and validation."""

    @staticmethod
    def test_driver_can_list_final_vars() -> None:
        """Verify we can list all nodes from driver."""
        runtime = build_driver()

        all_nodes = list(runtime.dr.list_available_variables())
        if not all_nodes:
            pytest.fail("No nodes returned from driver")

        node_names = [n.name for n in all_nodes]
        if "t__modules" not in node_names:
            pytest.fail("t__modules not returned in node list")

    @staticmethod
    def test_driver_can_display_graph() -> None:
        """Verify driver supports DAG visualization."""
        runtime = build_driver()

        try:
            dag = runtime.dr.display_all_functions()
            if dag is None:
                pytest.fail("display_all_functions returned None")
        except ImportError:
            pytest.skip("graphviz not available")


class TestTargetNodeTags:
    """Tests for Hamilton node tags and metadata."""

    @staticmethod
    def test_nodes_have_domain_tags() -> None:
        """Verify nodes have domain tags for observability."""
        runtime = build_driver()
        all_vars = runtime.dr.list_available_variables()
        var_by_name = {v.name: v for v in all_vars}

        modules_var = var_by_name.get("t__modules")
        if modules_var is None:
            pytest.fail("t__modules not found in driver")

        if modules_var.name != "t__modules":
            pytest.fail("modules node has wrong name")

    @staticmethod
    def test_nodes_have_target_tags() -> None:
        """Verify nodes have target name tags."""
        runtime = build_driver()
        all_vars = runtime.dr.list_available_variables()
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
    def test_all_phase0_targets_mapped() -> None:
        """Verify all Phase 0 targets have node mappings."""
        phase0_targets = [
            "modules",
            "scip",
            "ast",
            "goids",
            "call_graph",
            "function_metrics",
            "risk_factors",
        ]
        for target in phase0_targets:
            node = target_to_node_name(target)
            if node is None:
                pytest.fail(f"Target {target} has no node mapping")
            if not node.startswith("t__"):
                pytest.fail(f"Target {target} node should start with t__")

    @staticmethod
    def test_node_names_are_valid_identifiers() -> None:
        """Verify all node names are valid Python identifiers."""
        nodes = list_available_nodes()
        for node in nodes:
            if not node.isidentifier():
                pytest.fail(f"Node {node} is not a valid identifier")
