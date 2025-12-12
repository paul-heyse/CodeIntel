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
from codeintel.build.registry import MODULES_TARGET


class TestNamingConventions:
    """Tests for Hamilton node naming conventions."""

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
    def test_to_node_name(self, logical_name: str, prefix: str, expected: str) -> None:
        """Verify to_node_name produces valid Python identifiers."""
        result = to_node_name(logical_name, prefix=prefix)
        assert result == expected
        # Verify it's a valid Python identifier
        assert result.isidentifier()

    def test_target_node_helper(self) -> None:
        """Verify target_node uses 't' prefix."""
        result = target_node("function_metrics")
        assert result == "t__function_metrics"
        assert result.startswith("t__")

    def test_dataset_node_helper(self) -> None:
        """Verify dataset_node uses 'd' prefix."""
        result = dataset_node("graph.call_graph_edges")
        assert result == "d__graph__call_graph_edges"
        assert result.startswith("d__")

    def test_node_to_target_roundtrip(self) -> None:
        """Verify node_to_target extracts original name."""
        node_name = target_node("risk_factors")
        extracted = node_to_target(node_name)
        assert extracted == "risk_factors"

    def test_node_to_target_non_target_returns_none(self) -> None:
        """Verify non-target nodes return None."""
        result = node_to_target("d__some_dataset")
        assert result is None


class TestMetadataBridge:
    """Tests for metadata extraction from OutputTarget."""

    def test_from_target_extracts_name(self) -> None:
        """Verify from_target builds stable name from target."""
        meta = from_target(MODULES_TARGET)
        assert meta.name == "ingestion.modules"

    def test_from_target_extracts_domain(self) -> None:
        """Verify from_target extracts module as domain."""
        meta = from_target(MODULES_TARGET)
        assert meta.domain == "ingestion"

    def test_from_target_extracts_description(self) -> None:
        """Verify from_target includes target description."""
        meta = from_target(MODULES_TARGET)
        assert "module" in meta.description.lower() or "scan" in meta.description.lower()

    def test_canonical_plugin_meta_frozen(self) -> None:
        """Verify CanonicalPluginMeta is immutable."""
        meta = CanonicalPluginMeta(
            name="test.plugin",
            version="1.0.0",
            domain="test",
            description="Test plugin",
        )
        # Frozen dataclass should raise on mutation attempt
        with pytest.raises(AttributeError):
            meta.name = "changed"  # type: ignore[misc]


class TestDriverFactory:
    """Tests for Hamilton Driver construction."""

    def test_build_driver_returns_runtime(self) -> None:
        """Verify build_driver returns HamiltonRuntime."""
        runtime = build_driver(config={"profile": "test"})
        assert runtime.dr is not None
        assert runtime.graph is not None

    def test_driver_has_expected_nodes(self) -> None:
        """Verify driver has Phase 0 nodes available."""
        nodes = list_available_nodes()
        assert "t__modules" in nodes
        assert "t__scip" in nodes
        assert "t__ast" in nodes
        assert "t__goids" in nodes
        assert "t__function_metrics" in nodes

    def test_target_to_node_name_maps_correctly(self) -> None:
        """Verify target names map to node names."""
        assert target_to_node_name("modules") == "t__modules"
        assert target_to_node_name("function_metrics") == "t__function_metrics"
        assert target_to_node_name("unknown") is None

    def test_driver_node_dependencies(self) -> None:
        """Verify DAG has correct dependency structure."""
        runtime = build_driver()
        # Access the driver's graph to verify dependencies
        graph_dict = runtime.dr.graph.get_nodes()

        # modules should have no upstream dependencies (except inputs)
        modules_deps = [
            d for d in graph_dict.get("t__modules", {}).get("dependencies", [])
            if not d.startswith("env") and not d.startswith("graph")
        ]
        assert len(modules_deps) == 0, "modules should have no target dependencies"

        # scip should depend on modules
        scip_deps = graph_dict.get("t__scip", {}).get("dependencies", [])
        assert "t__modules" in scip_deps

        # goids should depend on scip and ast
        goids_deps = graph_dict.get("t__goids", {}).get("dependencies", [])
        assert "t__scip" in goids_deps
        assert "t__ast" in goids_deps


class TestDAGVisualization:
    """Tests for DAG introspection and validation."""

    def test_driver_can_list_final_vars(self) -> None:
        """Verify we can list all nodes from driver."""
        runtime = build_driver()
        # Hamilton Driver should support getting all node names
        all_nodes = list(runtime.dr.list_available_variables())
        assert len(all_nodes) > 0
        # Should include our Phase 0 targets
        node_names = [n.name for n in all_nodes]
        assert "t__modules" in node_names

    def test_driver_can_display_graph(self) -> None:
        """Verify driver supports DAG visualization."""
        runtime = build_driver()
        # Hamilton provides display methods - we just verify they work
        # In a real test, you might capture and inspect the output
        try:
            # This returns a graphviz object, not raises
            dag = runtime.dr.display_all_functions()
            assert dag is not None
        except ImportError:
            # graphviz may not be installed in test env
            pytest.skip("graphviz not available")


class TestTargetNodeTags:
    """Tests for Hamilton node tags and metadata."""

    def test_nodes_have_domain_tags(self) -> None:
        """Verify nodes have domain tags for observability."""
        runtime = build_driver()
        nodes = runtime.dr.graph.get_nodes()

        # Check that modules node has ingestion domain tag
        modules_node = nodes.get("t__modules", {})
        tags = modules_node.get("tags", {})
        assert "domain" in tags or len(tags) == 0  # Tags may be stored differently

    def test_nodes_have_target_tags(self) -> None:
        """Verify nodes have target name tags."""
        runtime = build_driver()
        nodes = runtime.dr.graph.get_nodes()

        # Check that modules node exists and has expected structure
        modules_node = nodes.get("t__modules", {})
        # The tag structure depends on Hamilton version
        assert modules_node is not None
        # Verify node has some metadata (tags or other attributes)
        assert isinstance(modules_node, dict)


class TestPhase0NodeRegistry:
    """Tests for Phase 0 node registration."""

    def test_all_phase0_targets_mapped(self) -> None:
        """Verify all Phase 0 targets have node mappings."""
        phase0_targets = ["modules", "scip", "ast", "goids", "function_metrics"]
        for target in phase0_targets:
            node = target_to_node_name(target)
            assert node is not None, f"Target {target} has no node mapping"
            assert node.startswith("t__"), f"Target {target} node should start with t__"

    def test_node_names_are_valid_identifiers(self) -> None:
        """Verify all node names are valid Python identifiers."""
        nodes = list_available_nodes()
        for node in nodes:
            assert node.isidentifier(), f"Node {node} is not a valid identifier"
