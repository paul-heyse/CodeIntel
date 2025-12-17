"""Test target registry and Hamilton DAG consistency."""

from __future__ import annotations

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.naming import target_node
from codeintel.build.registry import get_target_graph
from tests._helpers.assertions import expect_true


class TestRegistryConsistency:
    """Ensure the build registry and Hamilton runtime stay aligned."""

    @staticmethod
    def test_all_targets_have_hamilton_nodes() -> None:
        """Every OutputTarget must have a corresponding Hamilton target node."""
        runtime = build_driver()
        graph = get_target_graph()
        missing = {
            t.name for t in graph.all_targets if target_node(t.name) not in runtime.dr.graph.nodes
        }
        expect_true(
            len(missing) == 0,
            message=f"Targets missing from Hamilton DAG: {sorted(missing)}",
        )

    @staticmethod
    def test_targets_do_not_declare_plugin_implementations() -> None:
        """Targets should not declare plugin implementations in Hamilton-first execution."""
        graph = get_target_graph()
        non_empty = {t.name: t.plugin for t in graph.all_targets if t.plugin}
        expect_true(
            len(non_empty) == 0,
            message=f"Targets still declare plugin implementations: {non_empty}",
        )
