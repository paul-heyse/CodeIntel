"""Test target registry and Hamilton DAG consistency."""

from __future__ import annotations

from codeintel.build.hamilton.naming import target_node
from codeintel.build.target_system import load_target_system
from tests._helpers.assertions import expect_true


class TestRegistryConsistency:
    """Ensure the build registry and Hamilton runtime stay aligned."""

    @staticmethod
    def test_all_targets_have_hamilton_nodes() -> None:
        """Every OutputTarget must have a corresponding Hamilton target node."""
        target_system = load_target_system()
        runtime = target_system.runtime
        graph = target_system.graph
        missing = {
            t.name for t in graph.all_targets if target_node(t.name) not in runtime.dr.graph.nodes
        }
        expect_true(
            len(missing) == 0,
            message=f"Targets missing from Hamilton DAG: {sorted(missing)}",
        )
