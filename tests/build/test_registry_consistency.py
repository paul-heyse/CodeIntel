"""Test target registry and Hamilton DAG consistency."""

from __future__ import annotations

from codeintel.build.hamilton.naming import target_node
from codeintel.runtime.runtime_bundle import RuntimeBundle
from tests._helpers.assertions import expect_true


class TestRegistryConsistency:
    """Ensure the build registry and Hamilton runtime stay aligned."""

    @staticmethod
    def test_all_targets_have_hamilton_nodes(hamilton_runtime: RuntimeBundle) -> None:
        """Every target descriptor must have a corresponding Hamilton target node."""
        catalog = hamilton_runtime.catalog
        missing = {
            t.name
            for t in catalog.all_targets
            if target_node(t.name) not in hamilton_runtime.dr.graph.nodes
        }
        expect_true(
            len(missing) == 0,
            message=f"Targets missing from Hamilton DAG: {sorted(missing)}",
        )
