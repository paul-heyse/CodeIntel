"""Test target registry and Hamilton DAG consistency."""

from __future__ import annotations

from codeintel.build.hamilton.naming import target_node
from codeintel.build.target_metadata import get_target_metadata_service
from tests._helpers.assertions import expect_true


class TestRegistryConsistency:
    """Ensure the build registry and Hamilton runtime stay aligned."""

    @staticmethod
    def test_all_targets_have_hamilton_nodes() -> None:
        """Every target descriptor must have a corresponding Hamilton target node."""
        target_system = get_target_metadata_service().system
        runtime = target_system.runtime
        catalog = target_system.catalog
        missing = {
            t.name
            for t in catalog.all_targets
            if target_node(t.name) not in runtime.dr.graph.nodes
        }
        expect_true(
            len(missing) == 0,
            message=f"Targets missing from Hamilton DAG: {sorted(missing)}",
        )
