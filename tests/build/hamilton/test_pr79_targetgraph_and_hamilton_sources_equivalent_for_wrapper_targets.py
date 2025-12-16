"""Tests for PR-79: Hamilton and TargetGraph sources remain equivalent for wrapper targets.

NOTE: With the Phase 2 migration of ingestion targets to native Hamilton modules,
wrapper targets that depend on native targets will naturally have different closures
in the Hamilton-derived graph (which includes native dependencies) vs the base
TargetGraph (which only tracks registered dependencies). This test now only verifies
that the wrapper-specific portion of the closure is preserved.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.introspect import target_graph_from_hamilton
from codeintel.build.hamilton.native.registry import is_native_target
from codeintel.build.registry import get_target_graph


def test_pr79_targetgraph_and_hamilton_sources_equivalent_for_wrapper_targets() -> None:
    """Ensure wrapper-target closures contain all non-native deps from base graph.

    After Phase 2 migration, Hamilton-derived graphs may include additional native
    targets as upstream dependencies. This test verifies that the Hamilton closure
    is a superset of the non-native portion of the base graph closure.
    """
    base_graph = get_target_graph()
    runtime = build_driver(mode="generated")
    derived_graph = target_graph_from_hamilton(runtime, base_graph=base_graph, strict=True)

    wrapper_targets = sorted(t.name for t in base_graph.all_targets if not is_native_target(t.name))
    sample_targets = wrapper_targets[:10]

    for target in sample_targets:
        closure_targetgraph = set(base_graph.topological_order([target]))
        closure_hamilton = set(derived_graph.topological_order([target]))

        # Filter to only non-native targets for comparison
        # After Phase 2 migration, Hamilton may resolve additional native upstream targets
        wrapper_only_targetgraph = {t for t in closure_targetgraph if not is_native_target(t)}
        wrapper_only_hamilton = {t for t in closure_hamilton if not is_native_target(t)}

        # Wrapper targets from base graph should all be present in Hamilton-derived graph
        missing_in_hamilton = wrapper_only_targetgraph - wrapper_only_hamilton
        if missing_in_hamilton:
            pytest.fail(
                f"Wrapper targets missing in Hamilton closure for {target}: "
                f"missing={sorted(missing_in_hamilton)} "
                f"base_wrapper={sorted(wrapper_only_targetgraph)} "
                f"hamilton_wrapper={sorted(wrapper_only_hamilton)}"
            )
