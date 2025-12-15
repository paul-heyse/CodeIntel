"""PR-65: Hamilton-derived TargetGraph matches TargetGraph for a small subset."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.introspect import target_graph_from_hamilton


def test_pr65_hamilton_graph_matches_targetgraph_for_small_subset() -> None:
    """Derived dependency closure should match declarative TargetGraph for a native target."""
    runtime = build_driver(mode="auto")
    base = runtime.graph
    derived = target_graph_from_hamilton(runtime)

    requested = ["risk_factors"]
    derived_order = derived.topological_order(requested)
    base_order = base.topological_order(requested)
    if derived_order != base_order:
        pytest.fail("Hamilton-derived TargetGraph order does not match declarative TargetGraph order")

    for target_name in base.topological_order(requested):
        derived_deps = set(derived.get(target_name).dependencies)
        base_deps = set(base.get(target_name).dependencies)
        if derived_deps != base_deps:
            pytest.fail(f"{target_name} dependencies differ: {derived_deps} != {base_deps}")
