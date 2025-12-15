"""Tests for PR-79: Hamilton and TargetGraph sources remain equivalent for wrapper targets."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.introspect import target_graph_from_hamilton
from codeintel.build.hamilton.native.registry import is_native_target
from codeintel.build.registry import get_target_graph


def test_pr79_targetgraph_and_hamilton_sources_equivalent_for_wrapper_targets() -> None:
    """Ensure wrapper-target closures are unchanged when using Hamilton-derived deps."""
    base_graph = get_target_graph()
    runtime = build_driver(mode="generated")
    derived_graph = target_graph_from_hamilton(runtime, base_graph=base_graph, strict=True)

    wrapper_targets = sorted(t.name for t in base_graph.all_targets if not is_native_target(t.name))
    sample_targets = wrapper_targets[:10]

    for target in sample_targets:
        closure_targetgraph = set(base_graph.topological_order([target]))
        closure_hamilton = set(derived_graph.topological_order([target]))
        if closure_targetgraph != closure_hamilton:
            pytest.fail(
                f"Closure mismatch for wrapper target {target}: "
                f"targetgraph={sorted(closure_targetgraph)} hamilton={sorted(closure_hamilton)}"
            )
