"""Regression test: Hamilton is the single source of target dependencies.

Validate that the TargetGraph dependency edges match Hamilton-derived edges to
avoid drift or reintroduction of static dependency sources.
"""

from __future__ import annotations

from codeintel.build.hamilton.introspect import derive_target_dependencies
from codeintel.build.target_metadata import get_target_metadata_service
from tests._helpers.assertions import expect_true


def test_target_graph_dependencies_match_hamilton() -> None:
    """Ensure TargetGraph dependency edges match Hamilton-derived edges."""
    service = get_target_metadata_service()
    graph = service.system.graph
    derived = derive_target_dependencies(service.system.runtime)
    mismatches = {
        name: (tuple(graph.dependencies_of(name)), derived.get(name, ()))
        for name in graph
        if tuple(graph.dependencies_of(name)) != tuple(derived.get(name, ()))
    }
    expect_true(
        not mismatches,
        message="TargetGraph dependencies differ from Hamilton: " + str(sorted(mismatches.items())),
    )
