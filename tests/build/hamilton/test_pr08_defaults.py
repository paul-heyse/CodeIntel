"""Tests for Hamilton driver defaults and node availability.

The build system uses a single Hamilton composition path:

- Native `t__*` target nodes loaded from the unified registry.
- Generated support nodes (`d__*`, `q__*`, `a__*`) derived from
  target contracts.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import list_available_nodes
from codeintel.runtime.runtime_bundle import RuntimeBundle


class TestHamiltonDriverDefaults:
    """Driver composition invariants."""

    @staticmethod
    def test_runtime_constructs_runtime(hamilton_runtime: RuntimeBundle) -> None:
        """Verify runtime constructs a runtime with registered targets."""
        if not hamilton_runtime.catalog.target_nodes:
            pytest.fail("Expected catalog target_nodes to be non-empty")

        if "modules" not in hamilton_runtime.catalog.target_nodes:
            pytest.fail("Expected 'modules' to be present in catalog target_nodes")

    @staticmethod
    def test_list_available_nodes_includes_target_and_support_nodes(
        hamilton_runtime: RuntimeBundle,
    ) -> None:
        """Verify driver exposes both target and support nodes."""
        nodes = list_available_nodes(runtime=hamilton_runtime)

        if not nodes:
            pytest.fail("No nodes returned from list_available_nodes")

        required = [
            "t__modules",
            "d__analytics__function_metrics",
            "q__analytics__function_metrics",
        ]
        missing = [name for name in required if name not in nodes]
        if missing:
            pytest.fail(f"Missing expected nodes: {missing}")
