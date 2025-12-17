"""Tests for Hamilton driver defaults and node availability.

The build system uses a single Hamilton composition path:

- Native `t__*` target nodes loaded from the unified registry.
- Generated support nodes (`d__*`, `q__*`, `df__*`, `a__*`) derived from
  target contracts.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver, list_available_nodes
from codeintel.build.hamilton.nodes.support_factory import clear_support_module_cache


class TestHamiltonDriverDefaults:
    """Driver composition invariants."""

    @staticmethod
    def test_build_driver_constructs_runtime() -> None:
        """Verify build_driver constructs a runtime with registered targets."""
        clear_support_module_cache()
        runtime = build_driver()

        if not runtime.target_to_node:
            pytest.fail("Expected target_to_node mapping to be non-empty")

        if "modules" not in runtime.target_to_node:
            pytest.fail("Expected 'modules' to be present in target_to_node mapping")

    @staticmethod
    def test_list_available_nodes_includes_target_and_support_nodes() -> None:
        """Verify driver exposes both target and support nodes."""
        clear_support_module_cache()
        nodes = list_available_nodes()

        if not nodes:
            pytest.fail("No nodes returned from list_available_nodes")

        required = [
            "t__modules",
            "d__analytics__function_metrics",
            "q__analytics__function_metrics",
            "df__analytics__function_metrics",
        ]
        missing = [name for name in required if name not in nodes]
        if missing:
            pytest.fail(f"Missing expected nodes: {missing}")
