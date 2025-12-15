"""Tests for PR-08: Hamilton default mode and CLI options.

Validate that Hamilton is the default build engine and that
generated mode is the default node mode.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver, list_available_nodes
from codeintel.build.hamilton.executor import HamiltonBuildExecutor
from codeintel.build.hamilton.nodes.node_factory import clear_generated_module_cache


class TestHamiltonDefaultMode:
    """Tests for PR-08: Default mode verification."""

    @staticmethod
    def test_build_driver_defaults_to_generated_mode() -> None:
        """Verify build_driver defaults to generated mode when not specified."""
        clear_generated_module_cache()
        runtime = build_driver()
        if runtime.mode != "generated":
            pytest.fail(f"Expected default mode='generated', got '{runtime.mode}'")

    @staticmethod
    def test_list_available_nodes_defaults_to_generated() -> None:
        """Verify list_available_nodes defaults to generated mode."""
        clear_generated_module_cache()
        nodes = list_available_nodes()

        if len(nodes) == 0:
            pytest.fail("No nodes returned from list_available_nodes")

        target_nodes = [n for n in nodes if n.startswith("t__")]
        if len(target_nodes) == 0:
            pytest.fail("No target nodes found in generated mode")

    @staticmethod
    def test_executor_defaults_to_generated_mode() -> None:
        """Verify HamiltonBuildExecutor defaults to generated mode."""
        executor = HamiltonBuildExecutor(profile="default")
        if executor.mode != "generated":
            pytest.fail(f"Expected executor mode='generated', got '{executor.mode}'")

    @staticmethod
    def test_generated_mode_includes_all_targets() -> None:
        """Verify generated mode includes nodes for all registered targets."""
        clear_generated_module_cache()
        runtime = build_driver(mode="generated")

        if not runtime.target_to_node:
            pytest.fail("Generated mode should have target_to_node mapping")

        expected_targets = ["modules", "scip", "ast", "goids", "function_metrics"]
        for target in expected_targets:
            if target not in runtime.target_to_node:
                pytest.fail(f"Generated mode missing mapping for target: {target}")


class TestHamiltonModeConsistency:
    """Tests for mode consistency across driver and executor."""

    @staticmethod
    def test_driver_and_executor_mode_match() -> None:
        """Verify driver and executor use the same default mode."""
        clear_generated_module_cache()
        runtime = build_driver()
        executor = HamiltonBuildExecutor(profile="default")
        if runtime.mode != executor.mode:
            pytest.fail(f"Mode mismatch: driver={runtime.mode}, executor={executor.mode}")

    @staticmethod
    def test_explicit_mode_propagates() -> None:
        """Verify explicit mode is respected by both driver and executor."""
        runtime = build_driver(mode="generated")
        executor = HamiltonBuildExecutor(profile="default", mode="generated")
        if runtime.mode != "generated":
            pytest.fail(f"Expected mode='generated', got {runtime.mode}")
        if executor.mode != runtime.mode:
            pytest.fail(f"Executor mode should match runtime mode, got {executor.mode}")
