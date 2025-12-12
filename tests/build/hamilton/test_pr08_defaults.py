"""Tests for PR-08: Hamilton default mode and CLI options.

Validates that Hamilton is the default build engine and that
generated mode is the default node mode.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import (
    build_driver,
    list_available_nodes,
)
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
        # Generated mode should include more nodes than just phase0 nodes
        if len(nodes) == 0:
            pytest.fail("No nodes returned from list_available_nodes")
        # Should have target nodes
        target_nodes = [n for n in nodes if n.startswith("t__")]
        if len(target_nodes) == 0:
            pytest.fail("No target nodes found in generated mode")

    @staticmethod
    def test_executor_defaults_to_generated_mode() -> None:
        """Verify HamiltonBuildExecutor defaults to generated mode."""
        # Check the default value in the class signature
        executor = HamiltonBuildExecutor(profile="default")
        if executor._mode != "generated":
            pytest.fail(f"Expected executor mode='generated', got '{executor._mode}'")

    @staticmethod
    def test_hamilton_mode_phase0_still_works() -> None:
        """Verify phase0 mode still works when explicitly specified."""
        runtime = build_driver(mode="phase0")
        if runtime.mode != "phase0":
            pytest.fail(f"Expected mode='phase0', got '{runtime.mode}'")
        # Phase0 should have the explicit nodes
        nodes = list_available_nodes(mode="phase0")
        if "t__modules" not in nodes:
            pytest.fail("Phase0 mode missing t__modules node")

    @staticmethod
    def test_generated_mode_includes_all_targets() -> None:
        """Verify generated mode includes nodes for all registered targets."""
        clear_generated_module_cache()
        runtime = build_driver(mode="generated")
        # Should have mappings for all targets
        if not runtime.target_to_node:
            pytest.fail("Generated mode should have target_to_node mapping")
        # Verify key targets are mapped
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
        if runtime.mode != executor._mode:
            pytest.fail(f"Mode mismatch: driver={runtime.mode}, executor={executor._mode}")

    @staticmethod
    def test_explicit_mode_propagates() -> None:
        """Verify explicit mode is respected by both driver and executor."""
        runtime = build_driver(mode="phase0")
        executor = HamiltonBuildExecutor(profile="default", mode="phase0")
        if runtime.mode != "phase0":
            pytest.fail(f"Driver mode should be phase0, got {runtime.mode}")
        if executor._mode != "phase0":
            pytest.fail(f"Executor mode should be phase0, got {executor._mode}")
