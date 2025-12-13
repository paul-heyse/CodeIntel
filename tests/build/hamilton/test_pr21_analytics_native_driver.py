"""Tests for PR-21: Native analytics driver composition.

This module validates that the auto driver mode correctly loads and composes
native analytics modules with the generated assets and wrapper modules.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver, list_available_nodes


def test_auto_driver_includes_native_analytics_nodes() -> None:
    """Verify auto driver includes compute and materialize nodes for native analytics."""
    # Build driver with mode="auto"
    runtime = build_driver(mode="auto")

    # List all available nodes
    all_nodes = list_available_nodes(runtime, mode="auto")

    # Expected native analytics compute nodes
    expected_compute_nodes = [
        "t__coverage_functions__compute",
        "t__hotspots__compute",
        "t__subsystems__compute",
    ]

    # Expected native analytics materialize nodes
    expected_materialize_nodes = [
        "t__coverage_functions",
        "t__hotspots",
        "t__subsystems",
    ]

    # Check compute nodes
    for node_name in expected_compute_nodes:
        if node_name not in all_nodes:
            pytest.fail(f"Expected compute node '{node_name}' not found in auto driver")

    # Check materialize nodes
    for node_name in expected_materialize_nodes:
        if node_name not in all_nodes:
            pytest.fail(f"Expected materialize node '{node_name}' not found in auto driver")


def test_auto_driver_excludes_wrapper_for_native_targets() -> None:
    """Verify auto driver does not include wrapper t__ nodes for native targets."""
    # Build driver with mode="auto"
    runtime = build_driver(mode="auto")

    # List all available nodes
    all_nodes = list_available_nodes(runtime, mode="auto")

    # Native targets should not have duplicate wrapper nodes
    # The driver should exclude these from the wrapper module
    native_target_names = ["coverage_functions", "hotspots", "subsystems"]

    # Check that we have the native nodes, not wrapper duplicates
    for target_name in native_target_names:
        node_name = f"t__{target_name}"

        # The node should exist (from native module)
        if node_name not in all_nodes:
            pytest.fail(f"Expected node '{node_name}' not found in auto driver")

        # Verify it's the native version by checking for compute node existence
        compute_node_name = f"t__{target_name}__compute"
        if compute_node_name not in all_nodes:
            pytest.fail(
                f"Found '{node_name}' but no '{compute_node_name}', "
                f"suggesting wrapper instead of native"
            )


def test_auto_driver_includes_loader_nodes() -> None:
    """Verify auto driver includes loader nodes from assets module."""
    # Build driver with mode="auto"
    runtime = build_driver(mode="auto")

    # List all available nodes
    all_nodes = list_available_nodes(runtime, mode="auto")

    # Loader nodes should be present from the assets module
    # Check for a few key loader nodes used by native analytics

    # These should exist if the analytics targets are registered
    expected_loader_patterns = [
        "q__graph__goids",
        "q__analytics__coverage_lines",
        "q__core__modules",
        "q__core__file_state",
        "q__graph__import_graph_edges",
    ]

    missing_loaders = [node for node in expected_loader_patterns if node not in all_nodes]

    # It's okay if some loaders are missing (depends on what's generated)
    # but we should have at least one
    if len(missing_loaders) == len(expected_loader_patterns):
        pytest.fail(
            f"No expected loader nodes found. Missing: {missing_loaders}. "
            f"Assets module may not be loaded."
        )


def test_risk_factors_native_still_present_in_wave2() -> None:
    """Verify Wave 1 native target (risk_factors) is still present in auto driver."""
    # Build driver with mode="auto"
    runtime = build_driver(mode="auto")

    # List all available nodes
    all_nodes = list_available_nodes(runtime, mode="auto")

    # risk_factors should have its native nodes
    expected_risk_factors_nodes = [
        "t__risk_factors__compute",
        "t__risk_factors",
    ]

    for node_name in expected_risk_factors_nodes:
        if node_name not in all_nodes:
            pytest.fail(
                f"Expected risk_factors node '{node_name}' not found. "
                f"Wave 1 target may have been broken by Wave 2 changes."
            )
