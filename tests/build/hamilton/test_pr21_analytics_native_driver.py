"""Tests for PR-21: Native analytics driver composition."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import (
    list_available_nodes,
    target_to_node_name,
)
from codeintel.runtime.runtime_bundle import RuntimeBundle


def test_auto_driver_includes_native_analytics_nodes(hamilton_runtime: RuntimeBundle) -> None:
    """Verify auto driver includes target anchor nodes for native analytics."""
    all_nodes = list_available_nodes(runtime=hamilton_runtime)

    native_targets = [
        "coverage_functions",
        "external_deps",
        "function_metrics",
        "profiles",
        "risk_factors",
    ]
    expected_target_nodes = [f"t__{target_name}" for target_name in native_targets]

    for node_name in expected_target_nodes:
        if node_name not in all_nodes:
            pytest.fail(f"Expected target node '{node_name}' not found in auto driver")


def test_auto_driver_resolves_native_target_nodes(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify native targets resolve to canonical target nodes."""
    all_nodes = list_available_nodes(runtime=hamilton_runtime)

    native_targets = [
        "coverage_functions",
        "external_deps",
        "function_metrics",
        "profiles",
        "risk_factors",
    ]

    for target_name in native_targets:
        node_name = target_to_node_name(target_name, runtime=hamilton_runtime)
        if node_name is None:
            pytest.fail(f"Expected target '{target_name}' to resolve to a node")
        if node_name not in all_nodes:
            pytest.fail(f"Expected node '{node_name}' not found in auto driver")


def test_risk_factors_native_still_present_in_wave2(hamilton_runtime: RuntimeBundle) -> None:
    """Verify Wave 1 native target (risk_factors) is still present in auto driver."""
    all_nodes = list_available_nodes(runtime=hamilton_runtime)

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
