"""Tests for PR-21: Native analytics targets show correct impl_kind in plan."""

from __future__ import annotations

from dataclasses import replace

import pytest

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.planner import compute_plan
from codeintel.build.target_metadata import get_target_metadata_service
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness


def _make_env(harness: HamiltonBuildHarness, config: BuildConfig) -> BuildEnv:
    """Create a stubbed BuildEnv suitable for planning tests.

    Returns
    -------
    BuildEnv
        Build environment configured for planning-only scenarios.
    """
    return replace(
        harness.build_env(),
        config=config,
        manifest_index={},
    )


def test_native_analytics_marked_in_plan(build_harness: HamiltonBuildHarness) -> None:
    """Verify plan marks migrated analytics targets as native."""
    # Build driver in auto mode
    # Create minimal config and snapshot for planning
    config = BuildConfig()
    env = _make_env(build_harness, config)

    graph = get_target_metadata_service().system.graph

    # Compute plan for each migrated target
    native_targets = ["coverage_functions", "hotspots", "subsystems"]

    for target_name in native_targets:
        plan = compute_plan(env=env, graph=graph, requested=(target_name,))

        # Find the entry for this target in the plan
        target_entry = next((e for e in plan.entries if e.target == target_name), None)

        if target_entry is None:
            pytest.fail(f"Target '{target_name}' not found in plan")

        # Assert impl_kind is "native"
        if target_entry.impl_kind != "native":
            pytest.fail(
                f"Expected impl_kind='native' for {target_name}, got '{target_entry.impl_kind}'"
            )


def test_function_metrics_now_native_after_phase4(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Verify function_metrics is now native after Phase 4 migration.

    This target was migrated from wrapper to native in Phase 4 of the
    Hamilton Native Implementation Plan.
    """
    # Build driver in auto mode
    # Create minimal config and snapshot for planning
    config = BuildConfig()
    env = _make_env(build_harness, config)

    graph = get_target_metadata_service().system.graph

    # function_metrics was migrated in Phase 4
    target_name = "function_metrics"

    # Check if target exists in graph
    if graph.get(target_name) is None:
        pytest.skip(f"Target '{target_name}' not in graph")

    plan = compute_plan(env=env, graph=graph, requested=(target_name,))

    # Find the entry for this target in the plan
    target_entry = next((e for e in plan.entries if e.target == target_name), None)

    if target_entry is None:
        pytest.fail(f"Target '{target_name}' not found in plan")

    # Assert impl_kind is now "native" after Phase 4 migration
    if target_entry.impl_kind != "native":
        pytest.fail(
            f"Expected impl_kind='native' for {target_name}, got '{target_entry.impl_kind}'"
        )


def test_risk_factors_still_native_after_wave2(build_harness: HamiltonBuildHarness) -> None:
    """Verify Wave 1 native target (risk_factors) remains native."""
    # Build driver in auto mode
    # Create minimal config and snapshot for planning
    config = BuildConfig()
    env = _make_env(build_harness, config)

    graph = get_target_metadata_service().system.graph

    plan = compute_plan(env=env, graph=graph, requested=("risk_factors",))

    # Find the entry for risk_factors
    target_entry = next((e for e in plan.entries if e.target == "risk_factors"), None)

    if target_entry is None:
        pytest.fail("Target 'risk_factors' not found in plan")

    # Assert impl_kind is "native"
    if target_entry.impl_kind != "native":
        pytest.fail(f"Expected impl_kind='native' for risk_factors, got '{target_entry.impl_kind}'")
