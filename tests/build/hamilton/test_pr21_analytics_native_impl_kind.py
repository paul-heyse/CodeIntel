"""Tests for PR-21: Native analytics targets show correct impl_kind in plan."""

from __future__ import annotations

from dataclasses import replace

import pytest

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.impl_kind import target_impl_kind
from codeintel.build.hamilton.planner import compute_plan
from codeintel.build.planning.model import PlanRequest
from codeintel.runtime.runtime_bundle import RuntimeBundle
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


def test_native_analytics_marked_in_plan(
    build_harness: HamiltonBuildHarness,
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify plan marks migrated analytics targets as native."""
    # Build driver in auto mode
    # Create minimal config and snapshot for planning
    config = BuildConfig()
    env = _make_env(build_harness, config)

    # Compute plan for each migrated target
    native_targets = ["coverage_functions", "external_deps", "module_profile"]

    for target_name in native_targets:
        request = PlanRequest(
            requested_targets=(target_name,),
            mode="predict",
            include_node_details=False,
            include_io_details=False,
            include_cache_details=False,
        )
        plan = compute_plan(env=env, plan_request=request, runtime=hamilton_runtime)

        # Find the entry for this target in the plan
        target_entry = next((e for e in plan.entries if e.target == target_name), None)

        if target_entry is None:
            pytest.fail(f"Target '{target_name}' not found in plan")

        impl_kind = target_impl_kind(hamilton_runtime, target_name=target_name)
        if impl_kind != "native":
            pytest.fail(f"Expected impl_kind='native' for {target_name}, got '{impl_kind}'")


def test_function_metrics_now_native_after_phase4(
    build_harness: HamiltonBuildHarness,
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify function_metrics is now native after Phase 4 migration.

    This target was migrated from wrapper to native in Phase 4 of the
    Hamilton Native Implementation Plan.
    """
    # Build driver in auto mode
    # Create minimal config and snapshot for planning
    config = BuildConfig()
    env = _make_env(build_harness, config)

    # function_metrics was migrated in Phase 4
    target_name = "function_metrics"

    # Check if target exists in catalog
    if hamilton_runtime.catalog.get_target(target_name) is None:
        pytest.skip(f"Target '{target_name}' not in catalog")

    request = PlanRequest(
        requested_targets=(target_name,),
        mode="predict",
        include_node_details=False,
        include_io_details=False,
        include_cache_details=False,
    )
    plan = compute_plan(env=env, plan_request=request, runtime=hamilton_runtime)

    # Find the entry for this target in the plan
    target_entry = next((e for e in plan.entries if e.target == target_name), None)

    if target_entry is None:
        pytest.fail(f"Target '{target_name}' not found in plan")

    impl_kind = target_impl_kind(hamilton_runtime, target_name=target_name)
    if impl_kind != "native":
        pytest.fail(f"Expected impl_kind='native' for {target_name}, got '{impl_kind}'")


def test_risk_factors_still_native_after_wave2(
    build_harness: HamiltonBuildHarness,
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify Wave 1 native target (risk_factors) remains native."""
    # Build driver in auto mode
    # Create minimal config and snapshot for planning
    config = BuildConfig()
    env = _make_env(build_harness, config)

    request = PlanRequest(
        requested_targets=("risk_factors",),
        mode="predict",
        include_node_details=False,
        include_io_details=False,
        include_cache_details=False,
    )
    plan = compute_plan(env=env, plan_request=request, runtime=hamilton_runtime)

    # Find the entry for risk_factors
    target_entry = next((e for e in plan.entries if e.target == "risk_factors"), None)

    if target_entry is None:
        pytest.fail("Target 'risk_factors' not found in plan")

    impl_kind = target_impl_kind(hamilton_runtime, target_name="risk_factors")
    if impl_kind != "native":
        pytest.fail(f"Expected impl_kind='native' for risk_factors, got '{impl_kind}'")
