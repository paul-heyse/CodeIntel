"""Tests for PR-21: Native analytics targets show correct impl_kind in plan.

This module validates that the migrated analytics targets (coverage_functions,
hotspots, subsystems) are correctly marked as "native" in build plans, while
non-migrated targets remain as "wrapper".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.planner import compute_plan
from codeintel.build.registry import get_target_graph
from codeintel.config.primitives import BuildConfig, SnapshotRef

if TYPE_CHECKING:
    from pathlib import Path


def _snapshot(tmp_path: Path) -> SnapshotRef:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    return SnapshotRef(repo="test/repo", commit="abc123", repo_root=repo_root)


def test_native_analytics_marked_in_plan(tmp_path: Path) -> None:
    """Verify plan marks migrated analytics targets as native."""
    # Build driver in auto mode
    runtime = build_driver(mode="auto")

    # Create minimal config and snapshot for planning
    config = BuildConfig()
    snapshot = _snapshot(tmp_path)

    graph = get_target_graph()

    # Compute plan for each migrated target
    native_targets = ["coverage_functions", "hotspots", "subsystems"]

    for target_name in native_targets:
        plan = compute_plan(
            runtime=runtime,
            graph=graph,
            targets=(target_name,),
            snapshot=snapshot,
            config=config,
        )

        # Find the entry for this target in the plan
        target_entry = next((e for e in plan.entries if e.target == target_name), None)

        if target_entry is None:
            pytest.fail(f"Target '{target_name}' not found in plan")

        # Assert impl_kind is "native"
        if target_entry.impl_kind != "native":
            pytest.fail(
                f"Expected impl_kind='native' for {target_name}, "
                f"got '{target_entry.impl_kind}'"
            )


def test_wrapper_targets_still_marked_wrapper(tmp_path: Path) -> None:
    """Verify non-migrated analytics targets remain wrapper."""
    # Build driver in auto mode
    runtime = build_driver(mode="auto")

    # Create minimal config and snapshot for planning
    config = BuildConfig()
    snapshot = _snapshot(tmp_path)

    graph = get_target_graph()

    # Test a non-native analytics target
    # Choose a target that's likely to remain wrapper in Wave 2
    wrapper_targets = ["function_metrics"]  # Should still be wrapper

    for target_name in wrapper_targets:
        # Check if target exists in graph
        if graph.get(target_name) is None:
            pytest.skip(f"Target '{target_name}' not in graph")

        plan = compute_plan(
            runtime=runtime,
            graph=graph,
            targets=(target_name,),
            snapshot=snapshot,
            config=config,
        )

        # Find the entry for this target in the plan
        target_entry = next((e for e in plan.entries if e.target == target_name), None)

        if target_entry is None:
            pytest.fail(f"Target '{target_name}' not found in plan")

        # Assert impl_kind is "wrapper"
        if target_entry.impl_kind != "wrapper":
            pytest.fail(
                f"Expected impl_kind='wrapper' for {target_name}, "
                f"got '{target_entry.impl_kind}'"
            )


def test_risk_factors_still_native_after_wave2(tmp_path: Path) -> None:
    """Verify Wave 1 native target (risk_factors) remains native."""
    # Build driver in auto mode
    runtime = build_driver(mode="auto")

    # Create minimal config and snapshot for planning
    config = BuildConfig()
    snapshot = _snapshot(tmp_path)

    graph = get_target_graph()

    plan = compute_plan(
        runtime=runtime,
        graph=graph,
        targets=("risk_factors",),
        snapshot=snapshot,
        config=config,
    )

    # Find the entry for risk_factors
    target_entry = next((e for e in plan.entries if e.target == "risk_factors"), None)

    if target_entry is None:
        pytest.fail("Target 'risk_factors' not found in plan")

    # Assert impl_kind is "native"
    if target_entry.impl_kind != "native":
        pytest.fail(
            f"Expected impl_kind='native' for risk_factors, got '{target_entry.impl_kind}'"
        )
