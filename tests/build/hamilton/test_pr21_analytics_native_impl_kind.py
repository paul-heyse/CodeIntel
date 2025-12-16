"""Tests for PR-21: Native analytics targets show correct impl_kind in plan.

This module validates that the migrated analytics targets (coverage_functions,
hotspots, subsystems) are correctly marked as "native" in build plans, while
non-migrated targets remain as "wrapper".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.planner import compute_plan
from codeintel.build.registry import get_target_graph
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.storage.gateway import open_memory_gateway

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.providers import Providers


def _snapshot(tmp_path: Path) -> SnapshotRef:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    return SnapshotRef(repo="test/repo", commit="abc123", repo_root=repo_root)


def _make_env(tmp_path: Path, config: BuildConfig, snapshot: SnapshotRef) -> BuildEnv:
    """Create a stubbed BuildEnv suitable for planning tests.

    Parameters
    ----------
    tmp_path
        Temporary directory for building artifacts.
    config
        Build configuration for the plan.
    snapshot
        Snapshot reference for repo/commit under test.

    Returns
    -------
    BuildEnv
        Build environment configured for planning-only scenarios.
    """
    gateway = open_memory_gateway(validate_schema=False)
    paths = BuildPaths.from_explicit(build_dir=tmp_path / "build")
    return BuildEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        providers=cast("Providers", None),
        config=config,
        manifest_index={},
    )


def test_native_analytics_marked_in_plan(tmp_path: Path) -> None:
    """Verify plan marks migrated analytics targets as native."""
    # Build driver in auto mode
    # Create minimal config and snapshot for planning
    config = BuildConfig()
    snapshot = _snapshot(tmp_path)
    env = _make_env(tmp_path, config, snapshot)

    graph = get_target_graph()

    # Compute plan for each migrated target
    native_targets = ["coverage_functions", "hotspots", "subsystems"]

    for target_name in native_targets:
        plan = compute_plan(env=env, graph=graph, requested=(target_name,), mode="auto")

        # Find the entry for this target in the plan
        target_entry = next((e for e in plan.entries if e.target == target_name), None)

        if target_entry is None:
            pytest.fail(f"Target '{target_name}' not found in plan")

        # Assert impl_kind is "native"
        if target_entry.impl_kind != "native":
            pytest.fail(
                f"Expected impl_kind='native' for {target_name}, got '{target_entry.impl_kind}'"
            )


def test_function_metrics_now_native_after_phase4(tmp_path: Path) -> None:
    """Verify function_metrics is now native after Phase 4 migration.

    This target was migrated from wrapper to native in Phase 4 of the
    Hamilton Native Implementation Plan.
    """
    # Build driver in auto mode
    # Create minimal config and snapshot for planning
    config = BuildConfig()
    snapshot = _snapshot(tmp_path)
    env = _make_env(tmp_path, config, snapshot)

    graph = get_target_graph()

    # function_metrics was migrated in Phase 4
    target_name = "function_metrics"

    # Check if target exists in graph
    if graph.get(target_name) is None:
        pytest.skip(f"Target '{target_name}' not in graph")

    plan = compute_plan(env=env, graph=graph, requested=(target_name,), mode="auto")

    # Find the entry for this target in the plan
    target_entry = next((e for e in plan.entries if e.target == target_name), None)

    if target_entry is None:
        pytest.fail(f"Target '{target_name}' not found in plan")

    # Assert impl_kind is now "native" after Phase 4 migration
    if target_entry.impl_kind != "native":
        pytest.fail(
            f"Expected impl_kind='native' for {target_name}, got '{target_entry.impl_kind}'"
        )


def test_risk_factors_still_native_after_wave2(tmp_path: Path) -> None:
    """Verify Wave 1 native target (risk_factors) remains native."""
    # Build driver in auto mode
    # Create minimal config and snapshot for planning
    config = BuildConfig()
    snapshot = _snapshot(tmp_path)
    env = _make_env(tmp_path, config, snapshot)

    graph = get_target_graph()

    plan = compute_plan(env=env, graph=graph, requested=("risk_factors",), mode="auto")

    # Find the entry for risk_factors
    target_entry = next((e for e in plan.entries if e.target == "risk_factors"), None)

    if target_entry is None:
        pytest.fail("Target 'risk_factors' not found in plan")

    # Assert impl_kind is "native"
    if target_entry.impl_kind != "native":
        pytest.fail(f"Expected impl_kind='native' for risk_factors, got '{target_entry.impl_kind}'")
