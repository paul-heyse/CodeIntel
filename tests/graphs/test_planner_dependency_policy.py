"""Tests for graph planner dependency policies and stubs."""

from __future__ import annotations

import pytest

from codeintel.graphs.core.registry import (
    DependencyPolicy,
    PlanningOptions,
    plan_graph_plugins,
    reset_graph_registry,
)
from tests._helpers.assertions import expect_in, expect_true


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    """Reset registry between tests to avoid plugin leakage."""
    reset_graph_registry()


def test_strict_policy_raises_on_missing_dependency() -> None:
    """Strict mode should raise when required deps are absent."""
    with pytest.raises(ValueError, match="not registered"):
        plan_graph_plugins(
            plugin_names=("goid_builder",),
            plan_options=PlanningOptions(
                allow_missing_dependencies=False,
                dependency_policy=DependencyPolicy.STRICT,
                use_stubs=False,
            ),
        )


def test_skip_policy_records_skipped_dependencies() -> None:
    """Skip policy should not raise and should mark missing deps as skipped."""
    plan = plan_graph_plugins(
        plugin_names=("goid_builder",),
        plan_options=PlanningOptions(
            allow_missing_dependencies=False,
            dependency_policy=DependencyPolicy.SKIP,
            use_stubs=False,
            requested_required=False,
        ),
    )
    skipped_names = {skip.name for skip in plan.skipped_plugins}
    expect_in("goid_builder", skipped_names, label="skipped_plugin")
    expect_in("goid_builder", set(plan.dep_graph), label="dep_graph")


def test_default_plan_includes_ingest_stubs() -> None:
    """Default planning should register ingest stubs to satisfy deps."""
    plan = plan_graph_plugins(plugin_names=("goid_builder",))
    expect_true(
        bool({"scip_ingest", "ast_extract", "repo_scan"} & set(plan.dep_graph)),
        message="Expected ingest stubs to be present in dependency graph",
    )
    expect_true(not plan.skipped_plugins, message="Did not expect skipped plugins when stubs exist")
