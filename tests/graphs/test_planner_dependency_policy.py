"""Tests for graph planner dependency policies and stubs."""

from __future__ import annotations

import pytest

from codeintel.graphs.core.registry import (
    DependencyPolicy,
    PlanningOptions,
    plan_graph_plugins,
    reset_graph_registry,
)


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
    if "goid_builder" not in skipped_names:
        pytest.fail("Expected missing builder to be recorded as skipped")
    if "goid_builder" not in plan.dep_graph:
        pytest.fail("Expected goid_builder to appear in dep graph")


def test_default_plan_includes_ingest_stubs() -> None:
    """Default planning should register ingest stubs to satisfy deps."""
    plan = plan_graph_plugins(plugin_names=("goid_builder",))
    if not ({"scip_ingest", "ast_extract", "repo_scan"} & set(plan.dep_graph)):
        pytest.fail("Expected ingest stubs to be present in dependency graph")
    if plan.skipped_plugins:
        pytest.fail("Did not expect skipped plugins when stubs are available")
