"""Tests for Hamilton build planner."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.planner import HamiltonBuildPlan, PlanEntry, compute_plan
from tests._helpers.build import (
    TEST_BUILD_SETTINGS,
    make_build_config,
    make_build_paths,
    make_snapshot,
)
from tests._helpers.fakes.fake_providers import FakeProviders
from tests._helpers.harnesses.hamilton_build import BuildEnvSpec, build_test_env

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.providers import Providers
    from codeintel.storage.gateway import StorageGateway
    from tests.build.hamilton.conftest import FakeGateway


def make_test_build_env(
    gateway: FakeGateway,
    tmp_path: Path,
) -> BuildEnv:
    """Create a minimal BuildEnv for testing.

    Returns
    -------
    BuildEnv
        Build environment configured for planner tests.
    """
    snapshot = make_snapshot(tmp_path)
    paths = make_build_paths(tmp_path)
    config = make_build_config()
    providers = cast("Providers", FakeProviders.defaults())

    return build_test_env(
        BuildEnvSpec(
            gateway=cast("StorageGateway", gateway),
            snapshot=snapshot,
            paths=paths,
            providers=providers,
            build_config=config,
            settings=TEST_BUILD_SETTINGS,
        )
    )


class TestPlanEntryStructure:
    """Tests for PlanEntry dataclass structure."""

    @staticmethod
    def test_plan_entry_has_required_fields() -> None:
        """Verify PlanEntry has all required fields."""
        entry = PlanEntry(
            target="modules",
            node="t__modules",
            module="ingestion",
            status="compute",
            reason="scheduled",
            dependencies=(),
            table_keys=("ingestion.modules",),
            artifact_keys=(),
        )
        assert entry.target == "modules"
        assert entry.node == "t__modules"
        assert entry.status == "compute"
        assert entry.reason == "scheduled"
        payload = entry.to_dict()
        assert payload["target"] == "modules"
        assert payload["status"] == "compute"


class TestHamiltonBuildPlanStructure:
    """Tests for HamiltonBuildPlan dataclass structure."""

    @staticmethod
    def test_plan_has_entries() -> None:
        """Verify HamiltonBuildPlan has entries field."""
        entry = PlanEntry(
            target="a",
            node="t__a",
            module="ingestion",
            status="compute",
            reason="scheduled",
            dependencies=(),
            table_keys=(),
            artifact_keys=(),
        )
        plan = HamiltonBuildPlan(
            requested=("a",),
            closure=("a",),
            entries=(entry,),
        )
        assert len(plan.entries) == 1
        assert plan.requested == ("a",)
        assert plan.closure == ("a",)
        assert plan.to_skip == ()

    @staticmethod
    def test_plan_get_entry_method() -> None:
        """Verify get_entry returns matching entry."""
        entry_a = PlanEntry(
            target="a",
            node="t__a",
            module="ingestion",
            status="compute",
            reason="scheduled",
            dependencies=(),
            table_keys=(),
            artifact_keys=(),
        )
        entry_b = PlanEntry(
            target="b",
            node="t__b",
            module="graphs",
            status="compute",
            reason="scheduled",
            dependencies=("a",),
            table_keys=(),
            artifact_keys=(),
        )
        plan = HamiltonBuildPlan(
            requested=("b",),
            closure=("a", "b"),
            entries=(entry_a, entry_b),
        )
        found = plan.get_entry("b")
        assert found is not None
        assert found.target == "b"


class TestPlanComputation:
    """Tests for plan generation."""

    @staticmethod
    def test_compute_plan_closure(
        fake_gateway: FakeGateway,
        minimal_target_graph: DagCatalog,
        tmp_path: Path,
    ) -> None:
        """Verify compute_plan returns a closure entry for the target graph."""
        env = make_test_build_env(fake_gateway, tmp_path)
        plan = compute_plan(env=env, catalog=minimal_target_graph, requested=("c",))
        assert plan.to_compute == plan.closure
        assert plan.to_skip == ()
        assert plan.blocked == ()
        assert plan.missing == ()
