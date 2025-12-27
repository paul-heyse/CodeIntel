"""Tests for Hamilton build planner."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.build.hamilton.planner import compute_plan
from codeintel.build.planning.model import BuildPlan, PlanRequest, PlanTargetEntry
from codeintel.runtime.runtime_bundle import RuntimeBundle
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
    """Tests for PlanTargetEntry dataclass structure."""

    @staticmethod
    def test_plan_entry_has_required_fields() -> None:
        """Verify PlanTargetEntry has all required fields."""
        entry = PlanTargetEntry(
            target="modules",
            domain="ingestion",
            deps=(),
            reads=(),
            writes_tables=("ingestion.modules",),
            writes_artifacts=(),
            predicted_action="compute",
            block_reasons=(),
            cache_hit_ratio=None,
            miss_nodes=(),
        )
        assert entry.target == "modules"
        assert entry.domain == "ingestion"
        assert entry.predicted_action == "compute"
        payload = entry.to_dict()
        assert payload["target"] == "modules"
        assert payload["predicted_action"] == "compute"


class TestHamiltonBuildPlanStructure:
    """Tests for BuildPlan dataclass structure."""

    @staticmethod
    def test_plan_has_entries() -> None:
        """Verify BuildPlan has entries field."""
        request = PlanRequest(
            requested_targets=("a",),
            mode="predict",
            include_node_details=False,
            include_io_details=False,
            include_cache_details=False,
        )
        entry = PlanTargetEntry(
            target="a",
            domain="ingestion",
            deps=(),
            reads=(),
            writes_tables=(),
            writes_artifacts=(),
            predicted_action="compute",
            block_reasons=(),
            cache_hit_ratio=None,
            miss_nodes=(),
        )
        plan = BuildPlan(
            request=request,
            closure=("a",),
            entries=(entry,),
            created_at_utc="2024-01-01T00:00:00Z",
            build_fingerprint="test-fingerprint",
        )
        assert len(plan.entries) == 1
        assert plan.request.requested_targets == ("a",)
        assert plan.closure == ("a",)

    @staticmethod
    def test_plan_get_entry_method() -> None:
        """Verify BuildPlan entries can be located by target."""
        request = PlanRequest(
            requested_targets=("b",),
            mode="predict",
            include_node_details=False,
            include_io_details=False,
            include_cache_details=False,
        )
        entry_a = PlanTargetEntry(
            target="a",
            domain="ingestion",
            deps=(),
            reads=(),
            writes_tables=(),
            writes_artifacts=(),
            predicted_action="compute",
            block_reasons=(),
            cache_hit_ratio=None,
            miss_nodes=(),
        )
        entry_b = PlanTargetEntry(
            target="b",
            domain="graphs",
            deps=("a",),
            reads=(),
            writes_tables=(),
            writes_artifacts=(),
            predicted_action="compute",
            block_reasons=(),
            cache_hit_ratio=None,
            miss_nodes=(),
        )
        plan = BuildPlan(
            request=request,
            closure=("a", "b"),
            entries=(entry_a, entry_b),
            created_at_utc="2024-01-01T00:00:00Z",
            build_fingerprint="test-fingerprint",
        )
        found = next((entry for entry in plan.entries if entry.target == "b"), None)
        assert found is not None
        assert found.target == "b"


class TestPlanComputation:
    """Tests for plan generation."""

    @staticmethod
    def test_compute_plan_closure(
        fake_gateway: FakeGateway,
        tmp_path: Path,
        hamilton_runtime: RuntimeBundle,
    ) -> None:
        """Verify compute_plan returns a closure entry for the target graph."""
        env = make_test_build_env(fake_gateway, tmp_path)
        request = PlanRequest(
            requested_targets=("modules",),
            mode="predict",
            include_node_details=False,
            include_io_details=False,
            include_cache_details=False,
        )
        plan = compute_plan(
            env=env,
            plan_request=request,
            runtime=hamilton_runtime,
            materialize=False,
        )
        assert "modules" in plan.closure
