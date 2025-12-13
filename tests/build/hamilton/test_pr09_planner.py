"""Tests for PR-09: Hamilton build planner.

Validates the planner status matrix, entry structure, and plan generation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.build.contracts import OutputContract
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.planner import (
    HamiltonBuildPlan,
    PlanEntry,
    compute_plan,
)
from codeintel.build.manifest import OutputManifest
from codeintel.build.targets import OutputTarget, TargetGraph
from tests._helpers.build import make_build_config, make_build_paths, make_snapshot
from tests._helpers.fakes.fake_providers import FakeProviders

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.providers import Providers
    from codeintel.storage.gateway import StorageGateway
    from tests.build.hamilton.conftest import FakeGateway

EXPECTED_LINEAR_CLOSURE = 3


def make_test_build_env(
    gateway: FakeGateway,
    tmp_path: Path,
    manifest_index: dict[str, OutputManifest] | None = None,
    force_targets: frozenset[str] | None = None,
) -> BuildEnv:
    """Create a minimal BuildEnv for testing.

    Parameters
    ----------
    gateway
        Fake gateway for testing.
    tmp_path
        Temporary path for test artifacts.
    manifest_index
        Pre-loaded manifests.
    force_targets
        Targets to force rebuild.

    Returns
    -------
    BuildEnv
        A BuildEnv instance.
    """
    snapshot = make_snapshot(tmp_path)
    paths = make_build_paths(tmp_path)
    config = make_build_config()
    providers = cast("Providers", FakeProviders.defaults())

    return BuildEnv(
        gateway=cast("StorageGateway", gateway),
        snapshot=snapshot,
        paths=paths,
        providers=providers,
        config=config,
        force_targets=force_targets or frozenset(),
        manifest_index=manifest_index,
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
            reason="no_manifest",
            input_hash=None,
            options_hash=None,
            prior_input_hash=None,
            dependencies=(),
            table_keys=("ingestion.modules",),
        )
        if entry.target != "modules":
            pytest.fail("PlanEntry.target not set correctly")
        if entry.node != "t__modules":
            pytest.fail("PlanEntry.node not set correctly")
        if entry.status != "compute":
            pytest.fail("PlanEntry.status not set correctly")
        if entry.reason != "no_manifest":
            pytest.fail("PlanEntry.reason not set correctly")

    @staticmethod
    def test_plan_entry_has_dep_hashes_field() -> None:
        """Verify PlanEntry has dep_hashes field for explain support."""
        entry = PlanEntry(
            target="b",
            node="t__b",
            module="graphs",
            status="compute",
            reason="hash_changed",
            input_hash="current123",
            options_hash="opts",
            prior_input_hash="prior123",
            dependencies=("a",),
            table_keys=(),
            dep_hashes={"a": "hash_a"},
            prior_dep_hashes={"a": "old_hash_a"},
        )
        if not entry.dep_hashes:
            pytest.fail("PlanEntry.dep_hashes should be populated")
        if entry.dep_hashes.get("a") != "hash_a":
            pytest.fail("PlanEntry.dep_hashes['a'] incorrect")
        if not entry.prior_dep_hashes:
            pytest.fail("PlanEntry.prior_dep_hashes should be populated")


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
            reason="no_manifest",
            input_hash=None,
            options_hash=None,
            prior_input_hash=None,
            dependencies=(),
            table_keys=(),
        )
        plan = HamiltonBuildPlan(
            requested=("a",),
            closure=("a",),
            entries=(entry,),
        )
        if len(plan.entries) != 1:
            pytest.fail("HamiltonBuildPlan.entries should have 1 entry")
        if plan.requested != ("a",):
            pytest.fail("HamiltonBuildPlan.requested not set correctly")
        if plan.closure != ("a",):
            pytest.fail("HamiltonBuildPlan.closure not set correctly")

    @staticmethod
    def test_plan_get_entry_method() -> None:
        """Verify get_entry returns matching entry."""
        entry_a = PlanEntry(
            target="a",
            node="t__a",
            module="ingestion",
            status="compute",
            reason="no_manifest",
            input_hash=None,
            options_hash=None,
            prior_input_hash=None,
            dependencies=(),
            table_keys=(),
        )
        entry_b = PlanEntry(
            target="b",
            node="t__b",
            module="graphs",
            status="skip",
            reason="up_to_date",
            input_hash="hash123",
            options_hash="opts",
            prior_input_hash="hash123",
            dependencies=("a",),
            table_keys=(),
        )
        plan = HamiltonBuildPlan(
            requested=("b",),
            closure=("a", "b"),
            entries=(entry_a, entry_b),
        )
        found = plan.get_entry("b")
        if found is None:
            pytest.fail("get_entry should find entry 'b'")
        if found.status != "skip":
            pytest.fail("get_entry returned wrong entry")


class TestPlanStatusMatrix:
    """Tests for plan status determination based on manifest state."""

    @staticmethod
    def test_plan_status_no_manifest_returns_compute(
        fake_gateway: FakeGateway,
        minimal_target_graph: TargetGraph,
        tmp_path: Path,
    ) -> None:
        """Verify target with no manifest gets status=compute, reason=no_manifest."""
        env = make_test_build_env(fake_gateway, tmp_path, {})

        plan = compute_plan(
            env=env,
            graph=minimal_target_graph,
            requested=("a",),
            mode="generated",
        )

        entry = plan.get_entry("a")
        if entry is None:
            pytest.fail("Plan should contain entry for 'a'")
        if entry.status != "compute":
            pytest.fail(f"Expected status='compute', got '{entry.status}'")
        if entry.reason != "no_manifest":
            pytest.fail(f"Expected reason='no_manifest', got '{entry.reason}'")

    @staticmethod
    def test_plan_status_forced_returns_compute(
        fake_gateway: FakeGateway,
        minimal_target_graph: TargetGraph,
        tmp_path: Path,
    ) -> None:
        """Verify forced target gets status=compute, reason=forced."""
        manifest = OutputManifest(
            target="a",
            repo="test/repo",
            commit="abc123",
            plugin="ingestion.a",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="existing_hash",
            output_hash="out123",
            row_count=100,
        )

        env = make_test_build_env(
            fake_gateway,
            tmp_path,
            manifest_index={"a": manifest},
            force_targets=frozenset({"a"}),
        )

        plan = compute_plan(
            env=env,
            graph=minimal_target_graph,
            requested=("a",),
            mode="generated",
        )

        entry = plan.get_entry("a")
        if entry is None:
            pytest.fail("Plan should contain entry for 'a'")
        if entry.status != "compute":
            pytest.fail(f"Expected status='compute' for forced, got '{entry.status}'")
        if entry.reason != "forced":
            pytest.fail(f"Expected reason='forced', got '{entry.reason}'")

    @staticmethod
    def test_plan_status_upstream_missing_raises_or_blocks(
        fake_gateway: FakeGateway,
        tmp_path: Path,
    ) -> None:
        """Verify target with missing upstream dependency is handled.

        The planner may either raise KeyError for missing targets
        or mark them as blocked. Both are acceptable behaviors.
        """
        graph = TargetGraph()
        graph.register(
            OutputTarget(
                name="downstream",
                module="analytics",
                plugin="analytics.downstream",
                dependencies=("nonexistent",),
                description="Has missing dep",
            )
        )

        env = make_test_build_env(fake_gateway, tmp_path, {})

        try:
            plan = compute_plan(
                env=env,
                graph=graph,
                requested=("downstream",),
                mode="generated",
            )

            entry = plan.get_entry("downstream")
            if entry is not None and entry.status == "blocked":
                pass
            elif entry is not None:
                pytest.fail(f"Expected status='blocked' or KeyError, got '{entry.status}'")
        except KeyError:
            pass


class TestPlanClosure:
    """Tests for plan closure computation."""

    @staticmethod
    def test_plan_closure_matches_topological_order(
        fake_gateway: FakeGateway,
        minimal_target_graph: TargetGraph,
        tmp_path: Path,
    ) -> None:
        """Verify plan closure is in topological order."""
        env = make_test_build_env(fake_gateway, tmp_path, {})

        plan = compute_plan(
            env=env,
            graph=minimal_target_graph,
            requested=("c",),
            mode="generated",
        )

        if len(plan.closure) != EXPECTED_LINEAR_CLOSURE:
            pytest.fail(
                f"Expected {EXPECTED_LINEAR_CLOSURE} targets in closure, got {len(plan.closure)}"
            )

        a_idx = plan.closure.index("a")
        b_idx = plan.closure.index("b")
        c_idx = plan.closure.index("c")

        if not (a_idx < b_idx < c_idx):
            pytest.fail(f"Closure not in topological order: {plan.closure}")

    @staticmethod
    def test_plan_entry_includes_table_keys(
        fake_gateway: FakeGateway,
        tmp_path: Path,
    ) -> None:
        """Verify plan entries include table_keys from target contract."""
        graph = TargetGraph()
        graph.register(
            OutputTarget(
                name="with_tables",
                module="analytics",
                plugin="analytics.with_tables",
                description="Has contract",
                contract=OutputContract.simple(table_keys=("analytics.output_table",)),
            )
        )

        env = make_test_build_env(fake_gateway, tmp_path, {})

        plan = compute_plan(
            env=env,
            graph=graph,
            requested=("with_tables",),
            mode="generated",
        )

        entry = plan.get_entry("with_tables")
        if entry is None:
            pytest.fail("Plan should contain entry for 'with_tables'")
        if "analytics.output_table" not in entry.table_keys:
            pytest.fail(f"Expected table_keys to include contract tables: {entry.table_keys}")


class TestDryRunParity:
    """Tests for dry-run parity with build plan command."""

    @staticmethod
    def test_plan_to_dict_produces_consistent_output(
        fake_gateway: FakeGateway,
        minimal_target_graph: TargetGraph,
        tmp_path: Path,
    ) -> None:
        """Verify plan.to_dict() produces consistent serializable output.

        The dry-run output should be based on the same plan computation
        as the build plan command. This test verifies the serialization.
        """
        env = make_test_build_env(fake_gateway, tmp_path, {})

        plan = compute_plan(
            env=env,
            graph=minimal_target_graph,
            requested=("c",),
            mode="generated",
        )

        plan_dict = plan.to_dict()

        if "requested" not in plan_dict:
            pytest.fail("plan.to_dict() should include 'requested'")
        if "closure" not in plan_dict:
            pytest.fail("plan.to_dict() should include 'closure'")
        if "entries" not in plan_dict:
            pytest.fail("plan.to_dict() should include 'entries'")
        if "to_compute" not in plan_dict:
            pytest.fail("plan.to_dict() should include 'to_compute'")
        if "to_skip" not in plan_dict:
            pytest.fail("plan.to_dict() should include 'to_skip'")

    @staticmethod
    def test_plan_entries_match_closure_order(
        fake_gateway: FakeGateway,
        minimal_target_graph: TargetGraph,
        tmp_path: Path,
    ) -> None:
        """Verify plan entries are in the same order as closure.

        Both dry-run and plan should produce entries in topological order.
        """
        env = make_test_build_env(fake_gateway, tmp_path, {})

        plan = compute_plan(
            env=env,
            graph=minimal_target_graph,
            requested=("c",),
            mode="generated",
        )

        entry_targets = [e.target for e in plan.entries]
        closure_list = list(plan.closure)

        if entry_targets != closure_list:
            pytest.fail(f"Entry order {entry_targets} doesn't match closure {closure_list}")

    @staticmethod
    def test_multiple_plan_calls_produce_same_result(
        fake_gateway: FakeGateway,
        minimal_target_graph: TargetGraph,
        tmp_path: Path,
    ) -> None:
        """Verify compute_plan is deterministic across calls.

        Both dry-run and plan commands call compute_plan, so results
        must be identical.
        """
        env = make_test_build_env(fake_gateway, tmp_path, {})

        plan1 = compute_plan(
            env=env,
            graph=minimal_target_graph,
            requested=("c",),
            mode="generated",
        )

        plan2 = compute_plan(
            env=env,
            graph=minimal_target_graph,
            requested=("c",),
            mode="generated",
        )

        if plan1.closure != plan2.closure:
            pytest.fail("Plans should have same closure")
        if plan1.to_compute != plan2.to_compute:
            pytest.fail("Plans should have same to_compute list")
        if plan1.to_skip != plan2.to_skip:
            pytest.fail("Plans should have same to_skip list")
