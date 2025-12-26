"""Tests for the unified StateComputer.

This module tests the StateComputer class which provides the single source
of truth for target state computation. Tests verify:

1. Correct status computation (current, stale, missing, blocked)
2. Proper blocking propagation through dependencies
3. Session caching efficiency
4. Equivalence with StateValidator results (both now use unified types)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.contracts import EMPTY_CONTRACT
from codeintel.build.session import BuildSession
from codeintel.build.state import StateValidationOptions, StateValidator
from codeintel.build.state_computer import StateComputer
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.primitives import SnapshotRef
from codeintel.core.build_manifest import OutputManifest
from codeintel.core.config.settings import BuildSettings, ExportAuditSettings
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway

TEST_BUILD_SETTINGS = BuildSettings(
    engine_version="test",
    export_audit=ExportAuditSettings(),
)


def make_snapshot(tmp_path: Path, repo: str = "test/repo", commit: str = "abc123") -> SnapshotRef:
    """Create a test snapshot reference.

    Returns
    -------
    SnapshotRef
        Snapshot reference pointing at the temporary repo root.
    """
    return SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path)


def make_manifest(
    target: str,
    repo: str = "test/repo",
    commit: str = "abc123",
    input_hash: str = "hash123",
) -> OutputManifest:
    """Create a test manifest.

    Returns
    -------
    OutputManifest
        Manifest populated with defaults for the provided target.
    """
    return OutputManifest(
        target=target,
        repo=repo,
        commit=commit,
        impl_kind=f"{target}_impl",
        computed_at=datetime.now(tz=UTC),
        duration_ms=100.0,
        input_hash=input_hash,
    )


class TestStateComputer:
    """Tests for StateComputer."""

    @staticmethod
    def test_missing_status_when_no_manifest(
        fresh_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """Target without manifest has missing status."""
        # Create a simple target
        target = OutputTarget(
            name="test_target",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )

        graph = TargetGraph()
        graph.register(target)

        snapshot = make_snapshot(tmp_path)
        session = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )
        computer = StateComputer(graph=graph, session=session)

        state = computer.compute_all()

        expect_in("test_target", state.targets)
        expect_equal(state.targets["test_target"].status, "missing")
        expect_is_none(state.targets["test_target"].manifest)

    @staticmethod
    def test_current_status_when_hash_matches(
        fresh_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """Target with matching hash has current status."""
        target = OutputTarget(
            name="test_target",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )

        graph = TargetGraph()
        graph.register(target)

        snapshot = make_snapshot(tmp_path)
        session = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )
        computer = StateComputer(graph=graph, session=session)

        # Compute the actual input hash
        actual_hash = session.get_input_hash(target)

        # Create manifest with matching hash
        manifest = make_manifest("test_target", input_hash=actual_hash)
        fresh_gateway.build.save_manifest(manifest)

        state = computer.compute_all()

        expect_equal(state.targets["test_target"].status, "current")
        expect_is_not_none(state.targets["test_target"].manifest)
        expect_equal(state.targets["test_target"].current_hash, actual_hash)

    @staticmethod
    def test_stale_status_when_hash_differs(
        fresh_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """Target with different hash has stale status."""
        target = OutputTarget(
            name="test_target",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )

        graph = TargetGraph()
        graph.register(target)

        snapshot = make_snapshot(tmp_path)

        # Create manifest with different hash
        manifest = make_manifest("test_target", input_hash="old_hash_123")
        fresh_gateway.build.save_manifest(manifest)

        session = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )
        computer = StateComputer(graph=graph, session=session)

        state = computer.compute_all()

        expect_equal(state.targets["test_target"].status, "stale")
        expect_equal(state.targets["test_target"].blocking_reason, "input_hash_mismatch")

    @staticmethod
    def test_blocked_when_dependency_missing(
        fresh_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """Target blocked when dependency is missing."""
        dep_target = OutputTarget(
            name="dependency",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )
        main_target = OutputTarget(
            name="main",
            module="ingestion",
            dependencies=("dependency",),
            contract=EMPTY_CONTRACT,
        )

        graph = TargetGraph()
        graph.register(dep_target)
        graph.register(main_target)

        snapshot = make_snapshot(tmp_path)
        session = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )

        # Save manifest only for main, not for dependency
        main_hash = session.get_input_hash(main_target)
        manifest = make_manifest("main", input_hash=main_hash)
        fresh_gateway.build.save_manifest(manifest)

        computer = StateComputer(graph=graph, session=session)
        state = computer.compute_all()

        expect_equal(state.targets["dependency"].status, "missing")
        expect_equal(state.targets["main"].status, "blocked")
        expect_equal(state.targets["main"].blocking_reason, "dependency_missing")
        expect_in("dependency", state.targets["main"].blocking_deps)

    @staticmethod
    def test_blocked_when_dependency_stale(
        fresh_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """Target blocked when dependency is stale."""
        dep_target = OutputTarget(
            name="dependency",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )
        main_target = OutputTarget(
            name="main",
            module="ingestion",
            dependencies=("dependency",),
            contract=EMPTY_CONTRACT,
        )

        graph = TargetGraph()
        graph.register(dep_target)
        graph.register(main_target)

        snapshot = make_snapshot(tmp_path)
        session = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )

        # Save manifest for dependency with wrong hash (stale)
        dep_manifest = make_manifest("dependency", input_hash="stale_hash")
        fresh_gateway.build.save_manifest(dep_manifest)

        # Save manifest for main with correct hash
        main_hash = session.get_input_hash(main_target)
        main_manifest = make_manifest("main", input_hash=main_hash)
        fresh_gateway.build.save_manifest(main_manifest)

        computer = StateComputer(graph=graph, session=session)
        state = computer.compute_all()

        expect_equal(state.targets["dependency"].status, "stale")
        expect_equal(state.targets["main"].status, "blocked")
        expect_equal(state.targets["main"].blocking_reason, "dependency_stale")

    @staticmethod
    def test_build_state_query_methods(
        fresh_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """BuildState query methods return correct results."""
        t1 = OutputTarget(
            name="t1",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )
        t2 = OutputTarget(
            name="t2",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )
        t3 = OutputTarget(
            name="t3",
            module="ingestion",
            dependencies=("t1",),
            contract=EMPTY_CONTRACT,
        )

        graph = TargetGraph()
        graph.register(t1)
        graph.register(t2)
        graph.register(t3)

        snapshot = make_snapshot(tmp_path)
        session = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )

        # Mark t1 as current via a matching manifest.
        t1_hash = session.get_input_hash(t1)
        fresh_gateway.build.save_manifest(make_manifest("t1", input_hash=t1_hash))

        # Leave t2 and t3 without manifests (missing).

        computer = StateComputer(graph=graph, session=session)
        state = computer.compute_all()

        expect_equal(state.by_status("current"), ("t1",))
        expect_equal(state.by_status("missing"), ("t2", "t3"))
        expect_true(state.is_current("t1"))
        expect_false(state.is_current("t2"))

    @staticmethod
    def test_compute_single_target(
        fresh_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """compute_single returns correct state for individual target."""
        target = OutputTarget(
            name="single",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )

        graph = TargetGraph()
        graph.register(target)

        snapshot = make_snapshot(tmp_path)
        session = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )
        computer = StateComputer(graph=graph, session=session)

        state = computer.compute_single("single")

        expect_equal(state.name, "single")
        expect_equal(state.status, "missing")

    @staticmethod
    def test_session_caching_efficiency(
        fresh_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """Session caches hashes to avoid redundant computation."""
        target = OutputTarget(
            name="cached",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )

        graph = TargetGraph()
        graph.register(target)

        snapshot = make_snapshot(tmp_path)
        session = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )
        computer = StateComputer(graph=graph, session=session)

        # Compute multiple times
        state1 = computer.compute_all()
        state2 = computer.compute_all()

        # Session should have cached the hash
        expect_true(session.cached_hash_count >= 1, message="Expected cached hash count")
        expect_equal(state1.targets["cached"].status, state2.targets["cached"].status)


class TestStateValidatorEquivalence:
    """Tests ensuring StateValidator produces equivalent results to StateComputer.

    Note: Both now use unified types (BuildState, TargetState).
    """

    @staticmethod
    def test_validator_and_computer_agree_on_missing(
        fresh_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """StateValidator and StateComputer agree on missing targets."""
        target = OutputTarget(
            name="equiv_test",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )

        graph = TargetGraph()
        graph.register(target)

        snapshot = make_snapshot(tmp_path)

        # Use StateValidator
        validator = StateValidator(
            graph=graph,
            gateway=fresh_gateway,
            snapshot=snapshot,
            options=StateValidationOptions(settings=TEST_BUILD_SETTINGS),
        )
        validator_state = validator.validate()

        # Use StateComputer
        session = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )
        computer = StateComputer(graph=graph, session=session)
        computer_state = computer.compute_all()

        # Results should be equivalent (both use by_status now)
        expect_equal(validator_state.by_status("missing"), computer_state.by_status("missing"))

    @staticmethod
    def test_validator_and_computer_agree_on_current(
        fresh_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """StateValidator and StateComputer agree on current targets."""
        target = OutputTarget(
            name="equiv_current",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )

        graph = TargetGraph()
        graph.register(target)

        snapshot = make_snapshot(tmp_path)
        session = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )

        # Save manifest with correct hash
        hash_val = session.get_input_hash(target)
        manifest = make_manifest("equiv_current", input_hash=hash_val)
        fresh_gateway.build.save_manifest(manifest)

        # Use StateValidator
        validator = StateValidator(
            graph=graph,
            gateway=fresh_gateway,
            snapshot=snapshot,
            options=StateValidationOptions(settings=TEST_BUILD_SETTINGS),
        )
        validator_state = validator.validate()

        # Use StateComputer
        session2 = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )
        computer = StateComputer(graph=graph, session=session2)
        computer_state = computer.compute_all()

        # Both use "current" status now
        expect_equal(validator_state.by_status("current"), computer_state.by_status("current"))

    @staticmethod
    def test_validator_and_computer_agree_on_blocked(
        fresh_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """StateValidator and StateComputer agree on blocked targets."""
        dep = OutputTarget(
            name="dep",
            module="ingestion",
            dependencies=(),
            contract=EMPTY_CONTRACT,
        )
        main = OutputTarget(
            name="main",
            module="ingestion",
            dependencies=("dep",),
            contract=EMPTY_CONTRACT,
        )

        graph = TargetGraph()
        graph.register(dep)
        graph.register(main)

        snapshot = make_snapshot(tmp_path)
        session = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )

        # Save manifest for main with correct hash (but dep is missing)
        main_hash = session.get_input_hash(main)
        manifest = make_manifest("main", input_hash=main_hash)
        fresh_gateway.build.save_manifest(manifest)

        # Use StateValidator
        validator = StateValidator(
            graph=graph,
            gateway=fresh_gateway,
            snapshot=snapshot,
            options=StateValidationOptions(settings=TEST_BUILD_SETTINGS),
        )
        validator_state = validator.validate()

        # Use StateComputer
        session2 = BuildSession(
            snapshot=snapshot,
            gateway=fresh_gateway,
            settings=TEST_BUILD_SETTINGS,
        )
        computer = StateComputer(graph=graph, session=session2)
        computer_state = computer.compute_all()

        expect_equal(validator_state.by_status("blocked"), computer_state.by_status("blocked"))
