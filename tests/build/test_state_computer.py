"""Tests for the unified StateComputer.

This module tests the StateComputer class which provides the single source
of truth for target state computation. Tests verify:

1. Correct status computation (current, stale, missing, blocked)
2. Proper blocking propagation through dependencies
3. Session caching efficiency
4. Equivalence with legacy StateValidator results
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.contracts import EMPTY_CONTRACT
from codeintel.build.manifest import OutputManifest
from codeintel.build.session import BuildSession
from codeintel.build.state import StateValidator
from codeintel.build.state_computer import StateComputer
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.primitives import SnapshotRef

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def make_snapshot(tmp_path: Path, repo: str = "test/repo", commit: str = "abc123") -> SnapshotRef:
    """Create a test snapshot reference."""
    return SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path)


def make_manifest(
    target: str,
    repo: str = "test/repo",
    commit: str = "abc123",
    input_hash: str = "hash123",
) -> OutputManifest:
    """Create a test manifest."""
    return OutputManifest(
        target=target,
        repo=repo,
        commit=commit,
        plugin=f"{target}_plugin",
        computed_at=datetime.now(tz=UTC),
        duration_ms=100.0,
        input_hash=input_hash,
    )


class TestStateComputer:
    """Tests for StateComputer."""

    @staticmethod
    def test_missing_status_when_no_manifest(
        analytics_gateway: StorageGateway,
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
        session = BuildSession(snapshot=snapshot, gateway=analytics_gateway)
        computer = StateComputer(graph=graph, session=session)

        state = computer.compute_all()

        assert "test_target" in state.targets
        assert state.targets["test_target"].status == "missing"
        assert state.targets["test_target"].manifest is None

    @staticmethod
    def test_current_status_when_hash_matches(
        analytics_gateway: StorageGateway,
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
        session = BuildSession(snapshot=snapshot, gateway=analytics_gateway)
        computer = StateComputer(graph=graph, session=session)

        # Compute the actual input hash
        actual_hash = session.get_input_hash(target)

        # Create manifest with matching hash
        manifest = make_manifest("test_target", input_hash=actual_hash)
        analytics_gateway.build.save_manifest(manifest)

        state = computer.compute_all()

        assert state.targets["test_target"].status == "current"
        assert state.targets["test_target"].manifest is not None
        assert state.targets["test_target"].current_hash == actual_hash

    @staticmethod
    def test_stale_status_when_hash_differs(
        analytics_gateway: StorageGateway,
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
        analytics_gateway.build.save_manifest(manifest)

        session = BuildSession(snapshot=snapshot, gateway=analytics_gateway)
        computer = StateComputer(graph=graph, session=session)

        state = computer.compute_all()

        assert state.targets["test_target"].status == "stale"
        assert state.targets["test_target"].blocking_reason == "input_hash_mismatch"

    @staticmethod
    def test_blocked_when_dependency_missing(
        analytics_gateway: StorageGateway,
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
        session = BuildSession(snapshot=snapshot, gateway=analytics_gateway)

        # Save manifest only for main, not for dependency
        main_hash = session.get_input_hash(main_target)
        manifest = make_manifest("main", input_hash=main_hash)
        analytics_gateway.build.save_manifest(manifest)

        computer = StateComputer(graph=graph, session=session)
        state = computer.compute_all()

        assert state.targets["dependency"].status == "missing"
        assert state.targets["main"].status == "blocked"
        assert state.targets["main"].blocking_reason == "dependency_missing"
        assert "dependency" in state.targets["main"].blocking_deps

    @staticmethod
    def test_blocked_when_dependency_stale(
        analytics_gateway: StorageGateway,
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
        session = BuildSession(snapshot=snapshot, gateway=analytics_gateway)

        # Save manifest for dependency with wrong hash (stale)
        dep_manifest = make_manifest("dependency", input_hash="stale_hash")
        analytics_gateway.build.save_manifest(dep_manifest)

        # Save manifest for main with correct hash
        main_hash = session.get_input_hash(main_target)
        main_manifest = make_manifest("main", input_hash=main_hash)
        analytics_gateway.build.save_manifest(main_manifest)

        computer = StateComputer(graph=graph, session=session)
        state = computer.compute_all()

        assert state.targets["dependency"].status == "stale"
        assert state.targets["main"].status == "blocked"
        assert state.targets["main"].blocking_reason == "dependency_stale"

    @staticmethod
    def test_build_state_query_methods(
        analytics_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """BuildState query methods return correct results."""
        t1 = OutputTarget(name="t1", module="ingestion", dependencies=(), contract=EMPTY_CONTRACT)
        t2 = OutputTarget(name="t2", module="ingestion", dependencies=(), contract=EMPTY_CONTRACT)
        t3 = OutputTarget(name="t3", module="ingestion", dependencies=("t1",), contract=EMPTY_CONTRACT)

        graph = TargetGraph()
        graph.register(t1)
        graph.register(t2)
        graph.register(t3)

        snapshot = make_snapshot(tmp_path)
        session = BuildSession(snapshot=snapshot, gateway=analytics_gateway)

        # t1: current
        t1_hash = session.get_input_hash(t1)
        analytics_gateway.build.save_manifest(make_manifest("t1", input_hash=t1_hash))

        # t2: missing (no manifest)
        # t3: blocked (t1 needs to be rechecked after we make it current)

        computer = StateComputer(graph=graph, session=session)
        state = computer.compute_all()

        assert state.by_status("current") == ("t1",)
        assert state.by_status("missing") == ("t2", "t3")
        assert state.is_current("t1")
        assert not state.is_current("t2")

    @staticmethod
    def test_compute_single_target(
        analytics_gateway: StorageGateway,
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
        session = BuildSession(snapshot=snapshot, gateway=analytics_gateway)
        computer = StateComputer(graph=graph, session=session)

        state = computer.compute_single("single")

        assert state.name == "single"
        assert state.status == "missing"

    @staticmethod
    def test_session_caching_efficiency(
        analytics_gateway: StorageGateway,
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
        session = BuildSession(snapshot=snapshot, gateway=analytics_gateway)
        computer = StateComputer(graph=graph, session=session)

        # Compute multiple times
        state1 = computer.compute_all()
        state2 = computer.compute_all()

        # Session should have cached the hash
        assert session.cached_hash_count >= 1
        assert state1.targets["cached"].status == state2.targets["cached"].status


class TestStateValidatorEquivalence:
    """Tests ensuring StateValidator produces equivalent results to StateComputer."""

    @staticmethod
    def test_validator_and_computer_agree_on_missing(
        analytics_gateway: StorageGateway,
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

        # Use StateValidator (legacy)
        validator = StateValidator(graph=graph, gateway=analytics_gateway, snapshot=snapshot)
        legacy_state = validator.validate()

        # Use StateComputer (new)
        session = BuildSession(snapshot=snapshot, gateway=analytics_gateway)
        computer = StateComputer(graph=graph, session=session)
        new_state = computer.compute_all()

        # Results should be equivalent
        assert legacy_state.missing_targets() == new_state.missing_targets()

    @staticmethod
    def test_validator_and_computer_agree_on_current(
        analytics_gateway: StorageGateway,
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
        session = BuildSession(snapshot=snapshot, gateway=analytics_gateway)

        # Save manifest with correct hash
        hash_val = session.get_input_hash(target)
        manifest = make_manifest("equiv_current", input_hash=hash_val)
        analytics_gateway.build.save_manifest(manifest)

        # Use StateValidator
        validator = StateValidator(graph=graph, gateway=analytics_gateway, snapshot=snapshot)
        legacy_state = validator.validate()

        # Use StateComputer
        session2 = BuildSession(snapshot=snapshot, gateway=analytics_gateway)
        computer = StateComputer(graph=graph, session=session2)
        new_state = computer.compute_all()

        # "computed" in legacy = "current" in new
        assert legacy_state.computed_targets() == new_state.current_targets()

    @staticmethod
    def test_validator_and_computer_agree_on_blocked(
        analytics_gateway: StorageGateway,
        tmp_path: Path,
    ) -> None:
        """StateValidator and StateComputer agree on blocked targets."""
        dep = OutputTarget(name="dep", module="ingestion", dependencies=(), contract=EMPTY_CONTRACT)
        main = OutputTarget(name="main", module="ingestion", dependencies=("dep",), contract=EMPTY_CONTRACT)

        graph = TargetGraph()
        graph.register(dep)
        graph.register(main)

        snapshot = make_snapshot(tmp_path)
        session = BuildSession(snapshot=snapshot, gateway=analytics_gateway)

        # Save manifest for main with correct hash (but dep is missing)
        main_hash = session.get_input_hash(main)
        manifest = make_manifest("main", input_hash=main_hash)
        analytics_gateway.build.save_manifest(manifest)

        # Use StateValidator
        validator = StateValidator(graph=graph, gateway=analytics_gateway, snapshot=snapshot)
        legacy_state = validator.validate()

        # Use StateComputer
        session2 = BuildSession(snapshot=snapshot, gateway=analytics_gateway)
        computer = StateComputer(graph=graph, session=session2)
        new_state = computer.compute_all()

        assert legacy_state.blocked_targets() == new_state.blocked_targets()
