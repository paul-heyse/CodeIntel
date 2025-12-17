"""Unit tests for state validation module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.hashing import compute_input_hash
from codeintel.build.registry import get_target_graph
from codeintel.build.state import BuildState, StateValidator, TargetState
from codeintel.build.targets import OutputTarget, TargetGraph, TargetOptions
from codeintel.config.primitives import SnapshotRef
from codeintel.core.build_manifest import OutputManifest
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway


def _create_test_graph() -> TargetGraph:
    """Create a minimal test graph for state validation tests.

    Returns
    -------
    TargetGraph
        Graph with modules -> ast -> goids chain and independent typing target.
    """
    graph = TargetGraph()

    modules_target = OutputTarget.from_tables(
        name="modules",
        module="ingestion",
        plugin="repo_scan",
        tables=("core.modules",),
        options=TargetOptions(description="Repository module index"),
    )

    ast_target = OutputTarget.from_tables(
        name="ast",
        module="ingestion",
        plugin="ast_extract",
        tables=("core.ast_nodes",),
        options=TargetOptions(dependencies=("modules",), description="AST extraction"),
    )

    goids_target = OutputTarget.from_tables(
        name="goids",
        module="graphs",
        plugin="goid_builder",
        tables=("core.goids",),
        options=TargetOptions(dependencies=("ast",), description="GOID construction"),
    )

    typing_target = OutputTarget.from_tables(
        name="typing",
        module="ingestion",
        plugin="typing_ingest",
        tables=("analytics.typedness",),
        options=TargetOptions(dependencies=("modules",), description="Type analysis"),
    )

    metrics_target = OutputTarget.from_tables(
        name="function_metrics",
        module="analytics",
        plugin="function_metrics",
        tables=("analytics.function_metrics",),
        options=TargetOptions(dependencies=("goids", "ast"), description="Function metrics"),
    )

    graph.register(modules_target)
    graph.register(ast_target)
    graph.register(goids_target)
    graph.register(typing_target)
    graph.register(metrics_target)

    return graph


@pytest.fixture
def test_graph() -> TargetGraph:
    """Provide a minimal test graph for state validation tests.

    Returns
    -------
    TargetGraph
        Graph with modules -> ast -> goids chain.
    """
    return _create_test_graph()


@pytest.fixture
def snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide a snapshot reference for tests.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.

    Returns
    -------
    SnapshotRef
        Snapshot reference with test repo and commit.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    return SnapshotRef(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=repo_root,
    )


@pytest.fixture
def validator(
    test_graph: TargetGraph,
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> StateValidator:
    """Provide a StateValidator for tests.

    Parameters
    ----------
    test_graph
        Minimal test graph.
    fresh_gateway
        Fresh gateway with schema applied.
    snapshot
        Snapshot reference.

    Returns
    -------
    StateValidator
        Validator configured for testing.
    """
    return StateValidator(test_graph, fresh_gateway, snapshot)


class TestTargetState:
    """Tests for unified TargetState dataclass."""

    @staticmethod
    def test_create_missing_state() -> None:
        """Create a target state for missing target."""
        state = TargetState(
            name="test_target",
            status="missing",
            manifest=None,
        )
        expect_equal(state.name, "test_target")
        expect_equal(state.status, "missing")
        expect_true(state.manifest is None)
        expect_true(state.is_missing)
        expect_false(state.is_current)

    @staticmethod
    def test_create_current_state() -> None:
        """Create a target state for current target."""
        manifest = OutputManifest(
            target="test_target",
            repo="test/repo",
            commit="abc123",
            plugin="test_plugin",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="hash123",
        )
        state = TargetState(
            name="test_target",
            status="current",
            manifest=manifest,
            current_hash="hash123",
        )
        expect_equal(state.status, "current")
        expect_true(state.manifest is manifest)
        expect_true(state.is_current)

    @staticmethod
    def test_create_stale_state() -> None:
        """Create a target state for stale target."""
        manifest = OutputManifest(
            target="test_target",
            repo="test/repo",
            commit="abc123",
            plugin="test_plugin",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="old_hash",
        )
        state = TargetState(
            name="test_target",
            status="stale",
            manifest=manifest,
            current_hash="new_hash",
            blocking_reason="input_hash_mismatch",
        )
        expect_equal(state.status, "stale")
        expect_true(state.is_stale)
        expect_equal(state.blocking_reason, "input_hash_mismatch")

    @staticmethod
    def test_create_blocked_state() -> None:
        """Create a target state for blocked target."""
        state = TargetState(
            name="goids",
            status="blocked",
            manifest=None,
            blocking_reason="dependency_missing",
            blocking_deps=("ast",),
        )
        expect_equal(state.status, "blocked")
        expect_equal(state.blocking_deps, ("ast",))
        expect_true(state.is_blocked)


class TestBuildState:
    """Tests for unified BuildState dataclass."""

    @staticmethod
    def test_create_build_state() -> None:
        """Create a build state with targets."""
        states = {
            "modules": TargetState(
                name="modules",
                status="current",
                manifest=None,
                current_hash="hash1",
            ),
            "ast": TargetState(
                name="ast",
                status="missing",
                manifest=None,
            ),
        }
        build_state = BuildState(
            repo="test/repo",
            commit="abc123",
            targets=states,
        )
        expect_equal(build_state.repo, "test/repo")
        expect_equal(build_state.commit, "abc123")
        expect_equal(len(build_state.targets), 2)

    @staticmethod
    def test_get_existing_target() -> None:
        """Get state for existing target."""
        target_state = TargetState(
            name="modules",
            status="current",
            manifest=None,
            current_hash="hash1",
        )
        build_state = BuildState(
            repo="test/repo",
            commit="abc123",
            targets={"modules": target_state},
        )
        result = build_state.get("modules")
        expect_true(result is target_state)

    @staticmethod
    def test_get_nonexistent_target_raises() -> None:
        """Getting nonexistent target raises KeyError."""
        build_state = BuildState(
            repo="test/repo",
            commit="abc123",
            targets={},
        )
        with pytest.raises(KeyError, match="not found"):
            build_state.get("nonexistent")

    @staticmethod
    def test_by_status_missing() -> None:
        """Query missing targets from build state."""
        states = {
            "modules": TargetState(
                name="modules",
                status="current",
                manifest=None,
                current_hash="hash1",
            ),
            "ast": TargetState(
                name="ast",
                status="missing",
                manifest=None,
            ),
            "goids": TargetState(
                name="goids",
                status="missing",
                manifest=None,
            ),
        }
        build_state = BuildState(repo="test/repo", commit="abc123", targets=states)
        missing = build_state.by_status("missing")
        expect_equal(missing, ("ast", "goids"))

    @staticmethod
    def test_by_status_stale() -> None:
        """Query stale targets from build state."""
        states = {
            "modules": TargetState(
                name="modules",
                status="stale",
                manifest=None,
                blocking_reason="input_hash_mismatch",
                current_hash="hash1",
            ),
            "ast": TargetState(
                name="ast",
                status="current",
                manifest=None,
                current_hash="hash2",
            ),
        }
        build_state = BuildState(repo="test/repo", commit="abc123", targets=states)
        stale = build_state.by_status("stale")
        expect_equal(stale, ("modules",))

    @staticmethod
    def test_by_status_current() -> None:
        """Query current targets from build state."""
        states = {
            "modules": TargetState(
                name="modules",
                status="current",
                manifest=None,
                current_hash="hash1",
            ),
            "ast": TargetState(
                name="ast",
                status="current",
                manifest=None,
                current_hash="hash2",
            ),
            "goids": TargetState(
                name="goids",
                status="missing",
                manifest=None,
            ),
        }
        build_state = BuildState(repo="test/repo", commit="abc123", targets=states)
        current = build_state.by_status("current")
        expect_equal(current, ("ast", "modules"))

    @staticmethod
    def test_by_status_blocked() -> None:
        """Query blocked targets from build state."""
        states = {
            "modules": TargetState(
                name="modules",
                status="current",
                manifest=None,
                current_hash="hash1",
            ),
            "goids": TargetState(
                name="goids",
                status="blocked",
                manifest=None,
                blocking_reason="dependency_missing",
                blocking_deps=("ast",),
            ),
        }
        build_state = BuildState(repo="test/repo", commit="abc123", targets=states)
        blocked = build_state.by_status("blocked")
        expect_equal(blocked, ("goids",))

    @staticmethod
    def test_is_current() -> None:
        """Check if target is current."""
        states = {
            "modules": TargetState(
                name="modules",
                status="current",
                manifest=None,
                current_hash="hash1",
            ),
            "ast": TargetState(
                name="ast",
                status="stale",
                manifest=None,
                blocking_reason="input_hash_mismatch",
                current_hash="hash2",
            ),
        }
        build_state = BuildState(repo="test/repo", commit="abc123", targets=states)
        expect_true(build_state.is_current("modules"))
        expect_false(build_state.is_current("ast"))
        expect_false(build_state.is_current("nonexistent"))


class TestStateValidatorInit:
    """Tests for StateValidator initialization."""

    @staticmethod
    def test_create_validator(
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Create a state validator with valid inputs."""
        validator = StateValidator(test_graph, fresh_gateway, snapshot)
        expect_is_not_none(validator)

    @staticmethod
    def test_invalid_graph_raises(
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Creating validator with invalid graph raises ValueError."""
        graph = TargetGraph()
        target = OutputTarget.from_tables(
            name="target",
            module="ingestion",
            plugin="plugin",
            tables=("core.table",),
            options=TargetOptions(dependencies=("nonexistent",)),
        )
        graph.register(target)

        with pytest.raises(ValueError, match="validation failed"):
            StateValidator(graph, fresh_gateway, snapshot)


class TestValidateEmptyDatabase:
    """Tests for validating with no manifests."""

    @staticmethod
    def test_all_targets_missing(
        validator: StateValidator,
    ) -> None:
        """All targets should be missing when database is empty."""
        state = validator.validate()

        expect_equal(len(state.by_status("missing")), 5)
        expect_equal(len(state.by_status("current")), 0)
        expect_equal(len(state.by_status("stale")), 0)
        expect_equal(len(state.by_status("blocked")), 0)


class TestValidateCurrentTargets:
    """Tests for targets with valid manifests."""

    @staticmethod
    def test_single_current_target(
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Target with matching hash should be current."""
        target = test_graph.get("modules")
        correct_hash = compute_input_hash(target, snapshot, fresh_gateway)

        manifest = OutputManifest(
            target="modules",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="repo_scan",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash=correct_hash,
        )
        fresh_gateway.build.save_manifest(manifest)

        validator = StateValidator(test_graph, fresh_gateway, snapshot)
        state = validator.validate()

        expect_equal(state.get("modules").status, "current")
        expect_is_not_none(state.get("modules").manifest)

    @staticmethod
    def test_chain_of_current_targets(
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Chain of targets with valid hashes should all be current."""
        modules_target = test_graph.get("modules")
        modules_hash = compute_input_hash(modules_target, snapshot, fresh_gateway)
        modules_manifest = OutputManifest(
            target="modules",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="repo_scan",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash=modules_hash,
            output_hash="modules_output",
        )
        fresh_gateway.build.save_manifest(modules_manifest)

        ast_target = test_graph.get("ast")
        ast_hash = compute_input_hash(ast_target, snapshot, fresh_gateway)
        ast_manifest = OutputManifest(
            target="ast",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="ast_extract",
            computed_at=datetime.now(tz=UTC),
            duration_ms=200.0,
            input_hash=ast_hash,
            output_hash="ast_output",
        )
        fresh_gateway.build.save_manifest(ast_manifest)

        validator = StateValidator(test_graph, fresh_gateway, snapshot)
        state = validator.validate()

        expect_equal(state.get("modules").status, "current")
        expect_equal(state.get("ast").status, "current")


class TestValidateStaleTargets:
    """Tests for targets with mismatched hashes."""

    @staticmethod
    def test_stale_due_to_input_hash_mismatch(
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Target with wrong input hash should be stale."""
        manifest = OutputManifest(
            target="modules",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="repo_scan",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="wrong_hash_value",
        )
        fresh_gateway.build.save_manifest(manifest)

        validator = StateValidator(test_graph, fresh_gateway, snapshot)
        state = validator.validate()

        modules_state = state.get("modules")
        expect_equal(modules_state.status, "stale")
        expect_equal(modules_state.blocking_reason, "input_hash_mismatch")


class TestValidateBlockedTargets:
    """Tests for targets blocked by dependencies."""

    @staticmethod
    def test_blocked_by_missing_dependency(
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Target should be blocked when dependency is missing."""
        ast_target = test_graph.get("ast")
        ast_hash = compute_input_hash(ast_target, snapshot, fresh_gateway)
        ast_manifest = OutputManifest(
            target="ast",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="ast_extract",
            computed_at=datetime.now(tz=UTC),
            duration_ms=200.0,
            input_hash=ast_hash,
        )
        fresh_gateway.build.save_manifest(ast_manifest)

        validator = StateValidator(test_graph, fresh_gateway, snapshot)
        state = validator.validate()

        expect_equal(state.get("modules").status, "missing")

        ast_state = state.get("ast")
        expect_equal(ast_state.status, "blocked")
        expect_in("modules", ast_state.blocking_deps)
        expect_equal(ast_state.blocking_reason, "dependency_missing")

    @staticmethod
    def test_blocked_by_stale_dependency(
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Target should be blocked when dependency is stale."""
        modules_manifest = OutputManifest(
            target="modules",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="repo_scan",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="wrong_hash",
        )
        fresh_gateway.build.save_manifest(modules_manifest)

        ast_target = test_graph.get("ast")
        ast_hash = compute_input_hash(ast_target, snapshot, fresh_gateway)
        ast_manifest = OutputManifest(
            target="ast",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="ast_extract",
            computed_at=datetime.now(tz=UTC),
            duration_ms=200.0,
            input_hash=ast_hash,
        )
        fresh_gateway.build.save_manifest(ast_manifest)

        validator = StateValidator(test_graph, fresh_gateway, snapshot)
        state = validator.validate()

        expect_equal(state.get("modules").status, "stale")

        ast_state = state.get("ast")
        expect_equal(ast_state.status, "blocked")
        expect_in("modules", ast_state.blocking_deps)
        expect_equal(ast_state.blocking_reason, "dependency_stale")

    @staticmethod
    def test_cascade_blocking(
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Blocking should cascade through dependency chain."""
        modules_manifest = OutputManifest(
            target="modules",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="repo_scan",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="wrong_hash",
        )
        fresh_gateway.build.save_manifest(modules_manifest)

        ast_target = test_graph.get("ast")
        ast_hash = compute_input_hash(ast_target, snapshot, fresh_gateway)
        ast_manifest = OutputManifest(
            target="ast",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="ast_extract",
            computed_at=datetime.now(tz=UTC),
            duration_ms=200.0,
            input_hash=ast_hash,
        )
        fresh_gateway.build.save_manifest(ast_manifest)

        goids_target = test_graph.get("goids")
        goids_hash = compute_input_hash(goids_target, snapshot, fresh_gateway)
        goids_manifest = OutputManifest(
            target="goids",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="goid_builder",
            computed_at=datetime.now(tz=UTC),
            duration_ms=300.0,
            input_hash=goids_hash,
        )
        fresh_gateway.build.save_manifest(goids_manifest)

        validator = StateValidator(test_graph, fresh_gateway, snapshot)
        state = validator.validate()

        expect_equal(state.get("modules").status, "stale")

        expect_equal(state.get("ast").status, "blocked")

        goids_state = state.get("goids")
        expect_equal(goids_state.status, "blocked")
        expect_in("ast", goids_state.blocking_deps)
        expect_equal(goids_state.blocking_reason, "dependency_blocked")

    @staticmethod
    def test_multiple_blocking_dependencies(
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Target with multiple blocking deps should list all."""
        modules_manifest = OutputManifest(
            target="modules",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="repo_scan",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="wrong_hash",
        )
        fresh_gateway.build.save_manifest(modules_manifest)

        ast_manifest = OutputManifest(
            target="ast",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="ast_extract",
            computed_at=datetime.now(tz=UTC),
            duration_ms=200.0,
            input_hash="wrong_ast_hash",
        )
        fresh_gateway.build.save_manifest(ast_manifest)

        metrics_target = test_graph.get("function_metrics")
        metrics_hash = compute_input_hash(metrics_target, snapshot, fresh_gateway)
        metrics_manifest = OutputManifest(
            target="function_metrics",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="function_metrics",
            computed_at=datetime.now(tz=UTC),
            duration_ms=400.0,
            input_hash=metrics_hash,
        )
        fresh_gateway.build.save_manifest(metrics_manifest)

        validator = StateValidator(test_graph, fresh_gateway, snapshot)
        state = validator.validate()

        metrics_state = state.get("function_metrics")
        expect_equal(metrics_state.status, "blocked")

        expect_true(len(metrics_state.blocking_deps) >= 1)


class TestValidateTarget:
    """Tests for single-target validation."""

    @staticmethod
    def test_validate_single_target(
        validator: StateValidator,
    ) -> None:
        """Validate a single target by name."""
        state = validator.validate_target("modules")
        expect_equal(state.name, "modules")
        expect_equal(state.status, "missing")

    @staticmethod
    def test_validate_nonexistent_target_raises(
        validator: StateValidator,
    ) -> None:
        """Validating nonexistent target raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            validator.validate_target("nonexistent")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @staticmethod
    def test_manifest_for_unknown_target_logged(
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Manifest for unknown target should be logged but not crash."""
        unknown_manifest = OutputManifest(
            target="unknown_target",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="unknown_plugin",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="hash123",
        )
        fresh_gateway.build.save_manifest(unknown_manifest)

        validator = StateValidator(test_graph, fresh_gateway, snapshot)

        with caplog.at_level("WARNING"):
            state = validator.validate()

        expect_is_not_none(state)

        expect_true("unknown_target" in caplog.text or len(caplog.records) > 0)

    @staticmethod
    def test_empty_graph(
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Validating empty graph should work."""
        empty_graph = TargetGraph()

        validator = StateValidator(empty_graph, fresh_gateway, snapshot)
        state = validator.validate()

        expect_equal(len(state.targets), 0)
        expect_equal(state.by_status("missing"), ())
        expect_equal(state.by_status("current"), ())


class TestWithRealRegistry:
    """Integration tests using the full target registry."""

    @staticmethod
    def test_validate_with_real_registry(
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Validate using the full target registry."""
        graph = get_target_graph()
        validator = StateValidator(graph, fresh_gateway, snapshot)
        state = validator.validate()

        expect_equal(len(state.by_status("missing")), len(graph))
        expect_equal(len(state.by_status("current")), 0)

    @staticmethod
    def test_real_registry_target_count(
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Verify state covers all registered targets."""
        graph = get_target_graph()
        validator = StateValidator(graph, fresh_gateway, snapshot)
        state = validator.validate()

        expect_equal(len(state.targets), len(graph))
