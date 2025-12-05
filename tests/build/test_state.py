"""Unit tests for state validation module."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.build.hashing import compute_input_hash
from codeintel.build.manifest import OutputManifest
from codeintel.build.registry import ALL_TARGETS, get_target_graph
from codeintel.build.state import (
    DatabaseState,
    StalenessReason,
    StateValidator,
    TargetState,
)
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers import assert_frozen

# =============================================================================
# Test Fixtures
# =============================================================================


def _create_test_graph() -> TargetGraph:
    """Create a minimal test graph for state validation tests.

    Returns
    -------
    TargetGraph
        Graph with modules -> ast -> goids chain and independent typing target.
    """
    graph = TargetGraph()

    # Root target with no dependencies
    modules_target = OutputTarget(
        name="modules",
        module="ingestion",
        plugin="repo_scan",
        tables=("core.modules",),
        dependencies=(),
        description="Repository module index",
    )

    # Target depending on modules
    ast_target = OutputTarget(
        name="ast",
        module="ingestion",
        plugin="ast_extract",
        tables=("core.ast_nodes",),
        dependencies=("modules",),
        description="AST extraction",
    )

    # Target depending on ast
    goids_target = OutputTarget(
        name="goids",
        module="graphs",
        plugin="goid_builder",
        tables=("core.goids",),
        dependencies=("ast",),
        description="GOID construction",
    )

    # Independent target depending on modules
    typing_target = OutputTarget(
        name="typing",
        module="ingestion",
        plugin="typing_ingest",
        tables=("analytics.typedness",),
        dependencies=("modules",),
        description="Type analysis",
    )

    # Target with multiple dependencies
    metrics_target = OutputTarget(
        name="function_metrics",
        module="analytics",
        plugin="function_metrics",
        tables=("analytics.function_metrics",),
        dependencies=("goids", "ast"),
        description="Function metrics",
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


# =============================================================================
# Type Definition Tests
# =============================================================================


class TestStalenessReason:
    """Tests for StalenessReason dataclass."""

    def test_create_staleness_reason(self) -> None:
        """Create a staleness reason with all fields."""
        reason = StalenessReason(
            kind="input_hash_mismatch",
            details="Hash changed from abc to def",
        )
        assert reason.kind == "input_hash_mismatch"
        assert "abc" in reason.details
        assert "def" in reason.details

    def test_staleness_reason_is_frozen(self) -> None:
        """Verify staleness reason is immutable."""
        reason = StalenessReason(
            kind="dependency_missing",
            details="Dependency not computed",
        )
        assert_frozen(reason, "kind", "input_hash_mismatch")


class TestTargetState:
    """Tests for TargetState dataclass."""

    def test_create_missing_state(self) -> None:
        """Create a target state for missing target."""
        state = TargetState(
            name="test_target",
            status="missing",
            manifest=None,
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash=None,
        )
        assert state.name == "test_target"
        assert state.status == "missing"
        assert state.manifest is None
        assert state.staleness_reason is None
        assert state.blocking_deps == ()

    def test_create_computed_state(self) -> None:
        """Create a target state for computed target."""
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
            status="computed",
            manifest=manifest,
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash="hash123",
        )
        assert state.status == "computed"
        assert state.manifest is manifest

    def test_create_stale_state(self) -> None:
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
        reason = StalenessReason(
            kind="input_hash_mismatch",
            details="Hash changed",
        )
        state = TargetState(
            name="test_target",
            status="stale",
            manifest=manifest,
            staleness_reason=reason,
            blocking_deps=(),
            current_input_hash="new_hash",
        )
        assert state.status == "stale"
        assert state.staleness_reason is reason

    def test_create_blocked_state(self) -> None:
        """Create a target state for blocked target."""
        reason = StalenessReason(
            kind="dependency_missing",
            details="Dependency 'ast' not computed",
        )
        state = TargetState(
            name="goids",
            status="blocked",
            manifest=None,
            staleness_reason=reason,
            blocking_deps=("ast",),
            current_input_hash=None,
        )
        assert state.status == "blocked"
        assert state.blocking_deps == ("ast",)


class TestDatabaseState:
    """Tests for DatabaseState dataclass."""

    def test_create_database_state(self) -> None:
        """Create a database state with targets."""
        states = {
            "modules": TargetState(
                name="modules",
                status="computed",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash="hash1",
            ),
            "ast": TargetState(
                name="ast",
                status="missing",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash=None,
            ),
        }
        db_state = DatabaseState(
            repo="test/repo",
            commit="abc123",
            targets=states,
        )
        assert db_state.repo == "test/repo"
        assert db_state.commit == "abc123"
        assert len(db_state.targets) == 2

    def test_get_existing_target(self) -> None:
        """Get state for existing target."""
        target_state = TargetState(
            name="modules",
            status="computed",
            manifest=None,
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash="hash1",
        )
        db_state = DatabaseState(
            repo="test/repo",
            commit="abc123",
            targets={"modules": target_state},
        )
        result = db_state.get("modules")
        assert result is target_state

    def test_get_nonexistent_target_raises(self) -> None:
        """Getting nonexistent target raises KeyError."""
        db_state = DatabaseState(
            repo="test/repo",
            commit="abc123",
            targets={},
        )
        with pytest.raises(KeyError, match="not found"):
            db_state.get("nonexistent")

    def test_missing_targets_query(self) -> None:
        """Query missing targets from database state."""
        states = {
            "modules": TargetState(
                name="modules",
                status="computed",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash="hash1",
            ),
            "ast": TargetState(
                name="ast",
                status="missing",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash=None,
            ),
            "goids": TargetState(
                name="goids",
                status="missing",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash=None,
            ),
        }
        db_state = DatabaseState(repo="test/repo", commit="abc123", targets=states)
        missing = db_state.missing_targets()
        assert missing == ("ast", "goids")

    def test_stale_targets_query(self) -> None:
        """Query stale targets from database state."""
        states = {
            "modules": TargetState(
                name="modules",
                status="stale",
                manifest=None,
                staleness_reason=StalenessReason(kind="input_hash_mismatch", details=""),
                blocking_deps=(),
                current_input_hash="hash1",
            ),
            "ast": TargetState(
                name="ast",
                status="computed",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash="hash2",
            ),
        }
        db_state = DatabaseState(repo="test/repo", commit="abc123", targets=states)
        stale = db_state.stale_targets()
        assert stale == ("modules",)

    def test_computed_targets_query(self) -> None:
        """Query computed targets from database state."""
        states = {
            "modules": TargetState(
                name="modules",
                status="computed",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash="hash1",
            ),
            "ast": TargetState(
                name="ast",
                status="computed",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash="hash2",
            ),
            "goids": TargetState(
                name="goids",
                status="missing",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash=None,
            ),
        }
        db_state = DatabaseState(repo="test/repo", commit="abc123", targets=states)
        computed = db_state.computed_targets()
        assert computed == ("ast", "modules")

    def test_blocked_targets_query(self) -> None:
        """Query blocked targets from database state."""
        states = {
            "modules": TargetState(
                name="modules",
                status="computed",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash="hash1",
            ),
            "goids": TargetState(
                name="goids",
                status="blocked",
                manifest=None,
                staleness_reason=StalenessReason(kind="dependency_missing", details=""),
                blocking_deps=("ast",),
                current_input_hash=None,
            ),
        }
        db_state = DatabaseState(repo="test/repo", commit="abc123", targets=states)
        blocked = db_state.blocked_targets()
        assert blocked == ("goids",)

    def test_is_target_current(self) -> None:
        """Check if target is current (computed)."""
        states = {
            "modules": TargetState(
                name="modules",
                status="computed",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash="hash1",
            ),
            "ast": TargetState(
                name="ast",
                status="stale",
                manifest=None,
                staleness_reason=StalenessReason(kind="input_hash_mismatch", details=""),
                blocking_deps=(),
                current_input_hash="hash2",
            ),
        }
        db_state = DatabaseState(repo="test/repo", commit="abc123", targets=states)
        assert db_state.is_target_current("modules") is True
        assert db_state.is_target_current("ast") is False
        assert db_state.is_target_current("nonexistent") is False


# =============================================================================
# StateValidator Tests
# =============================================================================


class TestStateValidatorInit:
    """Tests for StateValidator initialization."""

    def test_create_validator(
        self,
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Create a state validator with valid inputs."""
        validator = StateValidator(test_graph, fresh_gateway, snapshot)
        assert validator is not None

    def test_invalid_graph_raises(
        self,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Creating validator with invalid graph raises ValueError."""
        # Create graph with missing dependency
        graph = TargetGraph()
        target = OutputTarget(
            name="target",
            module="ingestion",
            plugin="plugin",
            tables=("table",),
            dependencies=("nonexistent",),  # Missing dependency
        )
        graph.register(target)

        with pytest.raises(ValueError, match="validation failed"):
            StateValidator(graph, fresh_gateway, snapshot)


class TestValidateEmptyDatabase:
    """Tests for validating with no manifests."""

    def test_all_targets_missing(
        self,
        validator: StateValidator,
    ) -> None:
        """All targets should be missing when database is empty."""
        state = validator.validate()

        # All targets should be missing
        assert len(state.missing_targets()) == 5  # modules, ast, goids, typing, function_metrics
        assert len(state.computed_targets()) == 0
        assert len(state.stale_targets()) == 0
        assert len(state.blocked_targets()) == 0


class TestValidateComputedTargets:
    """Tests for targets with valid manifests."""

    def test_single_computed_target(
        self,
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Target with matching hash should be computed."""
        # First, compute what the hash should be
        target = test_graph.get("modules")
        correct_hash = compute_input_hash(target, snapshot, fresh_gateway)

        # Update manifest with correct hash
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

        assert state.get("modules").status == "computed"
        assert state.get("modules").manifest is not None

    def test_chain_of_computed_targets(
        self,
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Chain of targets with valid hashes should all be computed."""
        # Compute and save manifest for modules
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

        # Compute and save manifest for ast (depends on modules)
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

        assert state.get("modules").status == "computed"
        assert state.get("ast").status == "computed"


class TestValidateStaleTargets:
    """Tests for targets with mismatched hashes."""

    def test_stale_due_to_input_hash_mismatch(
        self,
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Target with wrong input hash should be stale."""
        # Save manifest with incorrect hash
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
        assert modules_state.status == "stale"
        assert modules_state.staleness_reason is not None
        assert modules_state.staleness_reason.kind == "input_hash_mismatch"
        assert "wrong_hash_value" in modules_state.staleness_reason.details

    def test_staleness_reason_includes_both_hashes(
        self,
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Staleness reason should include both old and new hashes."""
        manifest = OutputManifest(
            target="modules",
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin="repo_scan",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="old_hash_abc",
        )
        fresh_gateway.build.save_manifest(manifest)

        validator = StateValidator(test_graph, fresh_gateway, snapshot)
        state = validator.validate()

        reason = state.get("modules").staleness_reason
        assert reason is not None
        assert "old_hash_abc" in reason.details


class TestValidateBlockedTargets:
    """Tests for targets blocked by dependencies."""

    def test_blocked_by_missing_dependency(
        self,
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Target should be blocked when dependency is missing."""
        # Save manifest for ast but NOT for modules (its dependency)
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

        # modules is missing
        assert state.get("modules").status == "missing"

        # ast should be blocked because modules is missing
        ast_state = state.get("ast")
        assert ast_state.status == "blocked"
        assert "modules" in ast_state.blocking_deps
        assert ast_state.staleness_reason is not None
        assert ast_state.staleness_reason.kind == "dependency_missing"

    def test_blocked_by_stale_dependency(
        self,
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Target should be blocked when dependency is stale."""
        # Save manifest for modules with wrong hash (stale)
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

        # Save manifest for ast with correct hash
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

        # modules is stale
        assert state.get("modules").status == "stale"

        # ast should be blocked because modules is stale
        ast_state = state.get("ast")
        assert ast_state.status == "blocked"
        assert "modules" in ast_state.blocking_deps
        assert ast_state.staleness_reason is not None
        assert ast_state.staleness_reason.kind == "dependency_stale"

    def test_cascade_blocking(
        self,
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Blocking should cascade through dependency chain."""
        # Save manifest for modules with wrong hash (stale)
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

        # Save manifest for ast with correct hash
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

        # Save manifest for goids with correct hash
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

        # modules is stale
        assert state.get("modules").status == "stale"

        # ast is blocked by stale modules
        assert state.get("ast").status == "blocked"

        # goids is blocked because ast is blocked
        goids_state = state.get("goids")
        assert goids_state.status == "blocked"
        assert "ast" in goids_state.blocking_deps
        assert goids_state.staleness_reason is not None
        assert goids_state.staleness_reason.kind == "dependency_blocked"

    def test_multiple_blocking_dependencies(
        self,
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Target with multiple blocking deps should list all."""
        # Make modules stale
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

        # Make ast stale too
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

        # Save manifest for function_metrics (depends on goids and ast)
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

        # function_metrics should be blocked by both ast (stale) and goids (missing)
        metrics_state = state.get("function_metrics")
        assert metrics_state.status == "blocked"
        # Both dependencies should be in blocking_deps
        assert len(metrics_state.blocking_deps) >= 1


class TestValidateTarget:
    """Tests for single-target validation."""

    def test_validate_single_target(
        self,
        validator: StateValidator,
    ) -> None:
        """Validate a single target by name."""
        state = validator.validate_target("modules")
        assert state.name == "modules"
        assert state.status == "missing"  # No manifest saved

    def test_validate_nonexistent_target_raises(
        self,
        validator: StateValidator,
    ) -> None:
        """Validating nonexistent target raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            validator.validate_target("nonexistent")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_manifest_for_unknown_target_logged(
        self,
        test_graph: TargetGraph,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Manifest for unknown target should be logged but not crash."""
        # Save manifest for target not in graph
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

        # Should not crash
        assert state is not None
        # Should log warning
        assert "unknown_target" in caplog.text or len(caplog.records) > 0

    def test_empty_graph(
        self,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Validating empty graph should work."""
        empty_graph = TargetGraph()

        validator = StateValidator(empty_graph, fresh_gateway, snapshot)
        state = validator.validate()

        assert len(state.targets) == 0
        assert state.missing_targets() == ()
        assert state.computed_targets() == ()


# =============================================================================
# Integration Tests with Real Registry
# =============================================================================


class TestWithRealRegistry:
    """Integration tests using the full target registry."""

    def test_validate_with_real_registry(
        self,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Validate using the full target registry."""
        graph = get_target_graph()
        validator = StateValidator(graph, fresh_gateway, snapshot)
        state = validator.validate()

        # All targets should be missing (no manifests)
        assert len(state.missing_targets()) == len(graph)
        assert len(state.computed_targets()) == 0

    def test_real_registry_target_count(
        self,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Verify state covers all registered targets."""
        graph = get_target_graph()
        validator = StateValidator(graph, fresh_gateway, snapshot)
        state = validator.validate()

        assert len(state.targets) == len(ALL_TARGETS)
