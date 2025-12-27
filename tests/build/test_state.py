"""Unit tests for state validation module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.state import BuildState, StateValidationOptions, StateValidator, TargetState
from codeintel.config.primitives import SnapshotRef
from codeintel.core.build_manifest import OutputManifest
from tests._helpers.assertions import expect_equal, expect_false, expect_true
from tests._helpers.catalog import build_catalog, make_target_descriptor

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway


def _create_test_graph() -> DagCatalog:
    """Create a minimal test catalog for state validation tests.

    Returns
    -------
    DagCatalog
        Catalog with modules -> ast -> goids chain and independent typing target.
    """
    modules_target = make_target_descriptor(
        name="modules",
        module="ingestion",
        description="Repository module index",
    )

    ast_target = make_target_descriptor(
        name="ast",
        module="ingestion",
        dependencies=("modules",),
        description="AST extraction",
    )

    goids_target = make_target_descriptor(
        name="goids",
        module="graphs",
        dependencies=("ast",),
        description="GOID construction",
    )

    typing_target = make_target_descriptor(
        name="typing",
        module="ingestion",
        dependencies=("modules",),
        description="Type analysis",
    )

    return build_catalog(
        targets=(modules_target, ast_target, goids_target, typing_target),
        table_keys_by_target={
            "modules": ("core.modules",),
            "ast": ("core.ast_nodes",),
            "goids": ("core.goids",),
            "typing": ("analytics.typedness",),
        },
    )


def _save_manifest(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    target: str,
    input_hash: str,
) -> OutputManifest:
    manifest = OutputManifest(
        target=target,
        repo=snapshot.repo,
        commit=snapshot.commit,
        impl_kind="native",
        computed_at=datetime.now(tz=UTC),
        duration_ms=0.0,
        input_hash=input_hash,
    )
    gateway.build.save_manifest(manifest)
    return manifest


@pytest.fixture
def test_graph() -> DagCatalog:
    """Provide a minimal test catalog for state validation tests.

    Returns
    -------
    DagCatalog
        Catalog for state validation tests.
    """
    return _create_test_graph()


@pytest.fixture
def snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide a snapshot reference for tests.

    Returns
    -------
    SnapshotRef
        Snapshot reference for state validation tests.
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
    test_graph: DagCatalog,
    fresh_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> StateValidator:
    """Provide a StateValidator for tests.

    Returns
    -------
    StateValidator
        Validator for state validation tests.
    """
    return StateValidator(
        test_graph,
        fresh_gateway,
        snapshot,
        options=StateValidationOptions(),
    )


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
        expect_true(state.needs_computation)

    @staticmethod
    def test_create_current_state() -> None:
        """Create a target state for current target."""
        manifest = OutputManifest(
            target="test_target",
            repo="test/repo",
            commit="abc123",
            impl_kind="native",
            computed_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            input_hash="cache-key",
        )
        state = TargetState(
            name="test_target",
            status="current",
            manifest=manifest,
        )
        expect_equal(state.status, "current")
        expect_true(state.manifest is manifest)
        expect_true(state.is_current)
        expect_equal(state.stored_hash, "cache-key")

    @staticmethod
    def test_blocked_state_flags() -> None:
        """Blocked targets should not be runnable."""
        state = TargetState(
            name="blocked_target",
            status="blocked",
            manifest=None,
            blocking_reason="dependency_missing",
            blocking_deps=("modules",),
        )
        expect_true(state.is_blocked)
        expect_false(state.can_run)


class TestBuildState:
    """Tests for BuildState helpers."""

    @staticmethod
    def test_build_state_filters() -> None:
        """BuildState should expose status-based filters."""
        manifest = OutputManifest(
            target="modules",
            repo="test/repo",
            commit="abc123",
            impl_kind="native",
            computed_at=datetime.now(tz=UTC),
            duration_ms=10.0,
            input_hash="modules",
        )
        state = BuildState(
            repo="test/repo",
            commit="abc123",
            targets={
                "modules": TargetState(
                    name="modules",
                    status="current",
                    manifest=manifest,
                ),
                "ast": TargetState(
                    name="ast",
                    status="missing",
                    manifest=None,
                ),
                "goids": TargetState(
                    name="goids",
                    status="blocked",
                    manifest=None,
                    blocking_reason="dependency_missing",
                    blocking_deps=("ast",),
                ),
            },
        )
        expect_equal(state.current_targets(), ("modules",))
        expect_equal(state.missing_targets(), ("ast",))
        expect_equal(state.blocked_targets(), ("goids",))
        expect_equal(state.runnable_targets(), ("ast",))
        summary = state.summary
        expect_equal(summary["current"], 1)
        expect_equal(summary["missing"], 1)
        expect_equal(summary["blocked"], 1)


class TestStateValidator:
    """Tests for StateValidator behavior."""

    @staticmethod
    def test_empty_state_is_missing(validator: StateValidator) -> None:
        """When no manifests exist, all targets are missing."""
        state = validator.validate()
        expect_equal(state.missing_targets(), tuple(state.targets))
        expect_equal(state.current_targets(), ())
        expect_equal(state.blocked_targets(), ())

    @staticmethod
    def test_manifest_presence_marks_current(
        validator: StateValidator,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Targets with manifests should be current."""
        _save_manifest(fresh_gateway, snapshot, target="modules", input_hash="modules")
        state = validator.validate()
        expect_equal(state.current_targets(), ("modules",))
        expect_true("ast" in state.missing_targets())

    @staticmethod
    def test_blocked_when_dependency_missing(
        validator: StateValidator,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Targets with missing dependencies should be blocked."""
        _save_manifest(fresh_gateway, snapshot, target="goids", input_hash="goids")
        state = validator.validate()
        goids_state = state.get("goids")
        expect_equal(goids_state.status, "blocked")
        expect_equal(goids_state.blocking_reason, "dependency_missing")

    @staticmethod
    def test_blocked_when_dependency_blocked(
        validator: StateValidator,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Targets should surface dependency_blocked when deps are blocked."""
        _save_manifest(fresh_gateway, snapshot, target="ast", input_hash="ast")
        _save_manifest(fresh_gateway, snapshot, target="goids", input_hash="goids")
        state = validator.validate()
        ast_state = state.get("ast")
        goids_state = state.get("goids")
        expect_equal(ast_state.status, "blocked")
        expect_equal(ast_state.blocking_reason, "dependency_missing")
        expect_equal(goids_state.status, "blocked")
        expect_equal(goids_state.blocking_reason, "dependency_blocked")
