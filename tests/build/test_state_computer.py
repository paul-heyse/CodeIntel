"""Tests for the unified StateComputer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.session import BuildSession
from codeintel.build.state_computer import StateComputer
from codeintel.config.primitives import SnapshotRef
from codeintel.core.build_manifest import OutputManifest
from tests._helpers.assertions import expect_equal
from tests._helpers.catalog import build_catalog, make_target_descriptor

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.storage.gateway import StorageGateway


def _create_test_graph() -> DagCatalog:
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
    return build_catalog(
        targets=(modules_target, ast_target, goids_target),
        table_keys_by_target={
            "modules": ("core.modules",),
            "ast": ("core.ast_nodes",),
            "goids": ("core.goids",),
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
def snapshot(tmp_path: Path) -> SnapshotRef:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    return SnapshotRef(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=repo_root,
    )


@pytest.fixture
def test_graph() -> DagCatalog:
    return _create_test_graph()


@pytest.fixture
def session(snapshot: SnapshotRef, fresh_gateway: StorageGateway) -> BuildSession:
    return BuildSession(snapshot=snapshot, gateway=fresh_gateway)


@pytest.fixture
def computer(test_graph: DagCatalog, session: BuildSession) -> StateComputer:
    return StateComputer(catalog=test_graph, session=session)


class TestStateComputer:
    """Tests for StateComputer behavior."""

    @staticmethod
    def test_compute_all_missing(computer: StateComputer) -> None:
        """When no manifests exist, all targets are missing."""
        state = computer.compute_all()
        expect_equal(state.current_targets(), ())
        expect_equal(state.blocked_targets(), ())
        expect_equal(state.missing_targets(), tuple(state.targets))

    @staticmethod
    def test_blocked_when_dependency_missing(
        computer: StateComputer,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Targets with missing dependencies should be blocked."""
        _save_manifest(fresh_gateway, snapshot, target="goids", input_hash="goids")
        state = computer.compute_all()
        goids_state = state.get("goids")
        expect_equal(goids_state.status, "blocked")
        expect_equal(goids_state.blocking_reason, "dependency_missing")

    @staticmethod
    def test_compute_single_missing(computer: StateComputer) -> None:
        """compute_single should return missing state when no manifest exists."""
        state = computer.compute_single("modules")
        expect_equal(state.status, "missing")

    @staticmethod
    def test_compute_single_blocked(
        computer: StateComputer,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """compute_single should return blocked state when deps are missing."""
        _save_manifest(fresh_gateway, snapshot, target="goids", input_hash="goids")
        state = computer.compute_single("goids")
        expect_equal(state.status, "blocked")
        expect_equal(state.blocking_reason, "dependency_missing")


class TestTargetStateEquivalence:
    """Ensure compute_all and compute_single align for present manifests."""

    @staticmethod
    def test_current_state_consistency(
        computer: StateComputer,
        fresh_gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        _save_manifest(fresh_gateway, snapshot, target="modules", input_hash="modules")
        state_all = computer.compute_all().get("modules")
        state_single = computer.compute_single("modules")
        expect_equal(state_all.status, "current")
        expect_equal(state_single.status, "current")
        expect_equal(state_all.stored_hash, state_single.stored_hash)
