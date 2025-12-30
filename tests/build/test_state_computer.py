"""Tests for the unified StateComputer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.session import BuildSession
from codeintel.build.state_computer import StateComputer
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions import expect_equal
from tests._helpers.cache import make_cache_key_resolver, make_cache_store, seed_cache_store
from tests._helpers.catalog import build_catalog, make_target_descriptor

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.hamilton.cache_adapter import CacheStore
    from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
    from codeintel.build.hamilton.dag_catalog import DagCatalog


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


def _node_dependencies(catalog: DagCatalog) -> dict[str, tuple[str, ...]]:
    dependencies: dict[str, tuple[str, ...]] = {}
    for target in catalog.all_targets:
        node_name = catalog.target_nodes[target.name]
        dependencies[node_name] = tuple(catalog.target_nodes[dep] for dep in target.dependencies)
    return dependencies


@pytest.fixture
def snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide a snapshot reference for tests.

    Returns
    -------
    SnapshotRef
        Snapshot reference for state computer tests.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    return SnapshotRef(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=repo_root,
    )


@pytest.fixture
def cache_store(tmp_path: Path) -> CacheStore:
    """Provide a cache store for state computer tests.

    Returns
    -------
    CacheStore
        Cache store rooted under the temporary path.
    """
    return make_cache_store(tmp_path / "cache")


@pytest.fixture
def cache_key_resolver(
    test_graph: DagCatalog,
    cache_store: CacheStore,
) -> CacheKeyResolver:
    """Provide a cache key resolver for state computer tests.

    Returns
    -------
    CacheKeyResolver
        Resolver configured with the test graph dependencies.
    """
    return make_cache_key_resolver(
        node_dependencies=_node_dependencies(test_graph),
        cache_store=cache_store,
    )


@pytest.fixture
def test_graph() -> DagCatalog:
    """Provide a minimal test catalog for state computer tests.

    Returns
    -------
    DagCatalog
        Catalog for state computer tests.
    """
    return _create_test_graph()


@pytest.fixture
def session(
    snapshot: SnapshotRef,
    cache_store: CacheStore,
    cache_key_resolver: CacheKeyResolver,
) -> BuildSession:
    """Provide a BuildSession for state computer tests.

    Returns
    -------
    BuildSession
        Build session for state computer tests.
    """
    return BuildSession(
        snapshot=snapshot,
        cache_index=cache_store,
        cache_key_resolver=cache_key_resolver,
        input_values={},
    )


@pytest.fixture
def computer(test_graph: DagCatalog, session: BuildSession) -> StateComputer:
    """Provide a StateComputer for tests.

    Returns
    -------
    StateComputer
        State computer for test execution.
    """
    return StateComputer(catalog=test_graph, session=session)


class TestStateComputer:
    """Tests for StateComputer behavior."""

    @staticmethod
    def test_compute_all_missing(computer: StateComputer) -> None:
        """When no cache entries exist, all targets are missing."""
        state = computer.compute_all()
        expect_equal(state.current_targets(), ())
        expect_equal(state.blocked_targets(), ("ast", "goids"))
        expect_equal(state.missing_targets(), ("modules",))

    @staticmethod
    def test_blocked_when_dependency_missing(
        computer: StateComputer,
    ) -> None:
        """Targets with missing dependencies should be blocked."""
        state = computer.compute_all()
        ast_state = state.get("ast")
        expect_equal(ast_state.status, "blocked")
        expect_equal(ast_state.blocking_reason, "dependency_missing")

    @staticmethod
    def test_compute_single_missing(computer: StateComputer) -> None:
        """compute_single should return missing state when no cache entry exists."""
        state = computer.compute_single("modules")
        expect_equal(state.status, "missing")

    @staticmethod
    def test_compute_single_blocked(
        computer: StateComputer,
    ) -> None:
        """compute_single should return blocked state when deps are missing."""
        state = computer.compute_single("ast")
        expect_equal(state.status, "blocked")
        expect_equal(state.blocking_reason, "dependency_missing")


class TestTargetStateEquivalence:
    """Ensure compute_all and compute_single align for cached targets."""

    @staticmethod
    def test_current_state_consistency(
        computer: StateComputer,
        cache_store: CacheStore,
        cache_key_resolver: CacheKeyResolver,
    ) -> None:
        """Compute_all and compute_single should agree on current state."""
        seed_cache_store(
            cache_store,
            cache_key_resolver,
            nodes=("t__modules",),
        )
        state_all = computer.compute_all().get("modules")
        state_single = computer.compute_single("modules")
        expect_equal(state_all.status, "current")
        expect_equal(state_single.status, "current")
        expect_equal(state_all.stored_hash, state_single.stored_hash)
