"""Unit tests for state validation module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.session import BuildSession
from codeintel.build.state import BuildState, StateValidationOptions, StateValidator, TargetState
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions import expect_equal, expect_false, expect_true
from tests._helpers.cache import make_cache_key_resolver, make_cache_store, seed_cache_store
from tests._helpers.catalog import build_catalog, make_target_descriptor

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.hamilton.cache_adapter import CacheStore
    from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver


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


def _node_dependencies(catalog: DagCatalog) -> dict[str, tuple[str, ...]]:
    dependencies: dict[str, tuple[str, ...]] = {}
    for target in catalog.all_targets:
        node_name = catalog.target_nodes[target.name]
        dependencies[node_name] = tuple(
            catalog.target_nodes[dep] for dep in target.dependencies
        )
    return dependencies


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
def cache_store(tmp_path: Path) -> CacheStore:
    return make_cache_store(tmp_path / "cache")


@pytest.fixture
def cache_key_resolver(
    test_graph: DagCatalog,
    cache_store: CacheStore,
) -> CacheKeyResolver:
    return make_cache_key_resolver(
        node_dependencies=_node_dependencies(test_graph),
        cache_store=cache_store,
    )


@pytest.fixture
def validator(
    test_graph: DagCatalog,
    snapshot: SnapshotRef,
    cache_store: CacheStore,
    cache_key_resolver: CacheKeyResolver,
) -> StateValidator:
    """Provide a StateValidator for tests.

    Returns
    -------
    StateValidator
        Validator for state validation tests.
    """
    session = BuildSession(
        snapshot=snapshot,
        cache_index=cache_store,
        cache_key_resolver=cache_key_resolver,
        input_values={},
    )
    return StateValidator(
        test_graph,
        session,
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
        )
        expect_equal(state.name, "test_target")
        expect_equal(state.status, "missing")
        expect_true(state.is_missing)
        expect_false(state.is_current)
        expect_true(state.needs_computation)

    @staticmethod
    def test_create_current_state() -> None:
        """Create a target state for current target."""
        state = TargetState(
            name="test_target",
            status="current",
            current_hash="cache-key",
        )
        expect_equal(state.status, "current")
        expect_true(state.is_current)
        expect_equal(state.stored_hash, "cache-key")

    @staticmethod
    def test_blocked_state_flags() -> None:
        """Blocked targets should not be runnable."""
        state = TargetState(
            name="blocked_target",
            status="blocked",
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
        state = BuildState(
            repo="test/repo",
            commit="abc123",
            targets={
                "modules": TargetState(
                    name="modules",
                    status="current",
                    current_hash="modules",
                ),
                "ast": TargetState(
                    name="ast",
                    status="missing",
                ),
                "goids": TargetState(
                    name="goids",
                    status="blocked",
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
        """When no cache entries exist, all targets are missing."""
        state = validator.validate()
        expect_equal(state.missing_targets(), tuple(state.targets))
        expect_equal(state.current_targets(), ())
        expect_equal(state.blocked_targets(), ())

    @staticmethod
    def test_cache_presence_marks_current(
        validator: StateValidator,
        cache_store: CacheStore,
        cache_key_resolver: CacheKeyResolver,
    ) -> None:
        """Targets with cache entries should be current."""
        seed_cache_store(
            cache_store,
            cache_key_resolver,
            nodes=("t__modules",),
        )
        state = validator.validate()
        expect_equal(state.current_targets(), ("modules",))
        expect_true("ast" in state.missing_targets())

    @staticmethod
    def test_blocked_when_dependency_missing(
        validator: StateValidator,
    ) -> None:
        """Targets with missing dependencies should be blocked."""
        state = validator.validate()
        ast_state = state.get("ast")
        expect_equal(ast_state.status, "blocked")
        expect_equal(ast_state.blocking_reason, "dependency_missing")

    @staticmethod
    def test_blocked_when_dependency_blocked(
        validator: StateValidator,
    ) -> None:
        """Targets should surface dependency_blocked when deps are blocked."""
        state = validator.validate()
        ast_state = state.get("ast")
        goids_state = state.get("goids")
        expect_equal(ast_state.status, "blocked")
        expect_equal(ast_state.blocking_reason, "dependency_missing")
        expect_equal(goids_state.status, "blocked")
        expect_equal(goids_state.blocking_reason, "dependency_blocked")
