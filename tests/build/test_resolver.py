"""Unit tests for the minimal work resolver module."""

from __future__ import annotations

import pytest

from codeintel.build.registry import get_target_graph
from codeintel.build.resolver import (
    BuildResolver,
    ResolutionReason,
    ResolutionResult,
)
from codeintel.build.state import (
    DatabaseState,
    StalenessReason,
    TargetState,
)
from codeintel.build.targets import OutputTarget, TargetGraph
from tests._helpers import assert_frozen

# =============================================================================
# Test Fixtures
# =============================================================================


def _create_test_graph() -> TargetGraph:
    r"""Create a minimal test graph for resolver tests.

    Graph structure:
        modules (root)
           |
           v
          ast
         /   \
        v     v
      goids  typing
        |
        v
    function_metrics

    Returns
    -------
    TargetGraph
        Graph with modules -> ast -> goids/typing -> function_metrics chain.
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

    # Independent target depending on ast
    typing_target = OutputTarget(
        name="typing",
        module="ingestion",
        plugin="typing_ingest",
        tables=("analytics.typedness",),
        dependencies=("ast",),
        description="Type analysis",
    )

    # Target depending on goids
    metrics_target = OutputTarget(
        name="function_metrics",
        module="analytics",
        plugin="function_metrics",
        tables=("analytics.function_metrics",),
        dependencies=("goids",),
        description="Function metrics",
    )

    graph.register(modules_target)
    graph.register(ast_target)
    graph.register(goids_target)
    graph.register(typing_target)
    graph.register(metrics_target)

    return graph


@pytest.fixture
def resolver_graph() -> TargetGraph:
    """Provide the test graph for resolver tests.

    Returns
    -------
    TargetGraph
        Graph with modules -> ast -> goids/typing -> function_metrics chain.
    """
    return _create_test_graph()


def _create_all_missing_state(graph: TargetGraph) -> DatabaseState:
    """Create a state where all targets are missing.

    Parameters
    ----------
    graph
        Target graph to create state for.

    Returns
    -------
    DatabaseState
        State with all targets marked as missing.
    """
    targets: dict[str, TargetState] = {}
    for name in graph:
        targets[name] = TargetState(
            name=name,
            status="missing",
            manifest=None,
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash=None,
        )
    return DatabaseState(repo="test/repo", commit="abc123", targets=targets)


def _create_all_computed_state(graph: TargetGraph) -> DatabaseState:
    """Create a state where all targets are computed.

    Parameters
    ----------
    graph
        Target graph to create state for.

    Returns
    -------
    DatabaseState
        State with all targets marked as computed.
    """
    targets: dict[str, TargetState] = {}
    for name in graph:
        targets[name] = TargetState(
            name=name,
            status="computed",
            manifest=None,
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash=f"hash_{name}",
        )
    return DatabaseState(repo="test/repo", commit="abc123", targets=targets)


def _create_state_with_stale(
    graph: TargetGraph,
    stale_targets: set[str],
) -> DatabaseState:
    """Create a state with specified targets marked as stale.

    Parameters
    ----------
    graph
        Target graph to create state for.
    stale_targets
        Names of targets to mark as stale.

    Returns
    -------
    DatabaseState
        State with specified targets stale, others computed.
    """
    targets: dict[str, TargetState] = {}
    for name in graph:
        if name in stale_targets:
            targets[name] = TargetState(
                name=name,
                status="stale",
                manifest=None,
                staleness_reason=StalenessReason(
                    kind="input_hash_mismatch",
                    details="Input hash changed",
                ),
                blocking_deps=(),
                current_input_hash=f"new_hash_{name}",
            )
        else:
            targets[name] = TargetState(
                name=name,
                status="computed",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash=f"hash_{name}",
            )
    return DatabaseState(repo="test/repo", commit="abc123", targets=targets)


def _create_state_with_blocked(
    graph: TargetGraph,
    blocked_map: dict[str, tuple[str, ...]],
) -> DatabaseState:
    """Create a state with specified targets marked as blocked.

    Parameters
    ----------
    graph
        Target graph to create state for.
    blocked_map
        Mapping of target name to blocking dependencies.

    Returns
    -------
    DatabaseState
        State with specified targets blocked, others computed.
    """
    targets: dict[str, TargetState] = {}
    for name in graph:
        if name in blocked_map:
            targets[name] = TargetState(
                name=name,
                status="blocked",
                manifest=None,
                staleness_reason=StalenessReason(
                    kind="dependency_missing",
                    details=f"Blocked by {blocked_map[name]}",
                ),
                blocking_deps=blocked_map[name],
                current_input_hash=None,
            )
        else:
            targets[name] = TargetState(
                name=name,
                status="computed",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash=f"hash_{name}",
            )
    return DatabaseState(repo="test/repo", commit="abc123", targets=targets)


@pytest.fixture
def all_missing_state(resolver_graph: TargetGraph) -> DatabaseState:
    """Provide state with all targets missing.

    Parameters
    ----------
    resolver_graph
        Test graph.

    Returns
    -------
    DatabaseState
        All targets missing.
    """
    return _create_all_missing_state(resolver_graph)


@pytest.fixture
def all_computed_state(resolver_graph: TargetGraph) -> DatabaseState:
    """Provide state with all targets computed.

    Parameters
    ----------
    resolver_graph
        Test graph.

    Returns
    -------
    DatabaseState
        All targets computed.
    """
    return _create_all_computed_state(resolver_graph)


# =============================================================================
# Type Definition Tests
# =============================================================================


class TestResolutionReason:
    """Tests for ResolutionReason dataclass."""

    def test_create_resolution_reason(self) -> None:
        """Create a resolution reason with all fields."""
        reason = ResolutionReason(
            kind="cascade",
            details="Dependency 'ast' is being recomputed",
        )
        assert reason.kind == "cascade"
        assert "ast" in reason.details

    def test_resolution_reason_is_frozen(self) -> None:
        """Verify resolution reason is immutable."""
        reason = ResolutionReason(kind="missing", details="No manifest")
        assert_frozen(reason, "kind", "stale")


class TestResolutionResult:
    """Tests for ResolutionResult dataclass."""

    def test_create_result(self) -> None:
        """Create a resolution result with all fields."""
        result = ResolutionResult(
            requested=("function_metrics",),
            to_compute=("modules", "ast", "goids", "function_metrics"),
            to_skip=(),
            blocked=(),
            reasons={
                "modules": ResolutionReason(kind="missing", details="No manifest"),
            },
        )
        assert result.requested == ("function_metrics",)
        assert len(result.to_compute) == 4

    def test_total_work(self) -> None:
        """Test total_work property."""
        result = ResolutionResult(
            requested=("x",),
            to_compute=("a", "b", "c"),
            to_skip=("d",),
            blocked=(),
            reasons={},
        )
        assert result.total_work == 3

    def test_total_skipped(self) -> None:
        """Test total_skipped property."""
        result = ResolutionResult(
            requested=("x",),
            to_compute=("a",),
            to_skip=("b", "c", "d"),
            blocked=(),
            reasons={},
        )
        assert result.total_skipped == 3

    def test_is_empty_true(self) -> None:
        """Test is_empty returns True when no work."""
        result = ResolutionResult(
            requested=("x",),
            to_compute=(),
            to_skip=("x",),
            blocked=(),
            reasons={},
        )
        assert result.is_empty() is True

    def test_is_empty_false(self) -> None:
        """Test is_empty returns False when work exists."""
        result = ResolutionResult(
            requested=("x",),
            to_compute=("x",),
            to_skip=(),
            blocked=(),
            reasons={},
        )
        assert result.is_empty() is False

    def test_get_reason(self) -> None:
        """Test get_reason retrieves correct reason."""
        reason = ResolutionReason(kind="missing", details="No manifest")
        result = ResolutionResult(
            requested=("x",),
            to_compute=("x",),
            to_skip=(),
            blocked=(),
            reasons={"x": reason},
        )
        assert result.get_reason("x") is reason

    def test_get_reason_raises_key_error(self) -> None:
        """Test get_reason raises KeyError for unknown target."""
        result = ResolutionResult(
            requested=("x",),
            to_compute=("x",),
            to_skip=(),
            blocked=(),
            reasons={},
        )
        with pytest.raises(KeyError, match="not found"):
            result.get_reason("nonexistent")


# =============================================================================
# Basic Resolution Tests
# =============================================================================


class TestBasicResolution:
    """Tests for basic resolution scenarios."""

    def test_resolve_empty_goals(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Empty goals returns empty result."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve([])

        assert result.requested == ()
        assert result.to_compute == ()
        assert result.to_skip == ()
        assert result.blocked == ()
        assert result.is_empty() is True

    def test_resolve_single_root_missing(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Single root target missing results in one compute."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["modules"])

        assert result.requested == ("modules",)
        assert result.to_compute == ("modules",)
        assert result.total_work == 1
        assert result.get_reason("modules").kind == "missing"

    def test_resolve_single_with_deps_all_missing(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Goal with deps, all missing, computes full chain."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["function_metrics"])

        # Should compute modules -> ast -> goids -> function_metrics
        assert result.requested == ("function_metrics",)
        assert len(result.to_compute) == 4
        assert "modules" in result.to_compute
        assert "ast" in result.to_compute
        assert "goids" in result.to_compute
        assert "function_metrics" in result.to_compute

        # Should be in topological order
        compute_list = list(result.to_compute)
        assert compute_list.index("modules") < compute_list.index("ast")
        assert compute_list.index("ast") < compute_list.index("goids")
        assert compute_list.index("goids") < compute_list.index("function_metrics")

    def test_resolve_all_computed(
        self,
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """All current, nothing to compute."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["function_metrics"])

        assert result.is_empty() is True
        assert result.total_work == 0
        assert result.total_skipped == 4  # modules, ast, goids, function_metrics
        assert result.get_reason("function_metrics").kind == "current"

    def test_resolve_invalid_goal_raises(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Invalid goal raises KeyError."""
        resolver = BuildResolver(resolver_graph, all_missing_state)

        with pytest.raises(KeyError, match="nonexistent"):
            resolver.resolve(["nonexistent"])

    def test_resolve_multiple_invalid_goals(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Multiple invalid goals all listed in error."""
        resolver = BuildResolver(resolver_graph, all_missing_state)

        with pytest.raises(KeyError, match=r"foo.*bar|bar.*foo"):
            resolver.resolve(["foo", "bar"])


# =============================================================================
# Cascade Invalidation Tests
# =============================================================================


class TestCascadeInvalidation:
    """Tests for cascade invalidation behavior."""

    def test_cascade_from_stale_root(
        self,
        resolver_graph: TargetGraph,
    ) -> None:
        """Stale root causes cascade to all dependents."""
        state = _create_state_with_stale(resolver_graph, {"modules"})
        resolver = BuildResolver(resolver_graph, state)

        result = resolver.resolve(["function_metrics"])

        # All should be computed due to cascade
        assert "modules" in result.to_compute
        assert "ast" in result.to_compute
        assert "goids" in result.to_compute
        assert "function_metrics" in result.to_compute

        # Root is stale, others are cascade
        assert result.get_reason("modules").kind == "stale"
        assert result.get_reason("ast").kind == "cascade"
        assert result.get_reason("goids").kind == "cascade"
        assert result.get_reason("function_metrics").kind == "cascade"

    def test_cascade_from_stale_middle(
        self,
        resolver_graph: TargetGraph,
    ) -> None:
        """Stale middle target cascades only downstream."""
        state = _create_state_with_stale(resolver_graph, {"ast"})
        resolver = BuildResolver(resolver_graph, state)

        result = resolver.resolve(["function_metrics"])

        # modules should be skipped (upstream of stale)
        assert "modules" in result.to_skip
        # ast and downstream should compute
        assert "ast" in result.to_compute
        assert "goids" in result.to_compute
        assert "function_metrics" in result.to_compute

        assert result.get_reason("modules").kind == "current"
        assert result.get_reason("ast").kind == "stale"
        assert result.get_reason("goids").kind == "cascade"

    def test_no_cascade_when_root_current(
        self,
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """Current root doesn't cascade - all can skip."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["function_metrics"])

        assert result.is_empty() is True
        assert result.total_skipped == 4

    def test_partial_cascade_only_affected_subtree(
        self,
        resolver_graph: TargetGraph,
    ) -> None:
        """Only affected subtree cascades, parallel branches unaffected."""
        # Make goids stale - should cascade to function_metrics
        # but typing (parallel branch from ast) should be unaffected if not requested
        state = _create_state_with_stale(resolver_graph, {"goids"})
        resolver = BuildResolver(resolver_graph, state)

        result = resolver.resolve(["function_metrics"])

        # Only goids and function_metrics should compute
        assert "modules" in result.to_skip
        assert "ast" in result.to_skip
        assert "goids" in result.to_compute
        assert "function_metrics" in result.to_compute

        assert result.get_reason("goids").kind == "stale"
        assert result.get_reason("function_metrics").kind == "cascade"

    def test_cascade_with_multiple_goals(
        self,
        resolver_graph: TargetGraph,
    ) -> None:
        """Cascade affects all goals sharing the stale dependency."""
        # Make ast stale - should cascade to both goids and typing
        state = _create_state_with_stale(resolver_graph, {"ast"})
        resolver = BuildResolver(resolver_graph, state)

        # Request both typing and function_metrics
        result = resolver.resolve(["typing", "function_metrics"])

        # Both branches should cascade from stale ast
        assert "ast" in result.to_compute
        assert "typing" in result.to_compute
        assert "goids" in result.to_compute
        assert "function_metrics" in result.to_compute

        assert result.get_reason("typing").kind == "cascade"
        assert result.get_reason("goids").kind == "cascade"


# =============================================================================
# Force Recompute Tests
# =============================================================================


class TestForceRecompute:
    """Tests for force_recompute behavior."""

    def test_force_single_target(
        self,
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """Force recomputes a single target."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["modules"], force_recompute=["modules"])

        assert "modules" in result.to_compute
        assert result.get_reason("modules").kind == "forced"

    def test_force_causes_cascade(
        self,
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """Force triggers cascade to dependents."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["function_metrics"], force_recompute=["ast"])

        # ast forced, should cascade downstream
        assert "ast" in result.to_compute
        assert "goids" in result.to_compute
        assert "function_metrics" in result.to_compute

        # modules unaffected
        assert "modules" in result.to_skip

        assert result.get_reason("ast").kind == "forced"
        assert result.get_reason("goids").kind == "cascade"

    def test_force_unknown_target_ignored(
        self,
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unknown force target is logged and ignored."""
        resolver = BuildResolver(resolver_graph, all_computed_state)

        with caplog.at_level("WARNING"):
            resolver.resolve(["modules"], force_recompute=["nonexistent"])

        # Should still work, just ignoring the unknown force target
        # modules is current (not forced because nonexistent isn't a valid target)
        assert "nonexistent" in caplog.text or len(caplog.records) > 0

    def test_force_irrelevant_target_ignored(
        self,
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Force target not in transitive deps is ignored."""
        resolver = BuildResolver(resolver_graph, all_computed_state)

        # Force typing but only request modules (typing is not a dep of modules)
        with caplog.at_level("WARNING"):
            result = resolver.resolve(["modules"], force_recompute=["typing"])

        # typing should not be forced because it's not needed for modules
        assert "typing" not in result.to_compute


# =============================================================================
# Blocked Target Tests
# =============================================================================


class TestBlockedTargets:
    """Tests for blocked target handling."""

    def test_blocked_becomes_computable(
        self,
        resolver_graph: TargetGraph,
    ) -> None:
        """Blocked target resolves when blocking deps will compute."""
        # ast is blocked by modules (which is missing)
        # When we resolve, modules will be computed, so ast becomes computable
        state = _create_state_with_blocked(
            resolver_graph,
            blocked_map={"ast": ("modules",)},
        )
        # Also mark modules as missing
        targets = dict(state.targets)
        targets["modules"] = TargetState(
            name="modules",
            status="missing",
            manifest=None,
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash=None,
        )
        state = DatabaseState(repo="test/repo", commit="abc123", targets=targets)

        resolver = BuildResolver(resolver_graph, state)
        result = resolver.resolve(["ast"])

        # Both should be computable now
        assert "modules" in result.to_compute
        assert "ast" in result.to_compute
        assert len(result.blocked) == 0

    def test_truly_blocked_external(self) -> None:
        """External block prevents computation."""
        # Create a scenario where ast is blocked by something that won't be computed
        # This is a bit artificial - in real usage, blocking deps should be in the graph
        # But we can simulate by having a target blocked by a computed target
        # that we don't need to recompute

        # Let's create a more realistic scenario:
        # Add an external_data target that's blocked by something not in our goals
        graph = _create_test_graph()
        external_target = OutputTarget(
            name="external_data",
            module="ingestion",
            plugin="external_ingest",
            tables=("core.external",),
            dependencies=(),  # No deps in graph
            description="External data",
        )
        graph.register(external_target)

        # Create state where external_data is blocked by something
        # that won't be resolved (simulating external constraint)
        targets: dict[str, TargetState] = {}
        for name in graph:
            if name == "external_data":
                targets[name] = TargetState(
                    name=name,
                    status="blocked",
                    manifest=None,
                    staleness_reason=StalenessReason(
                        kind="dependency_missing",
                        details="Blocked by external system",
                    ),
                    blocking_deps=("external_system",),  # Not in graph
                    current_input_hash=None,
                )
            else:
                targets[name] = TargetState(
                    name=name,
                    status="computed",
                    manifest=None,
                    staleness_reason=None,
                    blocking_deps=(),
                    current_input_hash=f"hash_{name}",
                )

        state = DatabaseState(repo="test/repo", commit="abc123", targets=targets)
        resolver = BuildResolver(graph, state)
        result = resolver.resolve(["external_data"])

        # external_data should be blocked
        assert "external_data" in result.blocked
        assert result.get_reason("external_data").kind == "blocked_external"


# =============================================================================
# Module Filtering Tests
# =============================================================================


class TestResolveAll:
    """Tests for resolve_all method."""

    def test_resolve_all_no_filter(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Resolve all targets when no filter."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve_all()

        # All 5 targets should be in to_compute
        assert len(result.to_compute) == 5
        assert "modules" in result.to_compute
        assert "ast" in result.to_compute
        assert "goids" in result.to_compute
        assert "typing" in result.to_compute
        assert "function_metrics" in result.to_compute

    def test_resolve_all_ingestion_filter(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Resolve only ingestion targets."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve_all(module="ingestion")

        # Only ingestion targets requested (modules, ast, typing)
        assert "modules" in result.requested
        assert "ast" in result.requested
        assert "typing" in result.requested

        # But dependencies still computed
        assert "modules" in result.to_compute
        assert "ast" in result.to_compute
        assert "typing" in result.to_compute

    def test_resolve_all_analytics_filter(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Resolve only analytics targets."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve_all(module="analytics")

        # Only function_metrics is analytics
        assert result.requested == ("function_metrics",)

        # But should include all deps
        assert "modules" in result.to_compute
        assert "ast" in result.to_compute
        assert "goids" in result.to_compute
        assert "function_metrics" in result.to_compute


# =============================================================================
# Reason Tracking Tests
# =============================================================================


class TestReasonTracking:
    """Tests for reason tracking accuracy."""

    def test_reasons_for_missing(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Missing targets have 'missing' reason."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["modules"])

        reason = result.get_reason("modules")
        assert reason.kind == "missing"
        assert "manifest" in reason.details.lower() or "compute" in reason.details.lower()

    def test_reasons_for_stale(
        self,
        resolver_graph: TargetGraph,
    ) -> None:
        """Stale targets have 'stale' reason."""
        state = _create_state_with_stale(resolver_graph, {"modules"})
        resolver = BuildResolver(resolver_graph, state)
        result = resolver.resolve(["modules"])

        reason = result.get_reason("modules")
        assert reason.kind == "stale"

    def test_reasons_for_cascade(
        self,
        resolver_graph: TargetGraph,
    ) -> None:
        """Cascade targets have 'cascade' reason."""
        state = _create_state_with_stale(resolver_graph, {"modules"})
        resolver = BuildResolver(resolver_graph, state)
        result = resolver.resolve(["ast"])

        reason = result.get_reason("ast")
        assert reason.kind == "cascade"
        assert "modules" in reason.details

    def test_reasons_for_current(
        self,
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """Skipped targets have 'current' reason."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["modules"])

        reason = result.get_reason("modules")
        assert reason.kind == "current"
        assert "up-to-date" in reason.details.lower()

    def test_reasons_for_forced(
        self,
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """Forced targets have 'forced' reason."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["modules"], force_recompute=["modules"])

        reason = result.get_reason("modules")
        assert reason.kind == "forced"

    def test_reasons_include_goal_annotation(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Requested goals are annotated in reason details."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["function_metrics"])

        # function_metrics is a goal, should be noted
        reason = result.get_reason("function_metrics")
        assert "goal" in reason.details.lower()


# =============================================================================
# Topological Order Tests
# =============================================================================


class TestTopologicalOrder:
    """Tests for correct topological ordering."""

    def test_to_compute_in_topological_order(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """to_compute respects dependency order."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["function_metrics"])

        compute_list = list(result.to_compute)

        # modules must come before ast
        assert compute_list.index("modules") < compute_list.index("ast")
        # ast must come before goids
        assert compute_list.index("ast") < compute_list.index("goids")
        # goids must come before function_metrics
        assert compute_list.index("goids") < compute_list.index("function_metrics")

    def test_parallel_branches_both_included(
        self,
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Parallel branches are both computed when needed."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["typing", "goids"])

        # Both typing and goids depend on ast
        assert "ast" in result.to_compute
        assert "typing" in result.to_compute
        assert "goids" in result.to_compute

        # ast must come before both
        compute_list = list(result.to_compute)
        assert compute_list.index("ast") < compute_list.index("typing")
        assert compute_list.index("ast") < compute_list.index("goids")


# =============================================================================
# Integration Tests with Real Registry
# =============================================================================


class TestWithRealRegistry:
    """Integration tests using the full target registry."""

    def test_resolve_with_real_registry(self) -> None:
        """Resolve using the full target registry."""
        graph = get_target_graph()

        # Create all-missing state for real graph
        targets: dict[str, TargetState] = {}
        for name in graph:
            targets[name] = TargetState(
                name=name,
                status="missing",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash=None,
            )
        state = DatabaseState(repo="test/repo", commit="abc123", targets=targets)

        resolver = BuildResolver(graph, state)

        # Resolve a real target
        result = resolver.resolve(["function_metrics"])

        # Should include all its dependencies
        assert "modules" in result.to_compute
        assert "function_metrics" in result.to_compute
        assert len(result.to_compute) > 1  # Has dependencies

    def test_real_registry_all_targets_resolvable(self) -> None:
        """All targets in real registry can be resolved."""
        graph = get_target_graph()

        # Create all-missing state
        targets: dict[str, TargetState] = {}
        for name in graph:
            targets[name] = TargetState(
                name=name,
                status="missing",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash=None,
            )
        state = DatabaseState(repo="test/repo", commit="abc123", targets=targets)

        resolver = BuildResolver(graph, state)

        # resolve_all should work without errors
        result = resolver.resolve_all()

        # All targets should be in to_compute (since all are missing)
        assert result.total_work == len(list(graph))
