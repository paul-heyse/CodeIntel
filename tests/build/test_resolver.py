"""Unit tests for the minimal work resolver module."""

from __future__ import annotations

import pytest
from codeintel.build.resolver import (
    BuildResolver,
    ResolutionReason,
    ResolutionResult,
)

from codeintel.build.registry import get_target_graph
from codeintel.build.state import (
    DatabaseState,
    StalenessReason,
    TargetState,
)
from codeintel.build.targets import OutputTarget, TargetGraph, TargetOptions
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_length,
    expect_true,
)


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
        options=TargetOptions(dependencies=("ast",), description="Type analysis"),
    )

    metrics_target = OutputTarget.from_tables(
        name="function_metrics",
        module="analytics",
        plugin="function_metrics",
        tables=("analytics.function_metrics",),
        options=TargetOptions(dependencies=("goids",), description="Function metrics"),
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


class TestResolutionReason:
    """Tests for ResolutionReason dataclass."""

    @staticmethod
    def test_create_resolution_reason() -> None:
        """Create a resolution reason with all fields."""
        reason = ResolutionReason(
            kind="cascade",
            details="Dependency 'ast' is being recomputed",
        )
        expect_equal(reason.kind, "cascade")
        expect_in("ast", reason.details)

    @staticmethod
    def test_resolution_reason_is_frozen() -> None:
        """Verify resolution reason is immutable."""
        reason = ResolutionReason(kind="missing", details="No manifest")
        assert_frozen(reason, "kind", "stale")


class TestResolutionResult:
    """Tests for ResolutionResult dataclass."""

    @staticmethod
    def test_create_result() -> None:
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
        expect_equal(result.requested, ("function_metrics",))
        expect_length(result.to_compute, 4)

    @staticmethod
    def test_total_work() -> None:
        """Test total_work property."""
        result = ResolutionResult(
            requested=("x",),
            to_compute=("a", "b", "c"),
            to_skip=("d",),
            blocked=(),
            reasons={},
        )
        expect_equal(result.total_work, 3)

    @staticmethod
    def test_total_skipped() -> None:
        """Test total_skipped property."""
        result = ResolutionResult(
            requested=("x",),
            to_compute=("a",),
            to_skip=("b", "c", "d"),
            blocked=(),
            reasons={},
        )
        expect_equal(result.total_skipped, 3)

    @staticmethod
    def test_is_empty_true() -> None:
        """Test is_empty returns True when no work."""
        result = ResolutionResult(
            requested=("x",),
            to_compute=(),
            to_skip=("x",),
            blocked=(),
            reasons={},
        )
        expect_true(result.is_empty())

    @staticmethod
    def test_is_empty_false() -> None:
        """Test is_empty returns False when work exists."""
        result = ResolutionResult(
            requested=("x",),
            to_compute=("x",),
            to_skip=(),
            blocked=(),
            reasons={},
        )
        expect_false(result.is_empty())

    @staticmethod
    def test_get_reason() -> None:
        """Test get_reason retrieves correct reason."""
        reason = ResolutionReason(kind="missing", details="No manifest")
        result = ResolutionResult(
            requested=("x",),
            to_compute=("x",),
            to_skip=(),
            blocked=(),
            reasons={"x": reason},
        )
        expect_equal(result.get_reason("x"), reason)

    @staticmethod
    def test_get_reason_raises_key_error() -> None:
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


class TestBasicResolution:
    """Tests for basic resolution scenarios."""

    @staticmethod
    def test_resolve_empty_goals(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Empty goals returns empty result."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve([])

        expect_equal(result.requested, ())
        expect_equal(result.to_compute, ())
        expect_equal(result.to_skip, ())
        expect_equal(result.blocked, ())
        expect_true(result.is_empty())

    @staticmethod
    def test_resolve_single_root_missing(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Single root target missing results in one compute."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["modules"])

        expect_equal(result.requested, ("modules",))
        expect_equal(result.to_compute, ("modules",))
        expect_equal(result.total_work, 1)
        expect_equal(result.get_reason("modules").kind, "missing")

    @staticmethod
    def test_resolve_single_with_deps_all_missing(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Goal with deps, all missing, computes full chain."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["function_metrics"])

        expect_equal(result.requested, ("function_metrics",))
        expect_length(result.to_compute, 4)
        expect_in("modules", result.to_compute)
        expect_in("ast", result.to_compute)
        expect_in("goids", result.to_compute)
        expect_in("function_metrics", result.to_compute)

        compute_list = list(result.to_compute)
        expect_true(compute_list.index("modules") < compute_list.index("ast"))
        expect_true(compute_list.index("ast") < compute_list.index("goids"))
        expect_true(
            compute_list.index("goids") < compute_list.index("function_metrics"),
        )

    @staticmethod
    def test_resolve_all_computed(
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """All current, nothing to compute."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["function_metrics"])

        expect_true(result.is_empty())
        expect_equal(result.total_work, 0)
        expect_equal(result.total_skipped, 4)
        expect_equal(result.get_reason("function_metrics").kind, "current")

    @staticmethod
    def test_resolve_invalid_goal_raises(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Invalid goal raises KeyError."""
        resolver = BuildResolver(resolver_graph, all_missing_state)

        with pytest.raises(KeyError, match="nonexistent"):
            resolver.resolve(["nonexistent"])

    @staticmethod
    def test_resolve_multiple_invalid_goals(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Multiple invalid goals all listed in error."""
        resolver = BuildResolver(resolver_graph, all_missing_state)

        with pytest.raises(KeyError, match=r"foo.*bar|bar.*foo"):
            resolver.resolve(["foo", "bar"])


class TestCascadeInvalidation:
    """Tests for cascade invalidation behavior."""

    @staticmethod
    def test_cascade_from_stale_root(
        resolver_graph: TargetGraph,
    ) -> None:
        """Stale root causes cascade to all dependents."""
        state = _create_state_with_stale(resolver_graph, {"modules"})
        resolver = BuildResolver(resolver_graph, state)

        result = resolver.resolve(["function_metrics"])

        expect_in("modules", result.to_compute)
        expect_in("ast", result.to_compute)
        expect_in("goids", result.to_compute)
        expect_in("function_metrics", result.to_compute)

        expect_equal(result.get_reason("modules").kind, "stale")
        expect_equal(result.get_reason("ast").kind, "cascade")
        expect_equal(result.get_reason("goids").kind, "cascade")
        expect_equal(result.get_reason("function_metrics").kind, "cascade")

    @staticmethod
    def test_cascade_from_stale_middle(
        resolver_graph: TargetGraph,
    ) -> None:
        """Stale middle target cascades only downstream."""
        state = _create_state_with_stale(resolver_graph, {"ast"})
        resolver = BuildResolver(resolver_graph, state)

        result = resolver.resolve(["function_metrics"])

        expect_in("modules", result.to_skip)

        expect_in("ast", result.to_compute)
        expect_in("goids", result.to_compute)
        expect_in("function_metrics", result.to_compute)

        expect_equal(result.get_reason("modules").kind, "current")
        expect_equal(result.get_reason("ast").kind, "stale")
        expect_equal(result.get_reason("goids").kind, "cascade")

    @staticmethod
    def test_no_cascade_when_root_current(
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """Current root doesn't cascade - all can skip."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["function_metrics"])

        expect_true(result.is_empty())
        expect_equal(result.total_skipped, 4)

    @staticmethod
    def test_partial_cascade_only_affected_subtree(
        resolver_graph: TargetGraph,
    ) -> None:
        """Only affected subtree cascades, parallel branches unaffected."""
        state = _create_state_with_stale(resolver_graph, {"goids"})
        resolver = BuildResolver(resolver_graph, state)

        result = resolver.resolve(["function_metrics"])

        expect_in("modules", result.to_skip)
        expect_in("ast", result.to_skip)
        expect_in("goids", result.to_compute)
        expect_in("function_metrics", result.to_compute)

        expect_equal(result.get_reason("goids").kind, "stale")
        expect_equal(result.get_reason("function_metrics").kind, "cascade")

    @staticmethod
    def test_cascade_with_multiple_goals(
        resolver_graph: TargetGraph,
    ) -> None:
        """Cascade affects all goals sharing the stale dependency."""
        state = _create_state_with_stale(resolver_graph, {"ast"})
        resolver = BuildResolver(resolver_graph, state)

        result = resolver.resolve(["typing", "function_metrics"])

        expect_in("ast", result.to_compute)
        expect_in("typing", result.to_compute)
        expect_in("goids", result.to_compute)
        expect_in("function_metrics", result.to_compute)

        expect_equal(result.get_reason("typing").kind, "cascade")
        expect_equal(result.get_reason("goids").kind, "cascade")


class TestForceRecompute:
    """Tests for force_recompute behavior."""

    @staticmethod
    def test_force_single_target(
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """Force recomputes a single target."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["modules"], force_recompute=["modules"])

        expect_in("modules", result.to_compute)
        expect_equal(result.get_reason("modules").kind, "forced")

    @staticmethod
    def test_force_causes_cascade(
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """Force triggers cascade to dependents."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["function_metrics"], force_recompute=["ast"])

        expect_in("ast", result.to_compute)
        expect_in("goids", result.to_compute)
        expect_in("function_metrics", result.to_compute)

        expect_in("modules", result.to_skip)

        expect_equal(result.get_reason("ast").kind, "forced")
        expect_equal(result.get_reason("goids").kind, "cascade")

    @staticmethod
    def test_force_unknown_target_ignored(
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unknown force target is logged and ignored."""
        resolver = BuildResolver(resolver_graph, all_computed_state)

        with caplog.at_level("WARNING"):
            resolver.resolve(["modules"], force_recompute=["nonexistent"])

        expect_true("nonexistent" in caplog.text or len(caplog.records) > 0)

    @staticmethod
    def test_force_irrelevant_target_ignored(
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Force target not in transitive deps is ignored."""
        resolver = BuildResolver(resolver_graph, all_computed_state)

        with caplog.at_level("WARNING"):
            result = resolver.resolve(["modules"], force_recompute=["typing"])

        expect_false("typing" in result.to_compute)


class TestBlockedTargets:
    """Tests for blocked target handling."""

    @staticmethod
    def test_blocked_becomes_computable(
        resolver_graph: TargetGraph,
    ) -> None:
        """Blocked target resolves when blocking deps will compute."""
        state = _create_state_with_blocked(
            resolver_graph,
            blocked_map={"ast": ("modules",)},
        )

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

        expect_in("modules", result.to_compute)
        expect_in("ast", result.to_compute)
        expect_length(result.blocked, 0)

    @staticmethod
    def test_truly_blocked_external() -> None:
        """External block prevents computation."""
        graph = _create_test_graph()
        external_target = OutputTarget.from_tables(
            name="external_data",
            module="ingestion",
            plugin="external_ingest",
            tables=("core.external",),
            options=TargetOptions(description="External data"),
        )
        graph.register(external_target)

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
                    blocking_deps=("external_system",),
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

        expect_in("external_data", result.blocked)
        expect_equal(result.get_reason("external_data").kind, "blocked_external")


class TestResolveAll:
    """Tests for resolve_all method."""

    @staticmethod
    def test_resolve_all_no_filter(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Resolve all targets when no filter."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve_all()

        expect_length(result.to_compute, 5)
        expect_in("modules", result.to_compute)
        expect_in("ast", result.to_compute)
        expect_in("goids", result.to_compute)
        expect_in("typing", result.to_compute)
        expect_in("function_metrics", result.to_compute)

    @staticmethod
    def test_resolve_all_ingestion_filter(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Resolve only ingestion targets."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve_all(module="ingestion")

        expect_in("modules", result.requested)
        expect_in("ast", result.requested)
        expect_in("typing", result.requested)

        expect_in("modules", result.to_compute)
        expect_in("ast", result.to_compute)
        expect_in("typing", result.to_compute)

    @staticmethod
    def test_resolve_all_analytics_filter(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Resolve only analytics targets."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve_all(module="analytics")

        expect_equal(result.requested, ("function_metrics",))

        expect_in("modules", result.to_compute)
        expect_in("ast", result.to_compute)
        expect_in("goids", result.to_compute)
        expect_in("function_metrics", result.to_compute)


class TestReasonTracking:
    """Tests for reason tracking accuracy."""

    @staticmethod
    def test_reasons_for_missing(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Missing targets have 'missing' reason."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["modules"])

        reason = result.get_reason("modules")
        expect_equal(reason.kind, "missing")
        expect_true(
            "manifest" in reason.details.lower() or "compute" in reason.details.lower(),
        )

    @staticmethod
    def test_reasons_for_stale(
        resolver_graph: TargetGraph,
    ) -> None:
        """Stale targets have 'stale' reason."""
        state = _create_state_with_stale(resolver_graph, {"modules"})
        resolver = BuildResolver(resolver_graph, state)
        result = resolver.resolve(["modules"])

        reason = result.get_reason("modules")
        expect_equal(reason.kind, "stale")

    @staticmethod
    def test_reasons_for_cascade(
        resolver_graph: TargetGraph,
    ) -> None:
        """Cascade targets have 'cascade' reason."""
        state = _create_state_with_stale(resolver_graph, {"modules"})
        resolver = BuildResolver(resolver_graph, state)
        result = resolver.resolve(["ast"])

        reason = result.get_reason("ast")
        expect_equal(reason.kind, "cascade")
        expect_in("modules", reason.details)

    @staticmethod
    def test_reasons_for_current(
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """Skipped targets have 'current' reason."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["modules"])

        reason = result.get_reason("modules")
        expect_equal(reason.kind, "current")
        expect_in("up-to-date", reason.details.lower())

    @staticmethod
    def test_reasons_for_forced(
        resolver_graph: TargetGraph,
        all_computed_state: DatabaseState,
    ) -> None:
        """Forced targets have 'forced' reason."""
        resolver = BuildResolver(resolver_graph, all_computed_state)
        result = resolver.resolve(["modules"], force_recompute=["modules"])

        reason = result.get_reason("modules")
        expect_equal(reason.kind, "forced")

    @staticmethod
    def test_reasons_include_goal_annotation(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Requested goals are annotated in reason details."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["function_metrics"])

        reason = result.get_reason("function_metrics")
        expect_in("goal", reason.details.lower())


class TestTopologicalOrder:
    """Tests for correct topological ordering."""

    @staticmethod
    def test_to_compute_in_topological_order(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """to_compute respects dependency order."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["function_metrics"])

        compute_list = list(result.to_compute)

        expect_true(compute_list.index("modules") < compute_list.index("ast"))

        expect_true(compute_list.index("ast") < compute_list.index("goids"))

        expect_true(compute_list.index("goids") < compute_list.index("function_metrics"))

    @staticmethod
    def test_parallel_branches_both_included(
        resolver_graph: TargetGraph,
        all_missing_state: DatabaseState,
    ) -> None:
        """Parallel branches are both computed when needed."""
        resolver = BuildResolver(resolver_graph, all_missing_state)
        result = resolver.resolve(["typing", "goids"])

        expect_in("ast", result.to_compute)
        expect_in("typing", result.to_compute)
        expect_in("goids", result.to_compute)

        compute_list = list(result.to_compute)
        expect_true(compute_list.index("ast") < compute_list.index("typing"))
        expect_true(compute_list.index("ast") < compute_list.index("goids"))


class TestWithRealRegistry:
    """Integration tests using the full target registry."""

    @staticmethod
    def test_resolve_with_real_registry() -> None:
        """Resolve using the full target registry."""
        graph = get_target_graph()

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

        result = resolver.resolve(["function_metrics"])

        expect_in("modules", result.to_compute)
        expect_in("function_metrics", result.to_compute)
        expect_true(len(result.to_compute) > 1)

    @staticmethod
    def test_real_registry_all_targets_resolvable() -> None:
        """All targets in real registry can be resolved."""
        graph = get_target_graph()

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

        result = resolver.resolve_all()

        expect_equal(result.total_work, len(list(graph)))
