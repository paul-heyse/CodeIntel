"""Additional tests for build resolver edge cases."""

from __future__ import annotations

import logging

import pytest

from codeintel.build.resolver import BuildResolver
from codeintel.build.state import DatabaseState, StalenessReason, TargetState, TargetStatus
from codeintel.build.targets import OutputTarget, TargetGraph, TargetOptions
from tests._helpers.assertions import expect_equal, expect_in, expect_true
from tests._helpers.build import sample_manifest


def _make_graph() -> TargetGraph:
    """Create a small graph with a linear chain and extra node.

    Returns
    -------
    TargetGraph
        Graph containing root -> mid -> leaf chain and an extra target.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget.from_tables(
            name="root",
            module="ingestion",
            plugin="root_plugin",
            tables=("core.root",),
            options=TargetOptions(description="root"),
        )
    )
    graph.register(
        OutputTarget.from_tables(
            name="mid",
            module="graphs",
            plugin="mid_plugin",
            tables=("core.mid",),
            options=TargetOptions(dependencies=("root",), description="mid"),
        )
    )
    graph.register(
        OutputTarget.from_tables(
            name="leaf",
            module="analytics",
            plugin="leaf_plugin",
            tables=("core.leaf",),
            options=TargetOptions(dependencies=("mid",), description="leaf"),
        )
    )
    graph.register(
        OutputTarget.from_tables(
            name="extra",
            module="analytics",
            plugin="extra_plugin",
            tables=("core.extra",),
            options=TargetOptions(description="extra"),
        )
    )
    return graph


def _state_for(
    root_status: TargetStatus,
    mid_status: TargetStatus,
    leaf_status: TargetStatus,
    *,
    mid_blocking: tuple[str, ...] = (),
) -> DatabaseState:
    """Create a DatabaseState for the graph with varying statuses.

    Returns
    -------
    DatabaseState
        State with configured statuses for all targets.
    """
    targets: dict[str, TargetState] = {
        "root": TargetState(
            name="root",
            status=root_status,
            manifest=sample_manifest("root") if root_status == "computed" else None,
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash="root-hash" if root_status == "computed" else None,
        ),
        "mid": TargetState(
            name="mid",
            status=mid_status,
            manifest=sample_manifest("mid") if mid_status == "computed" else None,
            staleness_reason=StalenessReason(
                kind="input_hash_mismatch",
                details="inputs changed",
            )
            if mid_status == "stale"
            else None,
            blocking_deps=mid_blocking if mid_status == "blocked" else (),
            current_input_hash="mid-hash" if mid_status == "computed" else None,
        ),
        "leaf": TargetState(
            name="leaf",
            status=leaf_status,
            manifest=sample_manifest("leaf") if leaf_status == "computed" else None,
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash="leaf-hash" if leaf_status == "computed" else None,
        ),
        "extra": TargetState(
            name="extra",
            status="computed",
            manifest=sample_manifest("extra"),
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash="extra-hash",
        ),
    }
    return DatabaseState(repo="r", commit="c", targets=targets)


@pytest.mark.parametrize(
    ("root_status", "mid_status", "leaf_status", "expected_kinds", "expected_work"),
    [
        ("missing", "computed", "computed", ("missing", "cascade", "cascade"), 3),
        ("computed", "stale", "computed", ("current", "stale", "cascade"), 2),
    ],
)
def test_resolve_cascade_and_stale(
    root_status: TargetStatus,
    mid_status: TargetStatus,
    leaf_status: TargetStatus,
    expected_kinds: tuple[str, str, str],
    expected_work: int,
) -> None:
    """Resolve computes cascade and stale reasons in topological order."""
    graph = _make_graph()
    state = _state_for(root_status, mid_status, leaf_status)
    resolver = BuildResolver(graph, state)

    result = resolver.resolve(["leaf"])

    kinds = tuple(result.reasons[name].kind for name in ("root", "mid", "leaf"))
    expect_equal(kinds, expected_kinds)
    leaf_detail = result.reasons["leaf"].details
    expect_true(
        "requested goal" in leaf_detail or "Dependency cascade" in leaf_detail,
        message=f"unexpected detail: {leaf_detail}",
    )
    expect_equal(result.total_work, expected_work)


def test_resolve_handles_blocked_dependencies() -> None:
    """Blocked targets are reported with dependency or blocked_external kinds."""
    graph = _make_graph()
    state = _state_for(
        "computed",
        "blocked",
        "computed",
        mid_blocking=("root", "external"),
    )
    resolver = BuildResolver(graph, state)

    result = resolver.resolve(["leaf"])

    mid_reason = result.reasons["mid"]
    expect_equal(mid_reason.kind, "blocked_external")
    expect_in("external", mid_reason.details)


def test_dependency_reason_when_blocking_will_compute() -> None:
    """Blocked targets become dependency reasons when blockers are recomputed."""
    graph = _make_graph()
    state = _state_for("missing", "blocked", "computed", mid_blocking=("root",))
    resolver = BuildResolver(graph, state)

    result = resolver.resolve(["leaf"])

    expect_equal(result.reasons["mid"].kind, "dependency")
    expect_equal(result.reasons["root"].kind, "missing")
    expect_in("Dependencies", result.reasons["mid"].details)


def test_force_recompute_filters_irrelevant_targets(caplog: pytest.LogCaptureFixture) -> None:
    """Force recompute warnings are emitted for unknown or irrelevant targets."""
    graph = _make_graph()
    state = _state_for("computed", "computed", "computed")
    resolver = BuildResolver(graph, state)
    caplog.set_level(logging.WARNING)

    result = resolver.resolve(("leaf",), force_recompute=("unknown", "extra"))

    expect_equal(result.to_compute, ())
    expect_true(any("Force target 'unknown'" in rec.message for rec in caplog.records))
    expect_true(any("not in transitive deps" in rec.message for rec in caplog.records))


def test_resolve_all_filters_by_module() -> None:
    """resolve_all with module restricts goals to that module."""
    graph = _make_graph()

    computed_targets = {
        t.name: TargetState(
            name=t.name,
            status="computed",
            manifest=None,
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash=f"{t.name}-hash",
        )
        for t in graph.all_targets
    }
    state = DatabaseState(repo="r", commit="c", targets=computed_targets)

    resolver = BuildResolver(graph, state)
    result = resolver.resolve_all(module="ingestion")

    expect_equal(result.requested, ("root",))
    expect_equal(result.to_skip, ("root",))
    expect_equal(result.total_work, 0)


def test_get_reason_missing_key() -> None:
    """get_reason raises KeyError when target not in reasons."""
    graph = _make_graph()
    state = _state_for("computed", "computed", "computed")
    resolver = BuildResolver(graph, state)
    result = resolver.resolve(["leaf"])

    with pytest.raises(KeyError):
        result.get_reason("nonexistent")
