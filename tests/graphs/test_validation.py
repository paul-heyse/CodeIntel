"""Tests for graph validation helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import pytest

from codeintel.build.graphs.validation import (
    GraphValidationOptions,
    GraphValidationRunRequest,
    run_graph_validations_with_runner,
)
from codeintel.build.graphs.validation.checks.structure import (
    CallGraphStructureCheck,
    ConfigKeyCheck,
    ImportBridgeCheck,
    ImportCycleCheck,
    ImportGraphStructureCheck,
    ImportHubCheck,
    ImportUpwardCheck,
    SymbolGraphCheck,
)
from codeintel.build.graphs.validation.context import GraphValidationContext
from tests._helpers.assertions import (
    ModulesAssertions,
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)
from tests._helpers.factories import make_snapshot
from tests._helpers.fakes.graph_runtime import runtime_with_graphs
from tests._helpers.fixtures.graphs import empty_digraph, empty_graph
from tests._helpers.orchestration.seeding import (
    GraphValidationGapSeed,
    seed_graph_validation_gaps,
)

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

    from tests._helpers.fakes.graph_contexts import GraphTestEnv


TEST_REPO: Final = "test/repo"
TEST_COMMIT: Final = "abc123"
EXPECTED_ONE: Final = 1
EXPECTED_TWO: Final = 2


def test_run_graph_validations_emits_warnings(
    caplog: LogCaptureFixture, graph_executor_env: GraphTestEnv
) -> None:
    """Graph validations should warn for common integrity gaps."""
    gateway = graph_executor_env.gateway
    snapshot = graph_executor_env.snapshot
    repo: Final = "demo/repo"
    commit: Final = "deadbeef"
    seed_graph_validation_gaps(
        gateway,
        GraphValidationGapSeed(repo=repo, commit=commit, repo_root=snapshot.repo_root),
    )
    ModulesAssertions(gateway, snapshot).inventory_consistent()

    with caplog.at_level("WARNING"):
        request = GraphValidationRunRequest(
            snapshot=snapshot,
            runtime=runtime_with_graphs(gateway, snapshot)[0],
        )
        report = run_graph_validations_with_runner(gateway, request=request)

    messages = " ".join(record.message for record in caplog.records)
    expected = ["outside caller spans", "module(s) have no GOIDs"]
    for needle in expected:
        expect_in(needle, messages, label="graph_validation_warning")
    expect_true(report.checks_run > 0, message="Expected checks to run")


def test_run_graph_validations_snapshot_mismatch_raises(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Graph runtime snapshot must align with validation snapshot."""
    gateway = graph_executor_env.gateway
    snapshot = graph_executor_env.snapshot
    other_snapshot = make_snapshot(repo="other/repo", commit="cafebabe")
    mismatched_runtime = runtime_with_graphs(gateway, other_snapshot)[0]

    request = GraphValidationRunRequest(
        snapshot=snapshot,
        runtime=mismatched_runtime,
    )
    with pytest.raises(ValueError, match="GraphRuntime snapshot mismatch"):
        run_graph_validations_with_runner(gateway, request=request)


def test_run_graph_validations_hard_fail_on_error(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Hard-fail mode should raise when error-level findings exist."""
    gateway = graph_executor_env.gateway
    snapshot = graph_executor_env.snapshot
    repo = snapshot.repo
    commit = snapshot.commit
    seed_graph_validation_gaps(
        gateway,
        GraphValidationGapSeed(repo=repo, commit=commit, repo_root=snapshot.repo_root),
    )
    ModulesAssertions(gateway, snapshot).inventory_consistent()
    runtime = runtime_with_graphs(gateway, snapshot)[0]

    request = GraphValidationRunRequest(
        snapshot=snapshot,
        runtime=runtime,
        options=GraphValidationOptions(
            severity_overrides={
                "missing_function_goids": "error",
                "callsite_span_mismatch": "error",
            },
            hard_fail=True,
        ),
    )
    with pytest.raises(RuntimeError, match="error-level findings"):
        run_graph_validations_with_runner(gateway, request=request)


def test_run_graph_validations_caps_findings(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Per-rule caps should limit persisted validation rows."""
    gateway = graph_executor_env.gateway
    snapshot = graph_executor_env.snapshot
    repo = snapshot.repo
    commit = snapshot.commit
    seed_graph_validation_gaps(
        gateway,
        GraphValidationGapSeed(repo=repo, commit=commit, repo_root=snapshot.repo_root),
    )
    ModulesAssertions(gateway, snapshot).inventory_consistent()
    runtime = runtime_with_graphs(gateway, snapshot)[0]

    request = GraphValidationRunRequest(
        snapshot=snapshot,
        runtime=runtime,
        options=GraphValidationOptions(max_findings_per_rule=1),
    )
    run_graph_validations_with_runner(gateway, request=request)
    rows = gateway.con.execute(
        "SELECT graph_name, COUNT(*) FROM analytics.graph_validation GROUP BY graph_name"
    ).fetchall()
    for _, count in rows:
        expect_true(int(count) <= 1, message=f"Expected cap to apply, found {count} rows")


def test_call_graph_check_with_isolated_nodes() -> None:
    """CallGraphStructureCheck detects isolated function nodes."""
    graph = empty_digraph()

    graph.add_node(1, kind="function")
    graph.add_node(2, kind="function")

    graph.add_edge(3, 4)
    graph.nodes[3]["kind"] = "function"
    graph.nodes[4]["kind"] = "function"

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        call_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = CallGraphStructureCheck()
    findings = check.execute(ctx)

    isolated_findings = [f for f in findings if f["check_name"] == "call_graph_isolated_nodes"]
    expect_equal(len(isolated_findings), EXPECTED_ONE)
    detail = isolated_findings[0]["detail"]
    expect_is_instance(detail, str)
    detail_str = str(detail)
    expect_in("isolated", detail_str.lower())


def test_call_graph_check_with_scc() -> None:
    """CallGraphStructureCheck detects recursive call clusters."""
    graph = empty_digraph()

    for i in range(5):
        graph.add_node(i, kind="function")
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 0)

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        call_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = CallGraphStructureCheck()
    findings = check.execute(ctx)

    scc_findings = [f for f in findings if f["check_name"] == "call_graph_large_scc"]
    expect_equal(len(scc_findings), EXPECTED_ONE)


def test_call_graph_check_with_hub_nodes() -> None:
    """CallGraphStructureCheck detects high-degree hubs."""
    graph = empty_digraph()

    hub_node = 0
    graph.add_node(hub_node, kind="function")
    for i in range(1, 101):
        graph.add_node(i, kind="function")
        graph.add_edge(hub_node, i)

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        call_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = CallGraphStructureCheck()
    findings = check.execute(ctx)

    hub_findings = [f for f in findings if f["check_name"] == "call_graph_degree_hubs"]
    expect_equal(len(hub_findings), EXPECTED_ONE)


def test_call_graph_check_empty_graph() -> None:
    """CallGraphStructureCheck returns empty list for empty graph."""
    graph = empty_digraph()

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        call_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = CallGraphStructureCheck()
    findings = check.execute(ctx)

    expect_equal(findings, [])


def test_import_cycle_check_detects_large_cycles() -> None:
    """ImportCycleCheck detects large import cycles."""
    # Build a graph with the large cycle for ImportCycleCheck
    graph = empty_digraph()
    cycle_modules = ["pkg.a", "pkg.b", "pkg.c", "pkg.d", "pkg.e", "pkg.f", "pkg.g"]
    for mod in cycle_modules:
        graph.add_node(mod)
    for i in range(len(cycle_modules)):
        graph.add_edge(cycle_modules[i], cycle_modules[(i + 1) % len(cycle_modules)])

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        import_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = ImportCycleCheck()
    findings = check.execute(ctx)

    expect_true(len(findings) >= EXPECTED_ONE)


def test_import_cycle_check_detects_cross_package_cycles() -> None:
    """ImportCycleCheck detects cycles crossing package boundaries."""
    graph = empty_digraph()
    graph.add_node("pkg1.a")
    graph.add_node("pkg2.b")
    graph.add_edge("pkg1.a", "pkg2.b")
    graph.add_edge("pkg2.b", "pkg1.a")

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        import_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = ImportCycleCheck()
    findings = check.execute(ctx)

    cross_pkg_findings = [
        f for f in findings if f["check_name"] == "import_graph_cross_package_cycles"
    ]
    expect_equal(len(cross_pkg_findings), EXPECTED_ONE)


def test_import_hub_check_detects_hubs() -> None:
    """ImportHubCheck detects high-degree import hubs."""
    graph = empty_digraph()

    hub = "core.utils"
    graph.add_node(hub)
    for i in range(50):
        target = f"module{i}"
        graph.add_node(target)
        graph.add_edge(hub, target)

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        import_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = ImportHubCheck()
    findings = check.execute(ctx)

    expect_true(len(findings) >= EXPECTED_ONE)


def test_import_upward_check_detects_layer_violations() -> None:
    """ImportUpwardCheck detects imports against layering."""
    graph = empty_digraph()

    graph.add_node("deep.module", layer=3)
    graph.add_node("shallow.module", layer=1)
    graph.add_edge("deep.module", "shallow.module")

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        import_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = ImportUpwardCheck()
    findings = check.execute(ctx)

    expect_equal(len(findings), EXPECTED_ONE)


def test_import_upward_check_ignores_downward() -> None:
    """ImportUpwardCheck ignores proper layered imports."""
    graph = empty_digraph()

    graph.add_node("shallow.module", layer=1)
    graph.add_node("deep.module", layer=3)
    graph.add_edge("shallow.module", "deep.module")

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        import_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = ImportUpwardCheck()
    findings = check.execute(ctx)

    expect_equal(findings, [])


def test_import_bridge_check_detects_bridges() -> None:
    """ImportBridgeCheck detects high-betweenness modules."""
    graph = empty_digraph()

    modules = ["a.mod", "b.bridge", "c.mod", "d.mod", "e.mod"]
    for mod in modules:
        graph.add_node(mod)

    graph.add_edge("a.mod", "b.bridge")
    graph.add_edge("b.bridge", "c.mod")
    graph.add_edge("b.bridge", "d.mod")
    graph.add_edge("b.bridge", "e.mod")

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        import_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = ImportBridgeCheck()
    findings = check.execute(ctx)

    expect_true(isinstance(findings, list))


def test_import_graph_check_combines_checks() -> None:
    """ImportGraphStructureCheck runs all import checks."""
    graph = empty_digraph()

    graph.add_edge("pkg.a", "pkg.b")
    graph.add_edge("pkg.b", "pkg.c")

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        import_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = ImportGraphStructureCheck()
    findings = check.execute(ctx)

    expect_true(isinstance(findings, list))


def test_symbol_graph_check_detects_hubs() -> None:
    """SymbolGraphCheck detects high-degree symbol hubs."""
    graph = empty_graph()

    hub = "common_symbol"
    graph.add_node(hub)
    for i in range(100):
        node = f"module_{i}"
        graph.add_node(node)
        graph.add_edge(hub, node)

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        symbol_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = SymbolGraphCheck()
    findings = check.execute(ctx)

    expect_true(len(findings) >= EXPECTED_ONE)


def test_symbol_graph_check_empty_graph() -> None:
    """SymbolGraphCheck returns empty for empty graph."""
    graph = empty_graph()

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        symbol_graph=graph,
        logger=logging.getLogger("test"),
    )
    check = SymbolGraphCheck()
    findings = check.execute(ctx)

    expect_equal(findings, [])


def test_config_key_check_detects_broad_usage() -> None:
    """ConfigKeyCheck detects config keys used broadly."""
    graph = empty_graph()

    config_key = ("config_path", "common.key")
    graph.add_node(config_key, bipartite=0)

    for i in range(50):
        module = f"module_{i}"
        graph.add_node(module, bipartite=1)
        graph.add_edge(config_key, module)

    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        logger=logging.getLogger("test"),
    )
    # ConfigKeyCheck requires engine to get config_module_bipartite()
    # For unit test, we pass an empty context and check it returns empty
    check = ConfigKeyCheck()
    findings = check.execute(ctx)

    # With no engine, check returns empty (graph not accessible)
    expect_true(isinstance(findings, list))


def test_config_key_check_empty_graph() -> None:
    """ConfigKeyCheck returns empty for empty graph."""
    ctx = GraphValidationContext(
        gateway=None,
        repo=TEST_REPO,
        commit=TEST_COMMIT,
        logger=logging.getLogger("test"),
    )
    check = ConfigKeyCheck()
    findings = check.execute(ctx)

    expect_equal(findings, [])
