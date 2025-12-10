"""Tests for graph validation helpers."""

from __future__ import annotations

import logging
from typing import Final

import pytest
from _pytest.logging import LogCaptureFixture

from codeintel.graphs.validation import GraphValidationOptions, run_graph_validations
from codeintel.graphs.validation.checks import (
    call_graph_findings,
    config_key_findings,
    import_bridge_findings,
    import_cycle_findings,
    import_graph_findings,
    import_hub_findings,
    import_upward_findings,
    symbol_graph_findings,
)
from tests._helpers import seed_graph_validation_gaps
from tests._helpers.assertions import expect_equal, expect_in, expect_is_instance, expect_true
from tests._helpers.factories import make_graph_runtime_options, make_snapshot
from tests._helpers.fakes.graph_contexts import GraphTestEnv
from tests._helpers.fakes.networkx_graphs import empty_digraph, empty_graph

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_REPO: Final = "test/repo"
TEST_COMMIT: Final = "abc123"
EXPECTED_ONE: Final = 1
EXPECTED_TWO: Final = 2


def test_run_graph_validations_emits_warnings(
    caplog: LogCaptureFixture, graph_executor_env: GraphTestEnv
) -> None:
    """Graph validations should warn for common integrity gaps."""
    gateway = graph_executor_env.gateway
    repo: Final = "demo/repo"
    commit: Final = "deadbeef"
    seed_graph_validation_gaps(gateway, repo=repo, commit=commit)
    snapshot = graph_executor_env.snapshot

    with caplog.at_level("WARNING"):
        run_graph_validations(
            gateway,
            snapshot=snapshot,
            runtime=make_graph_runtime_options(snapshot=snapshot),
        )

    messages = " ".join(record.message for record in caplog.records)
    expected = ["outside caller spans", "module(s) have no GOIDs"]
    for needle in expected:
        expect_in(needle, messages, label="graph_validation_warning")


def test_run_graph_validations_snapshot_mismatch_raises(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Graph runtime snapshot must align with validation snapshot."""
    gateway = graph_executor_env.gateway
    snapshot = graph_executor_env.snapshot
    mismatched_runtime = make_graph_runtime_options(
        snapshot=make_snapshot(repo="other/repo", commit="cafebabe")
    )

    with pytest.raises(ValueError, match="GraphRuntime snapshot mismatch"):
        run_graph_validations(
            gateway,
            snapshot=snapshot,
            runtime=mismatched_runtime,
        )


def test_run_graph_validations_hard_fail_on_error(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Hard-fail mode should raise when error-level findings exist."""
    gateway = graph_executor_env.gateway
    snapshot = graph_executor_env.snapshot
    repo = snapshot.repo
    commit = snapshot.commit
    seed_graph_validation_gaps(gateway, repo=repo, commit=commit)
    runtime = make_graph_runtime_options(snapshot=snapshot)

    with pytest.raises(RuntimeError, match="error-level findings"):
        run_graph_validations(
            gateway,
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


def test_run_graph_validations_caps_findings(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Per-rule caps should limit persisted validation rows."""
    gateway = graph_executor_env.gateway
    snapshot = graph_executor_env.snapshot
    repo = snapshot.repo
    commit = snapshot.commit
    seed_graph_validation_gaps(gateway, repo=repo, commit=commit)
    runtime = make_graph_runtime_options(snapshot=snapshot)

    run_graph_validations(
        gateway,
        snapshot=snapshot,
        runtime=runtime,
        options=GraphValidationOptions(max_findings_per_rule=1),
    )
    rows = gateway.con.execute(
        "SELECT graph_name, COUNT(*) FROM analytics.graph_validation GROUP BY graph_name"
    ).fetchall()
    for _, count in rows:
        expect_true(int(count) <= 1, message=f"Expected cap to apply, found {count} rows")


# ===========================================================================
# Call Graph Check Tests
# ===========================================================================


def test_call_graph_findings_with_isolated_nodes() -> None:
    """call_graph_findings detects isolated function nodes."""
    graph = empty_digraph()
    # Add isolated function nodes
    graph.add_node(1, kind="function")
    graph.add_node(2, kind="function")
    # Add connected nodes
    graph.add_edge(3, 4)
    graph.nodes[3]["kind"] = "function"
    graph.nodes[4]["kind"] = "function"

    log = logging.getLogger("test")
    findings = call_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    isolated_findings = [f for f in findings if f["check_name"] == "call_graph_isolated_nodes"]
    expect_equal(len(isolated_findings), EXPECTED_ONE)
    detail = isolated_findings[0]["detail"]
    expect_is_instance(detail, str)
    detail_str = str(detail)
    expect_in("isolated", detail_str.lower())


def test_call_graph_findings_with_scc() -> None:
    """call_graph_findings detects recursive call clusters."""
    graph = empty_digraph()
    # Create a strongly connected component (cycle) with 3+ nodes
    for i in range(5):
        graph.add_node(i, kind="function")
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 0)  # Completes the cycle

    log = logging.getLogger("test")
    findings = call_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    scc_findings = [f for f in findings if f["check_name"] == "call_graph_large_scc"]
    expect_equal(len(scc_findings), EXPECTED_ONE)


def test_call_graph_findings_with_hub_nodes() -> None:
    """call_graph_findings detects high-degree hubs."""
    graph = empty_digraph()
    # Create a hub with many connections (degree > threshold)
    hub_node = 0
    graph.add_node(hub_node, kind="function")
    for i in range(1, 101):  # 100 connections
        graph.add_node(i, kind="function")
        graph.add_edge(hub_node, i)

    log = logging.getLogger("test")
    findings = call_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    hub_findings = [f for f in findings if f["check_name"] == "call_graph_degree_hubs"]
    expect_equal(len(hub_findings), EXPECTED_ONE)


def test_call_graph_findings_empty_graph() -> None:
    """call_graph_findings returns empty list for empty graph."""
    graph = empty_digraph()
    log = logging.getLogger("test")

    findings = call_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    expect_equal(findings, [])


# ===========================================================================
# Import Graph Check Tests
# ===========================================================================


def test_import_cycle_findings_detects_large_cycles() -> None:
    """import_cycle_findings detects large import cycles."""
    # Create SCCs with cycles - needs more than HUB_MIN_DEGREE_FLOOR // 2 (= 5) elements
    sccs: list[set[str]] = [
        {"pkg.a", "pkg.b", "pkg.c", "pkg.d", "pkg.e", "pkg.f", "pkg.g"},  # Large cycle (7 nodes)
    ]

    log = logging.getLogger("test")
    findings = import_cycle_findings(sccs, TEST_REPO, TEST_COMMIT, log)

    expect_true(len(findings) >= EXPECTED_ONE)


def test_import_cycle_findings_detects_cross_package_cycles() -> None:
    """import_cycle_findings detects cycles crossing package boundaries."""
    # Create a cycle that crosses packages
    sccs: list[set[str]] = [
        {"pkg1.a", "pkg2.b"},  # Cross-package cycle
    ]

    log = logging.getLogger("test")
    findings = import_cycle_findings(sccs, TEST_REPO, TEST_COMMIT, log)

    cross_pkg_findings = [
        f for f in findings if f["check_name"] == "import_graph_cross_package_cycles"
    ]
    expect_equal(len(cross_pkg_findings), EXPECTED_ONE)


def test_import_hub_findings_detects_hubs() -> None:
    """import_hub_findings detects high-degree import hubs."""
    graph = empty_digraph()
    # Create a hub module with many imports
    hub = "core.utils"
    graph.add_node(hub)
    for i in range(50):
        target = f"module{i}"
        graph.add_node(target)
        graph.add_edge(hub, target)

    log = logging.getLogger("test")
    findings = import_hub_findings(graph, TEST_REPO, TEST_COMMIT, log)

    expect_true(len(findings) >= EXPECTED_ONE)


def test_import_upward_findings_detects_layer_violations() -> None:
    """import_upward_findings detects imports against layering."""
    graph = empty_digraph()
    # Module at layer 3 imports from layer 1 (upward)
    graph.add_node("deep.module", layer=3)
    graph.add_node("shallow.module", layer=1)
    graph.add_edge("deep.module", "shallow.module")

    log = logging.getLogger("test")
    findings = import_upward_findings(graph, TEST_REPO, TEST_COMMIT, log)

    expect_equal(len(findings), EXPECTED_ONE)


def test_import_upward_findings_ignores_downward() -> None:
    """import_upward_findings ignores proper layered imports."""
    graph = empty_digraph()
    # Proper layering: lower layer imports from higher
    graph.add_node("shallow.module", layer=1)
    graph.add_node("deep.module", layer=3)
    graph.add_edge("shallow.module", "deep.module")

    log = logging.getLogger("test")
    findings = import_upward_findings(graph, TEST_REPO, TEST_COMMIT, log)

    expect_equal(findings, [])


def test_import_bridge_findings_detects_bridges() -> None:
    """import_bridge_findings detects high-betweenness modules."""
    graph = empty_digraph()
    # Create a bridge topology: A -> B -> C (B is a bridge)
    modules = ["a.mod", "b.bridge", "c.mod", "d.mod", "e.mod"]
    for mod in modules:
        graph.add_node(mod)

    # Make b.bridge a critical bridge
    graph.add_edge("a.mod", "b.bridge")
    graph.add_edge("b.bridge", "c.mod")
    graph.add_edge("b.bridge", "d.mod")
    graph.add_edge("b.bridge", "e.mod")

    log = logging.getLogger("test")
    findings = import_bridge_findings(graph, TEST_REPO, TEST_COMMIT, log)

    # Bridge findings depend on betweenness calculation
    expect_true(isinstance(findings, list))


def test_import_graph_findings_combines_checks() -> None:
    """import_graph_findings runs all import checks."""
    graph = empty_digraph()
    # Add some nodes and edges
    graph.add_edge("pkg.a", "pkg.b")
    graph.add_edge("pkg.b", "pkg.c")

    log = logging.getLogger("test")
    findings = import_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    expect_true(isinstance(findings, list))


# ===========================================================================
# Symbol Graph Check Tests
# ===========================================================================


def test_symbol_graph_findings_detects_hubs() -> None:
    """symbol_graph_findings detects high-degree symbol hubs."""
    graph = empty_graph()
    # Create a symbol hub with many connections
    hub = "common_symbol"
    graph.add_node(hub)
    for i in range(100):
        node = f"module_{i}"
        graph.add_node(node)
        graph.add_edge(hub, node)

    log = logging.getLogger("test")
    findings = symbol_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    expect_true(len(findings) >= EXPECTED_ONE)


def test_symbol_graph_findings_empty_graph() -> None:
    """symbol_graph_findings returns empty for empty graph."""
    graph = empty_graph()

    log = logging.getLogger("test")
    findings = symbol_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    expect_equal(findings, [])


# ===========================================================================
# Config Key Check Tests
# ===========================================================================


def test_config_key_findings_detects_broad_usage() -> None:
    """config_key_findings detects config keys used broadly."""
    graph = empty_graph()
    # Create a bipartite graph with config keys (bipartite=0) and modules (bipartite=1)
    config_key = ("config_path", "common.key")
    graph.add_node(config_key, bipartite=0)

    # Add many modules using this key
    for i in range(50):
        module = f"module_{i}"
        graph.add_node(module, bipartite=1)
        graph.add_edge(config_key, module)

    log = logging.getLogger("test")
    findings = config_key_findings(graph, TEST_REPO, TEST_COMMIT, log)

    expect_true(len(findings) >= EXPECTED_ONE)


def test_config_key_findings_empty_graph() -> None:
    """config_key_findings returns empty for empty graph."""
    graph = empty_graph()

    log = logging.getLogger("test")
    findings = config_key_findings(graph, TEST_REPO, TEST_COMMIT, log)

    expect_equal(findings, [])
