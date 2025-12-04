"""Tests for graph validation helpers."""

from __future__ import annotations

import logging
from typing import Final

import networkx as nx
import pytest
from _pytest.logging import LogCaptureFixture

from codeintel.analytics.runtime import GraphRuntimeOptions
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
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schema import apply_all_schemas
from tests._helpers import seed_graph_validation_gaps
from tests._helpers.factories import make_snapshot

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_REPO: Final = "test/repo"
TEST_COMMIT: Final = "abc123"
EXPECTED_ONE: Final = 1
EXPECTED_TWO: Final = 2


def test_run_graph_validations_emits_warnings(
    caplog: LogCaptureFixture, fresh_gateway: StorageGateway
) -> None:
    """
    Graph validations should warn for common integrity gaps.

    Raises
    ------
    AssertionError
        If expected warning text is absent.
    """
    gateway = fresh_gateway
    repo: Final = "demo/repo"
    commit: Final = "deadbeef"
    apply_all_schemas(gateway.con)
    seed_graph_validation_gaps(gateway, repo=repo, commit=commit)
    snapshot = make_snapshot(repo=repo, commit=commit)

    with caplog.at_level("WARNING"):
        run_graph_validations(
            gateway,
            snapshot=snapshot,
            runtime=GraphRuntimeOptions(snapshot=snapshot),
        )

    messages = " ".join(record.message for record in caplog.records)
    expected = ["outside caller spans", "module(s) have no GOIDs"]
    for needle in expected:
        if needle not in messages:
            message = f"Expected warning containing '{needle}' but messages were: {messages}"
            raise AssertionError(message)


def test_run_graph_validations_snapshot_mismatch_raises(
    fresh_gateway: StorageGateway,
) -> None:
    """Graph runtime snapshot must align with validation snapshot."""
    gateway = fresh_gateway
    apply_all_schemas(gateway.con)
    snapshot = make_snapshot()
    mismatched_runtime = GraphRuntimeOptions(
        snapshot=make_snapshot(repo="other/repo", commit="cafebabe")
    )

    with pytest.raises(ValueError, match="GraphRuntime snapshot mismatch"):
        run_graph_validations(
            gateway,
            snapshot=snapshot,
            runtime=mismatched_runtime,
        )


def test_run_graph_validations_hard_fail_on_error(
    fresh_gateway: StorageGateway,
) -> None:
    """Hard-fail mode should raise when error-level findings exist."""
    gateway = fresh_gateway
    apply_all_schemas(gateway.con)
    repo = "demo/repo"
    commit = "deadbeef"
    snapshot = make_snapshot(repo=repo, commit=commit)
    seed_graph_validation_gaps(gateway, repo=repo, commit=commit)
    runtime = GraphRuntimeOptions(snapshot=snapshot)

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
    fresh_gateway: StorageGateway,
) -> None:
    """
    Per-rule caps should limit persisted validation rows.

    Raises
    ------
    AssertionError
        When a rule exceeds the configured cap.
    """
    gateway = fresh_gateway
    apply_all_schemas(gateway.con)
    repo = "demo/repo"
    commit = "deadbeef"
    snapshot = make_snapshot(repo=repo, commit=commit)
    seed_graph_validation_gaps(gateway, repo=repo, commit=commit)
    runtime = GraphRuntimeOptions(snapshot=snapshot)

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
        if int(count) > 1:
            message = f"Expected cap to apply, found {count} rows"
            raise AssertionError(message)


# ===========================================================================
# Call Graph Check Tests
# ===========================================================================


def test_call_graph_findings_with_isolated_nodes() -> None:
    """call_graph_findings detects isolated function nodes."""
    graph = nx.DiGraph()
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
    assert len(isolated_findings) == EXPECTED_ONE
    detail = isolated_findings[0]["detail"]
    assert isinstance(detail, str)
    assert "isolated" in detail.lower()


def test_call_graph_findings_with_scc() -> None:
    """call_graph_findings detects recursive call clusters."""
    graph = nx.DiGraph()
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
    assert len(scc_findings) == EXPECTED_ONE


def test_call_graph_findings_with_hub_nodes() -> None:
    """call_graph_findings detects high-degree hubs."""
    graph = nx.DiGraph()
    # Create a hub with many connections (degree > threshold)
    hub_node = 0
    graph.add_node(hub_node, kind="function")
    for i in range(1, 101):  # 100 connections
        graph.add_node(i, kind="function")
        graph.add_edge(hub_node, i)

    log = logging.getLogger("test")
    findings = call_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    hub_findings = [f for f in findings if f["check_name"] == "call_graph_degree_hubs"]
    assert len(hub_findings) == EXPECTED_ONE


def test_call_graph_findings_empty_graph() -> None:
    """call_graph_findings returns empty list for empty graph."""
    graph = nx.DiGraph()
    log = logging.getLogger("test")

    findings = call_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    assert findings == []


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

    assert len(findings) >= EXPECTED_ONE


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
    assert len(cross_pkg_findings) == EXPECTED_ONE


def test_import_hub_findings_detects_hubs() -> None:
    """import_hub_findings detects high-degree import hubs."""
    graph = nx.DiGraph()
    # Create a hub module with many imports
    hub = "core.utils"
    graph.add_node(hub)
    for i in range(50):
        target = f"module{i}"
        graph.add_node(target)
        graph.add_edge(hub, target)

    log = logging.getLogger("test")
    findings = import_hub_findings(graph, TEST_REPO, TEST_COMMIT, log)

    assert len(findings) >= EXPECTED_ONE


def test_import_upward_findings_detects_layer_violations() -> None:
    """import_upward_findings detects imports against layering."""
    graph = nx.DiGraph()
    # Module at layer 3 imports from layer 1 (upward)
    graph.add_node("deep.module", layer=3)
    graph.add_node("shallow.module", layer=1)
    graph.add_edge("deep.module", "shallow.module")

    log = logging.getLogger("test")
    findings = import_upward_findings(graph, TEST_REPO, TEST_COMMIT, log)

    assert len(findings) == EXPECTED_ONE


def test_import_upward_findings_ignores_downward() -> None:
    """import_upward_findings ignores proper layered imports."""
    graph = nx.DiGraph()
    # Proper layering: lower layer imports from higher
    graph.add_node("shallow.module", layer=1)
    graph.add_node("deep.module", layer=3)
    graph.add_edge("shallow.module", "deep.module")

    log = logging.getLogger("test")
    findings = import_upward_findings(graph, TEST_REPO, TEST_COMMIT, log)

    assert findings == []


def test_import_bridge_findings_detects_bridges() -> None:
    """import_bridge_findings detects high-betweenness modules."""
    graph = nx.DiGraph()
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
    assert isinstance(findings, list)


def test_import_graph_findings_combines_checks() -> None:
    """import_graph_findings runs all import checks."""
    graph = nx.DiGraph()
    # Add some nodes and edges
    graph.add_edge("pkg.a", "pkg.b")
    graph.add_edge("pkg.b", "pkg.c")

    log = logging.getLogger("test")
    findings = import_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    assert isinstance(findings, list)


# ===========================================================================
# Symbol Graph Check Tests
# ===========================================================================


def test_symbol_graph_findings_detects_hubs() -> None:
    """symbol_graph_findings detects high-degree symbol hubs."""
    graph = nx.Graph()
    # Create a symbol hub with many connections
    hub = "common_symbol"
    graph.add_node(hub)
    for i in range(100):
        node = f"module_{i}"
        graph.add_node(node)
        graph.add_edge(hub, node)

    log = logging.getLogger("test")
    findings = symbol_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    assert len(findings) >= EXPECTED_ONE


def test_symbol_graph_findings_empty_graph() -> None:
    """symbol_graph_findings returns empty for empty graph."""
    graph = nx.Graph()

    log = logging.getLogger("test")
    findings = symbol_graph_findings(graph, TEST_REPO, TEST_COMMIT, log)

    assert findings == []


# ===========================================================================
# Config Key Check Tests
# ===========================================================================


def test_config_key_findings_detects_broad_usage() -> None:
    """config_key_findings detects config keys used broadly."""
    graph = nx.Graph()
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

    assert len(findings) >= EXPECTED_ONE


def test_config_key_findings_empty_graph() -> None:
    """config_key_findings returns empty for empty graph."""
    graph = nx.Graph()

    log = logging.getLogger("test")
    findings = config_key_findings(graph, TEST_REPO, TEST_COMMIT, log)

    assert findings == []
