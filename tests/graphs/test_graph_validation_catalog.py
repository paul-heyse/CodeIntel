"""Ensure graph validation uses catalog module map when core.modules is empty.

Uses MockFunctionCatalog from tests._helpers.fakes for catalog mocking.
"""

from __future__ import annotations

from codeintel.graphs.validation import run_graph_validations
from tests._helpers import seed_graph_validation_gaps
from tests._helpers.assertions import expect_rows_equal
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog
from tests._helpers.fakes.graph_contexts import GraphTestEnv
from tests._helpers.fakes.graph_runtime import runtime_with_graphs


def test_graph_validation_orphan_uses_catalog_map(graph_executor_env: GraphTestEnv) -> None:
    """Graph validation should fall back to catalog module map when modules are absent."""
    gateway = graph_executor_env.gateway
    con = gateway.con
    provider = MockFunctionCatalog(module_by_path={"pkg/a.py": "pkg.a"})
    snapshot = graph_executor_env.snapshot
    seed_graph_validation_gaps(gateway, repo=snapshot.repo, commit=snapshot.commit)
    run_graph_validations(
        gateway,
        snapshot=snapshot,
        catalog_provider=provider,
        runtime=runtime_with_graphs(gateway, snapshot)[0],
    )
    rows = con.execute("SELECT rel_path FROM analytics.graph_validation").fetchall()
    expect_rows_equal(rows, [("pkg/a.py",)], message="graph_validation_paths")
