"""Ensure graph validation uses catalog module map when core.modules is empty.

Uses MockFunctionCatalog from tests._helpers.fakes for catalog mocking.
"""

from __future__ import annotations

from codeintel.analytics.runtime import GraphRuntimeOptions
from codeintel.graphs.validation import run_graph_validations
from codeintel.storage.gateway import StorageGateway
from tests._helpers import seed_graph_validation_gaps
from tests._helpers.assertions import expect_equal
from tests._helpers.factories import make_snapshot
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog


def test_graph_validation_orphan_uses_catalog_map(graph_gateway: StorageGateway) -> None:
    """Graph validation should fall back to catalog module map when modules are absent."""
    gateway = graph_gateway
    con = gateway.con
    provider = MockFunctionCatalog(module_by_path={"pkg/a.py": "pkg.a"})
    repo = "r"
    commit = "c"
    seed_graph_validation_gaps(gateway, repo=repo, commit=commit)
    snapshot = make_snapshot(repo=repo, commit=commit)
    run_graph_validations(
        gateway,
        snapshot=snapshot,
        catalog_provider=provider,
        runtime=GraphRuntimeOptions(snapshot=snapshot),
    )
    rows = con.execute("SELECT rel_path FROM analytics.graph_validation").fetchall()
    expect_equal(rows, [("pkg/a.py",)], label="graph_validation_paths")
