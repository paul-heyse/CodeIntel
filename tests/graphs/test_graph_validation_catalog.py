"""Ensure graph validation uses catalog module map when core.modules is empty.

Uses MockFunctionCatalog from tests._helpers.fakes for catalog mocking.
"""

from __future__ import annotations

from codeintel.analytics.runtime import GraphRuntimeOptions
from codeintel.graphs.validation import run_graph_validations
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schema import apply_all_schemas
from tests._helpers import seed_graph_validation_gaps
from tests._helpers.factories import make_snapshot
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


def test_graph_validation_orphan_uses_catalog_map(fresh_gateway: StorageGateway) -> None:
    """Graph validation should fall back to catalog module map when modules are absent."""
    gateway = fresh_gateway
    con = gateway.con
    apply_all_schemas(con)
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
    _expect(condition=rows == [("pkg/a.py",)], detail=f"unexpected paths {rows}")
