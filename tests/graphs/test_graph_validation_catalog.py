"""Ensure graph validation uses catalog module map when core.modules is empty."""

from __future__ import annotations

from codeintel.graphs.function_catalog import FunctionCatalog
from codeintel.graphs.validation import run_graph_validations
from codeintel.storage.gateway import open_memory_gateway


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


class _CatalogProvider:
    def __init__(self, module_by_path: dict[str, str]) -> None:
        self._catalog = FunctionCatalog(functions=(), module_by_path=module_by_path)

    def catalog(self) -> FunctionCatalog:
        return self._catalog


def test_graph_validation_orphan_uses_catalog_map() -> None:
    """Graph validation should fall back to catalog module map when modules are absent."""
    gateway = open_memory_gateway(apply_schema=True)
    con = gateway.con
    provider = _CatalogProvider({"pkg/a.py": "pkg.a"})
    run_graph_validations(
        gateway,
        repo="r",
        commit="c",
        catalog_provider=provider,
    )
    rows = con.execute("SELECT path FROM analytics.graph_validation").fetchall()
    _expect(condition=rows == [("pkg/a.py",)], detail=f"unexpected paths {rows}")
