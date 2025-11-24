"""Symbol uses builder should handle partial module maps."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.config.models import SymbolUsesConfig
from codeintel.graphs.function_catalog import FunctionCatalog
from codeintel.graphs.function_catalog_service import FunctionCatalogService
from codeintel.graphs.symbol_uses import build_symbol_use_edges
from codeintel.storage.gateway import StorageConfig, open_gateway

SCIP_TEMPLATE = """
[
  {
    "relative_path": "pkg/a.py",
    "occurrences": [
      { "symbol": "sym#def", "symbol_roles": 1 }
    ]
  },
  {
    "relative_path": "pkg/b.py",
    "occurrences": [
      { "symbol": "sym#def", "symbol_roles": 2 }
    ]
  }
]
"""


def _write_scip(tmp_path: Path) -> Path:
    scip_path = tmp_path / "index.scip.json"
    scip_path.write_text(SCIP_TEMPLATE.strip(), encoding="utf8")
    return scip_path


def test_partial_catalog_map_falls_back_to_db(tmp_path: Path) -> None:
    """Def module from catalog, use module from DB should compute same_module correctly."""
    db_path = tmp_path / "db.duckdb"
    gateway = open_gateway(
        StorageConfig(db_path=db_path, apply_schema=True, ensure_views=True, validate_schema=True)
    )
    con = gateway.con
    gateway.core.insert_modules([("pkg.use", "pkg/b.py", "r", "c")])

    provider = FunctionCatalogService(
        FunctionCatalog(functions=(), module_by_path={"pkg/a.py": "pkg.def"})
    )
    scip_path = _write_scip(tmp_path)
    cfg = SymbolUsesConfig.from_paths(
        repo_root=tmp_path,
        repo="r",
        commit="c",
        scip_json_path=scip_path,
    )
    build_symbol_use_edges(con, cfg, catalog_provider=provider)

    row = con.execute("SELECT same_module FROM graph.symbol_use_edges").fetchone()
    if row is None:
        pytest.fail("symbol_use_edges row missing")
    same_module = bool(row[0])
    if same_module:
        pytest.fail("same_module should be False when modules differ")


def test_no_module_mapping_keeps_same_module_false(tmp_path: Path) -> None:
    """When neither catalog nor DB provide modules, same_module should be false."""
    db_path = tmp_path / "db.duckdb"
    gateway = open_gateway(
        StorageConfig(db_path=db_path, apply_schema=True, ensure_views=True, validate_schema=True)
    )
    con = gateway.con

    scip_path = _write_scip(tmp_path)
    cfg = SymbolUsesConfig.from_paths(
        repo_root=tmp_path,
        repo="r",
        commit="c",
        scip_json_path=scip_path,
    )
    build_symbol_use_edges(con, cfg)

    row = con.execute("SELECT same_module FROM graph.symbol_use_edges").fetchone()
    if row is None:
        pytest.fail("symbol_use_edges row missing")
    same_module = bool(row[0])
    if same_module:
        pytest.fail("same_module should remain False when modules are unknown")
