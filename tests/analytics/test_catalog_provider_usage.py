"""Ensure catalog provider injection is honored across analytics builders."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from codeintel.analytics.graph_metrics import compute_graph_metrics
from codeintel.analytics.profiles import build_function_profile, build_module_profile
from codeintel.config.models import GraphMetricsConfig, ProfilesAnalyticsConfig, SymbolUsesConfig
from codeintel.graphs.function_catalog import FunctionCatalog
from codeintel.graphs.symbol_uses import build_symbol_use_edges
from codeintel.storage.gateway import open_memory_gateway


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


class _FakeProvider:
    def __init__(self, module_by_path: dict[str, str]) -> None:
        self._catalog = FunctionCatalog(functions=(), module_by_path=module_by_path)

    def catalog(self) -> FunctionCatalog:
        return self._catalog


def test_symbol_uses_respects_catalog_module_map(tmp_path: Path) -> None:
    """Catalog module map toggles same_module when modules table is empty."""
    gateway = open_memory_gateway()
    con = gateway.con

    scip_path = tmp_path / "index.scip.json"
    scip_path.write_text(
        """
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
        """.strip(),
        encoding="utf8",
    )

    provider = _FakeProvider({"pkg/a.py": "pkg.mod", "pkg/b.py": "pkg.mod"})
    cfg = SymbolUsesConfig.from_paths(
        repo_root=tmp_path,
        repo="r",
        commit="c",
        scip_json_path=scip_path,
    )
    build_symbol_use_edges(con, cfg, catalog_provider=provider)

    row = con.execute("SELECT same_module FROM graph.symbol_use_edges").fetchone()
    _expect(condition=row is not None, detail="symbol_use_edges row missing")
    same_module = bool(row[0]) if row is not None else False
    _expect(
        condition=same_module is True,
        detail="same_module should be derived from catalog module map",
    )


def test_symbol_uses_falls_back_to_modules_when_catalog_partial(tmp_path: Path) -> None:
    """Catalog map may be partial; missing paths should fall back to core.modules."""
    gateway = open_memory_gateway()
    con = gateway.con
    con.execute(
        "INSERT INTO core.modules (module, path, repo, commit, language) VALUES (?, ?, ?, ?, ?)",
        ["pkg.use", "pkg/b.py", "r", "c", "python"],
    )
    scip_path = tmp_path / "index.scip.json"
    scip_path.write_text(
        """
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
        """.strip(),
        encoding="utf8",
    )

    # Provide only the defining module via catalog; use module comes from core.modules.
    provider = _FakeProvider({"pkg/a.py": "pkg.def"})
    cfg = SymbolUsesConfig.from_paths(
        repo_root=tmp_path,
        repo="r",
        commit="c",
        scip_json_path=scip_path,
    )
    build_symbol_use_edges(con, cfg, catalog_provider=provider)

    row = con.execute("SELECT same_module FROM graph.symbol_use_edges").fetchone()
    _expect(condition=row is not None, detail="symbol_use_edges row missing")
    same_module = bool(row[0]) if row is not None else False
    _expect(
        condition=same_module is False,
        detail="same_module should reflect mixed module sources",
    )


def test_graph_metrics_uses_catalog_for_symbol_modules() -> None:
    """Graph metrics module rows derive module names from injected catalog."""
    gateway = open_memory_gateway()
    con = gateway.con
    con.execute(
        """
        INSERT INTO graph.symbol_use_edges (symbol, def_path, use_path, same_file, same_module)
        VALUES ('sym#def', 'pkg/a.py', 'pkg/b.py', FALSE, FALSE)
        """
    )

    provider = _FakeProvider({"pkg/a.py": "pkg.mod", "pkg/b.py": "pkg.mod"})
    cfg = GraphMetricsConfig.from_paths(repo="r", commit="c")
    compute_graph_metrics(con, cfg, catalog_provider=provider)

    modules = {
        row[0]
        for row in con.execute(
            "SELECT module FROM analytics.graph_metrics_modules WHERE repo = 'r' AND commit = 'c'"
        ).fetchall()
    }
    _expect(
        condition=modules == {"pkg.mod"},
        detail=f"Expected module rows from catalog, got {modules}",
    )


def test_profiles_use_catalog_module_map_when_modules_table_empty() -> None:
    """Profiles builder should backfill modules from catalog when core.modules is empty."""
    gateway = open_memory_gateway()
    con = gateway.con
    now = datetime.now(tz=UTC)
    con.execute(
        """
        INSERT INTO analytics.goid_risk_factors (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            loc, logical_loc, cyclomatic_complexity, complexity_bucket, typedness_bucket,
            typedness_source, hotspot_score, file_typed_ratio, static_error_count,
            has_static_errors, executable_lines, covered_lines, coverage_ratio, tested,
            test_count, failing_test_count, last_test_status, risk_score, risk_level,
            tags, owners, created_at
        ) VALUES (
            1, 'urn:fn', 'r', 'c', 'pkg/a.py', 'python', 'function', 'pkg.a.fn',
            1, 1, 1, 'low', 'full', 'typed', 0.0, 1.0, 0, FALSE, 1, 1, 1.0, TRUE,
            1, 0, 'passed', 0.1, 'low', '[]', '[]', ?
        )
        """,
        [now],
    )

    provider = _FakeProvider({"pkg/a.py": "pkg.mod"})
    cfg = ProfilesAnalyticsConfig.from_paths(repo="r", commit="c")
    build_function_profile(con, cfg, catalog_provider=provider)
    build_module_profile(con, cfg, catalog_provider=provider)

    module_row = con.execute(
        """
        SELECT module FROM analytics.function_profile
        WHERE repo = 'r' AND commit = 'c' AND rel_path = 'pkg/a.py'
        """
    ).fetchone()
    _expect(condition=module_row is not None, detail="function_profile row missing")
    module_value = module_row[0] if module_row is not None else None
    _expect(
        condition=module_value == "pkg.mod",
        detail=f"function_profile module mismatch: {module_value}",
    )

    module_profile_modules = {
        row[0]
        for row in con.execute(
            "SELECT module FROM analytics.module_profile WHERE repo = 'r' AND commit = 'c'"
        ).fetchall()
    }
    _expect(
        condition=module_profile_modules == {"pkg.mod"},
        detail=f"module_profile modules mismatch: {module_profile_modules}",
    )
