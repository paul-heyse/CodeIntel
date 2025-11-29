"""Ensure catalog provider injection is honored across analytics builders."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from codeintel.analytics.graphs.graph_metrics import GraphMetricsDeps, compute_graph_metrics
from codeintel.analytics.profiles import build_function_profile, build_module_profile
from codeintel.config import ConfigBuilder
from codeintel.graphs.function_catalog import FunctionCatalog
from codeintel.graphs.symbol_uses import build_symbol_use_edges
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import (
    ModuleRow,
    RiskFactorRow,
    SymbolUseEdgeRow,
    insert_modules,
    insert_risk_factors,
    insert_symbol_use_edges,
)


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


class _FakeProvider:
    def __init__(self, module_by_path: dict[str, str]) -> None:
        self._catalog = FunctionCatalog(functions=(), module_by_path=module_by_path)

    def catalog(self) -> FunctionCatalog:
        return self._catalog

    def urn_for_goid(self, goid: int) -> str | None:
        return self._catalog.urn_for_goid(goid)

    def lookup_goid(
        self, rel_path: str, start_line: int, end_line: int | None, qualname: str | None
    ) -> int | None:
        return self._catalog.lookup_goid(rel_path, start_line, end_line, qualname)


def test_symbol_uses_respects_catalog_module_map(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Catalog module map toggles same_module when modules table is empty."""
    gateway = fresh_gateway
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
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=tmp_path)
    cfg = builder.symbol_uses(scip_json_path=scip_path)
    build_symbol_use_edges(gateway, cfg, catalog_provider=provider)

    row = con.execute("SELECT same_module FROM graph.symbol_use_edges").fetchone()
    _expect(condition=row is not None, detail="symbol_use_edges row missing")
    same_module = bool(row[0]) if row is not None else False
    _expect(
        condition=same_module is True,
        detail="same_module should be derived from catalog module map",
    )


def test_symbol_uses_falls_back_to_modules_when_catalog_partial(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Catalog map may be partial; missing paths should fall back to core.modules."""
    gateway = fresh_gateway
    con = gateway.con
    insert_modules(
        gateway,
        [
            ModuleRow(module="pkg.use", path="pkg/b.py", repo="r", commit="c"),
        ],
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
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=tmp_path)
    cfg = builder.symbol_uses(scip_json_path=scip_path)
    build_symbol_use_edges(gateway, cfg, catalog_provider=provider)

    row = con.execute("SELECT same_module FROM graph.symbol_use_edges").fetchone()
    _expect(condition=row is not None, detail="symbol_use_edges row missing")
    same_module = bool(row[0]) if row is not None else False
    _expect(
        condition=same_module is False,
        detail="same_module should reflect mixed module sources",
    )


def test_graph_metrics_uses_catalog_for_symbol_modules(
    fresh_gateway: StorageGateway,
) -> None:
    """Graph metrics module rows derive module names from injected catalog."""
    gateway = fresh_gateway
    con = gateway.con
    insert_symbol_use_edges(
        gateway,
        [
            SymbolUseEdgeRow(
                symbol="sym#def",
                def_path="pkg/a.py",
                use_path="pkg/b.py",
                same_file=False,
                same_module=False,
            )
        ],
    )

    provider = _FakeProvider({"pkg/a.py": "pkg.mod", "pkg/b.py": "pkg.mod"})
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path().resolve())
    cfg = builder.graph_metrics()
    compute_graph_metrics(
        gateway,
        cfg,
        deps=GraphMetricsDeps(catalog_provider=provider),
    )

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


def test_profiles_use_catalog_module_map_when_modules_table_empty(
    fresh_gateway: StorageGateway,
) -> None:
    """Profiles builder should backfill modules from catalog when core.modules is empty."""
    gateway = fresh_gateway
    con = gateway.con
    now = datetime.now(tz=UTC)
    insert_risk_factors(
        gateway,
        [
            RiskFactorRow(
                function_goid_h128=1,
                urn="urn:fn",
                repo="r",
                commit="c",
                rel_path="pkg/a.py",
                language="python",
                kind="function",
                qualname="pkg.a.fn",
                loc=1,
                logical_loc=1,
                cyclomatic_complexity=1,
                complexity_bucket="low",
                typedness_bucket="full",
                typedness_source="typed",
                hotspot_score=0.0,
                file_typed_ratio=1.0,
                static_error_count=0,
                has_static_errors=False,
                executable_lines=1,
                covered_lines=1,
                coverage_ratio=1.0,
                tested=True,
                test_count=1,
                failing_test_count=0,
                last_test_status="passed",
                risk_score=0.1,
                risk_level="low",
                tags="[]",
                owners="[]",
                created_at=now,
            )
        ],
    )

    provider = _FakeProvider({"pkg/a.py": "pkg.mod"})
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path().resolve())
    cfg = builder.profiles_analytics()
    build_function_profile(gateway, cfg, catalog_provider=provider)
    build_module_profile(gateway, cfg, catalog_provider=provider)

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
