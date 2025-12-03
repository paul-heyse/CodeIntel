"""Integration test ensuring pipeline graph steps use the shared function catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from coverage import Coverage

from codeintel.analytics.tests import compute_test_coverage_edges
from codeintel.config import TestCoverageStepConfig
from codeintel.graphs.catalog import load_function_catalog
from tests._helpers.pipeline_env import (
    build_graph_and_symbols,
    create_pipeline_env,
    generate_pipeline_coverage,
    load_coverage,
)

REPO: Final = "demo/repo"
COMMIT: Final = "deadbeef"


def test_pipeline_steps_use_function_catalog(tmp_path: Path) -> None:
    """Ensure pipeline steps and builders all consume the shared function catalog."""
    env = create_pipeline_env(tmp_path)
    build_graph_and_symbols(env)

    coverage_file = generate_pipeline_coverage(env)

    def _load_coverage(_cfg: TestCoverageStepConfig) -> Coverage:
        return load_coverage(coverage_file, _cfg)

    compute_test_coverage_edges(
        env.gateway,
        env.ctx.config_builder().test_coverage(
            coverage_file=coverage_file,
            coverage_loader=_load_coverage,
        ),
        coverage_loader=_load_coverage,
    )

    def _assert(condition: object, *, detail: str) -> None:
        if condition:
            return
        raise AssertionError(detail)

    catalog = load_function_catalog(env.gateway, repo=REPO, commit=COMMIT)
    callee_goid = catalog.lookup_goid("pkg/a.py", 1, 2, "pkg.a.callee")
    caller_goid = catalog.lookup_goid(
        "pkg/b.py", env.caller_lines[0], env.caller_lines[1], "pkg.b.caller"
    )
    _assert(callee_goid is not None, detail="Catalog missing callee GOID")
    _assert(caller_goid is not None, detail="Catalog missing caller GOID")
    _assert(
        catalog.module_by_path.get("pkg/a.py") == "pkg.a",
        detail="Catalog module mapping missing pkg.a",
    )

    _assert(
        caller_goid
        in {
            row[0]
            for row in env.gateway.con.execute(
                "SELECT caller_goid_h128 FROM graph.call_graph_edges"
            ).fetchall()
        },
        detail=f"Call graph edges missing caller GOID {caller_goid}",
    )

    _assert(
        caller_goid
        in {
            row[0]
            for row in env.gateway.con.execute(
                "SELECT function_goid_h128 FROM graph.cfg_blocks WHERE file_path = 'pkg/b.py'"
            ).fetchall()
        },
        detail=f"CFG blocks missing GOID {caller_goid}",
    )

    _assert(
        {
            row[0]
            for row in env.gateway.con.execute(
                "SELECT use_path FROM graph.symbol_use_edges WHERE def_path = 'pkg/a.py'"
            ).fetchall()
        }
        == {"pkg/b.py"},
        detail="Symbol uses not populated as expected: %s"
        % {
            row[0]
            for row in env.gateway.con.execute(
                "SELECT use_path FROM graph.symbol_use_edges WHERE def_path = 'pkg/a.py'"
            ).fetchall()
        },
    )

    coverage_goids = {
        row[0]
        for row in env.gateway.con.execute(
            "SELECT function_goid_h128 FROM analytics.test_coverage_edges"
        ).fetchall()
    }
    _assert(
        coverage_goids and caller_goid in coverage_goids,
        detail=f"Coverage GOIDs missing caller: {coverage_goids}",
    )
