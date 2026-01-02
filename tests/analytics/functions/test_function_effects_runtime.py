"""Integration-flavored tests for analytics.functions.function_effects."""

from __future__ import annotations

import ast
import json
import textwrap
from typing import TYPE_CHECKING

import polars as pl

from codeintel.build.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    FunctionEffectsOptions,
    build_function_effects_rows,
)
from codeintel.build.analytics.parsing.ast_cache import FunctionAst
from tests._helpers import TestScenario
from tests._helpers.assertions import assert_logged, expect_equal, expect_false, expect_true
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog
from tests._helpers.fixtures.rows import function_meta

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _build_function_ast_map(
    repo_root: Path, source: str, goids: dict[str, int]
) -> tuple[dict[int, FunctionAst], Path]:
    module_path = repo_root / "pkg" / "effects.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = textwrap.dedent(source).strip() + "\n"
    module_path.write_text(rendered, encoding="utf-8")
    tree = ast.parse(rendered)
    lines = rendered.splitlines()
    ast_by_goid: dict[int, FunctionAst] = {}
    for name, goid in goids.items():
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        start_line = getattr(node, "lineno", 0)
        end_line = getattr(node, "end_lineno", start_line)
        ast_by_goid[goid] = FunctionAst(
            goid=goid,
            rel_path=module_path.relative_to(repo_root).as_posix(),
            qualname=name,
            start_line=start_line,
            end_line=end_line,
            node=node,
            lines=lines,
        )
    return ast_by_goid, module_path


def test_build_function_effects_with_transitive_and_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """build_function_effects_rows records direct, transitive, and missing AST effects."""
    ctx = TestScenario.minimal().build(tmp_path)
    source = """
    import os
    import random
    import time
    import threading

    GLOBAL_STATE = 0

    def impure(target: str) -> int:
        os.getenv("HOME")
        random.random()
        time.time()
        threading.Thread(target=print).start()
        global GLOBAL_STATE
        GLOBAL_STATE += 1
        return 1

    def caller() -> int:
        return impure("x")

    def wrapper() -> int:
        return caller()

    def naive_unicode(name: str | None = None) -> str | None:
        return name

    def uses_nonlocal() -> int:
        value = 0

        def inner() -> int:
            nonlocal value
            value += 1
            return value

        return inner()
    """
    goids = {
        "impure": 1001,
        "caller": 1002,
        "uses_nonlocal": 1003,
        "missing": 1004,
        "wrapper": 1005,
        "naive_unicode": 1006,
    }
    ast_map, module_path = _build_function_ast_map(
        ctx.repo_root, source, {k: v for k, v in goids.items() if k != "missing"}
    )
    snapshot = ctx.to_snapshot_ref()
    options = FunctionEffectsOptions(
        max_call_depth=2,
        io_apis={"os": ["getenv"]},
        db_apis={},
        time_apis={"time": ["time"]},
        random_apis={"random": ["random"]},
        threading_apis={"threading": ["Thread"]},
    )

    edges_frame = pl.DataFrame(
        [
            {
                "repo": snapshot.repo,
                "commit": snapshot.commit,
                "caller_goid_h128": goids["caller"],
                "callee_goid_h128": goids["impure"],
                "callsite_path": ast_map[goids["caller"]].rel_path,
                "callsite_line": ast_map[goids["caller"]].start_line,
                "callsite_col": 0,
                "language": "python",
                "kind": "call",
                "resolved_via": "static",
                "confidence": 1.0,
                "evidence_json": None,
            },
            {
                "repo": snapshot.repo,
                "commit": snapshot.commit,
                "caller_goid_h128": goids["wrapper"],
                "callee_goid_h128": goids["caller"],
                "callsite_path": ast_map[goids["wrapper"]].rel_path,
                "callsite_line": ast_map[goids["wrapper"]].start_line,
                "callsite_col": 0,
                "language": "python",
                "kind": "call",
                "resolved_via": "static",
                "confidence": 1.0,
                "evidence_json": None,
            },
            {
                "repo": snapshot.repo,
                "commit": snapshot.commit,
                "caller_goid_h128": goids["caller"],
                "callee_goid_h128": None,
                "callsite_path": ast_map[goids["caller"]].rel_path,
                "callsite_line": ast_map[goids["caller"]].start_line,
                "callsite_col": 0,
                "language": "python",
                "kind": "call",
                "resolved_via": "static",
                "confidence": 1.0,
                "evidence_json": None,
            },
        ]
    )
    nodes_frame = pl.DataFrame(
        [
            {"goid_h128": goids["impure"], "kind": "function"},
            {"goid_h128": goids["caller"], "kind": "function"},
            {"goid_h128": goids["wrapper"], "kind": "function"},
        ]
    )

    inputs = FunctionEffectsInputs(
        catalog_provider=MockFunctionCatalog(
            functions=[
                function_meta(
                    goid=goids["impure"],
                    rel_path=module_path.relative_to(ctx.repo_root).as_posix(),
                    qualname="impure",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(
                        ast_map[goids["impure"]].start_line,
                        ast_map[goids["impure"]].end_line,
                    ),
                ),
                function_meta(
                    goid=goids["caller"],
                    rel_path=module_path.relative_to(ctx.repo_root).as_posix(),
                    qualname="caller",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(
                        ast_map[goids["caller"]].start_line,
                        ast_map[goids["caller"]].end_line,
                    ),
                ),
                function_meta(
                    goid=goids["uses_nonlocal"],
                    rel_path=module_path.relative_to(ctx.repo_root).as_posix(),
                    qualname="uses_nonlocal",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(
                        ast_map[goids["uses_nonlocal"]].start_line,
                        ast_map[goids["uses_nonlocal"]].end_line,
                    ),
                ),
                function_meta(
                    goid=goids["missing"],
                    rel_path=module_path.relative_to(ctx.repo_root).as_posix(),
                    qualname="missing",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(1, 1),
                ),
                function_meta(
                    goid=goids["wrapper"],
                    rel_path=module_path.relative_to(ctx.repo_root).as_posix(),
                    qualname="wrapper",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(
                        ast_map[goids["wrapper"]].start_line,
                        ast_map[goids["wrapper"]].end_line,
                    ),
                ),
                function_meta(
                    goid=goids["naive_unicode"],
                    rel_path=module_path.relative_to(ctx.repo_root).as_posix(),
                    qualname="naïve_unicode",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(
                        ast_map[goids["naive_unicode"]].start_line,
                        ast_map[goids["naive_unicode"]].end_line,
                    ),
                ),
            ],
            module_by_path={module_path.relative_to(ctx.repo_root).as_posix(): "pkg.effects"},
        ),
        ast_map=ast_map,
        missing_goids={goids["missing"]},
        call_graph_edges=edges_frame,
        call_graph_nodes=nodes_frame,
    )

    caplog.set_level("INFO")
    try:
        rows = build_function_effects_rows(snapshot, options=options, inputs=inputs)
        if rows:
            ctx.gateway.policy.delete_for_snapshot(
                "analytics.function_effects",
                repo=snapshot.repo,
                commit=snapshot.commit,
            )
            ctx.gateway.policy.bulk_insert_mappings("analytics.function_effects", rows)
        effects_by_goid = {
            int(row[0]): row
            for row in ctx.gateway.con.execute(
                """
                select
                  function_goid_h128,
                  is_pure,
                  uses_io,
                  touches_db,
                  uses_time,
                  uses_randomness,
                  modifies_globals,
                  modifies_closure,
                  spawns_threads_or_tasks,
                  has_transitive_effects,
                  purity_confidence,
                  effects_json
                from analytics.function_effects
                """
            ).fetchall()
        }
    finally:
        ctx.close()

    expect_false(effects_by_goid[goids["impure"]][1])
    expect_true(effects_by_goid[goids["impure"]][2])
    expect_true(effects_by_goid[goids["impure"]][4])
    expect_true(effects_by_goid[goids["impure"]][5])
    expect_true(effects_by_goid[goids["impure"]][6])
    expect_true(effects_by_goid[goids["impure"]][8])

    expect_false(effects_by_goid[goids["caller"]][1])
    expect_true(effects_by_goid[goids["caller"]][9])
    expect_true(effects_by_goid[goids["caller"]][10] < 1.0)
    expect_true(effects_by_goid[goids["wrapper"]][9])

    expect_true(effects_by_goid[goids["uses_nonlocal"]][7])

    expect_false(effects_by_goid[goids["missing"]][1])
    expect_equal(effects_by_goid[goids["missing"]][10], 0.0)
    expect_true(effects_by_goid[goids["naive_unicode"]][1])

    effects_json = effects_by_goid[goids["missing"]][11]
    parsed = effects_json if isinstance(effects_json, dict) else json.loads(effects_json)
    expect_equal(parsed["errors"][0]["details"]["kind"], "missing_ast")

    assert_logged(caplog.records, level="WARNING", containing="Missing AST for 1 functions")
    assert_logged(caplog.records, level="WARNING", containing=str(goids["missing"]))
    assert_logged(
        caplog.records,
        level="WARNING",
        containing="Unresolved call edges detected for 1 functions",
    )
    assert_logged(caplog.records, level="INFO", containing="function_effects populated")
