"""Integration-flavored tests for analytics.functions.function_effects."""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

import networkx as nx
import pytest

from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    FunctionEffectsStepConfig,
    compute_function_effects,
)
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.analytics.runtime.graph import GraphRuntime, GraphRuntimeOptions
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.sql.builder import ensure_schema
from tests._helpers.assertions import assert_logged, expect_equal, expect_false, expect_true
from tests._helpers.builders import CallGraphEdgeRow, insert_rows
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog
from tests._helpers.gateway import GatewayFactory
from tests._helpers.graphs import build_graph_engine_double
from tests._helpers.rows import function_meta


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


def test_compute_function_effects_with_transitive_and_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """compute_function_effects records direct, transitive, and missing AST effects."""
    repo_root = tmp_path / "repo"
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

    def uses_nonlocal() -> int:
        value = 0

        def inner() -> int:
            nonlocal value
            value += 1
            return value

        return inner()
    """
    goids = {"impure": 1001, "caller": 1002, "uses_nonlocal": 1003, "missing": 1004}
    ast_map, module_path = _build_function_ast_map(
        repo_root, source, {k: v for k, v in goids.items() if k != "missing"}
    )
    snapshot = SnapshotRef(repo="demo", commit="effects", repo_root=repo_root)
    gateway = GatewayFactory().with_snapshot(snapshot.repo, snapshot.commit).open()
    engine = build_graph_engine_double(
        gateway,
        snapshot,
        call_graph=nx.DiGraph([(goids["caller"], goids["impure"])]),
        copy_graphs=False,
    )
    runtime = GraphRuntime(GraphRuntimeOptions(snapshot=snapshot), engine)
    runtime.ensure_call_graph()
    cfg = FunctionEffectsStepConfig(
        snapshot=snapshot,
        max_call_depth=2,
        io_apis={"os": ["getenv"]},
        db_apis={},
        time_apis={"time": ["time"]},
        random_apis={"random": ["random"]},
        threading_apis={"threading": ["Thread"]},
    )

    ensure_schema(gateway.con, "graph.call_graph_edges")
    insert_rows(
        gateway,
        [
            CallGraphEdgeRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                caller_goid_h128=goids["caller"],
                callee_goid_h128=None,
                callsite_path=ast_map[goids["caller"]].rel_path,
                callsite_line=ast_map[goids["caller"]].start_line,
                callsite_col=0,
                language="python",
                kind="call",
                resolved_via="static",
                confidence=1.0,
            )
        ],
    )

    inputs = FunctionEffectsInputs(
        catalog_provider=MockFunctionCatalog(
            functions=[
                function_meta(
                    goid=goids["impure"],
                    rel_path=module_path.relative_to(repo_root).as_posix(),
                    qualname="impure",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(
                        ast_map[goids["impure"]].start_line,
                        ast_map[goids["impure"]].end_line,
                    ),
                ),
                function_meta(
                    goid=goids["caller"],
                    rel_path=module_path.relative_to(repo_root).as_posix(),
                    qualname="caller",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(
                        ast_map[goids["caller"]].start_line,
                        ast_map[goids["caller"]].end_line,
                    ),
                ),
                function_meta(
                    goid=goids["uses_nonlocal"],
                    rel_path=module_path.relative_to(repo_root).as_posix(),
                    qualname="uses_nonlocal",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(
                        ast_map[goids["uses_nonlocal"]].start_line,
                        ast_map[goids["uses_nonlocal"]].end_line,
                    ),
                ),
                function_meta(
                    goid=goids["missing"],
                    rel_path=module_path.relative_to(repo_root).as_posix(),
                    qualname="missing",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(1, 1),
                ),
            ],
            module_by_path={module_path.relative_to(repo_root).as_posix(): "pkg.effects"},
        ),
        runtime=runtime,
        ast_map=ast_map,
        missing_goids={goids["missing"]},
    )

    caplog.set_level("INFO")
    try:
        compute_function_effects(gateway, cfg, inputs=inputs)
        effects_by_goid = {
            int(row[0]): row
            for row in gateway.con.execute(
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
        gateway.close()

    expect_false(effects_by_goid[goids["impure"]][1])  # is_pure
    expect_true(effects_by_goid[goids["impure"]][2])  # uses_io
    expect_true(effects_by_goid[goids["impure"]][4])  # uses_time
    expect_true(effects_by_goid[goids["impure"]][5])  # uses_randomness
    expect_true(effects_by_goid[goids["impure"]][6])  # modifies_globals
    expect_true(effects_by_goid[goids["impure"]][8])  # spawns_threads_or_tasks

    expect_false(effects_by_goid[goids["caller"]][1])
    expect_true(effects_by_goid[goids["caller"]][9])  # has_transitive_effects
    expect_true(effects_by_goid[goids["caller"]][10] < 1.0)  # purity_confidence reduced

    expect_true(effects_by_goid[goids["uses_nonlocal"]][7])  # modifies_closure

    expect_false(effects_by_goid[goids["missing"]][1])  # is_pure should default to False
    expect_equal(effects_by_goid[goids["missing"]][10], 0.0)  # purity_confidence

    effects_json = effects_by_goid[goids["missing"]][11]
    parsed = effects_json if isinstance(effects_json, dict) else json.loads(effects_json)
    expect_equal(parsed["errors"][0]["details"]["kind"], "missing_ast")

    assert_logged(caplog.records, level="INFO", containing="function_effects populated")
