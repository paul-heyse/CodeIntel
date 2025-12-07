"""Integration-flavored tests for analytics.functions.function_effects."""

from __future__ import annotations

import ast
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx

from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    FunctionEffectsStepConfig,
    compute_function_effects,
)
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.analytics.runtime.graph import GraphRuntime, GraphRuntimeOptions
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.sql.builder import ensure_schema
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog, MockFunctionMeta
from tests._helpers.gateway import GatewayFactory
from tests._helpers.graphs import GraphStubEngine


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
        node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
        )
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


def test_compute_function_effects_with_transitive_and_missing(tmp_path: Path) -> None:
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
    call_graph = nx.DiGraph()
    call_graph.add_nodes_from(goids.values())
    call_graph.add_edge(goids["caller"], goids["impure"])
    engine = GraphStubEngine(
        gateway=gateway,
        snapshot=snapshot,
        call_graph_obj=call_graph,
        copy_graphs=False,
    )
    runtime = GraphRuntime(GraphRuntimeOptions(snapshot=snapshot), engine)
    runtime.ensure_call_graph()
    catalog = MockFunctionCatalog(
        functions=[
            MockFunctionMeta(
                goid=goids["impure"],
                urn="urn:pkg.effects.impure",
                rel_path=module_path.relative_to(repo_root).as_posix(),
                qualname="impure",
                start_line=ast_map[goids["impure"]].start_line,
                end_line=ast_map[goids["impure"]].end_line,
            ),
            MockFunctionMeta(
                goid=goids["caller"],
                urn="urn:pkg.effects.caller",
                rel_path=module_path.relative_to(repo_root).as_posix(),
                qualname="caller",
                start_line=ast_map[goids["caller"]].start_line,
                end_line=ast_map[goids["caller"]].end_line,
            ),
            MockFunctionMeta(
                goid=goids["uses_nonlocal"],
                urn="urn:pkg.effects.uses_nonlocal",
                rel_path=module_path.relative_to(repo_root).as_posix(),
                qualname="uses_nonlocal",
                start_line=ast_map[goids["uses_nonlocal"]].start_line,
                end_line=ast_map[goids["uses_nonlocal"]].end_line,
            ),
            MockFunctionMeta(
                goid=goids["missing"],
                urn="urn:pkg.effects.missing",
                rel_path=module_path.relative_to(repo_root).as_posix(),
                qualname="missing",
                start_line=1,
                end_line=1,
            ),
        ],
        module_by_path={module_path.relative_to(repo_root).as_posix(): "pkg.effects"},
    )
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
    gateway.con.execute(
        "INSERT INTO graph.call_graph_edges VALUES (?, ?, ?, ?)",
        (goids["caller"], None, snapshot.repo, snapshot.commit),
    )

    inputs = FunctionEffectsInputs(
        catalog_provider=catalog,
        runtime=runtime,
        ast_map=ast_map,
        missing_goids={goids["missing"]},
    )

    try:
        compute_function_effects(gateway, cfg, inputs=inputs)

        rows = gateway.con.execute(
            """
            SELECT function_goid_h128, is_pure, uses_io, touches_db, uses_time, uses_randomness,
                   modifies_globals, modifies_closure, spawns_threads_or_tasks,
                   has_transitive_effects, purity_confidence, evidence_json
            FROM analytics.function_effects
            WHERE repo = ? AND commit = ?
            ORDER BY function_goid_h128
            """,
            [snapshot.repo, snapshot.commit],
        ).fetchall()
    finally:
        gateway.close()

    effects_by_goid = {int(row[0]): row for row in rows}
    impure_row = effects_by_goid[goids["impure"]]
    caller_row = effects_by_goid[goids["caller"]]
    nonlocal_row = effects_by_goid[goids["uses_nonlocal"]]
    missing_row = effects_by_goid[goids["missing"]]

    assert impure_row[2] is True  # uses_io
    assert impure_row[4] is True  # uses_time
    assert impure_row[5] is True  # uses_randomness
    assert impure_row[6] is True  # modifies_globals
    assert impure_row[8] is True  # spawns_threads_or_tasks
    assert caller_row[9] is True
    assert caller_row[1] is False
    assert caller_row[10] < 1.0
    assert nonlocal_row[7] is True
    assert missing_row[1] is False
    assert missing_row[10] == 0.0
    assert missing_row[11]["errors"][0]["details"]["kind"] == "missing_ast"
