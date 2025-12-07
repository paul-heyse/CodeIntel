"""Integration-flavored tests for analytics.functions.function_effects."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    FunctionEffectsStepConfig,
    compute_function_effects,
)
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.analytics.runtime.graph import GraphRuntime, GraphRuntimeOptions
from codeintel.config.primitives import SnapshotRef
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.storage.sql.builder import ensure_schema
from tests._helpers.builders import CallGraphEdgeRow, insert_rows
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog
from tests._helpers.gateway import GatewayFactory
from tests._helpers.graphs import GraphStubEngine
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
    runtime = GraphRuntime(
        GraphRuntimeOptions(snapshot=snapshot),
        GraphStubEngine(
            gateway=gateway,
            snapshot=snapshot,
            call_graph_obj=call_graph,
            copy_graphs=False,
        ),
    )
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
                    line_span=(ast_map[goids["impure"]].start_line, ast_map[goids["impure"]].end_line),
                ),
                function_meta(
                    goid=goids["caller"],
                    rel_path=module_path.relative_to(repo_root).as_posix(),
                    qualname="caller",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(ast_map[goids["caller"]].start_line, ast_map[goids["caller"]].end_line),
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

    try:
        with patch.object(IngestStorageService, "run_batch", autospec=True) as run_batch:
            compute_function_effects(gateway, cfg, inputs=inputs)
        rows = run_batch.call_args.args[2]
    finally:
        gateway.close()

    effects_by_goid = {int(row[2]): row for row in rows}
    assert effects_by_goid[goids["impure"]][4] is True  # uses_io
    assert effects_by_goid[goids["impure"]][6] is True  # uses_time
    assert effects_by_goid[goids["impure"]][7] is True  # uses_randomness
    assert effects_by_goid[goids["impure"]][8] is True  # modifies_globals
    assert effects_by_goid[goids["impure"]][10] is True  # spawns_threads_or_tasks
    assert effects_by_goid[goids["caller"]][11] is True
    assert effects_by_goid[goids["caller"]][3] is False
    assert effects_by_goid[goids["caller"]][12] < 1.0
    assert effects_by_goid[goids["uses_nonlocal"]][9] is True
    assert effects_by_goid[goids["missing"]][3] is False
    assert effects_by_goid[goids["missing"]][12] == 0.0
    assert effects_by_goid[goids["missing"]][13]["errors"][0]["details"]["kind"] == "missing_ast"
