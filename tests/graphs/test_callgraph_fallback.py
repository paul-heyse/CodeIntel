"""Integration test to ensure AST fallback is invoked when CST yields no edges."""

from __future__ import annotations

from pathlib import Path

from codeintel.graphs import callgraph_builder
from codeintel.graphs.callgraph_builder import CallGraphRunScope
from codeintel.graphs.function_catalog import FunctionCatalog, FunctionMeta
from codeintel.storage.rows import CallGraphEdgeRow


def test_callgraph_falls_back_to_ast(tmp_path: Path) -> None:
    """
    When CST yields no edges, AST collector should still populate edges.

    Raises
    ------
    AssertionError
        If CST/AST collectors are not invoked or fallback edges are missing.
    """
    # Arrange: fake CST returns nothing, AST returns one edge
    cst_calls: list[Path] = []
    ast_calls: list[Path] = []

    def _fake_collect_edges_cst(
        rel_path: str,
        module: object | None = None,
        context: object | None = None,
        **_: object,
    ) -> list[CallGraphEdgeRow]:
        _ignored = (module, context, _)
        cst_calls.append(Path(rel_path))
        return []

    fake_edge: CallGraphEdgeRow = {
        "repo": "r",
        "commit": "c",
        "caller_goid_h128": 1,
        "callee_goid_h128": 2,
        "callsite_path": "mod.py",
        "callsite_line": 1,
        "callsite_col": 0,
        "language": "python",
        "kind": "direct",
        "resolved_via": "local_name",
        "confidence": 1.0,
        "evidence_json": {},
    }

    def _fake_collect_edges_ast(
        rel_path: str, file_path: Path, context: object | None = None
    ) -> list[CallGraphEdgeRow]:
        _ = (file_path, context)
        ast_calls.append(Path(rel_path))
        return [fake_edge]

    # Arrange: repo structure with one function span
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "mod.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    catalog = FunctionCatalog(
        functions=[
            FunctionMeta(
                goid=1,
                urn="urn:foo",
                rel_path="mod.py",
                qualname="mod.foo",
                start_line=1,
                end_line=2,
            )
        ],
        module_by_path={},
    )
    scope = CallGraphRunScope(repo="r", commit="c", repo_root=repo_root)

    # Act
    edges = callgraph_builder._collect_edges(  # noqa: SLF001
        catalog,
        scope,
        callgraph_builder.CallGraphInputs(
            global_callee_by_name={"mod.foo": 1, "foo": 1},
            scip_candidates_by_use={},
            def_goids_by_path={},
            cst_collector=_fake_collect_edges_cst,
            ast_collector=_fake_collect_edges_ast,
        ),
    )

    # Assert: AST path used to supply edges when CST returns none
    if not cst_calls:
        message = "CST collector was not invoked"
        raise AssertionError(message)
    if not ast_calls:
        message = "AST collector was not invoked"
        raise AssertionError(message)
    if edges != [fake_edge]:
        message = f"Expected AST edges fallback, got {edges}"
        raise AssertionError(message)
