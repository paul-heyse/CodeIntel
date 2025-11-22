"""Integration-style tests for callgraph alias and relative import resolution."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from codeintel.config.models import CallGraphConfig
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.storage.schemas import apply_all_schemas


def _write_file(path: Path, content: str) -> None:
    """Create parents and write content to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf8")


def test_callgraph_handles_aliases_and_relative_imports(tmp_path: Path) -> None:
    """
    Calls through import aliases and methods on imported classes are resolved.

    The fixture includes:
    - alias import: from .a import foo as f
    - module alias: import pkg.a as pa
    - method call via imported class C.helper
    - an unresolved call to ensure unresolved edges are emitted

    Raises
    ------
    AssertionError
        If expected call graph edges are missing or mis-resolved.
    """
    repo_root = tmp_path / "repo"
    pkg_dir = repo_root / "pkg"
    _write_file(pkg_dir / "__init__.py", "")
    _write_file(
        pkg_dir / "a.py",
        "\n".join(
            [
                "def foo():",
                "    return 1",
                "",
                "class C:",
                "    def helper(self):",
                "        return foo()",
            ]
        ),
    )
    _write_file(
        pkg_dir / "b.py",
        "\n".join(
            [
                "from .a import foo as f, C",
                "import pkg.a as pa",
                "",
                "def caller():",
                "    f()",
                "    obj = C()",
                "    obj.helper()",
                "    pa.foo()",
                "    unknown_call()",
            ]
        ),
    )

    repo = "demo/repo"
    commit = "deadbeef"
    con = duckdb.connect(":memory:")
    apply_all_schemas(con)

    # Seed GOIDs for functions referenced in the fixture.
    goids = [
        # goid, urn, repo, commit, rel_path, language, kind, qualname, start, end, created_at
        (
            100,
            "urn:pkg.a.foo",
            repo,
            commit,
            "pkg/a.py",
            "python",
            "function",
            "pkg.a.foo",
            1,
            2,
            datetime.now(UTC),
        ),
        (
            200,
            "urn:pkg.a.C.helper",
            repo,
            commit,
            "pkg/a.py",
            "python",
            "method",
            "pkg.a.C.helper",
            5,
            6,
            datetime.now(UTC),
        ),
        (
            300,
            "urn:pkg.b.caller",
            repo,
            commit,
            "pkg/b.py",
            "python",
            "function",
            "pkg.b.caller",
            4,
            9,
            datetime.now(UTC),
        ),
    ]
    con.executemany(
        """
        INSERT INTO core.goids
          (goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        goids,
    )

    cfg = CallGraphConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    build_call_graph(con, cfg)

    df_edges = con.execute(
        "SELECT caller_goid_h128, callee_goid_h128, kind, resolved_via, evidence_json "
        "FROM graph.call_graph_edges"
    ).fetch_df()
    if df_edges.empty:
        message = "expected call graph edges to be produced"
        raise AssertionError(message)

    edge_records = df_edges.to_dict("records")

    def _edge_to(callee: int | None) -> list[dict]:
        results: list[dict] = []
        for edge in edge_records:
            callee_val = edge["callee_goid_h128"]
            if callee_val != callee_val:  # NaN check
                callee_val = None
            if callee_val == callee:
                results.append(edge)
        return results

    alias_edges = _edge_to(100)
    if not alias_edges:
        message = "expected edge to foo via alias"
        raise AssertionError(message)
    if not any(e["resolved_via"] in {"local_name", "local_attr", "global_name", "global_attr"} for e in alias_edges):
        message = "expected foo edge to be resolved via name or attr"
        raise AssertionError(message)

    helper_edges = _edge_to(200)
    if not helper_edges:
        message = "expected edge to C.helper via attribute call"
        raise AssertionError(message)
    if not any(e["resolved_via"] in {"global_name", "local_attr", "import_alias"} for e in helper_edges):
        message = "expected helper edge to use global or alias resolution"
        raise AssertionError(message)

    unresolved = _edge_to(None)
    if not unresolved:
        message = "expected unresolved edge for unknown call"
        raise AssertionError(message)
    if not all(e["kind"] == "unresolved" for e in unresolved):
        message = "expected unresolved edges to have kind 'unresolved'"
        raise AssertionError(message)
