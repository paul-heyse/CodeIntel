"""Integration-style tests for callgraph alias and relative import resolution."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import duckdb

from codeintel.config.models import CallGraphConfig
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.storage.schemas import apply_all_schemas


def _write_file(path: Path, content: str) -> None:
    """Create parents and write content to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf8")


def _build_repo_fixture(repo_root: Path) -> None:
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


def _seed_goids(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> None:
    goids = [
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
    con.execute(
        """
        INSERT INTO graph.symbol_use_edges (symbol, def_path, use_path, same_file, same_module)
        VALUES ('sym', 'pkg/a.py', 'pkg/b.py', FALSE, FALSE)
        """
    )


def _normalize_callee(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return cast("int", value)


def _edge_to(edge_records: list[dict], callee: int | None) -> list[dict]:
    results: list[dict] = []
    for edge in edge_records:
        callee_val = _normalize_callee(edge["callee_goid_h128"])
        if callee_val == callee:
            results.append(edge)
    return results


def _assert_resolved_edge(
    edge_records: list[dict],
    callee: int,
    allowed_resolutions: set[str],
    missing_message: str,
    resolution_message: str,
) -> None:
    edges = _edge_to(edge_records, callee)
    if not edges:
        message = missing_message
        raise AssertionError(message)
    if not any(edge["resolved_via"] in allowed_resolutions for edge in edges):
        message = resolution_message
        raise AssertionError(message)


def _assert_unresolved_edge(edge_records: list[dict]) -> None:
    edges = _edge_to(edge_records, None)
    if not edges:
        message = "expected unresolved edge for unknown call"
        raise AssertionError(message)
    if not all(edge["kind"] == "unresolved" for edge in edges):
        message = "expected unresolved edges to have kind 'unresolved'"
        raise AssertionError(message)
    for edge in edges:
        evidence = edge.get("evidence_json")
        if isinstance(evidence, str):
            evidence = json.loads(evidence)
        if evidence is None:
            message = "expected evidence_json on unresolved edge"
            raise AssertionError(message)
        scip_candidates = evidence.get("scip_candidates")
        if scip_candidates != ["pkg/a.py"]:
            message = f"expected SCIP candidates ['pkg/a.py'], got {scip_candidates}"
            raise AssertionError(message)


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
    _build_repo_fixture(repo_root)

    repo = "demo/repo"
    commit = "deadbeef"
    con = duckdb.connect(":memory:")
    apply_all_schemas(con)

    # Seed GOIDs for functions referenced in the fixture.
    _seed_goids(con, repo, commit)

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

    _assert_resolved_edge(
        edge_records=edge_records,
        callee=100,
        allowed_resolutions={"local_name", "local_attr", "global_name", "global_attr"},
        missing_message="expected edge to foo via alias",
        resolution_message="expected foo edge to be resolved via name or attr",
    )

    _assert_resolved_edge(
        edge_records=edge_records,
        callee=200,
        allowed_resolutions={"global_name", "local_attr", "import_alias"},
        missing_message="expected edge to C.helper via attribute call",
        resolution_message="expected helper edge to use global or alias resolution",
    )

    _assert_unresolved_edge(edge_records)
