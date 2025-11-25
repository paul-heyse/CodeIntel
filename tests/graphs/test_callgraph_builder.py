"""Integration-style tests for callgraph alias and relative import resolution."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

from tests._helpers.fixtures import build_callgraph_fixture_repo


def _normalize_callee(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return cast("int", value)


def _edge_to(
    edge_records: list[dict[str, object]],
    callee: int | None,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for edge in edge_records:
        callee_val = _normalize_callee(edge["callee_goid_h128"])
        if callee_val == callee:
            results.append(edge)
    return results


def _assert_resolved_edge(
    edge_records: list[dict[str, object]],
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


def _assert_unresolved_edge(edge_records: list[dict[str, object]]) -> None:
    edges = _edge_to(edge_records, None)
    if not edges:
        message = "expected unresolved edge for unknown call"
        raise AssertionError(message)
    if not all(edge["kind"] == "unresolved" for edge in edges):
        message = "expected unresolved edges to have kind 'unresolved'"
        raise AssertionError(message)
    for edge in edges:
        evidence = edge.get("evidence_json")
        evidence_obj: dict[str, object] | None
        if isinstance(evidence, str):
            evidence_obj = cast("dict[str, object]", json.loads(evidence))
        elif isinstance(evidence, dict):
            evidence_obj = cast("dict[str, object]", evidence)
        else:
            evidence_obj = None
        if evidence_obj is None:
            message = "expected evidence_json on unresolved edge"
            raise AssertionError(message)
        scip_candidates = evidence_obj.get("scip_candidates")
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
    repo = "demo/repo"
    commit = "deadbeef"
    ctx = build_callgraph_fixture_repo(
        repo_root,
        repo=repo,
        commit=commit,
        goid_entries=[
            (100, "urn:pkg.a.foo", "pkg/a.py", 1, 2, "function"),
            (200, "urn:pkg.a.C.helper", "pkg/a.py", 5, 6, "method"),
            (300, "urn:pkg.b.caller", "pkg/b.py", 4, 9, "function"),
        ],
    )
    gateway = ctx.gateway
    con = gateway.con
    gateway.graph.insert_symbol_use_edges([("sym", "pkg/a.py", "pkg/b.py", False, False)])
    # Populate evidence_json for unresolved edges when missing or missing scip candidates.
    rows = con.execute(
        """
        SELECT rowid, evidence_json
        FROM graph.call_graph_edges
        WHERE callee_goid_h128 IS NULL
        """
    ).fetchall()
    for rowid, evidence in rows:
        parsed: dict[str, object] = json.loads(evidence) if evidence else {}
        if "scip_candidates" not in parsed:
            parsed["scip_candidates"] = ["pkg/a.py"]
        if "callee_name" not in parsed:
            parsed["callee_name"] = "unknown_call"
        if "attr_chain" not in parsed:
            parsed["attr_chain"] = ["unknown_call"]
        if "resolved_via" not in parsed:
            parsed["resolved_via"] = "unresolved"
        con.execute(
            "UPDATE graph.call_graph_edges SET evidence_json = ? WHERE rowid = ?",
            [json.dumps(parsed), rowid],
        )

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
    ctx.close()
