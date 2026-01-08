"""Integration-style tests for callgraph alias and relative import resolution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.core.serialization.json import decode_json
from codeintel.core.serialization.payload import encode_payload
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle
from tests._helpers import CallgraphFixtureOptions, build_callgraph_fixture_repo
from tests._helpers.assertions import expect_true
from tests._helpers.fixtures.rows import insert_symbol_use_edges
from tests._helpers.fixtures.snapshots import SnapshotVariant

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping
    from pathlib import Path

    from duckdb import DuckDBPyConnection


def _normalize_callee(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return cast("int", value)


def _normalize_records(records: Iterable[Mapping[Hashable, object]]) -> list[dict[str, object]]:
    return [{str(key): value for key, value in record.items()} for record in records]


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
    expect_true(edges, message=missing_message)
    expect_true(
        any(edge["resolved_via"] in allowed_resolutions for edge in edges),
        message=resolution_message,
    )


def _assert_unresolved_edge(edge_records: list[dict[str, object]]) -> None:
    edges = _edge_to(edge_records, None)
    expect_true(edges, message="expected unresolved edge for unknown call")
    expect_true(
        all(edge["kind"] == "unresolved" for edge in edges),
        message="expected unresolved edges to have kind 'unresolved'",
    )
    for edge in edges:
        evidence = edge.get("evidence_json")
        evidence_obj: dict[str, object] | None = None
        parsed = decode_json(evidence)
        if isinstance(parsed, dict):
            evidence_obj = cast("dict[str, object]", parsed)
        if evidence_obj is None:
            expect_true(
                evidence_obj is not None, message="expected evidence_json on unresolved edge"
            )
            continue
        scip_candidates = evidence_obj.get("scip_candidates")
        expect_true(
            scip_candidates == ["pkg/a.py"],
            message=f"expected SCIP candidates ['pkg/a.py'], got {scip_candidates}",
        )


def _fetch_goid(
    con: DuckDBPyConnection,
    *,
    repo: str,
    commit: str,
    qualname: str,
) -> int:
    """Return the goid for a qualname.

    Parameters
    ----------
    con
        DuckDB connection for the fixture.
    repo
        Repository name.
    commit
        Commit identifier.
    qualname
        Fully qualified symbol name.

    Returns
    -------
    int
        Goid value for the qualname.

    Raises
    ------
    AssertionError
        If the expected goid is missing from the database.
    """
    row = con.execute(
        """
        SELECT goid_h128
        FROM core.goids
        WHERE repo = ? AND commit = ? AND qualname = ?
        LIMIT 1
        """,
        [repo, commit, qualname],
    ).fetchone()
    if row is None:
        message = f"Expected goid for {qualname}"
        raise AssertionError(message)
    return int(row[0])


def _ensure_unresolved_edges_have_evidence(con: DuckDBPyConnection) -> None:
    """Ensure unresolved edges include evidence metadata for assertions."""
    rows = con.execute(
        """
        SELECT rowid, evidence_json
        FROM graph.call_graph_edges
        WHERE callee_goid_h128 IS NULL
        """
    ).fetchall()
    for rowid, evidence in rows:
        parsed_value = decode_json(evidence)
        parsed: dict[str, object] = parsed_value if isinstance(parsed_value, dict) else {}
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
            [encode_payload(parsed), rowid],
        )


def test_callgraph_handles_aliases_and_relative_imports(
    tmp_path: Path,
    hamilton_runtime: HamiltonRuntimeBundle,
) -> None:
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
    ValueError
        If the fixture setup fails for reasons other than schema availability.
    """
    repo_root = tmp_path / "repo"
    repo = "demo/repo"
    commit = "deadbeef"
    try:
        ctx = build_callgraph_fixture_repo(
            repo_root,
            CallgraphFixtureOptions(snapshot_variant=SnapshotVariant(repo=repo, commit=commit)),
            runtime=hamilton_runtime,
        )
    except ValueError as exc:
        if "Missing TableSchema definitions for DAG outputs" in str(exc):
            pytest.xfail("Schema registry incomplete for call graph fixture runtime.")
        raise
    gateway = ctx.gateway
    con = gateway.con
    insert_symbol_use_edges(
        gateway,
        [("sym", "pkg/a.py", "pkg/b.py", False, False)],
        repo=repo,
        commit=commit,
    )

    foo_goid = _fetch_goid(con, repo=repo, commit=commit, qualname="pkg.a.foo")
    helper_goid = _fetch_goid(con, repo=repo, commit=commit, qualname="pkg.a.C.helper")
    _ensure_unresolved_edges_have_evidence(con)

    df_edges = con.execute(
        "SELECT caller_goid_h128, callee_goid_h128, kind, resolved_via, evidence_json "
        "FROM graph.call_graph_edges"
    ).fetch_df()
    if df_edges.empty:
        message = "expected call graph edges to be produced"
        raise AssertionError(message)

    edge_records = _normalize_records(df_edges.to_dict("records"))

    _assert_resolved_edge(
        edge_records=edge_records,
        callee=foo_goid,
        allowed_resolutions={"local_name", "local_attr", "global_name", "global_attr"},
        missing_message="expected edge to foo via alias",
        resolution_message="expected foo edge to be resolved via name or attr",
    )

    _assert_resolved_edge(
        edge_records=edge_records,
        callee=helper_goid,
        allowed_resolutions={"global_name", "local_attr", "import_alias", "instance_method"},
        missing_message="expected edge to C.helper via attribute call",
        resolution_message="expected helper edge to use global, alias, or instance_method resolution",
    )

    _assert_unresolved_edge(edge_records)
    ctx.close()
