"""Validate docs views do not mix edges across repo/commit."""

from __future__ import annotations

from pathlib import Path

from tests._helpers.fixtures import docs_views_ready_gateway, seed_call_graph_scoping


def test_call_graph_view_scopes_edges_to_repo_commit(tmp_path: Path) -> None:
    """
    Edges from other snapshots should not join into v_call_graph_enriched.

    Raises
    ------
    AssertionError
        If caller repos leak across snapshots.
    """
    ctx = docs_views_ready_gateway(tmp_path / "docs_scoping", repo="r1", commit="c1")
    seed_call_graph_scoping(gateway=ctx.gateway, now_iso="2024-01-01T00:00:00Z")
    con = ctx.gateway.con

    rows_r1 = con.execute(
        "SELECT DISTINCT caller_repo FROM docs.v_call_graph_enriched WHERE caller_repo = 'r1'"
    ).fetchall()
    rows_r2 = con.execute(
        "SELECT DISTINCT caller_repo FROM docs.v_call_graph_enriched WHERE caller_repo = 'r2'"
    ).fetchall()

    if rows_r1 != [("r1",)]:
        message = f"Unexpected caller_repo rows for r1: {rows_r1}"
        raise AssertionError(message)
    if rows_r2 != [("r2",)]:
        message = f"Unexpected caller_repo rows for r2: {rows_r2}"
        raise AssertionError(message)
    ctx.close()
