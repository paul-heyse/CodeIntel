"""Validate docs views do not mix edges across repo/commit."""

from __future__ import annotations

from codeintel.storage.gateway import open_memory_gateway


def test_call_graph_view_scopes_edges_to_repo_commit() -> None:
    """
    Edges from other snapshots should not join into v_call_graph_enriched.

    Raises
    ------
    AssertionError
        If caller repos leak across snapshots.
    """
    con = open_memory_gateway(apply_schema=True, ensure_views=True).con

    now = "2024-01-01T00:00:00Z"
    con.execute(
        """
        INSERT INTO core.goids VALUES
        (1, 'urn:1', 'r1', 'c1', 'a.py', 'python', 'function', 'a.f', 1, 2, ?),
        (2, 'urn:2', 'r2', 'c2', 'b.py', 'python', 'function', 'b.f', 1, 2, ?)
        """,
        [now, now],
    )
    con.execute(
        """
        INSERT INTO analytics.goid_risk_factors (
            function_goid_h128,
            urn,
            repo,
            commit,
            rel_path,
            language,
            kind,
            qualname,
            risk_level,
            risk_score
        ) VALUES
        (1, 'urn:1', 'r1', 'c1', 'a.py', 'python', 'function', 'a.f', 'low', 0.1),
        (2, 'urn:2', 'r2', 'c2', 'b.py', 'python', 'function', 'b.f', 'high', 0.9)
        """
    )
    con.execute(
        """
        INSERT INTO graph.call_graph_edges VALUES
        ('r1','c1',1,NULL,'a.py',1,0,'python','direct','local',1.0,'{}'),
        ('r2','c2',2,NULL,'b.py',2,0,'python','direct','local',1.0,'{}')
        """
    )

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
