"""Tests for graph validation helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from _pytest.logging import LogCaptureFixture

from codeintel.graphs.validation import run_graph_validations
from codeintel.storage.gateway import open_memory_gateway


def test_run_graph_validations_emits_warnings(caplog: LogCaptureFixture) -> None:
    """
    Graph validations should warn for common integrity gaps.

    Raises
    ------
    AssertionError
        If expected warning text is absent.
    """
    con = open_memory_gateway().con
    repo: Final = "demo/repo"
    commit: Final = "deadbeef"

    # File with AST functions but no GOIDs -> missing GOIDs + orphan module.
    con.execute(
        """
        INSERT INTO core.ast_nodes (path, node_type, name, qualname, lineno, end_lineno, col_offset, end_col_offset, parent_qualname, decorators, docstring, hash)
        VALUES ('pkg/a.py', 'FunctionDef', 'foo', 'pkg.a.foo', 1, 2, 0, 0, 'pkg.a', '[]', NULL, 'h1')
        """
    )
    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES ('pkg.a', 'pkg/a.py', ?, ?, 'python', '[]', '[]')
        """,
        [repo, commit],
    )

    # Caller GOID with out-of-span callsite -> span mismatch.
    con.execute(
        """
        INSERT INTO core.goids (goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, created_at)
        VALUES (1, 'urn:pkg.b.caller', ?, ?, 'pkg/b.py', 'python', 'function', 'pkg.b.caller', 1, 5, ?)
        """,
        [repo, commit, datetime.now(UTC)],
    )
    con.execute(
        """
        INSERT INTO graph.call_graph_edges (
            repo, commit, caller_goid_h128, callee_goid_h128, callsite_path, callsite_line, callsite_col, language, kind, resolved_via, confidence, evidence_json
        ) VALUES (?, ?, 1, NULL, 'pkg/b.py', 50, 0, 'python', 'unresolved', 'unresolved', 0.0, '{}')
        """,
        [repo, commit],
    )

    with caplog.at_level("WARNING"):
        run_graph_validations(con, repo=repo, commit=commit)

    messages = " ".join(record.message for record in caplog.records)
    expected = [
        "functions without GOIDs",
        "outside caller spans",
        "module(s) have no GOIDs",
    ]
    for needle in expected:
        if needle not in messages:
            message = f"Expected warning containing '{needle}' but messages were: {messages}"
            raise AssertionError(message)
