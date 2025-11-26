"""Call graph resolution tests for import alias heuristics using real builders."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from codeintel.config import ConfigBuilder
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.storage.gateway import StorageGateway

REPO = "demo/repo"
COMMIT = "deadbeef"


def test_callgraph_resolves_import_alias(tmp_path: Path, fresh_gateway: StorageGateway) -> None:
    """
    Call graph builder should resolve calls made through an import alias.

    Raises
    ------
    AssertionError
        If alias-based resolution fails to return the expected GOID.
    """
    repo_root = tmp_path / "repo"
    _write_repo_files(repo_root)
    callee_goid, caller_goid = _seed_modules_and_goids(fresh_gateway, repo_root)

    builder = ConfigBuilder.from_snapshot(
        repo=REPO,
        commit=COMMIT,
        repo_root=repo_root,
        build_dir=repo_root / "build",
    )
    build_call_graph(fresh_gateway, builder.call_graph())

    con = fresh_gateway.con
    node_goids = {
        int(goid)
        for goid, rel_path in con.execute(
            "SELECT goid_h128, rel_path FROM graph.call_graph_nodes"
        ).fetchall()
        if rel_path in {"pkg/callee.py", "pkg/caller.py"}
    }
    if {callee_goid, caller_goid} != node_goids:
        message = f"Call graph nodes missing expected GOIDs: {node_goids}"
        raise AssertionError(message)

    edge_rows = {
        (int(caller), int(callee)): resolved_via
        for caller, callee, resolved_via in con.execute(
            """
            SELECT caller_goid_h128, callee_goid_h128, resolved_via
            FROM graph.call_graph_edges
            WHERE repo = ? AND commit = ?
            """,
            [REPO, COMMIT],
        ).fetchall()
    }
    resolved_via = edge_rows.get((caller_goid, callee_goid))
    if resolved_via is None:
        message = f"Expected edge from caller to callee, got {edge_rows}"
        raise AssertionError(message)
    if resolved_via not in {"import_alias", "import_alias_global"}:
        message = f"Unexpected resolution mode: {resolved_via}"
        raise AssertionError(message)


def _write_repo_files(repo_root: Path) -> tuple[Path, Path]:
    """
    Create minimal caller/callee modules for alias resolution.

    Returns
    -------
    tuple[Path, Path]
        Paths to the callee and caller Python files.
    """
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    callee_path = pkg_dir / "callee.py"
    callee_path.write_text(
        "def target():\n    return 1\n",
        encoding="utf-8",
    )
    caller_path = pkg_dir / "caller.py"
    caller_path.write_text(
        "import pkg.callee as cc\n\ndef caller():\n    return cc.target()\n",
        encoding="utf-8",
    )
    return callee_path, caller_path


def _seed_modules_and_goids(
    gateway: StorageGateway,
    repo_root: Path,
) -> tuple[int, int]:
    callee_goid = 1
    caller_goid = 2
    callee_rel = (repo_root / "pkg" / "callee.py").relative_to(repo_root).as_posix()
    caller_rel = (repo_root / "pkg" / "caller.py").relative_to(repo_root).as_posix()
    now = datetime.now(UTC)

    con = gateway.con
    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, 'python', '[]', '[]')
        """,
        [
            ("pkg.callee", callee_rel, REPO, COMMIT),
            ("pkg.caller", caller_rel, REPO, COMMIT),
        ],
    )
    con.executemany(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                callee_goid,
                "urn:pkg.callee.target",
                REPO,
                COMMIT,
                callee_rel,
                "python",
                "function",
                "pkg.callee.target",
                1,
                2,
                now,
            ),
            (
                caller_goid,
                "urn:pkg.caller.caller",
                REPO,
                COMMIT,
                caller_rel,
                "python",
                "function",
                "pkg.caller.caller",
                3,
                4,
                now,
            ),
        ],
    )
    return callee_goid, caller_goid
