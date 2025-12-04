"""Database integrity validation checks.

This module contains validation checks that verify data integrity
by querying the database for inconsistencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from codeintel.graphs.catalog import FunctionCatalog
    from codeintel.storage.gateway import StorageGateway


def warn_missing_function_goids(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for files with functions in AST that are missing GOIDs.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo
        Repository identifier.
    commit
        Commit identifier.
    log
        Logger for output.

    Returns
    -------
    list[dict[str, object]]
        Findings for files with missing function GOIDs.
    """
    try:
        rows = gateway.con.execute(
            """
            WITH funcs AS (
                SELECT path AS rel_path, COUNT(*) AS function_count
                FROM core.ast_nodes
                WHERE repo = ? AND commit = ? AND node_type IN ('FunctionDef', 'AsyncFunctionDef')
                GROUP BY path
            ),
            goids AS (
                SELECT rel_path, COUNT(*) AS goid_count
                FROM core.goids
                WHERE repo = ? AND commit = ? AND kind IN ('function', 'method')
                GROUP BY rel_path
            )
            SELECT f.rel_path, f.function_count, COALESCE(g.goid_count, 0) AS goid_count
            FROM funcs f
            LEFT JOIN goids g ON g.rel_path = f.rel_path
            WHERE COALESCE(g.goid_count, 0) < f.function_count
            ORDER BY f.rel_path
            """,
            [repo, commit, repo, commit],
        ).fetchall()
    except DuckDBError:
        return []

    if not rows:
        return []
    sample = ", ".join(path for path, _, _ in rows[:5])
    log.warning(
        "Validation: %d file(s) have functions without GOIDs (sample: %s)",
        len(rows),
        sample,
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "missing_function_goids",
            "severity": "warning",
            "path": path,
            "detail": f"{function_count} functions, {goid_count} GOIDs",
            "context": {"function_count": function_count, "goid_count": goid_count},
        }
        for path, function_count, goid_count in rows
    ]


def warn_callsite_span_mismatches(
    gateway: StorageGateway,
    catalog: FunctionCatalog,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for call graph edges whose callsites lie outside caller spans.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    catalog
        Function catalog with span information.
    repo
        Repository identifier.
    commit
        Commit identifier.
    log
        Logger for output.

    Returns
    -------
    list[dict[str, object]]
        Findings for callsite span mismatches.
    """
    spans_by_goid = {span.goid: span for span in catalog.function_spans}
    try:
        rows = gateway.con.execute(
            """
            SELECT
                e.caller_goid_h128,
                e.callsite_path,
                e.callsite_line
            FROM graph.call_graph_edges e
            WHERE e.callsite_line IS NOT NULL
              AND e.repo = ? AND e.commit = ?
            """,
            [repo, commit],
        ).fetchall()
    except DuckDBError:
        return []

    mismatches = []
    for goid, path, line in rows:
        span = spans_by_goid.get(int(goid)) if goid is not None else None
        if span is None:
            continue
        if line < span.start_line or line > span.end_line:
            mismatches.append((path, line, span.start_line, span.end_line))

    if not mismatches:
        return []
    sample = ", ".join(f"{path}:{line}" for path, line, _, _ in mismatches[:5])
    log.warning(
        "Validation: %d call graph edges fall outside caller spans (sample: %s)",
        len(mismatches),
        sample,
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "callsite_span_mismatch",
            "severity": "warning",
            "path": path,
            "detail": f"callsite {line} outside span {start}-{end}",
            "context": {"callsite_line": line, "start_line": start, "end_line": end},
        }
        for path, line, start, end in mismatches
    ]


def warn_orphan_modules(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    log: logging.Logger,
    catalog: FunctionCatalog,
) -> list[dict[str, object]]:
    """Check for modules with no GOIDs (orphans).

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo
        Repository identifier.
    commit
        Commit identifier.
    log
        Logger for output.
    catalog
        Function catalog for fallback module lookup.

    Returns
    -------
    list[dict[str, object]]
        Findings for orphan modules.
    """
    query_failed = False
    try:
        con = gateway.con
        rows = con.execute(
            """
            SELECT m.path
            FROM core.modules m
            LEFT JOIN core.goids g
              ON g.rel_path = m.path AND g.repo = ? AND g.commit = ? AND g.kind = 'module'
            WHERE m.repo = ? AND m.commit = ? AND g.goid_h128 IS NULL
            """,
            [repo, commit, repo, commit],
        ).fetchall()
        if rows:
            stats = con.execute(
                """
                WITH module_goids AS (
                    SELECT rel_path, COUNT(*) AS cnt
                    FROM core.goids
                    WHERE repo = ? AND commit = ? AND kind = 'module'
                    GROUP BY rel_path
                )
                SELECT m.path, COALESCE(g.cnt, 0) AS module_goids
                FROM core.modules m
                LEFT JOIN module_goids g ON g.rel_path = m.path
                WHERE m.repo = ? AND m.commit = ?
                ORDER BY module_goids ASC, m.path
                LIMIT 5
                """,
                [repo, commit, repo, commit],
            ).fetchall()
            sample_detail = ", ".join(f"{path} (module_goids={cnt})" for path, cnt in stats)
            log.info(
                "Orphan module debug: repo=%s commit=%s sample=%s",
                repo,
                commit,
                sample_detail,
            )
    except DuckDBError:
        query_failed = True
        rows = []

    if query_failed and catalog.module_by_path:
        rows = [(path,) for path in catalog.module_by_path]

    if not rows:
        return []
    sample = ", ".join(path for (path,) in rows[:5])
    log.warning("Validation: %d module(s) have no GOIDs (sample: %s)", len(rows), sample)
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "orphan_module",
            "severity": "warning",
            "path": path,
            "detail": "module has no GOIDs",
            "context": {},
        }
        for (path,) in rows
    ]


__all__ = [
    "warn_callsite_span_mismatches",
    "warn_missing_function_goids",
    "warn_orphan_modules",
]
