"""Lightweight validations for graph construction outputs."""

from __future__ import annotations

import logging

import duckdb

from codeintel.graphs.function_catalog import FunctionCatalog, load_function_catalog
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider

FINDINGS_TABLE = "analytics.graph_validation"


def run_graph_validations(
    con: duckdb.DuckDBPyConnection,
    *,
    repo: str,
    commit: str,
    catalog_provider: FunctionCatalogProvider | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """
    Emit warnings for common graph integrity issues.

    Checks include:
    - Files with functions in AST that are missing GOIDs.
    - Call graph edges whose callsites lie outside caller spans.
    - Modules with no GOIDs (orphans).
    """
    active_log = logger or logging.getLogger(__name__)
    catalog = (
        catalog_provider.catalog()
        if catalog_provider is not None
        else load_function_catalog(con, repo=repo, commit=commit)
    )
    findings = []
    findings.extend(_warn_missing_function_goids(con, repo, commit, active_log))
    findings.extend(_warn_callsite_span_mismatches(con, catalog, repo, commit, active_log))
    findings.extend(_warn_orphan_modules(con, repo, commit, active_log, catalog))
    _persist_findings(con, findings)
    active_log.info(
        "Graph validation completed for %s@%s: %d finding(s)",
        repo,
        commit,
        len(findings),
    )


def _warn_missing_function_goids(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    try:
        rows = con.execute(
            """
            WITH funcs AS (
                SELECT path AS rel_path, COUNT(*) AS function_count
                FROM core.ast_nodes
                WHERE node_type IN ('FunctionDef', 'AsyncFunctionDef')
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
            [repo, commit],
        ).fetchall()
    except duckdb.Error:
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


def _warn_callsite_span_mismatches(
    con: duckdb.DuckDBPyConnection,
    catalog: FunctionCatalog,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    spans_by_goid = {span.goid: span for span in catalog.function_spans}
    try:
        rows = con.execute(
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
    except duckdb.Error:
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


def _warn_orphan_modules(
    con: duckdb.DuckDBPyConnection,
    repo: str,
    commit: str,
    log: logging.Logger,
    catalog: FunctionCatalog,
) -> list[dict[str, object]]:
    try:
        rows = con.execute(
            """
            SELECT m.path
            FROM core.modules m
            LEFT JOIN core.goids g
              ON g.rel_path = m.path AND g.repo = ? AND g.commit = ?
            WHERE m.repo = ? AND m.commit = ? AND g.goid_h128 IS NULL
            """,
            [repo, commit, repo, commit],
        ).fetchall()
    except duckdb.Error:
        rows = []

    if not rows and catalog.module_by_path:
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


def _persist_findings(con: duckdb.DuckDBPyConnection, findings: list[dict[str, object]]) -> None:
    if not findings:
        return
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics.graph_validation (
            repo VARCHAR,
            commit VARCHAR,
            check_name VARCHAR,
            severity VARCHAR,
            path VARCHAR,
            detail VARCHAR,
            context JSON,
            created_at TIMESTAMP
        )
        """
    )
    con.executemany(
        """
        INSERT INTO analytics.graph_validation
        (repo, commit, check_name, severity, path, detail, context, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            (
                finding.get("repo"),
                finding.get("commit"),
                finding.get("check_name"),
                finding.get("severity"),
                finding.get("path"),
                finding.get("detail"),
                finding.get("context"),
            )
            for finding in findings
        ],
    )
