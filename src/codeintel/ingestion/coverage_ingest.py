"""Ingest coverage.py results into `analytics.coverage_lines`."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from coverage import Coverage

log = logging.getLogger(__name__)


def ingest_coverage_lines(
    con: duckdb.DuckDBPyConnection,
    repo_root: Path,
    repo: str,
    commit: str,
    *,
    coverage_file: Path | None = None,
) -> None:
    """
    Read a `.coverage` database and populate `analytics.coverage_lines`.

    The ingestion uses the `coverage.py` API and assumes data was collected
    with `dynamic_context = test_function` or similar so context_count can be
    computed. Rows contain repo, commit, path, line number, execution flags,
    hit counts, and created_at timestamps.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection to the DuckDB database.
    repo_root : Path
        Root of the repository whose coverage is being ingested.
    repo : str
        Repository slug.
    commit : str
        Commit SHA for the coverage snapshot.
    coverage_file : Path, optional
        Path to the `.coverage` database. Defaults to `<repo_root>/.coverage`.
    """
    repo_root = repo_root.resolve()
    if coverage_file is None:
        coverage_file = repo_root / ".coverage"

    if not coverage_file.is_file():
        log.warning("Coverage file %s not found; skipping coverage ingestion", coverage_file)
        return

    cov = Coverage(data_file=str(coverage_file))
    cov.load()
    data = cov.get_data()

    # Clear existing coverage rows for this repo/commit.
    con.execute(
        "DELETE FROM analytics.coverage_lines WHERE repo = ? AND commit = ?",
        [repo, commit],
    )

    insert_sql = """
        INSERT INTO analytics.coverage_lines (
            repo, commit, rel_path, line,
            is_executable, is_covered, hits,
            context_count, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    now = datetime.now(UTC)

    for abs_path in data.measured_files():
        abs_path = Path(abs_path).resolve()
        try:
            rel_path = abs_path.relative_to(repo_root).as_posix()
        except ValueError:
            # Outside repo; ignore (e.g. venv, stdlib)
            continue

        try:
            _, statements, _, missing, executed = cov.analysis2(str(abs_path))
        except Exception as exc:
            log.warning("coverage.analysis2 failed for %s: %s", abs_path, exc)
            continue

        statements_set = set(statements)
        executed_set = set(executed)
        # coverage.py 5+ has contexts_by_lineno; if missing, context_count=0.
        try:
            contexts_by_lineno: dict[int, set] = data.contexts_by_lineno(str(abs_path)) or {}
        except Exception:
            contexts_by_lineno = {}

        for line in sorted(statements_set):
            is_executable = True
            is_covered = line in executed_set
            hits = 1 if is_covered else 0
            contexts = contexts_by_lineno.get(line)
            context_count = len(contexts) if contexts else 0

            con.execute(
                insert_sql,
                [
                    repo,
                    commit,
                    rel_path,
                    line,
                    is_executable,
                    is_covered,
                    hits,
                    context_count,
                    now,
                ],
            )

    log.info("coverage_lines ingested for %s@%s", repo, commit)
