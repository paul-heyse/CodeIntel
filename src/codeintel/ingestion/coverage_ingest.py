"""Ingest coverage.py results into `analytics.coverage_lines`."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException


@dataclass(frozen=True)
class CoverageInsertContext:
    """Shared context for inserting coverage rows."""

    repo: str
    commit: str
    insert_sql: str
    now: datetime


@dataclass(frozen=True)
class CoverageFileInfo:
    """Resolved coverage file paths used during ingestion."""

    measured_path: Path
    rel_path: str


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

    insert_ctx = CoverageInsertContext(
        repo=repo,
        commit=commit,
        insert_sql=insert_sql,
        now=now,
    )

    for measured in data.measured_files():
        measured_path = Path(measured).resolve()
        try:
            rel_path = measured_path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        file_info = CoverageFileInfo(measured_path=measured_path, rel_path=rel_path)
        _ingest_file_coverage(con=con, cov=cov, data=data, file_info=file_info, ctx=insert_ctx)

    log.info("coverage_lines ingested for %s@%s", repo, commit)


def _ingest_file_coverage(
    *,
    con: duckdb.DuckDBPyConnection,
    cov: Coverage,
    data: CoverageData,
    file_info: CoverageFileInfo,
    ctx: CoverageInsertContext,
) -> None:
    try:
        _, statements, _, _missing, executed = cov.analysis2(str(file_info.measured_path))
    except CoverageException as exc:
        log.warning("coverage.analysis2 failed for %s: %s", file_info.measured_path, exc)
        return

    statements_set = set(statements)
    executed_set = set(executed)
    try:
        contexts_raw = data.contexts_by_lineno(str(file_info.measured_path)) or {}
        contexts_by_lineno: dict[int, set[str]] = {
            line: set(ctxs) for line, ctxs in contexts_raw.items()
        }
    except CoverageException:
        contexts_by_lineno = {}

    for line in sorted(statements_set):
        is_covered = line in executed_set
        hits = 1 if is_covered else 0
        contexts = contexts_by_lineno.get(line)
        context_count = len(contexts) if contexts else 0

        con.execute(
            ctx.insert_sql,
            [
                ctx.repo,
                ctx.commit,
                file_info.rel_path,
                line,
                True,
                is_covered,
                hits,
                context_count,
                ctx.now,
            ],
        )
