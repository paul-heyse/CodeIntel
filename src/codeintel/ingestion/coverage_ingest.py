"""Ingest coverage.py results into `analytics.coverage_lines`."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException

from codeintel.config.models import CoverageIngestConfig
from codeintel.config.schemas.sql_builder import ensure_schema, prepared_statements


@dataclass(frozen=True)
class CoverageInsertContext:
    """Shared context for inserting coverage rows."""

    repo: str
    commit: str
    insert_sql: str
    now: datetime
    coverage_file: Path


@dataclass(frozen=True)
class CoverageFileInfo:
    """Resolved coverage file paths used during ingestion."""

    measured_path: Path
    rel_path: str


log = logging.getLogger(__name__)


def ingest_coverage_lines(
    con: duckdb.DuckDBPyConnection,
    cfg: CoverageIngestConfig,
) -> None:
    """
    Read a `.coverage` database and populate `analytics.coverage_lines`.

    The ingestion uses the `coverage.py` API and assumes data was collected
    with `dynamic_context = test_function` or similar so context_count can be
    computed. Rows contain repo, commit, path, line number, execution flags,
    hit counts, and created_at timestamps.

    Parameters
    ----------
    con:
        Connection to the DuckDB database.
    cfg:
        Coverage ingestion configuration (paths and identifiers).
    """
    repo_root = cfg.repo_root
    coverage_file = cfg.coverage_file

    if coverage_file is None or not coverage_file.is_file():
        log.warning("Coverage file %s not found; skipping coverage ingestion", coverage_file)
        return

    cov = Coverage(data_file=str(coverage_file))
    cov.load()
    data = cov.get_data()

    ensure_schema(con, "analytics.coverage_lines")
    coverage_stmt = prepared_statements("analytics.coverage_lines")

    now = datetime.now(UTC)

    insert_ctx = CoverageInsertContext(
        repo=cfg.repo,
        commit=cfg.commit,
        insert_sql=coverage_stmt.insert_sql,
        now=now,
        coverage_file=coverage_file,
    )

    rows: list[list[object]] = []
    for measured in data.measured_files():
        measured_path = Path(measured).resolve()
        try:
            rel_path = measured_path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        file_info = CoverageFileInfo(measured_path=measured_path, rel_path=rel_path)
        rows.extend(_collect_file_coverage(cov=cov, data=data, file_info=file_info, ctx=insert_ctx))

    con.execute("BEGIN")
    try:
        if coverage_stmt.delete_sql is not None:
            con.execute(coverage_stmt.delete_sql, [cfg.repo, cfg.commit])
        if rows:
            con.executemany(coverage_stmt.insert_sql, rows)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    log.info("coverage_lines ingested for %s@%s", cfg.repo, cfg.commit)


def _collect_file_coverage(
    *,
    cov: Coverage,
    data: CoverageData,
    file_info: CoverageFileInfo,
    ctx: CoverageInsertContext,
) -> list[list[object]]:
    rows: list[list[object]] = []
    try:
        _, statements, _, _missing, executed = cov.analysis2(str(file_info.measured_path))
    except CoverageException as exc:
        log.warning("coverage.analysis2 failed for %s: %s", file_info.measured_path, exc)
        return rows

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

        rows.append(
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
            ]
        )
    return rows
