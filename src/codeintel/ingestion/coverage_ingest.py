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
from codeintel.ingestion.common import run_batch, should_skip_missing_file
from codeintel.models.rows import CoverageLineRow, coverage_line_to_tuple
from codeintel.utils.paths import normalize_rel_path


@dataclass(frozen=True)
class CoverageInsertContext:
    """Shared context for inserting coverage rows."""

    repo: str
    commit: str
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

    if coverage_file is None or should_skip_missing_file(
        coverage_file, logger=log, label="coverage file"
    ):
        return

    cov = Coverage(data_file=str(coverage_file))
    cov.load()
    data = cov.get_data()

    now = datetime.now(UTC)

    insert_ctx = CoverageInsertContext(
        repo=cfg.repo, commit=cfg.commit, now=now, coverage_file=coverage_file
    )

    rows: list[CoverageLineRow] = []
    for measured in data.measured_files():
        measured_path = Path(measured).resolve()
        try:
            rel_path = normalize_rel_path(measured_path.relative_to(repo_root))
        except ValueError:
            continue
        file_info = CoverageFileInfo(measured_path=measured_path, rel_path=rel_path)
        rows.extend(_collect_file_coverage(cov=cov, data=data, file_info=file_info, ctx=insert_ctx))

    run_batch(
        con,
        "analytics.coverage_lines",
        [coverage_line_to_tuple(r) for r in rows],
        delete_params=[cfg.repo, cfg.commit],
    )
    log.info("coverage_lines ingested for %s@%s", cfg.repo, cfg.commit)


def _collect_file_coverage(
    *,
    cov: Coverage,
    data: CoverageData,
    file_info: CoverageFileInfo,
    ctx: CoverageInsertContext,
) -> list[CoverageLineRow]:
    rows: list[CoverageLineRow] = []
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
            CoverageLineRow(
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=file_info.rel_path,
                line=line,
                is_executable=True,
                is_covered=is_covered,
                hits=hits,
                context_count=context_count,
                created_at=ctx.now,
            )
        )
    return rows
