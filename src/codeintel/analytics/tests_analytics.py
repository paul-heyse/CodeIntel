"""Build test_coverage_edges from coverage contexts and test catalog."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException

log = logging.getLogger(__name__)


@dataclass
class TestCoverageConfig:
    """
    Configuration for deriving test coverage edges.

    Parameters
    ----------
    repo:
        Repository slug.
    commit:
        Commit SHA.
    repo_root:
        Path to the repository root containing the coverage file.
    coverage_file:
        Optional explicit path to a coverage.py database (defaults to repo_root/.coverage).
    """

    repo: str
    commit: str
    repo_root: Path
    coverage_file: Path | None = None


@dataclass
class EdgeContext:
    """Shared context for building test coverage edges."""

    status_by_test: dict[str, str]
    cfg: TestCoverageConfig
    now: datetime


def _load_coverage_data(cfg: TestCoverageConfig) -> Coverage | None:
    coverage_path = cfg.coverage_file or (cfg.repo_root / ".coverage")
    if not coverage_path.is_file():
        log.warning("Coverage file %s not found; skipping test coverage edges", coverage_path)
        return None

    cov = Coverage(data_file=str(coverage_path))
    cov.load()
    return cov


def _functions_by_path(con: duckdb.DuckDBPyConnection, cfg: TestCoverageConfig) -> dict[str, list[dict]]:
    funcs = con.execute(
        """
        SELECT
            goid_h128,
            urn,
            rel_path,
            qualname,
            start_line,
            end_line
        FROM core.goids
        WHERE repo = ? AND commit = ? AND kind IN ('function', 'method')
        """,
        [cfg.repo, cfg.commit],
    ).fetch_df()

    funcs_by_path: dict[str, list[dict]] = {}
    for _, row in funcs.iterrows():
        rel_path = str(row["rel_path"]).replace("\\", "/")
        funcs_by_path.setdefault(rel_path, []).append(row.to_dict())
    return funcs_by_path


def _file_coverage(
    cov: Coverage,
    data: CoverageData,
    abs_file: Path,
) -> tuple[set[int], dict[int, set[str]]]:
    """
    Extract executable statements and contexts for a single file.

    Returns
    -------
    tuple[set[int], dict[int, set[str]]]
        Executable statement line numbers and context mappings.
    """
    try:
        _, statements, _, _missing, _executed = cov.analysis2(str(abs_file))
    except CoverageException as exc:
        log.warning("coverage.analysis2 failed for %s: %s", abs_file, exc)
        return set(), {}

    statements_set = set(statements)
    try:
        raw_contexts = data.contexts_by_lineno(str(abs_file)) or {}
        contexts_by_lineno: dict[int, set[str]] = {
            ln: set(ctxs) for ln, ctxs in raw_contexts.items()
        }
    except CoverageException:
        contexts_by_lineno = {}

    return statements_set, contexts_by_lineno


def _edges_for_file(
    file_funcs: list[dict],
    statements_set: set[int],
    contexts_by_lineno: dict[int, set[str]],
    rel_path: str,
    ctx: EdgeContext,
) -> list[tuple]:
    edges: list[tuple] = []
    for info in file_funcs:
        start_line = int(info["start_line"])
        end_line = int(info["end_line"]) if info["end_line"] is not None else start_line
        executable_lines = len([ln for ln in statements_set if start_line <= ln <= end_line])
        if executable_lines == 0:
            continue

        covered_by_test: dict[str, int] = defaultdict(int)
        for ln in range(start_line, end_line + 1):
            if ln not in statements_set:
                continue
            contexts = contexts_by_lineno.get(ln) or set()
            for ctx_name in contexts:
                covered_by_test[ctx_name] += 1

        for test_id, covered_lines in covered_by_test.items():
            last_status = ctx.status_by_test.get(test_id, "unknown")
            coverage_ratio = covered_lines / float(executable_lines) if executable_lines else None
            edges.append(
                (
                    test_id,
                    None,  # test_goid_h128 (future enhancement)
                    int(info["goid_h128"]),
                    info["urn"],
                    ctx.cfg.repo,
                    ctx.cfg.commit,
                    rel_path,
                    info["qualname"],
                    covered_lines,
                    executable_lines,
                    coverage_ratio,
                    last_status,
                    ctx.now,
                )
            )
    return edges


def compute_test_coverage_edges(
    con: duckdb.DuckDBPyConnection,
    cfg: TestCoverageConfig,
) -> None:
    """
    Populate analytics.test_coverage_edges by combining coverage contexts with GOIDs.

    This expects coverage.py to have been run with dynamic contexts enabled
    (e.g., dynamic_context = test_function) so contexts_by_lineno returns
    pytest nodeids.
    """
    log.info("Computing test_coverage_edges for repo=%s commit=%s", cfg.repo, cfg.commit)

    cov = _load_coverage_data(cfg)
    if cov is None:
        return

    funcs_by_path = _functions_by_path(con, cfg)
    if not funcs_by_path:
        log.info("No functions found; skipping test coverage edges")
        return

    status_rows = con.execute(
        """
        SELECT test_id, status
        FROM analytics.test_catalog
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()
    status_by_test = {row[0]: row[1] for row in status_rows}

    con.execute(
        "DELETE FROM analytics.test_coverage_edges WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    edge_ctx = EdgeContext(status_by_test=status_by_test, cfg=cfg, now=datetime.now(UTC))
    data = cov.get_data()
    insert_rows: list[tuple] = []

    for measured in data.measured_files():
        abs_file = Path(measured).resolve()
        try:
            rel_path = abs_file.relative_to(cfg.repo_root).as_posix()
        except ValueError:
            continue

        file_funcs = funcs_by_path.get(rel_path)
        if not file_funcs:
            continue

        statements_set, contexts_by_lineno = _file_coverage(cov, data, abs_file)
        if not statements_set:
            continue

        insert_rows.extend(
            _edges_for_file(
                file_funcs=file_funcs,
                statements_set=statements_set,
                contexts_by_lineno=contexts_by_lineno,
                rel_path=rel_path,
                ctx=edge_ctx,
            )
        )

    if insert_rows:
        con.executemany(
            """
            INSERT INTO analytics.test_coverage_edges (
                test_id,
                test_goid_h128,
                function_goid_h128,
                urn,
                repo,
                commit,
                rel_path,
                qualname,
                covered_lines,
                executable_lines,
                coverage_ratio,
                last_status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            insert_rows,
        )

    log.info(
        "test_coverage_edges populated: %d rows for %s@%s",
        len(insert_rows),
        cfg.repo,
        cfg.commit,
    )
