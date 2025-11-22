"""Build test_coverage_edges from coverage contexts and test catalog."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

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
    test_meta_by_id: dict[str, tuple[int | None, str | None]]


class FunctionRow(TypedDict):
    """Minimal function metadata used for coverage edge construction."""

    goid_h128: int
    urn: str
    rel_path: str
    qualname: str
    start_line: int
    end_line: int | None


def _load_coverage_data(cfg: TestCoverageConfig) -> Coverage | None:
    coverage_path = cfg.coverage_file or (cfg.repo_root / ".coverage")
    if not coverage_path.is_file():
        log.warning("Coverage file %s not found; skipping test coverage edges", coverage_path)
        return None

    cov = Coverage(data_file=str(coverage_path))
    cov.load()
    return cov


def _functions_by_path(
    con: duckdb.DuckDBPyConnection, cfg: TestCoverageConfig
) -> dict[str, list[FunctionRow]]:
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
    ).fetchall()

    funcs_by_path: dict[str, list[FunctionRow]] = {}
    for (
        goid_h128,
        urn,
        rel_path,
        qualname,
        start_line,
        end_line,
    ) in funcs:
        rel_path_str = str(rel_path).replace("\\", "/")
        funcs_by_path.setdefault(rel_path_str, []).append(
            FunctionRow(
                goid_h128=int(goid_h128),
                urn=str(urn),
                rel_path=rel_path_str,
                qualname=str(qualname),
                start_line=int(start_line),
                end_line=int(end_line) if end_line is not None else None,
            )
        )
    return funcs_by_path


def _backfill_test_goids(
    con: duckdb.DuckDBPyConnection,
    cfg: TestCoverageConfig,
) -> tuple[dict[str, int], dict[str, str]]:
    """
    Try to map test_catalog entries to GOIDs and update catalog rows.

    Returns
    -------
    tuple[dict[str, int], dict[str, str]]
        Mappings from test_id to GOID h128 and URN.
    """
    tests_rows = con.execute(
        """
        SELECT test_id, rel_path, qualname
        FROM analytics.test_catalog
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()
    if not tests_rows:
        return {}, {}

    goid_rows = con.execute(
        """
        SELECT goid_h128, urn, rel_path, qualname
        FROM core.goids
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()
    if not goid_rows:
        return {}, {}

    goid_index: dict[tuple[str, str], tuple[int, str]] = {
        (str(rel_path).replace("\\", "/"), str(qualname)): (int(goid_h128), str(urn))
        for goid_h128, urn, rel_path, qualname in goid_rows
    }

    goid_by_id: dict[str, int] = {}
    urn_by_id: dict[str, str] = {}
    updates: list[tuple[int, str, str, str]] = []

    for test_id_raw, rel_path_raw, qualname in tests_rows:
        normalized = None if qualname is None else str(qualname).replace("::", ".")
        if normalized is None:
            continue
        test_id = str(test_id_raw)
        rel_path = str(rel_path_raw).replace("\\", "/")
        hit = goid_index.get((rel_path, normalized))
        if hit:
            goid, urn = hit
            goid_by_id[test_id] = goid
            urn_by_id[test_id] = urn
            updates.append((goid, urn, test_id, rel_path))

    if updates:
        con.executemany(
            """
            UPDATE analytics.test_catalog
            SET test_goid_h128 = ?, urn = ?
            WHERE test_id = ? AND rel_path = ?
              AND repo = ? AND commit = ?
            """,
            [(g, u, tid, rel, cfg.repo, cfg.commit) for g, u, tid, rel in updates],
        )

    return goid_by_id, urn_by_id


def backfill_test_goids_for_catalog(
    con: duckdb.DuckDBPyConnection, cfg: TestCoverageConfig
) -> tuple[dict[str, int], dict[str, str]]:
    """
    Public wrapper to backfill GOIDs and URNs for tests in test_catalog.

    Returns
    -------
    tuple[dict[str, int], dict[str, str]]
        Mappings from test_id to GOID h128 and URN.
    """
    return _backfill_test_goids(con, cfg)


def build_edges_for_file_for_tests(
    file_funcs: list[FunctionRow],
    statements_set: set[int],
    contexts_by_lineno: dict[int, set[str]],
    rel_path: str,
    ctx: EdgeContext,
) -> list[tuple]:
    """
    Exposed wrapper around `_edges_for_file` for unit testing.

    Parameters
    ----------
    file_funcs:
        Functions within the file (dicts with GOID metadata).
    statements_set:
        Executable statement lines for the file.
    contexts_by_lineno:
        Mapping of line numbers to coverage contexts (pytest nodeids).
    rel_path:
        Repo-relative file path.
    ctx:
        EdgeContext holding repo metadata and test mappings.

    Returns
    -------
    list[tuple]
        Edge tuples mirroring analytics.test_coverage_edges schema.
    """
    return _edges_for_file(
        file_funcs=file_funcs,
        statements_set=statements_set,
        contexts_by_lineno=contexts_by_lineno,
        rel_path=rel_path,
        ctx=ctx,
    )


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
    file_funcs: list[FunctionRow],
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
            test_goid, test_urn = ctx.test_meta_by_id.get(test_id, (None, None))
            edges.append(
                (
                    test_id,
                    test_goid,
                    int(info["goid_h128"]),
                    test_urn or info["urn"],
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
    test_goid_by_id, test_urn_by_id = _backfill_test_goids(con, cfg)
    test_meta_by_id = {
        test_id: (test_goid_by_id.get(test_id), test_urn_by_id.get(test_id))
        for test_id in set(status_by_test.keys())
        | set(test_goid_by_id.keys())
        | set(test_urn_by_id.keys())
    }

    con.execute(
        "DELETE FROM analytics.test_coverage_edges WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    edge_ctx = EdgeContext(
        status_by_test=status_by_test,
        cfg=cfg,
        now=datetime.now(UTC),
        test_meta_by_id=test_meta_by_id,
    )
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
