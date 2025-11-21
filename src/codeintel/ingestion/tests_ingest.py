"""Ingest pytest JSON reports into `analytics.test_catalog`."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)


def _find_default_report(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "pytest-report.json",
        repo_root / "tests" / "pytest-report.json",
        repo_root / "build" / "pytest-report.json",
        repo_root / ".pytest-report.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_tests_from_report(report_path: Path) -> list[dict]:
    with report_path.open("r", encoding="utf8") as f:
        data = json.load(f)

    # pytest-json-report usually has tests at top-level "tests"
    tests = data.get("tests")
    if tests is None and "report" in data:
        tests = data["report"].get("tests")

    if not isinstance(tests, list):
        log.warning("Unexpected pytest report format; 'tests' missing or not a list")
        return []

    return tests


def _nodeid_to_path_and_qualname(nodeid: str) -> tuple[str, str | None]:
    """
    Split a pytest nodeid into a path and qualified test name.

    The input looks like `tests/test_app.py::TestFoo::test_bar[param]` and the
    result is a `(rel_path, qualname)` tuple.
    """
    parts = nodeid.split("::")
    rel_path = parts[0]
    qualname = "::".join(parts[1:]) if len(parts) > 1 else None
    return rel_path, qualname


def ingest_tests(
    con: duckdb.DuckDBPyConnection,
    repo_root: Path,
    repo: str,
    commit: str,
    *,
    pytest_report_path: Path | None = None,
) -> None:
    """
    Ingest a pytest JSON report into analytics.test_catalog.

    This step does NOT compute test_coverage_edges; those are derived
    later in an analytics step by combining coverage contexts with GOIDs.
    """
    repo_root = repo_root.resolve()
    if pytest_report_path is None:
        pytest_report_path = _find_default_report(repo_root)

    if pytest_report_path is None or not pytest_report_path.is_file():
        log.warning("Pytest JSON report not found; skipping test_catalog ingestion")
        return

    tests = _load_tests_from_report(pytest_report_path)
    if not tests:
        log.warning("No tests found in pytest report %s", pytest_report_path)
        return

    con.execute("DELETE FROM analytics.test_catalog WHERE repo = ? AND commit = ?", [repo, commit])

    insert_sql = """
        INSERT INTO analytics.test_catalog (
            test_id, test_goid_h128, urn,
            repo, commit, rel_path, qualname,
            kind, status, duration_ms, markers,
            parametrized, flaky, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    now = datetime.now(UTC)

    for t in tests:
        nodeid = t.get("nodeid")
        if not nodeid:
            continue

        rel_path, qualname = _nodeid_to_path_and_qualname(nodeid)

        status = t.get("outcome") or t.get("status") or "unknown"
        call = t.get("call") or {}
        duration_s = call.get("duration") or 0.0
        duration_ms = float(duration_s) * 1000.0

        keywords = t.get("keywords") or {}
        markers = sorted([k for k, v in keywords.items() if v])

        parametrized = "[" in nodeid and "]" in nodeid
        flaky = "flaky" in markers

        kind = "parametrized_case" if parametrized else "function"

        con.execute(
            insert_sql,
            [
                nodeid,          # test_id
                None,            # test_goid_h128 (filled later)
                None,            # urn (filled later)
                repo,
                commit,
                rel_path,
                qualname,
                kind,
                status,
                duration_ms,
                markers,
                parametrized,
                flaky,
                now,
            ],
        )

    log.info("test_catalog ingested from %s for %s@%s", pytest_report_path, repo, commit)
