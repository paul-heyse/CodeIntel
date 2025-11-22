"""Ingest pytest JSON reports into `analytics.test_catalog`."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from codeintel.config.models import TestsIngestConfig
from codeintel.config.schemas.sql_builder import ensure_schema, prepared_statements
from codeintel.types import PytestTestEntry

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


def _load_tests_from_report(report_path: Path) -> list[PytestTestEntry]:
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

    Parameters
    ----------
    nodeid : str
        Pytest node identifier to split.

    Returns
    -------
    tuple[str, str | None]
        Relative path and qualified name (None when missing).
    """
    parts = nodeid.split("::")
    rel_path = parts[0]
    qualname = "::".join(parts[1:]) if len(parts) > 1 else None
    return rel_path, qualname


@dataclass(frozen=True)
class TestCatalogRow:
    """Normalized representation of a pytest test case."""

    test_id: str
    rel_path: str
    qualname: str | None
    status: str
    duration_ms: float
    markers: list[str]
    parametrized: bool
    flaky: bool

    @property
    def kind(self) -> str:
        """Return canonical test kind based on parametrization."""
        return "parametrized_case" if self.parametrized else "function"

    def to_params(self, repo: str, commit: str, created_at: datetime) -> list[object]:
        """
        Convert the row into parameter order matching analytics.test_catalog.

        Parameters
        ----------
        repo : str
            Repository name for scoping.
        commit : str
            Commit hash for scoping.
        created_at : datetime
            Timestamp for the ingestion event.

        Returns
        -------
        list[object]
            Parameter list aligned with analytics.test_catalog schema.
        """
        return [
            self.test_id,
            None,  # test_goid_h128 (filled later)
            None,  # urn (filled later)
            repo,
            commit,
            self.rel_path,
            self.qualname,
            self.kind,
            self.status,
            self.duration_ms,
            self.markers,
            self.parametrized,
            self.flaky,
            created_at,
        ]


def _build_row(test: PytestTestEntry) -> TestCatalogRow | None:
    """
    Build a TestCatalogRow from a pytest JSON test entry.

    Parameters
    ----------
    test : PytestTestEntry
        Raw test entry from pytest-json-report.

    Returns
    -------
    TestCatalogRow | None
        Normalized row when nodeid is present, otherwise None.
    """
    nodeid = test.get("nodeid")
    if not nodeid:
        return None

    rel_path, qualname = _nodeid_to_path_and_qualname(nodeid)

    status = test.get("outcome") or test.get("status") or "unknown"
    call = test.get("call") or {}
    duration_s = call.get("duration") or 0.0
    duration_ms = float(duration_s) * 1000.0

    keywords = test.get("keywords") or {}
    markers = sorted([k for k, v in keywords.items() if v])

    parametrized = "[" in nodeid and "]" in nodeid
    flaky = "flaky" in markers

    return TestCatalogRow(
        test_id=nodeid,
        rel_path=rel_path,
        qualname=qualname,
        status=status,
        duration_ms=duration_ms,
        markers=markers,
        parametrized=parametrized,
        flaky=flaky,
    )


def ingest_tests(
    con: duckdb.DuckDBPyConnection,
    cfg: TestsIngestConfig,
) -> None:
    """
    Ingest a pytest JSON report into analytics.test_catalog.

    This step does NOT compute test_coverage_edges; those are derived
    later in an analytics step by combining coverage contexts with GOIDs.
    """
    repo_root = cfg.repo_root
    pytest_report_path = cfg.pytest_report_path or _find_default_report(repo_root)

    if pytest_report_path is None or not pytest_report_path.is_file():
        log.warning("Pytest JSON report not found; skipping test_catalog ingestion")
        return

    tests = _load_tests_from_report(pytest_report_path)
    if not tests:
        log.warning("No tests found in pytest report %s", pytest_report_path)
        return

    ensure_schema(con, "analytics.test_catalog")
    test_stmt = prepared_statements("analytics.test_catalog")
    if test_stmt.delete_sql is not None:
        con.execute(test_stmt.delete_sql, [cfg.repo, cfg.commit])

    now = datetime.now(UTC)
    rows: list[TestCatalogRow] = []
    for test in tests:
        row = _build_row(test)
        if row is not None:
            rows.append(row)

    if not rows:
        log.warning("No valid tests found in pytest report %s", pytest_report_path)
        return

    con.executemany(
        test_stmt.insert_sql,
        [row.to_params(cfg.repo, cfg.commit, now) for row in rows],
    )

    log.info("test_catalog ingested from %s for %s@%s", pytest_report_path, cfg.repo, cfg.commit)
