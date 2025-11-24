"""Ingest pytest JSON reports into `analytics.test_catalog`."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from codeintel.config.models import TestsIngestConfig
from codeintel.ingestion.common import run_batch, should_skip_missing_file
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.models.rows import TestCatalogRowModel, test_catalog_row_to_tuple
from codeintel.storage.gateway import StorageGateway
from codeintel.types import PytestTestEntry

log = logging.getLogger(__name__)


def _find_default_report(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "pytest-report.json",
        repo_root / "tests" / "pytest-report.json",
        repo_root / "build" / "pytest-report.json",
        repo_root / "build" / "test-results" / "pytest-report.json",
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


def _resolve_report_path(
    cfg: TestsIngestConfig, runner: ToolRunner | None, report_path: Path | None
) -> Path | None:
    """
    Determine the pytest report path, generating one if a runner is provided.

    Returns
    -------
    Path | None
        Path to an existing report or None when unavailable.
    """
    repo_root = cfg.repo_root
    default_report = cfg.pytest_report_path or _find_default_report(repo_root)
    pytest_report_path = report_path or default_report

    if runner is None:
        return pytest_report_path

    def _generate_report(target: Path, tool_runner: ToolRunner) -> Path | None:
        target.parent.mkdir(parents=True, exist_ok=True)
        result = tool_runner.run(
            "pytest",
            [
                "pytest",
                "--json-report",
                f"--json-report-file={target}",
            ],
            cwd=repo_root,
            output_path=target,
        )
        if result.returncode != 0:
            log.warning("pytest report generation failed (code %s)", result.returncode)
        return target if target.is_file() else None

    if pytest_report_path is None:
        target = default_report or repo_root / "build" / "test-results" / "pytest-report.json"
        return _generate_report(target, runner)

    if not pytest_report_path.is_file():
        log.info("pytest report missing at %s; attempting generation", pytest_report_path)
        return _generate_report(pytest_report_path, runner)

    return pytest_report_path


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

    def to_row(self, repo: str, commit: str, created_at: datetime) -> TestCatalogRowModel:
        """
        Convert the row into a typed dict matching analytics.test_catalog.

        Returns
        -------
        TestCatalogRowModel
            Typed representation aligned with test_catalog schema.
        """
        return TestCatalogRowModel(
            test_id=self.test_id,
            test_goid_h128=None,
            urn=None,
            repo=repo,
            commit=commit,
            rel_path=self.rel_path,
            qualname=self.qualname,
            kind=self.kind,
            status=self.status,
            duration_ms=self.duration_ms,
            markers=self.markers,
            parametrized=self.parametrized,
            flaky=self.flaky,
            created_at=created_at,
        )


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
    markers: list[str]
    if isinstance(keywords, dict):
        markers = sorted([k for k, v in keywords.items() if v])
    elif isinstance(keywords, list):
        markers = sorted([str(k) for k in keywords])
    else:
        log.debug("Unexpected keywords payload type %s for nodeid %s", type(keywords), nodeid)
        markers = []

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
    gateway: StorageGateway,
    cfg: TestsIngestConfig,
    runner: ToolRunner | None = None,
    report_path: Path | None = None,
) -> None:
    """
    Ingest a pytest JSON report into analytics.test_catalog.

    This step does NOT compute test_coverage_edges; those are derived
    later in an analytics step by combining coverage contexts with GOIDs.

    Parameters
    ----------
    gateway:
        StorageGateway providing access to the DuckDB database.
    cfg:
        Tests ingestion configuration (paths and identifiers).
    runner:
        Optional ToolRunner for generating a pytest JSON report when one is missing.
    report_path:
        Optional explicit path to write a pytest JSON report.
    """
    con = gateway.con
    pytest_report_path = _resolve_report_path(cfg, runner, report_path)

    if pytest_report_path is None or should_skip_missing_file(
        pytest_report_path, logger=log, label="pytest JSON report"
    ):
        return

    tests = _load_tests_from_report(pytest_report_path)
    if not tests:
        log.warning("No tests found in pytest report %s", pytest_report_path)
        return

    now = datetime.now(UTC)
    rows: list[TestCatalogRow] = []
    for test in tests:
        row = _build_row(test)
        if row is not None:
            rows.append(row)

    if not rows:
        log.warning("No valid tests found in pytest report %s", pytest_report_path)
        return

    run_batch(
        gateway,
        "analytics.test_catalog",
        [test_catalog_row_to_tuple(row.to_row(cfg.repo, cfg.commit, now)) for row in rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info("test_catalog ingested from %s for %s@%s", pytest_report_path, cfg.repo, cfg.commit)
