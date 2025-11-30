"""Ingest pytest JSON reports into `analytics.test_catalog`."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from codeintel.config import TestsIngestStepConfig
from codeintel.config.models import ToolsConfig
from codeintel.core.types import (
    PytestTestEntry,
    normalize_pytest_entry,
    validate_pytest_entry,
)
from codeintel.ingestion.change_tracker import (
    ChangeTracker,
    IncrementalIngestOps,
    run_incremental_ingest,
)
from codeintel.ingestion.common import ModuleRecord, run_batch, should_skip_missing_file
from codeintel.ingestion.tool_runner import ToolExecutionError, ToolNotFoundError, ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import TestCatalogRowModel, serialize_test_catalog_row

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
    """
    Load pytest-json-report entries, normalizing and filtering invalid rows.

    Returns
    -------
    list[PytestTestEntry]
        Normalized test entries (invalid entries skipped).
    """
    with report_path.open("r", encoding="utf8") as f:
        data = json.load(f)

    # pytest-json-report usually has tests at top-level "tests"
    tests = data.get("tests")
    if tests is None and "report" in data:
        tests = data["report"].get("tests")

    if not isinstance(tests, list):
        log.warning("Unexpected pytest report format; 'tests' missing or not a list")
        return []

    normalized: list[PytestTestEntry] = []
    skipped = 0
    for raw in tests:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        entry = normalize_pytest_entry(raw)
        if entry is None:
            skipped += 1
            continue
        try:
            validate_pytest_entry(entry)
        except ValueError as exc:
            log.debug("Skipping invalid pytest entry: %s", exc)
            skipped += 1
            continue
        normalized.append(entry)
    if skipped:
        log.warning("Skipped %d invalid pytest entries from %s", skipped, report_path)
    return normalized


def load_tests_from_report(report_path: Path) -> list[PytestTestEntry]:
    """
    Public wrapper around _load_tests_from_report for reuse.

    Returns
    -------
    list[PytestTestEntry]
        Normalized test entries (invalid entries skipped).
    """
    return _load_tests_from_report(report_path)


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
    call = test.get("call") if isinstance(test.get("call"), dict) else {}
    duration_s = call.get("duration") if isinstance(call, dict) else None
    duration_ms = float(duration_s) * 1000.0 if isinstance(duration_s, (int, float)) else 0.0

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


def _collect_test_rows(
    cfg: TestsIngestStepConfig,
    *,
    report_path: Path | None,
    tool_service: ToolService | None,
    created_at: datetime,
) -> tuple[list[TestCatalogRowModel], Path | None]:
    """
    Collect normalized test rows from a pytest JSON report.

    Returns
    -------
    tuple[list[TestCatalogRowModel], Path | None]
        Rows ready for ingestion and the resolved report path (may be None).
    """
    repo_root = cfg.repo_root
    pytest_report_path = report_path or cfg.pytest_report_path or _find_default_report(repo_root)
    service = tool_service
    if service is None:
        active_tools = ToolsConfig.model_validate({})
        runner = ToolRunner(
            tools_config=active_tools, cache_dir=repo_root / "build" / ".tool_cache"
        )
        service = ToolService(runner, active_tools)

    if pytest_report_path is None:
        pytest_report_path = repo_root / "build" / "test-results" / "pytest-report.json"

    if not pytest_report_path.is_file():
        try:
            asyncio.run(
                service.run_pytest_report(
                    repo_root,
                    json_report_path=pytest_report_path,
                )
            )
        except (ToolExecutionError, ToolNotFoundError) as exc:
            log.warning("pytest report generation failed: %s", exc)
            return [], pytest_report_path

    if pytest_report_path is None or should_skip_missing_file(
        pytest_report_path, logger=log, label="pytest JSON report"
    ):
        return [], pytest_report_path

    tests = _load_tests_from_report(pytest_report_path)
    log.info("Loaded %d pytest entries from %s", len(tests), pytest_report_path)
    if not tests:
        log.warning("No tests found in pytest report %s", pytest_report_path)
        return [], pytest_report_path

    rows: list[TestCatalogRowModel] = []
    for test in tests:
        row = _build_row(test)
        if row is not None:
            rows.append(row.to_row(cfg.repo, cfg.commit, created_at))

    if not rows:
        log.warning("No valid tests found in pytest report %s", pytest_report_path)
        return [], pytest_report_path

    return rows, pytest_report_path


@dataclass
class TestsIngestOps(IncrementalIngestOps[TestCatalogRowModel]):
    """Implement incremental ingest operations for analytics.test_catalog."""

    cfg: TestsIngestStepConfig
    rows_by_path: dict[str, list[TestCatalogRowModel]]
    dataset_name: str = field(init=False, default="analytics.test_catalog")

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """
        Restrict ingestion to test modules.

        Returns
        -------
        bool
            True when the module is within the tests/ path.
        """
        return module.rel_path.startswith("tests/")

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """Delete rows for removed test modules."""
        if rel_paths:
            gateway.con.execute(
                """
                DELETE FROM analytics.test_catalog
                WHERE repo = ? AND commit = ? AND rel_path IN (SELECT * FROM UNNEST(?))
                """,
                [self.cfg.repo, self.cfg.commit, list(rel_paths)],
            )
            return

        run_batch(
            gateway,
            "analytics.test_catalog",
            [],
            delete_params=[self.cfg.repo, self.cfg.commit],
        )

    def process_module(self, module: ModuleRecord) -> Iterable[TestCatalogRowModel]:
        """
        Return test rows for the specified module path.

        Returns
        -------
        Iterable[TestCatalogRowModel]
            Rows previously grouped by relative path.
        """
        return self.rows_by_path.get(module.rel_path, [])

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[TestCatalogRowModel]) -> None:
        """Insert test rows for changed modules."""
        if not rows:
            return

        run_batch(
            gateway,
            "analytics.test_catalog",
            [serialize_test_catalog_row(row) for row in rows],
            delete_params=None,
            scope=f"{self.cfg.repo}@{self.cfg.commit}",
        )


def ingest_tests(
    gateway: StorageGateway,
    cfg: TestsIngestStepConfig,
    report_path: Path | None = None,
    *,
    tool_service: ToolService | None = None,
    tracker: ChangeTracker | None = None,
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
    report_path:
        Optional explicit path to write a pytest JSON report.
    tool_service:
        Optional ToolService for running pytest; constructed from runner/tools when missing.
    tracker :
        Optional change tracker enabling incremental ingestion.
    """
    created_at = datetime.now(UTC)
    rows, resolved_report = _collect_test_rows(
        cfg,
        report_path=report_path,
        tool_service=tool_service,
        created_at=created_at,
    )

    if tracker is not None:
        rows_by_path: dict[str, list[TestCatalogRowModel]] = {}
        for row in rows:
            rows_by_path.setdefault(row["rel_path"], []).append(row)

        ops = TestsIngestOps(cfg=cfg, rows_by_path=rows_by_path)
        run_incremental_ingest(tracker, ops)
        log.info(
            "test_catalog ingested incrementally for %s@%s rows=%d",
            cfg.repo,
            cfg.commit,
            sum(len(bucket) for bucket in rows_by_path.values()),
        )
        return

    if not rows:
        log.warning("No valid tests found in pytest report %s", resolved_report)
        return

    run_batch(
        gateway,
        "analytics.test_catalog",
        [serialize_test_catalog_row(row) for row in rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info(
        "test_catalog ingested from %s for %s@%s",
        resolved_report,
        cfg.repo,
        cfg.commit,
    )
