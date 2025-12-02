"""Test results ingestion step with port injection.

This module provides a pure domain logic implementation for ingesting
pytest test results, using ports for all I/O operations.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.ingestion.steps.base import StepResult

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort

log = logging.getLogger(__name__)


class TestsIngestStep:
    """Test results ingestion step with port injection.

    This step ingests pytest JSON reports,
    using ports for all I/O operations.

    Parameters
    ----------
    storage
        Storage port for persisting data.
    """

    def __init__(
        self,
        storage: IngestStoragePort,
    ) -> None:
        """Initialize the step.

        Parameters
        ----------
        storage
            Storage port for persisting data.
        """
        self._storage = storage

    def execute(
        self,
        _modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
        json_report_path: Path,
    ) -> StepResult:
        """Execute test results ingestion.

        Parameters
        ----------
        _modules
            Modules for reference (not directly used).
        repo
            Repository identifier.
        commit
            Commit identifier.
        json_report_path
            Path to the pytest JSON report.

        Returns
        -------
        StepResult
            Execution result with row counts.
        """
        created_at = datetime.now(UTC)

        # Parse JSON report
        if not json_report_path.exists():
            log.warning("Test report not found: %s", json_report_path)
            return StepResult.skip("Test report not found")

        try:
            data = json.loads(json_report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to read test report: %s", exc)
            return StepResult.fail(f"Failed to read test report: {exc}")

        # Build rows
        all_rows: list[list[object]] = []
        tests = data.get("tests", [])

        for test in tests:
            nodeid = test.get("nodeid", "")
            outcome = test.get("outcome", "unknown")
            duration = test.get("duration", 0.0)
            longrepr = test.get("longrepr")

            # Extract file path from nodeid
            rel_path = nodeid.split("::")[0] if "::" in nodeid else nodeid

            all_rows.append([
                repo,
                commit,
                nodeid,
                rel_path,
                outcome,
                duration,
                longrepr[:1000] if longrepr else None,  # Truncate long repr
                created_at,
            ])

        # Persist rows
        table_counts: dict[str, int] = {}
        total_rows = 0

        if all_rows:
            scope = f"{repo}@{commit}"
            result = self._storage.write_batch("core.test_results", all_rows, scope=scope)
            table_counts["core.test_results"] = result.rows_written
            total_rows = result.rows_written

        # Also persist summary
        summary = data.get("summary", {})
        summary_rows: list[list[object]] = [[
            repo,
            commit,
            summary.get("passed", 0),
            summary.get("failed", 0),
            summary.get("skipped", 0),
            summary.get("error", 0),
            summary.get("duration", 0.0),
            created_at,
        ]]

        if summary_rows:
            result = self._storage.write_batch("core.test_summary", summary_rows)
            table_counts["core.test_summary"] = result.rows_written
            total_rows += result.rows_written

        log.info(
            "Tests ingest: repo=%s commit=%s tests=%d",
            repo,
            commit,
            len(tests),
        )

        return StepResult(
            rows_written=total_rows,
            table_counts=table_counts,
        )


__all__ = ["TestsIngestStep"]
