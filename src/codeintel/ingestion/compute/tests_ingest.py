"""Test results ingestion step with port injection.

This module provides a pure domain logic implementation for ingesting
pytest test results, using ports for all I/O operations.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.ingestion.compute.base import ExecutionResult

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort

log = logging.getLogger(__name__)
TEST_RESULTS_TABLE_KEY = "core.test_results"
TEST_SUMMARY_TABLE_KEY = "core.test_summary"


def _build_test_result_rows(
    tests: Sequence[Mapping[str, object]],
    *,
    repo: str,
    commit: str,
    created_at: datetime,
    serializer: Callable[[Mapping[str, object]], tuple[object, ...]],
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for test in tests:
        nodeid = str(test.get("nodeid", ""))
        outcome = str(test.get("outcome", "unknown"))
        duration = test.get("duration", 0.0)
        longrepr = test.get("longrepr")
        rel_path = nodeid.split("::", maxsplit=1)[0] if "::" in nodeid else nodeid

        rows.append(
            serializer(
                {
                    "repo": repo,
                    "commit": commit,
                    "nodeid": nodeid,
                    "rel_path": rel_path,
                    "outcome": outcome,
                    "duration": duration,
                    "longrepr": longrepr[:1000] if isinstance(longrepr, str) else None,
                    "created_at": created_at,
                }
            )
        )
    return rows


def _build_test_summary_rows(
    summary: Mapping[str, object],
    *,
    repo: str,
    commit: str,
    created_at: datetime,
    serializer: Callable[[Mapping[str, object]], tuple[object, ...]],
) -> list[tuple[object, ...]]:
    return [
        serializer(
            {
                "repo": repo,
                "commit": commit,
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "skipped": summary.get("skipped", 0),
                "error": summary.get("error", 0),
                "duration": summary.get("duration", 0.0),
                "created_at": created_at,
            }
        )
    ]


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
    ) -> ExecutionResult:
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
        ExecutionResult
            Execution result with row counts.
        """
        created_at = datetime.now(UTC)

        if not json_report_path.exists():
            log.warning("Test report not found: %s", json_report_path)
            return ExecutionResult.skip("Test report not found")

        try:
            data = json.loads(json_report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to read test report: %s", exc)
            return ExecutionResult.failed(f"Failed to read test report: {exc}")

        tests = data.get("tests", [])
        result_serializer = row_serializer_for_table_key(TEST_RESULTS_TABLE_KEY)
        summary_serializer = row_serializer_for_table_key(TEST_SUMMARY_TABLE_KEY)

        all_rows = _build_test_result_rows(
            tests,
            repo=repo,
            commit=commit,
            created_at=created_at,
            serializer=result_serializer,
        )

        table_counts: dict[str, int] = {}

        if all_rows:
            scope = f"{repo}@{commit}"
            result = self._storage.write_batch(TEST_RESULTS_TABLE_KEY, all_rows, scope=scope)
            table_counts[TEST_RESULTS_TABLE_KEY] = result.rows_affected

        summary = data.get("summary", {})
        summary_rows = _build_test_summary_rows(
            summary,
            repo=repo,
            commit=commit,
            created_at=created_at,
            serializer=summary_serializer,
        )

        if summary_rows:
            result = self._storage.write_batch(TEST_SUMMARY_TABLE_KEY, summary_rows)
            table_counts[TEST_SUMMARY_TABLE_KEY] = result.rows_affected

        log.info(
            "Tests ingest: repo=%s commit=%s tests=%d",
            repo,
            commit,
            len(tests),
        )

        return ExecutionResult.ok(table_counts=table_counts)


__all__ = ["TestsIngestStep"]
