"""Test catalog ingestion step with port injection.

This module provides a pure domain logic implementation for ingesting
pytest test metadata into analytics.test_catalog, using ports for all
I/O operations.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.ingestion.compute.base import ExecutionResult
from codeintel.ingestion.engine.results import parse_test_duration, parse_test_markers

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort

log = logging.getLogger(__name__)
TEST_CATALOG_TABLE_KEY = "analytics.test_catalog"


def _build_test_catalog_rows(
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
        if not nodeid:
            continue
        outcome = str(test.get("outcome", test.get("status", "unknown")))
        duration_s = parse_test_duration(test)
        if duration_s == 0.0:
            raw_duration = test.get("duration")
            if isinstance(raw_duration, (int, float)):
                duration_s = float(raw_duration)
        duration_ms = duration_s * 1000.0
        markers = parse_test_markers(test)
        rel_path = nodeid.split("::", maxsplit=1)[0] if "::" in nodeid else nodeid
        qualname = nodeid.split("::", maxsplit=1)[1] if "::" in nodeid else None
        parametrized = "[" in nodeid or "parametrize" in markers
        flaky = "flaky" in markers

        rows.append(
            serializer(
                {
                    "test_id": nodeid,
                    "test_goid_h128": None,
                    "urn": None,
                    "repo": repo,
                    "commit": commit,
                    "rel_path": rel_path,
                    "qualname": qualname,
                    "kind": "test",
                    "status": outcome,
                    "duration_ms": duration_ms,
                    "markers": list(markers),
                    "parametrized": parametrized,
                    "flaky": flaky,
                    "created_at": created_at,
                }
            )
        )
    return rows


class TestsIngestStep:
    """Test catalog ingestion step with port injection.

    This step ingests pytest JSON reports into analytics.test_catalog, using
    ports for all I/O operations.

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
        """Execute test catalog ingestion.

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
        catalog_serializer = row_serializer_for_table_key(TEST_CATALOG_TABLE_KEY)

        all_rows = _build_test_catalog_rows(
            tests,
            repo=repo,
            commit=commit,
            created_at=created_at,
            serializer=catalog_serializer,
        )

        table_counts: dict[str, int] = {}

        if all_rows:
            scope = f"{repo}@{commit}"
            result = self._storage.write_batch(TEST_CATALOG_TABLE_KEY, all_rows, scope=scope)
            table_counts[TEST_CATALOG_TABLE_KEY] = result.rows_affected

        log.info(
            "Test catalog ingest: repo=%s commit=%s tests=%d",
            repo,
            commit,
            len(tests),
        )

        return ExecutionResult.ok(table_counts=table_counts)


__all__ = ["TestsIngestStep"]
